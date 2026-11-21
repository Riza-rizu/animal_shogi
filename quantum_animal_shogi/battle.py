"""
量子どうぶつしょうぎ 対戦・自己対局評価スクリプト (battle.py)
"""

import os
import sys
import warnings
import argparse
from collections import Counter
import numpy as np

# 不要な TensorFlow ログを抑制
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

repo_dir = os.path.dirname(os.path.abspath(__file__))
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)

from constants import PLAYER_1, PLAYER_2, RESULT_PLAYER1, RESULT_PLAYER2, RESULT_DRAW
from board import QuantumBoard, state2vec
from engine import GameEngine
from player import ModelPlayer, RandomPlayer


def play_single_game(p1, p2, max_moves=200, gamma=0.6, collect_data=False):
    """
    1対局を実行する。
    
    Returns:
        winner (int): PLAYER_1, PLAYER_2, または RESULT_DRAW
        total_moves (int): 総手数
        history_records (list): [(state_vec, player), ...] (collect_data=Trueの場合)
    """
    board = QuantumBoard()
    board.setup()
    players = {PLAYER_1: p1, PLAYER_2: p2}
    current_p = PLAYER_1

    game_moves = 0
    winner = None
    state_counts = Counter()
    history_records = []

    # 初期局面カウント
    init_key = (board.to_hashable(), current_p)
    state_counts[init_key] = 1

    while game_moves < max_moves:
        legal_moves = GameEngine.get_legal_moves(board, current_p)
        if not legal_moves:
            # 合法手なし＝負け
            winner = 3 - current_p
            break

        if collect_data:
            s_vec = state2vec(board, current_p)
            history_records.append((s_vec, current_p))

        player_obj = players[current_p]
        if isinstance(player_obj, ModelPlayer):
            move = player_obj.choose_move(board, legal_moves, history_counts=state_counts, gamma=gamma)
        else:
            move = player_obj.choose_move(board, legal_moves)

        GameEngine.make_move(board, move, current_p)
        game_moves += 1

        winner = GameEngine.check_winner(board, last_player=current_p)
        if winner is not None:
            break

        current_p = 3 - current_p
        state_key = (board.to_hashable(), current_p)
        state_counts[state_key] += 1

    if winner is None:
        winner = RESULT_DRAW

    return winner, game_moves, history_records


def run_tournament(m1_path, m2_path, num_games=100, beta=10.0, depth1=0, depth2=0, gamma=0.6, save_data=None):
    """
    先後を半分ずつ入れ替えて対戦を実行する。
    """
    print(f"\n==========================================")
    print(f"量子どうぶつしょうぎ 対戦開始 ({num_games} 局)")
    print(f"Model 1: {m1_path} (depth={depth1})")
    print(f"Model 2: {m2_path} (depth={depth2})")
    print(f"Beta: {beta}, Gamma: {gamma}")
    print(f"==========================================")

    # モデルプレイヤーの初期化
    p_m1_as_p1 = ModelPlayer(PLAYER_1, model_path_or_model=m1_path, beta=beta, depth=depth1)
    p_m2_as_p2 = ModelPlayer(PLAYER_2, model_path_or_model=m2_path, beta=beta, depth=depth2)

    p_m2_as_p1 = ModelPlayer(PLAYER_1, model_path_or_model=m2_path, beta=beta, depth=depth2)
    p_m1_as_p2 = ModelPlayer(PLAYER_2, model_path_or_model=m1_path, beta=beta, depth=depth1)

    half = num_games // 2
    m1_wins = 0
    m2_wins = 0
    draws = 0
    move_lengths = []

    all_dataset_rows = []
    collect_data = (save_data is not None)

    # 第1部: M1 先手, M2 後手
    for g in range(half):
        w, moves, records = play_single_game(p_m1_as_p1, p_m2_as_p2, gamma=gamma, collect_data=collect_data)
        move_lengths.append(moves)
        if w == PLAYER_1:
            m1_wins += 1
        elif w == PLAYER_2:
            m2_wins += 1
        else:
            draws += 1

        if collect_data and records:
            # 評価値の割り当て (tanh: +1.0 for P1 win, -1.0 for P2 win, 0.0 for Draw)
            final_eval = 1.0 if w == PLAYER_1 else (-1.0 if w == PLAYER_2 else 0.0)
            for s_vec, step_p in records:
                all_dataset_rows.append(np.append(s_vec, final_eval))

        if (g + 1) % 10 == 0:
            print(f"進捗: {g + 1}/{num_games} (M1先手: M1勝 {m1_wins}, M2勝 {m2_wins}, 引分 {draws})")

    # 第2部: M2 先手, M1 後手
    for g in range(half, num_games):
        w, moves, records = play_single_game(p_m2_as_p1, p_m1_as_p2, gamma=gamma, collect_data=collect_data)
        move_lengths.append(moves)
        if w == PLAYER_1:
            m2_wins += 1
        elif w == PLAYER_2:
            m1_wins += 1
        else:
            draws += 1

        if collect_data and records:
            final_eval = 1.0 if w == PLAYER_1 else (-1.0 if w == PLAYER_2 else 0.0)
            for s_vec, step_p in records:
                all_dataset_rows.append(np.append(s_vec, final_eval))

        if (g + 1) % 10 == 0:
            print(f"進捗: {g + 1}/{num_games} (累計: M1勝 {m1_wins}, M2勝 {m2_wins}, 引分 {draws})")

    avg_moves = np.mean(move_lengths) if move_lengths else 0
    print("\n================== 結果 ==================")
    print(f"総対局数: {num_games}")
    print(f"Model 1 勝利: {m1_wins} ({m1_wins / num_games * 100:.1f}%)")
    print(f"Model 2 勝利: {m2_wins} ({m2_wins / num_games * 100:.1f}%)")
    print(f"引き分け: {draws} ({draws / num_games * 100:.1f}%)")
    print(f"平均手数: {avg_moves:.1f} 手")
    print("==========================================")

    if save_data and all_dataset_rows:
        dataset_array = np.array(all_dataset_rows, dtype=np.float32)
        np.save(save_data, dataset_array)
        print(f"学習データを保存しました: {save_data} (総局面数: {dataset_array.shape[0]})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m1", type=str, default="quantum_model_gen0.keras", help="Model 1 のパス")
    parser.add_argument("--m2", type=str, default="quantum_model_gen0.keras", help="Model 2 のパス")
    parser.add_argument("--games", type=int, default=50, help="対局数")
    parser.add_argument("--beta", type=float, default=10.0, help="逆温度パラメータ")
    parser.add_argument("--depth1", type=int, default=0, help="Model 1 の探索深さ")
    parser.add_argument("--depth2", type=int, default=0, help="Model 2 の探索深さ")
    parser.add_argument("--gamma", type=float, default=0.6, help="レピティションペナルティ減衰率")
    parser.add_argument("--save_data", type=str, default=None, help="自己対局棋譜データの保存先 (.npy)")
    args = parser.parse_args()

    run_tournament(
        m1_path=args.m1,
        m2_path=args.m2,
        num_games=args.games,
        beta=args.beta,
        depth1=args.depth1,
        depth2=args.depth2,
        gamma=args.gamma,
        save_data=args.save_data
    )


if __name__ == "__main__":
    main()
