"""
量子どうぶつしょうぎ メイン実行ファイル (main.py)
研究室サーバー／バッチ並列シミュレーション対応
"""

import sys
import pickle
import os
import yaml
import numpy as np
from collections import Counter

repo_dir = os.path.dirname(os.path.abspath(__file__))
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)

from constants import (
    COLS, ROWS, PLAYER_1, PLAYER_2, RESULT_DRAW, RESULT_PLAYER1, RESULT_PLAYER2
)
from board import QuantumBoard, state2vec
from engine import GameEngine
from player import HumanPlayer, RandomPlayer, ModelPlayer, load_keras_model_safe


def to_bytes_key(vec_state):
    """
    187次元の float32 ベクトルを bytes 型に変換（辞書キーの省メモリ化）
    """
    return vec_state.astype(np.float32).tobytes()


def run_simulation(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    seed = config.get('seed', 0)
    games = config.get('games', 1000)
    alpha = config.get('alpha', 0.95)  # 割引率
    p1_type = config.get('p1', 'ai')
    p2_type = config.get('p2', 'ai')
    model_path = config.get('model_path', 'quantum_model_gen0.keras')
    beta = config.get('beta', 10.0)
    depth = config.get('depth', 0)
    gamma = config.get('gamma', 0.6)

    import random
    random.seed(seed)
    np.random.seed(seed)

    print(f"[{os.path.basename(yaml_path)}] シミュレーション開始: seed={seed}, games={games}, depth={depth}, model={model_path}")

    # モデルのロード
    model = None
    if p1_type == 'ai' or p2_type == 'ai':
        if os.path.exists(model_path):
            model = load_keras_model_safe(model_path)
        else:
            print(f"エラー: モデルファイルが見つかりません: {model_path}")
            sys.exit(1)

    # プレイヤーの初期化
    def get_player(p_type, p_id):
        if p_type == 'ai':
            return ModelPlayer(p_id, model_path_or_model=model, beta=beta, depth=depth)
        elif p_type == 'random':
            return RandomPlayer(p_id)
        else:
            return RandomPlayer(p_id)

    player1 = get_player(p1_type, PLAYER_1)
    player2 = get_player(p2_type, PLAYER_2)
    players = {PLAYER_1: player1, PLAYER_2: player2}

    # 局面データの蓄積用辞書: {bytes_key: [visits, value_sum]}
    data_dict = {}
    match_records = []

    p1_wins = 0
    p2_wins = 0
    draws = 0

    for g in range(games):
        board = QuantumBoard()
        board.setup()
        current_p = PLAYER_1

        game_history = []  # [(bytes_key, player_id), ...]
        state_counts = Counter()

        init_key = (board.to_hashable(), current_p)
        state_counts[init_key] = 1

        game_moves = 0
        max_moves = 200
        winner = None
        win_reason = "Draw_MaxMoves"
        game_moves_list = []
        game_moves_coords = []
        game_probs_list = []
        p1_collapse_ply = -1
        p2_collapse_ply = -1

        while game_moves < max_moves:
            s_vec = state2vec(board, current_p)
            b_key = to_bytes_key(s_vec)
            game_history.append((b_key, current_p))

            # 現在の全8駒の確率分布 (8, 5: L, G, E, C, H) を記録
            probs_matrix = board.get_all_piece_probs_array()
            game_probs_list.append(probs_matrix)

            # ライオン確定タイミングの検出
            if p1_collapse_ply == -1 and board.get_determined_lion_id(PLAYER_1) is not None:
                p1_collapse_ply = game_moves
            if p2_collapse_ply == -1 and board.get_determined_lion_id(PLAYER_2) is not None:
                p2_collapse_ply = game_moves

            legal_moves = GameEngine.get_legal_moves(board, current_p)
            if not legal_moves:
                winner = 3 - current_p
                win_reason = "NoLegalMoves"
                break

            p_obj = players[current_p]
            if isinstance(p_obj, ModelPlayer):
                move = p_obj.choose_move(board, legal_moves, history_counts=state_counts, gamma=gamma)
            else:
                move = p_obj.choose_move(board, legal_moves)

            game_moves_list.append(str(move))
            # 着手座標情報の記録 [from_c, from_r, to_c, to_r, owner, pid, is_drop]
            fc = -1 if move.is_drop else move.from_pos[0]
            fr = -1 if move.is_drop else move.from_pos[1]
            tc, tr = move.to_pos[0], move.to_pos[1]
            m_owner, m_pid = move.piece_token
            game_moves_coords.append([fc, fr, tc, tr, m_owner, m_pid, 1 if move.is_drop else 0])

            GameEngine.make_move(board, move, current_p)
            game_moves += 1

            winner, reason = GameEngine.check_winner_detail(board, last_player=current_p)
            if winner is not None:
                win_reason = reason
                break

            current_p = 3 - current_p
            state_key = (board.to_hashable(), current_p)
            state_counts[state_key] += 1

        if winner is None:
            winner = RESULT_DRAW
            win_reason = "Draw_MaxMoves"

        if winner == PLAYER_1:
            p1_wins += 1
            final_eval = 1.0
        elif winner == PLAYER_2:
            p2_wins += 1
            final_eval = -1.0
        else:
            draws += 1
            final_eval = 0.0

        match_records.append({
            "game_id": g + 1,
            "winner": winner,
            "moves": game_moves,
            "win_reason": win_reason,
            "p1_collapse_ply": p1_collapse_ply,
            "p2_collapse_ply": p2_collapse_ply,
            "is_short_game": (game_moves <= 6 and winner != RESULT_DRAW),
            "moves_10": game_moves_list[:10],
            "moves_coords": np.array(game_moves_coords, dtype=np.int8) if game_moves_coords else np.empty((0, 7), dtype=np.int8),
            "all_probs": np.array(game_probs_list, dtype=np.float16) if game_probs_list else np.empty((0, 8, 5), dtype=np.float16)
        })

        # モンテカルロ価値更新（その局面の手番プレイヤー step_p から見た勝敗評価値を蓄積）
        for step_i, (b_key, step_p) in enumerate(game_history):
            dist_from_end = len(game_history) - 1 - step_i
            discount = alpha ** dist_from_end

            if winner == RESULT_DRAW:
                step_eval = 0.0
            elif winner == step_p:
                step_eval = 1.0 * discount
            else:
                step_eval = -1.0 * discount

            if b_key not in data_dict:
                data_dict[b_key] = [0, 0.0]
            data_dict[b_key][0] += 1
            data_dict[b_key][1] += step_eval

        if (g + 1) % 100 == 0 or (g + 1) == games:
            print(f"[{os.path.basename(yaml_path)}] 進捗: {g + 1}/{games} 局完了 (P1: {p1_wins}, P2: {p2_wins}, 引分: {draws})", flush=True)

    # 成果物の保存
    output_filename = f"output{seed:03d}.pkl"
    with open(output_filename, 'wb') as f:
        pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    match_log_filename = f"match_log_{seed:03d}.pkl"
    with open(match_log_filename, 'wb') as f:
        pickle.dump(match_records, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[{os.path.basename(yaml_path)}] 完了: {output_filename} と {match_log_filename} を保存しました。")


def main():
    if len(sys.argv) == 2 and (sys.argv[1].endswith('.yaml') or sys.argv[1].endswith('.yml')):
        run_simulation(sys.argv[1])
        return

    import argparse
    parser = argparse.ArgumentParser(description="量子どうぶつしょうぎ メイン実行")
    parser.add_argument("--config", type=str, required=True, help="YAML設定ファイルのパス")
    args = parser.parse_args()
    run_simulation(args.config)


if __name__ == "__main__":
    main()
