"""
元Gen7 同士の新旧探索頂上決戦:
【旧】元Gen7 (深さ1探索・当時の仕様) vs 【新】いまのGen7 (深さ2探索・自滅手=-1.0)
先後入れ替え各500局（計1,000局）の並列対戦ベンチマークスクリプト
"""

import os
import sys
import warnings
import logging

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

repo_dir = os.path.dirname(os.path.abspath(__file__))
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)

from board import Board, state2vec
from player import ModelPlayer, load_keras_model_safe
from engine import GameEngine
from constants import PLAYER_1, PLAYER_2, RESULT_DRAW

class Depth1ModelPlayer:
    """当時（旧世代）の深さ1探索プレイヤーを完全再現"""
    def __init__(self, player_id, model_path_or_model, beta=10.0):
        self.player_id = player_id
        self.beta = beta
        if isinstance(model_path_or_model, str):
            self.model = load_keras_model_safe(model_path_or_model)
        else:
            self.model = model_path_or_model

    def choose_move(self, board, legal_moves):
        if not legal_moves:
            return None
        if len(legal_moves) == 1:
            return legal_moves[0]

        next_states = []
        immediate_win_indices = set()
        opponent_id = 3 - self.player_id

        for idx, move in enumerate(legal_moves):
            next_board = board.copy()
            GameEngine.make_move(next_board, move, self.player_id)
            winner = GameEngine.check_winner(next_board)
            if winner == self.player_id:
                immediate_win_indices.add(idx)
                next_states.append(None)
            else:
                next_state_vec = state2vec(next_board, opponent_id)
                next_states.append(next_state_vec)

        evals = np.zeros(len(legal_moves), dtype=np.float32)
        batch_states = []
        batch_indices = []

        for idx, state_vec in enumerate(next_states):
            if idx in immediate_win_indices:
                evals[idx] = 1.0  # 即勝ち手のみ 1.0
            else:
                batch_states.append(state_vec)
                batch_indices.append(idx)

        if batch_states:
            batch_array = np.array(batch_states, dtype=np.float32)
            preds = self.model(batch_array, training=False).numpy().flatten()
            for idx, pred in zip(batch_indices, preds):
                evals[idx] = -pred  # 相手局面評価の単純反転（相手の反撃は読まない）

        best_val = np.max(evals)
        shifted_evals = evals - best_val
        exp_vals = np.exp(self.beta * shifted_evals)
        probs = exp_vals / np.sum(exp_vals)

        choice_idx = np.random.choice(len(legal_moves), p=probs)
        return legal_moves[choice_idx]

def play_matchup_worker(worker_id, num_games, model_path, p1_type, p2_type, beta, seed):
    import os
    import sys
    import warnings
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')

    np.random.seed(seed)
    
    # プレイヤーの初期化
    if p1_type == "depth1":
        p1 = Depth1ModelPlayer(PLAYER_1, model_path, beta=beta)
    else:
        p1 = ModelPlayer(PLAYER_1, model_path_or_model=model_path, beta=beta)

    if p2_type == "depth1":
        p2 = Depth1ModelPlayer(PLAYER_2, model_path, beta=beta)
    else:
        p2 = ModelPlayer(PLAYER_2, model_path_or_model=model_path, beta=beta)

    players = {PLAYER_1: p1, PLAYER_2: p2}

    move_counts = []
    winners = []

    for _ in range(num_games):
        board = Board()
        board.setup()
        current_p = PLAYER_1
        max_moves = 200
        winner = None
        game_moves = 0

        for _ in range(max_moves):
            legal_moves = GameEngine.get_legal_moves(board, current_p)
            if not legal_moves:
                winner = 3 - current_p
                break
            move = players[current_p].choose_move(board, legal_moves)
            game_moves += 1
            GameEngine.make_move(board, move, current_p)

            winner = GameEngine.check_winner(board)
            if winner:
                break
            current_p = 3 - current_p
        else:
            winner = RESULT_DRAW

        move_counts.append(game_moves)
        winners.append(winner)

    return move_counts, winners

def run_single_series(m_path, p1_name, p1_type, p2_name, p2_type, num_games, beta=10.0, num_workers=6):
    print(f"\n=================================================================")
    print(f" 対戦開始: [先手] {p1_name} vs [後手] {p2_name} ({num_games} 局)")
    print(f"=================================================================")
    sys.stdout.flush()

    chunk_size = 25
    task_games = []
    remaining = num_games
    while remaining > 0:
        c = min(chunk_size, remaining)
        task_games.append(c)
        remaining -= c

    all_moves = []
    all_winners = []
    completed_games = 0
    last_reported = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(play_matchup_worker, i, n, m_path, p1_type, p2_type, beta, 10000 + i * 37)
            for i, n in enumerate(task_games)
        ]
        for f in as_completed(futures):
            moves, winners = f.result()
            all_moves.extend(moves)
            all_winners.extend(winners)
            completed_games += len(moves)

            while completed_games >= last_reported + 100 and last_reported + 100 <= num_games:
                last_reported += 100
                print(f"  進捗: {last_reported:4d} / {num_games} 局完了")
                sys.stdout.flush()

    if last_reported < num_games and completed_games == num_games:
        print(f"  進捗: {num_games:4d} / {num_games} 局完了")
        sys.stdout.flush()

    all_moves = np.array(all_moves)
    all_winners = np.array(all_winners)

    p1_wins = int(np.sum(all_winners == PLAYER_1))
    p2_wins = int(np.sum(all_winners == PLAYER_2))
    draws = int(np.sum(all_winners == RESULT_DRAW))

    p3 = int(np.sum(all_moves == 3))
    p4 = int(np.sum(all_moves == 4))
    p5 = int(np.sum(all_moves == 5))
    p6 = int(np.sum(all_moves == 6))

    print(f"\n[終了] {p1_name}(先手) {p1_wins}勝 ({p1_wins/num_games*100:.1f}%) | "
          f"{p2_name}(後手) {p2_wins}勝 ({p2_wins/num_games*100:.1f}%) | "
          f"引分 {draws} ({draws/num_games*100:.1f}%)")
    print(f"平均手数: {np.mean(all_moves):.2f} 手 | 3〜6手計: {p3+p4+p5+p6} 局 ({(p3+p4+p5+p6)/num_games*100:.2f}%)")
    sys.stdout.flush()

    return {
        "p1_name": p1_name, "p2_name": p2_name,
        "p1_wins": p1_wins, "p2_wins": p2_wins, "draws": draws,
        "moves": all_moves, "winners": all_winners,
        "p3": p3, "p4": p4, "p5": p5, "p6": p6,
        "num_games": num_games
    }

def main():
    parser = argparse.ArgumentParser(description="元Gen7 (深さ1) vs いまのGen7 (深さ2) 対戦スクリプト")
    parser.add_argument("--model", type=str, default="legacy_v1/animal_shogi_model_server_gen7.keras", help="元Gen7モデルパス")
    parser.add_argument("--games", type=int, default=500, help="各手番の対戦局数 (デフォルト: 500)")
    parser.add_argument("--beta", type=float, default=10.0, help="逆温度 beta")
    parser.add_argument("--workers", type=int, default=min(multiprocessing.cpu_count(), 6), help="並列プロセス数")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"エラー: モデル {args.model} が見つかりません。")
        sys.exit(1)

    print("=================================================================")
    print(" 【同一モデル知能対決】元Gen7(深さ1:旧) vs いまのGen7(深さ2:新)")
    print(f" 使用モデル: {args.model} (※重みパラメータは全く同一)")
    print(f" 各手番局数: {args.games} 局 (合計 {args.games * 2} 局)")
    print(f" 逆温度 beta: {args.beta}")
    print(f" 並列ワーカー: {args.workers}")
    print("=================================================================")
    sys.stdout.flush()

    # 1. パターンA: 先手[元Gen7(深さ1)] vs 後手[いまのGen7(深さ2)]
    res_A = run_single_series(args.model, "元Gen7(深さ1)", "depth1", "新Gen7(深さ2)", "depth2",
                              args.games, args.beta, args.workers)

    # 2. パターンB: 先手[いまのGen7(深さ2)] vs 後手[元Gen7(深さ1)]
    res_B = run_single_series(args.model, "新Gen7(深さ2)", "depth2", "元Gen7(深さ1)", "depth1",
                              args.games, args.beta, args.workers)

    # 総合集計
    total_games = args.games * 2
    d1_wins = res_A["p1_wins"] + res_B["p2_wins"]  # 元Gen7(深さ1)の通算勝利
    d2_wins = res_A["p2_wins"] + res_B["p1_wins"]  # 新Gen7(深さ2)の通算勝利
    total_draws = res_A["draws"] + res_B["draws"]

    all_moves = np.concatenate([res_A["moves"], res_B["moves"]])
    tot_p3 = res_A["p3"] + res_B["p3"]
    tot_p4 = res_A["p4"] + res_B["p4"]
    tot_p5 = res_A["p5"] + res_B["p5"]
    tot_p6 = res_A["p6"] + res_B["p6"]
    tot_3to6 = tot_p3 + tot_p4 + tot_p5 + tot_p6

    # 7手以上の実力勝負集計
    mask_7_A = res_A["moves"] >= 7
    d1_wins_7_A = int(np.sum((res_A["winners"] == PLAYER_1) & mask_7_A))
    d2_wins_7_A = int(np.sum((res_A["winners"] == PLAYER_2) & mask_7_A))

    mask_7_B = res_B["moves"] >= 7
    d2_wins_7_B = int(np.sum((res_B["winners"] == PLAYER_1) & mask_7_B))
    d1_wins_7_B = int(np.sum((res_B["winners"] == PLAYER_2) & mask_7_B))

    tot_games_7 = int(np.sum(mask_7_A)) + int(np.sum(mask_7_B))
    tot_d1_wins_7 = d1_wins_7_A + d1_wins_7_B
    tot_d2_wins_7 = d2_wins_7_A + d2_wins_7_B

    summary_text = f"""
=================================================================================
     対戦結果: 元Gen7 (深さ1: 当時の探索) vs いまのGen7 (深さ2: 最新探索)
=================================================================================
【対戦条件】
  使用モデル: {args.model} (※ニューラルネットの重みは100%同一)
  先後入れ替え 各 {args.games} 局 (計 {total_games} 局) / beta = {args.beta}

---------------------------------------------------------------------------------
(A) 先手:元Gen7(深さ1) vs 後手:いまのGen7(深さ2) ({args.games} 局)
    元Gen7(深さ1)(先手) 勝利 : {res_A['p1_wins']:4d} 局 ({res_A['p1_wins']/args.games*100:5.1f}%)
    いまのGen7(深さ2)(後手) 勝利 : {res_A['p2_wins']:4d} 局 ({res_A['p2_wins']/args.games*100:5.1f}%)
    引き分け                 : {res_A['draws']:4d} 局 ({res_A['draws']/args.games*100:5.1f}%)
    3〜6手即死決着 計        : {res_A['p3']+res_A['p4']+res_A['p5']+res_A['p6']:4d} 局 ({(res_A['p3']+res_A['p4']+res_A['p5']+res_A['p6'])/args.games*100:5.2f}%)

(B) 先手:いまのGen7(深さ2) vs 後手:元Gen7(深さ1) ({args.games} 局)
    いまのGen7(深さ2)(先手) 勝利 : {res_B['p1_wins']:4d} 局 ({res_B['p1_wins']/args.games*100:5.1f}%)
    元Gen7(深さ1)(後手) 勝利 : {res_B['p2_wins']:4d} 局 ({res_B['p2_wins']/args.games*100:5.1f}%)
    引き分け                 : {res_B['draws']:4d} 局 ({res_B['draws']/args.games*100:5.1f}%)
    3〜6手即死決着 計        : {res_B['p3']+res_B['p4']+res_B['p5']+res_B['p6']:4d} 局 ({(res_B['p3']+res_B['p4']+res_B['p5']+res_B['p6'])/args.games*100:5.2f}%)

---------------------------------------------------------------------------------
【通算総合結果 ({total_games} 局)】
    ★ いまのGen7 (深さ2: 最新探索) 通算勝利 : {d2_wins:4d} 局 ({d2_wins/total_games*100:5.2f}%)
    ★ 元Gen7     (深さ1: 当時探索) 通算勝利 : {d1_wins:4d} 局 ({d1_wins/total_games*100:5.2f}%)
    ★ 通算引き分け                          : {total_draws:4d} 局 ({total_draws/total_games*100:5.2f}%)
    ★ 全体平均手数                          : {np.mean(all_moves):.2f} 手 (中央値: {np.median(all_moves):.1f} 手)
    -----------------------------------------------------------------------------
    ★ 3〜6手決着数                          : {tot_3to6:4d} 局 ({tot_3to6/total_games*100:5.2f}%)
    -----------------------------------------------------------------------------
    【純粋な実力勝負】7手以上の対局 (計 {tot_games_7} 局)
    ★ いまのGen7 (深さ2) 勝利 : {tot_d2_wins_7:4d} 局 ({tot_d2_wins_7/tot_games_7*100:5.2f}%)
    ★ 元Gen7     (深さ1) 勝利 : {tot_d1_wins_7:4d} 局 ({tot_d1_wins_7/tot_games_7*100:5.2f}%)
=================================================================================
"""
    print(summary_text)

    summary_file = os.path.join(repo_dir, "summary_gen7_depth1_vs_depth2.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"[サマリー保存完了] {summary_file}")

if __name__ == "__main__":
    main()
