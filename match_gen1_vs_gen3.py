"""
新旧世代頂上決戦: Gen1 (新アーキテクチャ) vs Gen3 (旧アーキテクチャ)
先後入れ替え各500局（計1,000局）の並列対戦評価スクリプト
（3手〜6手決着の完全詳細集計付き）
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

from board import Board
from player import ModelPlayer
from engine import GameEngine
from constants import PLAYER_1, PLAYER_2, RESULT_DRAW

def play_matchup_worker(worker_id, num_games, m1_path, m2_path, beta, seed):
    import os
    import sys
    import warnings
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')

    np.random.seed(seed)
    p1 = ModelPlayer(PLAYER_1, model_path_or_model=m1_path, beta=beta)
    p2 = ModelPlayer(PLAYER_2, model_path_or_model=m2_path, beta=beta)
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

def run_matchup(m1_name, m1_path, m2_name, m2_path, num_games, beta=10.0, num_workers=6):
    print(f"\n=================================================================")
    print(f" 対戦開始: [先手] {m1_name} vs [後手] {m2_name} ({num_games} 局)")
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
            executor.submit(play_matchup_worker, i, n, m1_path, m2_path, beta, 20000 + i * 23)
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

    p1_wins = np.sum(all_winners == PLAYER_1)
    p2_wins = np.sum(all_winners == PLAYER_2)
    draws = np.sum(all_winners == RESULT_DRAW)

    p3 = int(np.sum(all_moves == 3))
    p4 = int(np.sum(all_moves == 4))
    p5 = int(np.sum(all_moves == 5))
    p6 = int(np.sum(all_moves == 6))

    print(f"\n[対戦終了] {m1_name}(先手) {p1_wins}勝 ({p1_wins/num_games*100:.1f}%) | "
          f"{m2_name}(後手) {p2_wins}勝 ({p2_wins/num_games*100:.1f}%) | "
          f"引き分け {draws} ({draws/num_games*100:.1f}%)")
    print(f"平均手数: {np.mean(all_moves):.2f} 手 (中央値: {np.median(all_moves):.1f} 手)")
    print(f"超短手数: 3手={p3}局, 4手={p4}局, 5手={p5}局, 6手={p6}局 (3〜6手計: {p3+p4+p5+p6}局 / {(p3+p4+p5+p6)/num_games*100:.1f}%)")
    sys.stdout.flush()

    return {
        "p1_name": m1_name, "p2_name": m2_name,
        "p1_wins": p1_wins, "p2_wins": p2_wins, "draws": draws,
        "moves": all_moves,
        "p3": p3, "p4": p4, "p5": p5, "p6": p6,
        "num_games": num_games
    }

def main():
    parser = argparse.ArgumentParser(description="Gen1 (新) vs Gen3 (旧) 対戦スクリプト")
    parser.add_argument("--gen1", type=str, default="animal_shogi_model_gen1.keras", help="Gen1モデルパス")
    parser.add_argument("--gen3", type=str, default="legacy_v1/animal_shogi_model_server_gen3.keras", help="Gen3モデルパス")
    parser.add_argument("--games", type=int, default=500, help="各手番の対戦局数 (デフォルト: 500)")
    parser.add_argument("--beta", type=float, default=10.0, help="逆温度 beta (デフォルト: 10.0)")
    parser.add_argument("--workers", type=int, default=min(multiprocessing.cpu_count(), 6), help="並列プロセス数")
    args = parser.parse_args()

    # パス存在確認
    if not os.path.exists(args.gen1):
        print(f"エラー: Gen1 モデル {args.gen1} が見つかりません。")
        sys.exit(1)
    if not os.path.exists(args.gen3):
        print(f"エラー: Gen3 モデル {args.gen3} が見つかりません。")
        sys.exit(1)

    print("=================================================================")
    print(" 【新旧決戦】Gen1 (新CNN + Min-Max) vs Gen3 (旧全結合モデル)")
    print(f" Gen1 モデル: {args.gen1}")
    print(f" Gen3 モデル: {args.gen3}")
    print(f" 各手番局数 : {args.games} 局 (合計 {args.games * 2} 局)")
    print(f" 逆温度 beta: {args.beta}")
    print(f" 並列ワーカー: {args.workers}")
    print("=================================================================")
    sys.stdout.flush()

    # 1. パターンA: 先手 Gen1 vs 後手 Gen3
    res_1v3 = run_matchup("Gen1", args.gen1, "Gen3", args.gen3,
                          num_games=args.games, beta=args.beta, num_workers=args.workers)

    # 2. パターンB: 先手 Gen3 vs 後手 Gen1
    res_3v1 = run_matchup("Gen3", args.gen3, "Gen1", args.gen1,
                          num_games=args.games, beta=args.beta, num_workers=args.workers)

    # 3. 総合勝敗表
    total_g1_wins = res_1v3["p1_wins"] + res_3v1["p2_wins"]
    total_g3_wins = res_1v3["p2_wins"] + res_3v1["p1_wins"]
    total_draws = res_1v3["draws"] + res_3v1["draws"]
    total_games = args.games * 2

    all_moves_combined = np.concatenate([res_1v3["moves"], res_3v1["moves"]])
    tot_p3 = res_1v3["p3"] + res_3v1["p3"]
    tot_p4 = res_1v3["p4"] + res_3v1["p4"]
    tot_p5 = res_1v3["p5"] + res_3v1["p5"]
    tot_p6 = res_1v3["p6"] + res_3v1["p6"]
    tot_3to6 = tot_p3 + tot_p4 + tot_p5 + tot_p6

    summary_text = f"""
=================================================================================
                 新旧頂上決戦: Gen1 (新世代) vs Gen3 (旧世代) 総合勝敗表
=================================================================================

【対戦条件】
  先手後手入れ替え 各 {args.games} 局 (合計 {total_games} 局) / beta = {args.beta}

---------------------------------------------------------------------------------
(A) 先手:Gen1(新) vs 後手:Gen3(旧) ({args.games} 局)
    Gen1(先手) 勝利 : {res_1v3['p1_wins']:4d} 局 ({res_1v3['p1_wins']/args.games*100:5.1f}%)
    Gen3(後手) 勝利 : {res_1v3['p2_wins']:4d} 局 ({res_1v3['p2_wins']/args.games*100:5.1f}%)
    引き分け        : {res_1v3['draws']:4d} 局 ({res_1v3['draws']/args.games*100:5.1f}%)
    -----------------------------------------------------------------------------
    3手決着         : {res_1v3['p3']:4d} 局 ({res_1v3['p3']/args.games*100:5.2f}%)
    4手決着         : {res_1v3['p4']:4d} 局 ({res_1v3['p4']/args.games*100:5.2f}%)
    5手決着         : {res_1v3['p5']:4d} 局 ({res_1v3['p5']/args.games*100:5.2f}%)
    6手決着         : {res_1v3['p6']:4d} 局 ({res_1v3['p6']/args.games*100:5.2f}%)
    (3〜6手決着 計) : {res_1v3['p3']+res_1v3['p4']+res_1v3['p5']+res_1v3['p6']:4d} 局 ({(res_1v3['p3']+res_1v3['p4']+res_1v3['p5']+res_1v3['p6'])/args.games*100:5.2f}%)

(B) 先手:Gen3(旧) vs 後手:Gen1(新) ({args.games} 局)
    Gen1(後手) 勝利 : {res_3v1['p2_wins']:4d} 局 ({res_3v1['p2_wins']/args.games*100:5.1f}%)
    Gen3(先手) 勝利 : {res_3v1['p1_wins']:4d} 局 ({res_3v1['p1_wins']/args.games*100:5.1f}%)
    引き分け        : {res_3v1['draws']:4d} 局 ({res_3v1['draws']/args.games*100:5.1f}%)
    -----------------------------------------------------------------------------
    3手決着         : {res_3v1['p3']:4d} 局 ({res_3v1['p3']/args.games*100:5.2f}%)
    4手決着         : {res_3v1['p4']:4d} 局 ({res_3v1['p4']/args.games*100:5.2f}%)
    5手決着         : {res_3v1['p5']:4d} 局 ({res_3v1['p5']/args.games*100:5.2f}%)
    6手決着         : {res_3v1['p6']:4d} 局 ({res_3v1['p6']/args.games*100:5.2f}%)
    (3〜6手決着 計) : {res_3v1['p3']+res_3v1['p4']+res_3v1['p5']+res_3v1['p6']:4d} 局 ({(res_3v1['p3']+res_3v1['p4']+res_3v1['p5']+res_3v1['p6'])/args.games*100:5.2f}%)

---------------------------------------------------------------------------------
【通算総合結果 ({total_games} 局)】
    ★ Gen1 (新世代) 通算勝利 : {total_g1_wins:4d} 局 ({total_g1_wins/total_games*100:5.2f}%)
    ★ Gen3 (旧世代) 通算勝利 : {total_g3_wins:4d} 局 ({total_g3_wins/total_games*100:5.2f}%)
    ★ 通算引き分け           : {total_draws:4d} 局 ({total_draws/total_games*100:5.2f}%)
    ★ 全体平均手数           : {np.mean(all_moves_combined):.2f} 手 (中央値: {np.median(all_moves_combined):.1f} 手)
    -----------------------------------------------------------------------------
    ★ 3手決着数              : {tot_p3:4d} 局 ({tot_p3/total_games*100:5.2f}%)
    ★ 4手決着数              : {tot_p4:4d} 局 ({tot_p4/total_games*100:5.2f}%)
    ★ 5手決着数              : {tot_p5:4d} 局 ({tot_p5/total_games*100:5.2f}%)
    ★ 6手決着数              : {tot_p6:4d} 局 ({tot_p6/total_games*100:5.2f}%)
    ★ 3〜6手決着 合計        : {tot_3to6:4d} 局 ({tot_3to6/total_games*100:5.2f}%)
=================================================================================
"""
    print(summary_text)

    summary_path = os.path.join(repo_dir, "gen1_vs_gen3_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"[保存完了] 勝敗表テキスト: {summary_path}")

if __name__ == "__main__":
    main()
