"""
世代間対戦評価スクリプト (evaluate_generations.py)

ユーザー指定仕様:
1. TensorFlow / Keras の内部警告ログやモデルロード表示を完全非表示
2. 100局ごとに対戦進捗ログのみを表示
3. 3手〜6手決着の詳細内訳をすべて集計・出力
4. Gen1同士 1,000局対局（勝敗表 ＋ 手数分布グラフ gen1_1000games_distribution.png 出力）
5. Gen1 vs Gen0 先後入れ替え各500局（計1,000局の勝敗表出力）
"""

import os
import sys
import warnings
import logging

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import argparse
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

repo_dir = os.path.dirname(os.path.abspath(__file__))
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)

from board import Board
from player import ModelPlayer, RandomPlayer
from engine import GameEngine
from constants import PLAYER_1, PLAYER_2, RESULT_DRAW

def play_matchup_worker(worker_id, num_games, m1_path, m2_path, beta, seed):
    import os
    import sys
    import warnings
    warnings.filterwarnings("ignore")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')

    np.random.seed(seed)
    p1 = ModelPlayer(PLAYER_1, model_path_or_model=m1_path, beta=beta) if m1_path else RandomPlayer(PLAYER_1)
    p2 = ModelPlayer(PLAYER_2, model_path_or_model=m2_path, beta=beta) if m2_path else RandomPlayer(PLAYER_2)
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
            executor.submit(play_matchup_worker, i, n, m1_path, m2_path, beta, 10000 + i * 17)
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
    print(f"平均手数: {np.mean(all_moves):.2f} 手 | 3〜6手計: {p3+p4+p5+p6} 局 ({(p3+p4+p5+p6)/num_games*100:.2f}%)")
    sys.stdout.flush()

    return {
        "p1_name": m1_name, "p2_name": m2_name,
        "p1_wins": p1_wins, "p2_wins": p2_wins, "draws": draws,
        "moves": all_moves,
        "p3": p3, "p4": p4, "p5": p5, "p6": p6,
        "num_games": num_games
    }

def main():
    parser = argparse.ArgumentParser(description="どうぶつしょうぎ 世代間対戦ベンチマーク (Gen1 vs Gen1 & Gen1 vs Gen0)")
    parser.add_argument("--gen0", type=str, default="animal_shogi_model_gen0.keras", help="Gen0モデルパス")
    parser.add_argument("--gen1", type=str, default="animal_shogi_model_gen1.keras", help="Gen1モデルパス")
    parser.add_argument("--self_games", type=int, default=1000, help="Gen1同士の対戦局数 (デフォルト: 1000)")
    parser.add_argument("--vs_games", type=int, default=500, help="Gen1 vs Gen0 の各手番局数 (デフォルト: 500)")
    parser.add_argument("--beta", type=float, default=10.0, help="逆温度 beta (デフォルト: 10.0)")
    parser.add_argument("--workers", type=int, default=min(multiprocessing.cpu_count(), 6), help="並列プロセス数")
    args = parser.parse_args()

    if not os.path.exists(args.gen0):
        if os.path.exists("animal_shogi_model.keras"):
            args.gen0 = "animal_shogi_model.keras"
        else:
            print(f"エラー: Gen0 モデル {args.gen0} が見つかりません。")
            sys.exit(1)

    if not os.path.exists(args.gen1):
        if os.path.exists("animal_shogi_model_server_gen1.keras"):
            args.gen1 = "animal_shogi_model_server_gen1.keras"
        else:
            print(f"エラー: Gen1 モデル {args.gen1} が見つかりません。サーバーからダウンロードしてください。")
            sys.exit(1)

    print("=================================================================")
    print(" どうぶつしょうぎ ベンチマーク評価プログラム")
    print(f" Gen0 モデル: {args.gen0}")
    print(f" Gen1 モデル: {args.gen1}")
    print(f" 並列プロセス数: {args.workers} (CPU)")
    print("=================================================================")
    sys.stdout.flush()

    # 1. Gen1同士の対戦 (1,000局)
    res_self = run_matchup("Gen1", args.gen1, "Gen1", args.gen1,
                           num_games=args.self_games, beta=args.beta, num_workers=args.workers)

    bins = [0, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 200]
    hist, _ = np.histogram(res_self["moves"], bins=bins)
    print("\n[Gen1同士 手数分布テーブル]")
    for i in range(len(hist)):
        print(f"  {bins[i]:3d} - {bins[i+1]:3d} 手: {hist[i]:4d} 局 ({hist[i]/args.self_games*100:5.1f}%)")

    plt.figure(figsize=(10, 6))
    plt.hist(res_self["moves"], bins=range(min(res_self["moves"]), max(res_self["moves"]) + 2),
             color="crimson", edgecolor="black", alpha=0.8)
    plt.axvline(np.mean(res_self["moves"]), color="blue", linestyle="--", linewidth=2,
                label=f"平均手数: {np.mean(res_self['moves']):.1f}")
    plt.axvline(np.median(res_self["moves"]), color="green", linestyle=":", linewidth=2,
                label=f"中央値手数: {np.median(res_self['moves']):.1f}")
    plt.title(f"Gen1 (Trained CNN + 2-ply Min-Max) Move Distribution ({args.self_games} Games)", fontsize=14)
    plt.xlabel("Game Length (Ply)", fontsize=12)
    plt.ylabel("Number of Games", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    dist_plot_path = os.path.join(repo_dir, "gen1_1000games_distribution.png")
    plt.savefig(dist_plot_path, dpi=300)
    plt.close()
    print(f"\n[保存完了] Gen1同士の手数分布グラフ: {dist_plot_path}")

    # 2. Gen1 vs Gen0 (先手後手入れ替え 各500局)
    res_1v0 = run_matchup("Gen1", args.gen1, "Gen0", args.gen0,
                          num_games=args.vs_games, beta=args.beta, num_workers=args.workers)

    res_0v1 = run_matchup("Gen0", args.gen0, "Gen1", args.gen1,
                          num_games=args.vs_games, beta=args.beta, num_workers=args.workers)

    # 3. 総合勝敗表
    total_g1_wins = res_1v0["p1_wins"] + res_0v1["p2_wins"]
    total_g0_wins = res_1v0["p2_wins"] + res_0v1["p1_wins"]
    total_draws = res_1v0["draws"] + res_0v1["draws"]
    total_vs_games = args.vs_games * 2

    tot_p3 = res_1v0["p3"] + res_0v1["p3"]
    tot_p4 = res_1v0["p4"] + res_0v1["p4"]
    tot_p5 = res_1v0["p5"] + res_0v1["p5"]
    tot_p6 = res_1v0["p6"] + res_0v1["p6"]
    tot_3to6 = tot_p3 + tot_p4 + tot_p5 + tot_p6

    summary_text = f"""
=================================================================================
                           どうぶつしょうぎ 総合勝敗表
=================================================================================

【1. Gen1 同士の対戦結果 ({args.self_games} 局)】
---------------------------------------------------------------------------------
  先手(Gen1) 勝利 : {res_self['p1_wins']:4d} 局 ({res_self['p1_wins']/args.self_games*100:5.1f}%)
  後手(Gen1) 勝利 : {res_self['p2_wins']:4d} 局 ({res_self['p2_wins']/args.self_games*100:5.1f}%)
  引き分け        : {res_self['draws']:4d} 局 ({res_self['draws']/args.self_games*100:5.1f}%)
  平均手数        : {np.mean(res_self['moves']):.2f} 手 (中央値: {np.median(res_self['moves']):.1f} 手)
  3手決着         : {res_self['p3']:4d} 局 ({res_self['p3']/args.self_games*100:5.2f}%)
  4手決着         : {res_self['p4']:4d} 局 ({res_self['p4']/args.self_games*100:5.2f}%)
  5手決着         : {res_self['p5']:4d} 局 ({res_self['p5']/args.self_games*100:5.2f}%)
  6手決着         : {res_self['p6']:4d} 局 ({res_self['p6']/args.self_games*100:5.2f}%)
  (3〜6手決着 計) : {res_self['p3']+res_self['p4']+res_self['p5']+res_self['p6']:4d} 局 ({(res_self['p3']+res_self['p4']+res_self['p5']+res_self['p6'])/args.self_games*100:5.2f}%)
  ※ 分布グラフ   : gen1_1000games_distribution.png に保存済み

【2. Gen1 vs Gen0 対戦結果 (先後入替 各{args.vs_games}局 / 計{total_vs_games}局)】
---------------------------------------------------------------------------------
  (A) 先手:Gen1 vs 後手:Gen0 ({args.vs_games}局)
      Gen1(先手) 勝利 : {res_1v0['p1_wins']:4d} 局 ({res_1v0['p1_wins']/args.vs_games*100:5.1f}%)
      Gen0(後手) 勝利 : {res_1v0['p2_wins']:4d} 局 ({res_1v0['p2_wins']/args.vs_games*100:5.1f}%)
      引き分け        : {res_1v0['draws']:4d} 局 ({res_1v0['draws']/args.vs_games*100:5.1f}%)
      3手={res_1v0['p3']}局, 4手={res_1v0['p4']}局, 5手={res_1v0['p5']}局, 6手={res_1v0['p6']}局 (3〜6手計: {res_1v0['p3']+res_1v0['p4']+res_1v0['p5']+res_1v0['p6']}局 / {(res_1v0['p3']+res_1v0['p4']+res_1v0['p5']+res_1v0['p6'])/args.vs_games*100:5.2f}%)

  (B) 先手:Gen0 vs 後手:Gen1 ({args.vs_games}局)
      Gen1(後手) 勝利 : {res_0v1['p2_wins']:4d} 局 ({res_0v1['p2_wins']/args.vs_games*100:5.1f}%)
      Gen0(先手) 勝利 : {res_0v1['p1_wins']:4d} 局 ({res_0v1['p1_wins']/args.vs_games*100:5.1f}%)
      引き分け        : {res_0v1['draws']:4d} 局 ({res_0v1['draws']/args.vs_games*100:5.1f}%)
      3手={res_0v1['p3']}局, 4手={res_0v1['p4']}局, 5手={res_0v1['p5']}局, 6手={res_0v1['p6']}局 (3〜6手計: {res_0v1['p3']+res_0v1['p4']+res_0v1['p5']+res_0v1['p6']}局 / {(res_0v1['p3']+res_0v1['p4']+res_0v1['p5']+res_0v1['p6'])/args.vs_games*100:5.2f}%)

  (C) Gen1 vs Gen0 総合計 ({total_vs_games}局 通算)
      ★ Gen1 通算勝利 : {total_g1_wins:4d} 局 ({total_g1_wins/total_vs_games*100:5.2f}%)
      ★ Gen0 通算勝利 : {total_g0_wins:4d} 局 ({total_g0_wins/total_vs_games*100:5.2f}%)
      ★ 通算引き分け  : {total_draws:4d} 局 ({total_draws/total_vs_games*100:5.2f}%)
      ---------------------------------------------------------------------------
      ★ 3手決着数     : {tot_p3:4d} 局 ({tot_p3/total_vs_games*100:5.2f}%)
      ★ 4手決着数     : {tot_p4:4d} 局 ({tot_p4/total_vs_games*100:5.2f}%)
      ★ 5手決着数     : {tot_p5:4d} 局 ({tot_p5/total_vs_games*100:5.2f}%)
      ★ 6手決着数     : {tot_p6:4d} 局 ({tot_p6/total_vs_games*100:5.2f}%)
      ★ 3〜6手決着 計 : {tot_3to6:4d} 局 ({tot_3to6/total_vs_games*100:5.2f}%)
=================================================================================
"""
    print(summary_text)

    summary_path = os.path.join(repo_dir, "gen1_vs_gen0_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"[保存完了] 総合勝敗表テキスト: {summary_path}")

if __name__ == "__main__":
    main()
