"""
万能対戦評価スクリプト (battle.py)

ユーザー指定仕様:
- 別モデル同士: 先後入れ替え各500局 (計1,000局) + 3〜6手決着詳細
- 同一モデル同士: 1,000局 + 手数分布グラフ出力 + 3〜6手決着詳細
- 100局ごとの進捗表示のみ (警告・ロード表示は完全消音)
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
import matplotlib.pyplot as plt
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

def run_single_series(m1_name, m1_path, m2_name, m2_path, num_games, beta=10.0, num_workers=6):
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
            executor.submit(play_matchup_worker, i, n, m1_path, m2_path, beta, np.random.randint(1, 1000000) + i * 31)
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

    print(f"\n[終了] {m1_name}(先手) {p1_wins}勝 ({p1_wins/num_games*100:.1f}%) | "
          f"{m2_name}(後手) {p2_wins}勝 ({p2_wins/num_games*100:.1f}%) | "
          f"引分 {draws} ({draws/num_games*100:.1f}%)")
    print(f"平均手数: {np.mean(all_moves):.2f} 手 | 3〜6手計: {p3+p4+p5+p6} 局 ({(p3+p4+p5+p6)/num_games*100:.2f}%)")
    sys.stdout.flush()

    return {
        "p1_name": m1_name, "p2_name": m2_name,
        "p1_wins": p1_wins, "p2_wins": p2_wins, "draws": draws,
        "moves": all_moves,
        "p3": p3, "p4": p4, "p5": p5, "p6": p6,
        "num_games": num_games
    }

def run_versus_match(m1_name, m1_path, m2_name, m2_path, games_per_side=500, beta=10.0, num_workers=6, save_prefix=""):
    print(f"\n#################################################################")
    print(f" 【世代対決】 {m1_name} vs {m2_name} (先後各{games_per_side}局 / 計{games_per_side*2}局)")
    print(f"#################################################################")

    res_A = run_single_series(m1_name, m1_path, m2_name, m2_path, games_per_side, beta, num_workers)
    res_B = run_single_series(m2_name, m2_path, m1_name, m1_path, games_per_side, beta, num_workers)

    total_m1_wins = res_A["p1_wins"] + res_B["p2_wins"]
    total_m2_wins = res_A["p2_wins"] + res_B["p1_wins"]
    total_draws = res_A["draws"] + res_B["draws"]
    total_games = games_per_side * 2

    all_moves = np.concatenate([res_A["moves"], res_B["moves"]])
    tot_p3 = res_A["p3"] + res_B["p3"]
    tot_p4 = res_A["p4"] + res_B["p4"]
    tot_p5 = res_A["p5"] + res_B["p5"]
    tot_p6 = res_A["p6"] + res_B["p6"]
    tot_3to6 = tot_p3 + tot_p4 + tot_p5 + tot_p6

    summary_text = f"""
=================================================================================
                 対戦結果: {m1_name} vs {m2_name} 総合勝敗表
=================================================================================
【対戦条件】先後入れ替え 各 {games_per_side} 局 (計 {total_games} 局) / beta = {beta}

(A) 先手:{m1_name} vs 後手:{m2_name} ({games_per_side} 局)
    {m1_name}(先手) 勝利 : {res_A['p1_wins']:4d} 局 ({res_A['p1_wins']/games_per_side*100:5.1f}%)
    {m2_name}(後手) 勝利 : {res_A['p2_wins']:4d} 局 ({res_A['p2_wins']/games_per_side*100:5.1f}%)
    引き分け             : {res_A['draws']:4d} 局 ({res_A['draws']/games_per_side*100:5.1f}%)
    -----------------------------------------------------------------------------
    3手決着              : {res_A['p3']:4d} 局 ({res_A['p3']/games_per_side*100:5.2f}%)
    4手決着              : {res_A['p4']:4d} 局 ({res_A['p4']/games_per_side*100:5.2f}%)
    5手決着              : {res_A['p5']:4d} 局 ({res_A['p5']/games_per_side*100:5.2f}%)
    6手決着              : {res_A['p6']:4d} 局 ({res_A['p6']/games_per_side*100:5.2f}%)
    (3〜6手決着 計)      : {res_A['p3']+res_A['p4']+res_A['p5']+res_A['p6']:4d} 局 ({(res_A['p3']+res_A['p4']+res_A['p5']+res_A['p6'])/games_per_side*100:5.2f}%)

(B) 先手:{m2_name} vs 後手:{m1_name} ({games_per_side} 局)
    {m1_name}(後手) 勝利 : {res_B['p2_wins']:4d} 局 ({res_B['p2_wins']/games_per_side*100:5.1f}%)
    {m2_name}(先手) 勝利 : {res_B['p1_wins']:4d} 局 ({res_B['p1_wins']/games_per_side*100:5.1f}%)
    引き分け             : {res_B['draws']:4d} 局 ({res_B['draws']/games_per_side*100:5.1f}%)
    -----------------------------------------------------------------------------
    3手決着              : {res_B['p3']:4d} 局 ({res_B['p3']/games_per_side*100:5.2f}%)
    4手決着              : {res_B['p4']:4d} 局 ({res_B['p4']/games_per_side*100:5.2f}%)
    5手決着              : {res_B['p5']:4d} 局 ({res_B['p5']/games_per_side*100:5.2f}%)
    6手決着              : {res_B['p6']:4d} 局 ({res_B['p6']/games_per_side*100:5.2f}%)
    (3〜6手決着 計)      : {res_B['p3']+res_B['p4']+res_B['p5']+res_B['p6']:4d} 局 ({(res_B['p3']+res_B['p4']+res_B['p5']+res_B['p6'])/games_per_side*100:5.2f}%)

---------------------------------------------------------------------------------
【通算総合結果 ({total_games} 局)】
    ★ {m1_name} 通算勝利 : {total_m1_wins:4d} 局 ({total_m1_wins/total_games*100:5.2f}%)
    ★ {m2_name} 通算勝利 : {total_m2_wins:4d} 局 ({total_m2_wins/total_games*100:5.2f}%)
    ★ 通算引き分け       : {total_draws:4d} 局 ({total_draws/total_games*100:5.2f}%)
    ★ 全体平均手数       : {np.mean(all_moves):.2f} 手 (中央値: {np.median(all_moves):.1f} 手)
    -----------------------------------------------------------------------------
    ★ 3手決着数          : {tot_p3:4d} 局 ({tot_p3/total_games*100:5.2f}%)
    ★ 4手決着数          : {tot_p4:4d} 局 ({tot_p4/total_games*100:5.2f}%)
    ★ 5手決着数          : {tot_p5:4d} 局 ({tot_p5/total_games*100:5.2f}%)
    ★ 6手決着数          : {tot_p6:4d} 局 ({tot_p6/total_games*100:5.2f}%)
    ★ 3〜6手決着 合計    : {tot_3to6:4d} 局 ({tot_3to6/total_games*100:5.2f}%)
=================================================================================
"""
    print(summary_text)

    fname = f"summary_{save_prefix if save_prefix else m1_name + '_vs_' + m2_name}.txt"
    fpath = os.path.join(repo_dir, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"[保存完了] 勝敗表テキスト: {fpath}")

def run_self_match(m_name, m_path, num_games=1000, beta=10.0, num_workers=6, save_prefix=""):
    print(f"\n#################################################################")
    print(f" 【単独評価】 {m_name} 同士の自己対局 ({num_games} 局)")
    print(f"#################################################################")

    res = run_single_series(m_name, m_path, m_name, m_path, num_games, beta, num_workers)

    bins = [0, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 200]
    hist, _ = np.histogram(res["moves"], bins=bins)
    print(f"\n[{m_name} 同士 手数分布テーブル]")
    for i in range(len(hist)):
        print(f"  {bins[i]:3d} - {bins[i+1]:3d} 手: {hist[i]:4d} 局 ({hist[i]/num_games*100:5.1f}%)")

    # 手数分布グラフ
    tag = save_prefix if save_prefix else m_name
    plt.figure(figsize=(10, 6))
    plt.hist(res["moves"], bins=range(min(res["moves"]), max(res["moves"]) + 2),
             color="crimson", edgecolor="black", alpha=0.8)
    plt.axvline(np.mean(res["moves"]), color="blue", linestyle="--", linewidth=2,
                label=f"平均手数: {np.mean(res['moves']):.1f}")
    plt.axvline(np.median(res["moves"]), color="green", linestyle=":", linewidth=2,
                label=f"中央値手数: {np.median(res['moves']):.1f}")
    plt.title(f"{m_name} Self-Play Move Distribution ({num_games} Games)", fontsize=14)
    plt.xlabel("Game Length (Ply)", fontsize=12)
    plt.ylabel("Number of Games", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    dist_plot_path = os.path.join(repo_dir, f"{tag}_{num_games}games_distribution.png")
    plt.savefig(dist_plot_path, dpi=300)
    plt.close()
    print(f"\n[保存完了] 手数分布グラフ: {dist_plot_path}")

    tot_3to6 = res["p3"] + res["p4"] + res["p5"] + res["p6"]
    summary_text = f"""
=================================================================================
                 自己対戦結果: {m_name} 同士 ({num_games} 局)
=================================================================================
  先手 勝利       : {res['p1_wins']:4d} 局 ({res['p1_wins']/num_games*100:5.1f}%)
  後手 勝利       : {res['p2_wins']:4d} 局 ({res['p2_wins']/num_games*100:5.1f}%)
  引き分け        : {res['draws']:4d} 局 ({res['draws']/num_games*100:5.1f}%)
  平均手数        : {np.mean(res['moves']):.2f} 手 (中央値: {np.median(res['moves']):.1f} 手)
  -----------------------------------------------------------------------------
  3手決着         : {res['p3']:4d} 局 ({res['p3']/num_games*100:5.2f}%)
  4手決着         : {res['p4']:4d} 局 ({res['p4']/num_games*100:5.2f}%)
  5手決着         : {res['p5']:4d} 局 ({res['p5']/num_games*100:5.2f}%)
  6手決着         : {res['p6']:4d} 局 ({res['p6']/num_games*100:5.2f}%)
  3〜6手決着 合計 : {tot_3to6:4d} 局 ({tot_3to6/num_games*100:5.2f}%)
  ※ 分布グラフ   : {dist_plot_path}
=================================================================================
"""
    print(summary_text)

    fname = f"summary_{tag}_self.txt"
    fpath = os.path.join(repo_dir, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"[保存完了] 勝敗表テキスト: {fpath}")

def main():
    parser = argparse.ArgumentParser(description="どうぶつしょうぎ 万能対戦スクリプト")
    parser.add_argument("--preset", type=str, default=None,
                        choices=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "all"],
                        help="プリセット番号 (1:新Gen2 vs 新Gen1, 2:新Gen1 vs 元Gen5, 3:新Gen2 vs 元Gen6, 4:新Gen2同士, 5:新Gen3 vs 新Gen2, 6:新Gen3同士, 7:新Gen4 vs 新Gen3, 8:新Gen4同士, 9:新Gen4 vs 新Gen2, 10:新Gen4 vs 元Gen7, 11:新Gen5 vs 新Gen4, 12:新Gen5 vs 新Gen3, 13:新Gen5 vs 元Gen7, 14:新Gen5同士, 15:新Gen2 vs 旧Gen4, 16:新Gen3(0.80) vs Gen3(0.95), 17:新Gen4(0.80) vs Gen4(0.95), all:全て連続実行)")
    parser.add_argument("--m1", type=str, default=None, help="モデル1パス")
    parser.add_argument("--m2", type=str, default=None, help="モデル2パス")
    parser.add_argument("--name1", type=str, default="Model1", help="モデル1名称")
    parser.add_argument("--name2", type=str, default="Model2", help="モデル2名称")
    parser.add_argument("--self", dest="self_path", type=str, default=None, help="自己対戦モデルパス")
    parser.add_argument("--self_name", type=str, default="Model", help="自己対戦モデル名称")
    parser.add_argument("--games", type=int, default=500, help="対局数 (別モデル時は先後各500局、自己対戦時は1000局)")
    parser.add_argument("--beta", type=float, default=10.0, help="逆温度 beta")
    parser.add_argument("--workers", type=int, default=min(multiprocessing.cpu_count(), 6), help="並列プロセス数")
    args = parser.parse_args()

    # パス定義
    g1_path = "animal_shogi_model_gen1.keras"
    g2_path = "animal_shogi_model_gen2.keras"
    g3_path = "animal_shogi_model_gen3.keras"
    g4_path = "animal_shogi_model_gen4.keras"
    g5_new_path = "animal_shogi_model_gen5.keras"
    g4_old_path = "legacy_v1/animal_shogi_model_server_gen4.keras"
    g5_path = "legacy_v1/animal_shogi_model_server_gen5.keras"
    g6_path = "legacy_v1/animal_shogi_model_server_gen6.keras"
    g7_path = "legacy_v1/animal_shogi_model_server_gen7.keras"
    g3_decay095_path = "models_decay095/animal_shogi_model_gen3_decay095.keras"
    g4_decay095_path = "models_decay095/animal_shogi_model_gen4_decay095.keras"

    if args.preset == "1":
        # 1. 新Gen2 vs 新Gen1 (先後各500局)
        run_versus_match("新Gen2", g2_path, "新Gen1", g1_path,
                         games_per_side=500, beta=args.beta, num_workers=args.workers, save_prefix="gen2_vs_gen1")
    elif args.preset == "2":
        # 2. 新Gen1 vs 元Gen5 (先後各500局)
        run_versus_match("新Gen1", g1_path, "元Gen5", g5_path,
                         games_per_side=500, beta=args.beta, num_workers=args.workers, save_prefix="gen1_vs_old_gen5")
    elif args.preset == "3":
        # 3. 新Gen2 vs 元Gen6 (先後各500局)
        run_versus_match("新Gen2", g2_path, "元Gen6", g6_path,
                         games_per_side=500, beta=args.beta, num_workers=args.workers, save_prefix="gen2_vs_old_gen6")
    elif args.preset == "4":
        # 4. 新Gen2 同士 (1,000局)
        run_self_match("新Gen2", g2_path,
                       num_games=1000, beta=args.beta, num_workers=args.workers, save_prefix="gen2")
    elif args.preset == "5":
        # 5. 新Gen3 vs 新Gen2 (先後各500局)
        run_versus_match("新Gen3", g3_path, "新Gen2", g2_path,
                         games_per_side=500, beta=args.beta, num_workers=args.workers, save_prefix="gen3_vs_gen2")
    elif args.preset == "6":
        # 6. 新Gen3 同士 (1,000局)
        run_self_match("新Gen3", g3_path,
                       num_games=1000, beta=args.beta, num_workers=args.workers, save_prefix="gen3")
    elif args.preset == "7":
        # 7. 新Gen4 vs 新Gen3 (先後各500局)
        run_versus_match("新Gen4", g4_path, "新Gen3", g3_path,
                         games_per_side=500, beta=args.beta, num_workers=args.workers, save_prefix="gen4_vs_gen3")
    elif args.preset == "8":
        # 8. 新Gen4 同士 (1,000局)
        run_self_match("新Gen4", g4_path,
                       num_games=1000, beta=args.beta, num_workers=args.workers, save_prefix="gen4")
    elif args.preset == "9":
        # 9. 新Gen4 vs 新Gen2 (先後各500局)
        run_versus_match("新Gen4", g4_path, "新Gen2", g2_path,
                         games_per_side=500, beta=args.beta, num_workers=args.workers, save_prefix="gen4_vs_gen2")
    elif args.preset == "10":
        # 10. 新Gen4 vs 元Gen7 (先後各500局)
        run_versus_match("新Gen4", g4_path, "元Gen7", g7_path,
                         games_per_side=500, beta=args.beta, num_workers=args.workers, save_prefix="gen4_vs_old_gen7")
    elif args.preset == "11":
        # 11. 新Gen5 vs 新Gen4 (先後各500局)
        run_versus_match("新Gen5", g5_new_path, "新Gen4", g4_path,
                         games_per_side=500, beta=args.beta, num_workers=args.workers, save_prefix="gen5_vs_gen4")
    elif args.preset == "12":
        # 12. 新Gen5 vs 新Gen3 (先後各500局)
        run_versus_match("新Gen5", g5_new_path, "新Gen3", g3_path,
                         games_per_side=500, beta=args.beta, num_workers=args.workers, save_prefix="gen5_vs_gen3")
    elif args.preset == "13":
        # 13. 新Gen5 vs 元Gen7 (先後各500局)
        run_versus_match("新Gen5", g5_new_path, "元Gen7", g7_path,
                         games_per_side=500, beta=args.beta, num_workers=args.workers, save_prefix="gen5_vs_old_gen7")
    elif args.preset == "14":
        # 14. 新Gen5 同士 (1,000局)
        run_self_match("新Gen5", g5_new_path,
                       num_games=1000, beta=args.beta, num_workers=args.workers, save_prefix="gen5")
    elif args.preset == "15":
        # 15. 新Gen2 vs 旧Gen4 (先後各500局)
        run_versus_match("新Gen2", g2_path, "旧Gen4", g4_old_path,
                         games_per_side=500, beta=args.beta, num_workers=args.workers, save_prefix="gen2_vs_old_gen4")
    elif args.preset == "16":
        # 16. 新Gen3(0.80) vs Gen3(0.95) (先後各500局)
        run_versus_match("新Gen3(0.80)", g3_path, "Gen3(0.95)", g3_decay095_path,
                         games_per_side=500, beta=args.beta, num_workers=args.workers, save_prefix="gen3_080_vs_gen3_095")
    elif args.preset == "17":
        # 17. 新Gen4(0.80) vs Gen4(0.95) (先後各500局)
        run_versus_match("新Gen4(0.80)", g4_path, "Gen4(0.95)", g4_decay095_path,
                         games_per_side=500, beta=args.beta, num_workers=args.workers, save_prefix="gen4_080_vs_gen4_095")
    elif args.preset == "all":
        print("\n" + "=" * 70)
        print(" 【全4カード連続対戦ベンチマークを開始します】")
        print("=" * 70)
        run_versus_match("新Gen2", g2_path, "新Gen1", g1_path, games_per_side=500, beta=args.beta, num_workers=args.workers, save_prefix="gen2_vs_gen1")
        run_versus_match("新Gen1", g1_path, "元Gen5", g5_path, games_per_side=500, beta=args.beta, num_workers=args.workers, save_prefix="gen1_vs_old_gen5")
        run_versus_match("新Gen2", g2_path, "元Gen6", g6_path, games_per_side=500, beta=args.beta, num_workers=args.workers, save_prefix="gen2_vs_old_gen6")
        run_self_match("新Gen2", g2_path, num_games=1000, beta=args.beta, num_workers=args.workers, save_prefix="gen2")
    elif args.self_path:
        run_self_match(args.self_name, args.self_path, num_games=args.games if args.games != 500 else 1000, beta=args.beta, num_workers=args.workers)
    elif args.m1 and args.m2:
        run_versus_match(args.name1, args.m1, args.name2, args.m2, games_per_side=args.games, beta=args.beta, num_workers=args.workers)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
