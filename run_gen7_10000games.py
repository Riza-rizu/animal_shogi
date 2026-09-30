"""
元Gen7 同士 10,000局対戦 & 手数分布グラフ生成スクリプト
（最新の深さ2探索・tanh自滅手=-1.0適用版）
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

# 日本語フォント設定（Windows環境対応）
plt.rcParams['font.sans-serif'] = ['Meiryo', 'MS Gothic', 'Yu Gothic', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def play_batch_worker(worker_id, num_games, model_path, beta, seed):
    import os
    import sys
    import warnings
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')

    np.random.seed(seed)
    p1 = ModelPlayer(PLAYER_1, model_path_or_model=model_path, beta=beta)
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

def plot_move_distribution(move_counts, gen_name, stats, output_path):
    """手数の分布棒グラフ（ヒストグラム）を当時と完全同一のデザインでプロットして保存"""
    plt.figure(figsize=(10, 6))
    
    max_m = max(move_counts)
    bins = np.arange(1, max_m + 3) - 0.5
    
    plt.hist(
        move_counts, bins=bins, color='#3498db', edgecolor='#2980b9', alpha=0.85
    )
    
    # 平均値・中央値の縦線
    plt.axvline(stats['avg'], color='#e74c3c', linestyle='dashed', linewidth=2, label=f"平均手数: {stats['avg']:.1f} 手")
    plt.axvline(stats['med'], color='#2ecc71', linestyle='dotted', linewidth=2, label=f"中央値: {stats['med']:.0f} 手")
    
    plt.title(f"【{gen_name} 同士の対戦 (深さ2探索)】 手数の出現頻度分布 (総対戦数: {len(move_counts)} 局)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("決着までの手数 (Moves)", fontsize=12)
    plt.ylabel("対局数 (Games)", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    # 統計情報のテキストボックス (後手勝率を含む)
    info_text = (
        f"総対戦数: {len(move_counts)} 局\n"
        f"平均手数: {stats['avg']:.2f} 手\n"
        f"中央値: {stats['med']:.0f} 手\n"
        f"最短手数: {stats['min']} 手\n"
        f"最長手数: {stats['max']} 手\n"
        f"先手勝率: {stats['p1_win_rate']:.2f}%\n"
        f"後手勝率: {stats['p2_win_rate']:.2f}%\n"
        f"千日手 (200手): {stats['draws']} 局 ({stats['draw_rate']:.2f}%)\n"
        f"3〜6手即死: {stats['tot_3to6']} 局 ({stats['p_3to6']:.2f}%)"
    )
    plt.gca().text(
        0.72, 0.95, info_text, transform=plt.gca().transAxes,
        fontsize=10.5, verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.6', facecolor='#ffffff', edgecolor='#bdc3c7', alpha=0.9)
    )
    
    plt.legend(loc='upper right', bbox_to_anchor=(0.70, 0.98), fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"\n[グラフ保存完了] {output_path}")

def main():
    parser = argparse.ArgumentParser(description="元Gen7 同士 10,000局対戦スクリプト")
    parser.add_argument("--model", type=str, default="legacy_v1/animal_shogi_model_server_gen7.keras", help="モデルパス")
    parser.add_argument("--games", type=int, default=10000, help="対局数 (デフォルト: 10000)")
    parser.add_argument("--beta", type=float, default=10.0, help="逆温度 beta")
    parser.add_argument("--workers", type=int, default=min(multiprocessing.cpu_count(), 6), help="並列プロセス数")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"エラー: モデル {args.model} が見つかりません。")
        sys.exit(1)

    print("=================================================================")
    print(f" 元Gen7 同士 10,000局 対戦分析 (最新: 深さ2探索 / 自滅手=-1.0)")
    print(f" モデルパス: {args.model}")
    print(f" 総対戦数  : {args.games} 局")
    print(f" 逆温度 beta: {args.beta}")
    print(f" 並列ワーカー: {args.workers}")
    print("=================================================================")
    sys.stdout.flush()

    chunk_size = 50
    task_games = []
    remaining = args.games
    while remaining > 0:
        c = min(chunk_size, remaining)
        task_games.append(c)
        remaining -= c

    all_moves = []
    all_winners = []
    completed_games = 0
    last_reported = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(play_batch_worker, i, n, args.model, args.beta, 50000 + i * 47)
            for i, n in enumerate(task_games)
        ]
        for f in as_completed(futures):
            moves, winners = f.result()
            all_moves.extend(moves)
            all_winners.extend(winners)
            completed_games += len(moves)

            while completed_games >= last_reported + 500 and last_reported + 500 <= args.games:
                last_reported += 500
                print(f"  進捗: {last_reported:5d} / {args.games} 局完了 ({last_reported/args.games*100:.1f}%)")
                sys.stdout.flush()

    if last_reported < args.games and completed_games == args.games:
        print(f"  進捗: {args.games:5d} / {args.games} 局完了 (100.0%)")
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
    tot_3to6 = p3 + p4 + p5 + p6

    avg_moves = float(np.mean(all_moves))
    med_moves = float(np.median(all_moves))
    min_moves = int(np.min(all_moves))
    max_moves = int(np.max(all_moves))

    p1_rate = p1_wins / args.games * 100
    p2_rate = p2_wins / args.games * 100
    draw_rate = draws / args.games * 100
    p_3to6 = tot_3to6 / args.games * 100

    stats = {
        "avg": avg_moves,
        "med": med_moves,
        "min": min_moves,
        "max": max_moves,
        "p1_wins": p1_wins,
        "p2_wins": p2_wins,
        "draws": draws,
        "p1_win_rate": p1_rate,
        "p2_win_rate": p2_rate,
        "draw_rate": draw_rate,
        "tot_3to6": tot_3to6,
        "p_3to6": p_3to6
    }

    # グラフ生成・保存
    graph_filename = f"moves_distribution_Gen7_2ply_{args.games}games.png"
    graph_path = os.path.join(repo_dir, graph_filename)
    plot_move_distribution(all_moves, "元Gen7", stats, graph_path)

    # 手数分布テーブル
    bins = [0, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 200]
    hist, _ = np.histogram(all_moves, bins=bins)
    
    summary_text = f"""
=================================================================================
          元Gen7 同士 10,000局対戦分析結果 (最新: 深さ2探索 / 自滅手=-1.0)
=================================================================================
【対戦総数】 {args.games} 局 / 逆温度 beta = {args.beta}

【勝敗統計】
  ★ 先手(Player 1) 勝利 : {p1_wins:5d} 局 ({p1_rate:5.2f}%)
  ★ 後手(Player 2) 勝利 : {p2_wins:5d} 局 ({p2_rate:5.2f}%)
  ★ 千日手・引き分け     : {draws:5d} 局 ({draw_rate:5.2f}%)

【手数統計】
  ★ 平均手数 : {avg_moves:5.2f} 手
  ★ 中央値   : {med_moves:5.1f} 手
  ★ 最短手数 : {min_moves:5d} 手
  ★ 最長手数 : {max_moves:5d} 手

【超短手数（即死手）の内訳】
  ★ 3手決着 : {p3:4d} 局 ({p3/args.games*100:5.2f}%)
  ★ 4手決着 : {p4:4d} 局 ({p4/args.games*100:5.2f}%)
  ★ 5手決着 : {p5:4d} 局 ({p5/args.games*100:5.2f}%)
  ★ 6手決着 : {p6:4d} 局 ({p6/args.games*100:5.2f}%)
  ★ 3〜6手決着 合計: {tot_3to6:4d} 局 ({p_3to6:5.2f}%)

【手数分布テーブル】
"""
    for i in range(len(hist)):
        summary_text += f"  {bins[i]:3d} - {bins[i+1]:3d} 手: {hist[i]:5d} 局 ({hist[i]/args.games*100:5.2f}%)\n"

    summary_text += f"""
【グラフ画像保存先】
  {graph_path}
=================================================================================
"""
    print(summary_text)

    summary_file = os.path.join(repo_dir, f"summary_Gen7_2ply_{args.games}games.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"[サマリー保存完了] {summary_file}")

if __name__ == "__main__":
    main()
