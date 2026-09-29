import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# animal_shogi へのパス
repo_dir = r"c:\Users\youyo\OneDrive\ドキュメント\渡辺研\quantum_project\animal_shogi"
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)

from board import Board
from player import ModelPlayer
from engine import GameEngine
from constants import PLAYER_1, PLAYER_2, RESULT_DRAW

def play_games_worker(worker_id, num_games, model_path, beta, seed_offset):
    # 各ワーカープロセス内で乱数シードとモデルを初期化
    np.random.seed(seed_offset + worker_id)
    p1 = ModelPlayer(PLAYER_1, model_path_or_model=model_path, beta=beta)
    p2 = ModelPlayer(PLAYER_2, model_path_or_model=model_path, beta=beta)
    players = {PLAYER_1: p1, PLAYER_2: p2}

    move_counts = []
    winners = []
    four_ply_count = 0

    for g in range(num_games):
        board = Board()
        board.setup()
        current_p = PLAYER_1
        max_moves = 200
        winner = None
        game_moves = 0

        for m in range(max_moves):
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
        if game_moves == 4:
            four_ply_count += 1

    return move_counts, winners, four_ply_count

def main():
    num_games = 1000
    model_path = "animal_shogi_model.keras"
    beta = 10.0
    num_workers = min(multiprocessing.cpu_count(), 6)

    print(f"=== Gen0 モデルによる {num_games} 局対戦テストを開始します (並列ワーカー数: {num_workers}) ===")

    games_per_worker = [num_games // num_workers] * num_workers
    for i in range(num_games % num_workers):
        games_per_worker[i] += 1

    all_move_counts = []
    all_winners = []
    total_four_ply = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(play_games_worker, i, games_per_worker[i], model_path, beta, 1000 * i)
            for i in range(num_workers)
        ]
        for f in futures:
            move_counts, winners, four_ply_count = f.result()
            all_move_counts.extend(move_counts)
            all_winners.extend(winners)
            total_four_ply += 4 if four_ply_count > 0 and False else four_ply_count

    all_move_counts = np.array(all_move_counts)
    all_winners = np.array(all_winners)

    p1_wins = np.sum(all_winners == PLAYER_1)
    p2_wins = np.sum(all_winners == PLAYER_2)
    draws = np.sum(all_winners == RESULT_DRAW)

    print("\n" + "=" * 50)
    print(f"=== {num_games} 局対戦結果集計 (Gen0) ===")
    print("=" * 50)
    print(f"先手(P1) 勝利数: {p1_wins} ({p1_wins/num_games*100:.1f}%)")
    print(f"後手(P2) 勝利数: {p2_wins} ({p2_wins/num_games*100:.1f}%)")
    print(f"引き分け数    : {draws} ({draws/num_games*100:.1f}%)")
    print(f"平均手数      : {np.mean(all_move_counts):.2f} 手")
    print(f"中央値手数    : {np.median(all_move_counts):.1f} 手")
    print(f"最短手数      : {np.min(all_move_counts)} 手")
    print(f"最長手数      : {np.max(all_move_counts)} 手")
    print(f"4手決着数     : {total_four_ply} 局 ({total_four_ply/num_games*100:.2f}%)")
    print("=" * 50)

    # 手数分布の集計
    bins = [0, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 200]
    hist, _ = np.histogram(all_move_counts, bins=bins)
    print("\n手数分布テーブル:")
    for i in range(len(hist)):
        print(f"  {bins[i]:3d} - {bins[i+1]:3d} 手: {hist[i]:4d} 局 ({hist[i]/num_games*100:5.1f}%)")

    # グラフの保存
    plt.figure(figsize=(10, 6))
    plt.hist(all_move_counts, bins=range(min(all_move_counts), max(all_move_counts) + 2), color="royalblue", edgecolor="black", alpha=0.8)
    plt.axvline(np.mean(all_move_counts), color="red", linestyle="--", linewidth=2, label=f"平均手数: {np.mean(all_move_counts):.1f}")
    plt.axvline(np.median(all_move_counts), color="green", linestyle=":", linewidth=2, label=f"中央値手数: {np.median(all_move_counts):.1f}")
    plt.title(f"Gen0 (Random CNN + 2-ply Min-Max) Move Distribution ({num_games} Games)", fontsize=14)
    plt.xlabel("Game Length (Ply)", fontsize=12)
    plt.ylabel("Number of Games", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plot_path = os.path.join(repo_dir, "gen0_1000games_distribution.png")
    plt.savefig(plot_path, dpi=300)
    print(f"\n分布グラフを保存しました: {plot_path}")

    # 集計結果をファイルにも保存
    res_path = os.path.join(repo_dir, "gen0_1000games_summary.txt")
    with open(res_path, "w", encoding="utf-8") as f:
        f.write(f"Gen0 1000 Games Summary\n")
        f.write(f"P1 Wins: {p1_wins} ({p1_wins/num_games*100:.1f}%)\n")
        f.write(f"P2 Wins: {p2_wins} ({p2_wins/num_games*100:.1f}%)\n")
        f.write(f"Draws: {draws} ({draws/num_games*100:.1f}%)\n")
        f.write(f"Mean Moves: {np.mean(all_move_counts):.2f}\n")
        f.write(f"Median Moves: {np.median(all_move_counts):.1f}\n")
        f.write(f"Min Moves: {np.min(all_move_counts)}\n")
        f.write(f"Max Moves: {np.max(all_move_counts)}\n")
        f.write(f"4-ply Games: {total_four_ply}\n")
        for i in range(len(hist)):
            f.write(f"{bins[i]}-{bins[i+1]} moves: {hist[i]}\n")
    print(f"集計サマリーを保存しました: {res_path}")

if __name__ == "__main__":
    main()
