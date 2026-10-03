"""
いまのGen7 vs いまのGen6 対戦 & 全試合棋譜CSV出力スクリプト
（深さ2探索・tanh自滅手=-1.0適用）
・先後入れ替え各500局（計1,000局）
・全試合の初手〜10手目、決着手数、勝者、勝者モデルをCSV（utf-8-sig）に出力
"""

import os
import sys
import csv
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
from constants import PLAYER_1, PLAYER_2, RESULT_DRAW, PIECE_DISPLAY

def format_move(move):
    """着手を人間が読みやすい文字列にフォーマット (例: B3ヒB2, 打ヒA3)"""
    col_names = ["A", "B", "C"]
    row_names = ["1", "2", "3", "4"]
    p_name = PIECE_DISPLAY.get(move.piece, move.piece).strip()
    
    to_str = f"{col_names[move.to_pos[0]]}{row_names[move.to_pos[1]]}"
    if move.is_drop:
        return f"打{p_name}{to_str}"
    from_str = f"{col_names[move.from_pos[0]]}{row_names[move.from_pos[1]]}"
    return f"{from_str}{p_name}{to_str}"

def play_batch_worker(worker_id, num_games, m1_path, m2_path, beta, seed):
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

    results = []

    for _ in range(num_games):
        board = Board()
        board.setup()
        current_p = PLAYER_1
        max_moves = 200
        winner = None
        game_moves = 0
        history_moves = []

        for _ in range(max_moves):
            legal_moves = GameEngine.get_legal_moves(board, current_p)
            if not legal_moves:
                winner = 3 - current_p
                break
            move = players[current_p].choose_move(board, legal_moves)
            game_moves += 1

            if len(history_moves) < 10:
                history_moves.append(format_move(move))

            GameEngine.make_move(board, move, current_p)

            winner = GameEngine.check_winner(board)
            if winner:
                break
            current_p = 3 - current_p
        else:
            winner = RESULT_DRAW

        results.append((game_moves, winner, history_moves))

    return results

def run_series(m1_name, m1_path, m2_name, m2_path, num_games, beta, num_workers, seed_base):
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

    all_results = []
    completed_games = 0
    last_reported = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(play_batch_worker, i, n, m1_path, m2_path, beta, seed_base + i * 37)
            for i, n in enumerate(task_games)
        ]
        for f in as_completed(futures):
            batch_res = f.result()
            all_results.extend(batch_res)
            completed_games += len(batch_res)

            while completed_games >= last_reported + 100 and last_reported + 100 <= num_games:
                last_reported += 100
                print(f"  進捗: {last_reported:4d} / {num_games} 局完了")
                sys.stdout.flush()

    if last_reported < num_games and completed_games == num_games:
        print(f"  進捗: {num_games:4d} / {num_games} 局完了")
        sys.stdout.flush()

    return all_results

def main():
    parser = argparse.ArgumentParser(description="Gen7 vs Gen6 対戦 & 全棋譜CSV出力スクリプト")
    parser.add_argument("--g7", type=str, default="legacy_v1/animal_shogi_model_server_gen7.keras", help="Gen7モデルパス")
    parser.add_argument("--g6", type=str, default="legacy_v1/animal_shogi_model_server_gen6.keras", help="Gen6モデルパス")
    parser.add_argument("--games", type=int, default=500, help="各手番の対局数 (デフォルト: 500局、計1000局)")
    parser.add_argument("--beta", type=float, default=10.0, help="逆温度 beta")
    parser.add_argument("--workers", type=int, default=min(multiprocessing.cpu_count(), 6), help="並列プロセス数")
    args = parser.parse_args()

    if not os.path.exists(args.g7):
        print(f"エラー: Gen7 モデルが見つかりません: {args.g7}")
        sys.exit(1)
    if not os.path.exists(args.g6):
        print(f"エラー: Gen6 モデルが見つかりません: {args.g6}")
        sys.exit(1)

    total_games = args.games * 2
    print("=================================================================")
    print(" 【全試合棋譜記録】いまのGen7 vs いまのGen6 対戦ベンチマーク")
    print(f" Gen7 モデル : {args.g7}")
    print(f" Gen6 モデル : {args.g6}")
    print(f" 各手番対局数: {args.games} 局 (合計 {total_games} 局)")
    print(f" 逆温度 beta : {args.beta}")
    print(f" 並列ワーカー: {args.workers}")
    print("=================================================================")
    sys.stdout.flush()

    # 1. パターンA: 先手 Gen7 vs 後手 Gen6
    results_A = run_series("Gen7", args.g7, "Gen6", args.g6, args.games, args.beta, args.workers, 20000)

    # 2. パターンB: 先手 Gen6 vs 後手 Gen7
    results_B = run_series("Gen6", args.g6, "Gen7", args.g7, args.games, args.beta, args.workers, 60000)

    # レコードの統合
    game_records = []
    global_id = 1
    move_keys = [f"move_{i+1}" for i in range(10)]

    g7_wins = 0
    g6_wins = 0
    draws = 0
    all_moves = []

    # パターンA処理
    for m_count, winner, hist_moves in results_A:
        all_moves.append(m_count)
        if winner == PLAYER_1:
            g7_wins += 1
            w_name = "先手(1)"
            w_model = "Gen7"
        elif winner == PLAYER_2:
            g6_wins += 1
            w_name = "後手(2)"
            w_model = "Gen6"
        else:
            draws += 1
            w_name = "引き分け"
            w_model = "引き分け"

        rec = {
            "game_id": global_id,
            "series": "Gen7(先) vs Gen6(後)",
            "p1_model": "Gen7",
            "p2_model": "Gen6",
            "winner_name": w_name,
            "winner_model": w_model,
            "total_moves": m_count,
        }
        for i in range(10):
            rec[f"move_{i+1}"] = hist_moves[i] if i < len(hist_moves) else ""
        rec["opening_10ply"] = " -> ".join(hist_moves)
        game_records.append(rec)
        global_id += 1

    # パターンB処理
    for m_count, winner, hist_moves in results_B:
        all_moves.append(m_count)
        if winner == PLAYER_1:
            g6_wins += 1
            w_name = "先手(1)"
            w_model = "Gen6"
        elif winner == PLAYER_2:
            g7_wins += 1
            w_name = "後手(2)"
            w_model = "Gen7"
        else:
            draws += 1
            w_name = "引き分け"
            w_model = "引き分け"

        rec = {
            "game_id": global_id,
            "series": "Gen6(先) vs Gen7(後)",
            "p1_model": "Gen6",
            "p2_model": "Gen7",
            "winner_name": w_name,
            "winner_model": w_model,
            "total_moves": m_count,
        }
        for i in range(10):
            rec[f"move_{i+1}"] = hist_moves[i] if i < len(hist_moves) else ""
        rec["opening_10ply"] = " -> ".join(hist_moves)
        game_records.append(rec)
        global_id += 1

    # 1. 棋譜 CSV 保存 (Excelで文字化けしない utf-8-sig)
    csv_filename = f"match_log_Gen7_vs_Gen6_{total_games}games.csv"
    csv_path = os.path.join(repo_dir, csv_filename)
    fieldnames = [
        "game_id", "series", "p1_model", "p2_model",
        "winner_name", "winner_model", "total_moves"
    ] + move_keys + ["opening_10ply"]

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(game_records)

    print(f"\n[全試合棋譜 CSV 保存完了] {csv_path}")

    # 2. 統計の計算とサマリー出力
    all_moves = np.array(all_moves)
    tot_3to6 = int(np.sum((all_moves >= 3) & (all_moves <= 6)))
    mask_7 = all_moves >= 7
    tot_games_7 = int(np.sum(mask_7))

    # 7手以上の勝敗
    g7_wins_7 = sum(1 for r in game_records if r["winner_model"] == "Gen7" and r["total_moves"] >= 7)
    g6_wins_7 = sum(1 for r in game_records if r["winner_model"] == "Gen6" and r["total_moves"] >= 7)
    draws_7 = sum(1 for r in game_records if r["winner_model"] == "引き分け" and r["total_moves"] >= 7)

    summary_text = f"""
=================================================================================
         Gen7 vs Gen6 対戦結果サマリー (計 {total_games} 局 / 深さ2探索)
=================================================================================
【通算総合結果】
  ★ Gen7 勝利     : {g7_wins:4d} 局 ({g7_wins / total_games * 100:5.2f}%)
  ★ Gen6 勝利     : {g6_wins:4d} 局 ({g6_wins / total_games * 100:5.2f}%)
  ★ 引き分け       : {draws:4d} 局 ({draws / total_games * 100:5.2f}%)
  ★ 全体平均手数   : {np.mean(all_moves):.2f} 手 (中央値: {np.median(all_moves):.1f} 手)
  -----------------------------------------------------------------------------
  ★ 3〜6手即死決着 : {tot_3to6:4d} 局 ({tot_3to6 / total_games * 100:5.2f}%)
  -----------------------------------------------------------------------------
  【純粋な実力勝負】7手以上の対局 (計 {tot_games_7} 局)
  ★ Gen7 勝利     : {g7_wins_7:4d} 局 ({g7_wins_7 / tot_games_7 * 100:5.2f}%)
  ★ Gen6 勝利     : {g6_wins_7:4d} 局 ({g6_wins_7 / tot_games_7 * 100:5.2f}%)
  ★ 引き分け       : {draws_7:4d} 局 ({draws_7 / tot_games_7 * 100:5.2f}%)

【全試合棋譜 CSV ファイル】
  {csv_path}
=================================================================================
"""
    print(summary_text)

    summary_filename = f"summary_Gen7_vs_Gen6_{total_games}games.txt"
    summary_path = os.path.join(repo_dir, summary_filename)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"[サマリー保存完了] {summary_path}")

if __name__ == "__main__":
    main()
