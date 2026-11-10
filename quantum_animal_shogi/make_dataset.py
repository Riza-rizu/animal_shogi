"""
シミュレーション結果 (output*.pkl, match_log_*.pkl) をマージし、
学習用の NumPy 配列 (train_data.npy) および統計ログを生成するスクリプト。
"""

import os
import sys
import glob
import pickle
import numpy as np
import shutil
import json
import csv
import argparse

DECAY_FACTOR = 0.80
CUMULATIVE_FILE = "cumulative_dataset.pkl"
ARCHIVE_DIR = "processed_pkl"


def main():
    parser = argparse.ArgumentParser(description="対局ログのマージと学習データセット作成")
    parser.add_argument("--gen", type=int, default=None, help="目標世代番号")
    args = parser.parse_args()

    # ディレクトリ内の output*.pkl ファイルを取得
    pkl_files = sorted(glob.glob("output*.pkl"))
    log_files = sorted(glob.glob("match_log_*.pkl"))

    if not pkl_files and not os.path.exists(CUMULATIVE_FILE):
        print("エラー: output*.pkl が見つからず、cumulative_dataset.pkl も存在しません。")
        sys.exit(1)

    cumulative_data = {}
    if os.path.exists(CUMULATIVE_FILE):
        print(f"既存の累計データセット {CUMULATIVE_FILE} をロード中...")
        with open(CUMULATIVE_FILE, 'rb') as f:
            cumulative_data = pickle.load(f)

        # 既存データの減衰処理
        if pkl_files:
            print(f"既存データの訪問回数・評価値に減衰率 {DECAY_FACTOR} を適用します...")
            for k in list(cumulative_data.keys()):
                cumulative_data[k][0] *= DECAY_FACTOR
                cumulative_data[k][1] *= DECAY_FACTOR

    # 新規 output*.pkl のマージ
    current_gen_data = {}
    if pkl_files:
        print(f"新規の対戦ログ {len(pkl_files)} 個をマージ中...")
        for pkl_file in pkl_files:
            with open(pkl_file, 'rb') as f:
                worker_data = pickle.load(f)
            for k, (visits, val_sum) in worker_data.items():
                if k not in cumulative_data:
                    cumulative_data[k] = [0.0, 0.0]
                cumulative_data[k][0] += visits
                cumulative_data[k][1] += val_sum

                if k not in current_gen_data:
                    current_gen_data[k] = [0.0, 0.0]
                current_gen_data[k][0] += visits
                current_gen_data[k][1] += val_sum

        # 累計ファイルの保存
        print(f"累計データベースを保存しています: {CUMULATIVE_FILE} (ユニーク局面数: {len(cumulative_data)})")
        with open(CUMULATIVE_FILE, 'wb') as f:
            pickle.dump(cumulative_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    # train_data.npy の作成 (139次元 + 評価値1次元 = 140次元)
    print("train_data.npy を構築中...")
    num_samples = len(cumulative_data)
    dataset = np.zeros((num_samples, 140), dtype=np.float32)

    for i, (b_key, (visits, val_sum)) in enumerate(cumulative_data.items()):
        # bytes から 139次元 float32 ベクトルを復元
        s_vec = np.frombuffer(b_key, dtype=np.float32)
        # 評価値 (平均値) を -1.0 〜 1.0 にクリップ
        avg_eval = float(np.clip(val_sum / visits if visits > 0 else 0.0, -1.0, 1.0))
        dataset[i, :139] = s_vec
        dataset[i, 139] = avg_eval

    np.save("train_data.npy", dataset)
    print(f"train_data.npy を保存しました (形状: {dataset.shape})")

    # 対局ログ (match_log_*.pkl) の集約と統計保存
    if log_files:
        print(f"\n対局ログ {len(log_files)} 個を集約中...")
        all_records = []
        for log_f in log_files:
            try:
                with open(log_f, 'rb') as f:
                    records = pickle.load(f)
                    all_records.extend(records)
            except Exception as e:
                print(f"警告: {log_f} の読み込みに失敗しました: {e}")

        if all_records:
            total_games = len(all_records)
            p1_wins = sum(1 for r in all_records if r.get('winner') == 1)
            p2_wins = sum(1 for r in all_records if r.get('winner') == 2)
            draws = sum(1 for r in all_records if r.get('winner') not in (1, 2))
            short_games = sum(1 for r in all_records if r.get('is_short_game', False))
            avg_moves = float(np.mean([r.get('moves', 0) for r in all_records])) if total_games > 0 else 0.0

            player_gen = (args.gen - 1) if args.gen is not None else None
            match_stats = {
                "player_generation": f"Gen{player_gen}" if player_gen is not None else "Unknown",
                "created_model_generation": f"Gen{args.gen}" if args.gen is not None else "Unknown",
                "total_games": total_games,
                "p1_wins": p1_wins,
                "p1_win_rate": round(p1_wins / total_games * 100.0, 2) if total_games > 0 else 0.0,
                "p2_wins": p2_wins,
                "p2_win_rate": round(p2_wins / total_games * 100.0, 2) if total_games > 0 else 0.0,
                "draws": draws,
                "draw_rate": round(draws / total_games * 100.0, 2) if total_games > 0 else 0.0,
                "avg_moves": round(avg_moves, 2),
                "short_games_3_to_6": short_games,
                "short_game_rate": round(short_games / total_games * 100.0, 2) if total_games > 0 else 0.0
            }

            # match_stats.json 保存 (第6期と完全同一)
            with open("match_stats.json", 'w', encoding='utf-8') as f:
                json.dump(match_stats, f, ensure_ascii=False, indent=2)
            if player_gen is not None:
                gen_stats_filename = f"match_stats_gen{player_gen}.json"
                with open(gen_stats_filename, 'w', encoding='utf-8') as f:
                    json.dump(match_stats, f, ensure_ascii=False, indent=2)
                print(f"対局統計サマリーを保存しました: match_stats.json / {gen_stats_filename}")
            else:
                print(f"対局統計サマリーを保存しました: match_stats.json")
            print(f"  - 総対戦数: {total_games}, 先手勝率: {match_stats['p1_win_rate']}%, 後手勝率: {match_stats['p2_win_rate']}%, 千日手率: {match_stats['draw_rate']}%, 平均手数: {match_stats['avg_moves']}, 即死(3-6手): {short_games}局 ({match_stats['short_game_rate']}%)")

            # CSV 保存 (第6期と完全同一の10手目までのヘッダー)
            player_gen_str = f"_gen{player_gen}" if player_gen is not None else ""
            csv_filename = f"match_log_{total_games}games{player_gen_str}.csv"
            header = ["game_id", "winner", "moves", "is_short_game"] + [f"move_{i+1}" for i in range(10)]
            with open(csv_filename, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for idx, r in enumerate(all_records):
                    gid = r.get("game_id", idx + 1)
                    w = r.get("winner", 0)
                    m = r.get("moves", 0)
                    is_short = r.get("is_short_game", False)
                    moves_10 = r.get("moves_10", [])
                    row = [gid, w, m, is_short]
                    for step_idx in range(10):
                        row.append(moves_10[step_idx] if step_idx < len(moves_10) else "")
                    writer.writerow(row)
            print(f"対局ログCSVを保存しました: {csv_filename}")

    # 一時 pkl ファイルの退避
    if pkl_files:
        os.makedirs(ARCHIVE_DIR, exist_ok=True)
        for f in pkl_files + log_files:
            try:
                shutil.move(f, os.path.join(ARCHIVE_DIR, os.path.basename(f)))
            except Exception:
                pass


if __name__ == "__main__":
    main()
