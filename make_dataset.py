"""
シミュレーション結果（output*.pkl）をマージし、
0.1刻みの勝率ビンによるデータバランシング（各ビン最大1,000件）を行って
学習用の NumPy 配列（train_data.npy）を生成するスクリプト。
"""

import os
import sys
import glob
import pickle
import numpy as np
import shutil
import random
import json
import csv
import argparse

# 古いデータの影響力を落とす減衰率 (0.0 〜 1.0)
DECAY_FACTOR = 0.80
CUMULATIVE_FILE = "cumulative_dataset.pkl"
ARCHIVE_DIR = "processed_pkl"
MAX_SAMPLES_PER_BIN = 3000  # 各ビン（0.1刻み）あたりの最大サンプリング数 (合計最大30,000件)

def main():
    parser = argparse.ArgumentParser(description="対局ログのマージと学習データセット作成")
    parser.add_argument("--gen", type=int, default=None, help="現在作成中の目標世代番号（例: Gen4モデルを作成中なら --gen 4）")
    parser.add_argument("--balance", action="store_true", help="0.1刻みビン分割バランシングを行う（軽量実験用）")
    args = parser.parse_args()

    # ディレクトリ内の output*.pkl ファイルを取得
    pkl_files = glob.glob("output*.pkl")
    
    # 累計データベースのロード (存在しない場合は新規作成)
    cumulative_data = {}
    
    if os.path.exists(CUMULATIVE_FILE):
        try:
            with open(CUMULATIVE_FILE, 'rb') as f:
                cumulative_data = pickle.load(f)
            print(f"累計データベースをロードしました。現在の登録局面数: {len(cumulative_data)}")
        except Exception as e:
            print(f"警告: 既存の {CUMULATIVE_FILE} の読み込みに失敗しました。新規作成します。エラー: {e}")
    else:
        print("新規の累計データベースを構築します。")

    if not pkl_files:
        if not cumulative_data:
            print("エラー: 新しい対戦ログ（output*.pkl）が見つかりません。")
            print("先にシミュレーションを実行してデータを生成してください。")
            sys.exit(1)
        else:
            print("新しい対戦ログはありません。既存の累計データベースからサンプリングを行います。")
    else:
        print(f"検出された新しいデータファイル数: {len(pkl_files)}")
        print("新しいデータファイルを読み込み、マージしています...")

        # 新しいデータファイルの内容を一時的に集計
        new_data = {}
        for filepath in pkl_files:
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                    for state_key, values in data.items():
                        if state_key not in new_data:
                            new_data[state_key] = []
                        new_data[state_key].extend(values)
            except Exception as e:
                print(f"警告: {filepath} の読み込み中にエラーが発生しました: {e}")

        print(f"新データ集計完了。ユニーク局面数: {len(new_data)}")
        print("減衰付き加重平均を計算し、累計データベースを更新しています...")

        # 累計データベースの更新
        updated_count = 0
        new_added_count = 0
        
        for state_key, new_vals in new_data.items():
            new_count = len(new_vals)
            new_mean = float(np.mean(new_vals))
            
            if state_key in cumulative_data:
                old_count, old_mean = cumulative_data[state_key]
                decayed_old_count = old_count * DECAY_FACTOR
                merged_count = decayed_old_count + new_count
                merged_mean = (decayed_old_count * old_mean + new_count * new_mean) / merged_count
                cumulative_data[state_key] = (merged_count, merged_mean)
                updated_count += 1
            else:
                cumulative_data[state_key] = (float(new_count), new_mean)
                new_added_count += 1

        print(f"データベース更新完了 (更新: {updated_count}件, 新規追加: {new_added_count}件, 総数: {len(cumulative_data)}件)")

        # 累計データベースの保存
        print(f"累計データベースを保存しています: {CUMULATIVE_FILE} ...")
        with open(CUMULATIVE_FILE, 'wb') as f:
            pickle.dump(cumulative_data, f)

        # 今回の対局結果 S_n(x) [α=1] の生データセット配列を保存
        print("最新世代の対局評価値 S_n(x) [α=1] を配列化しています...")
        num_new = len(new_data)
        current_data_array = np.empty((num_new, 140), dtype=np.float32)
        for idx, (state_key, new_vals) in enumerate(new_data.items()):
            new_mean = float(np.mean(new_vals))
            scaled_sn = np.sign(new_mean) * (np.abs(new_mean) ** 0.5)
            current_data_array[idx, :139] = list(state_key)
            current_data_array[idx, 139] = scaled_sn

        current_data_filename = "current_gen_data.npy"
        np.save(current_data_filename, current_data_array)
        print(f"最新対局結果データ (S_n(x), α=1) を保存しました: {current_data_filename} (Shape: {current_data_array.shape})")

    # =========================================================
    # 0.1刻みの勝率ビンによるデータバランシング（サンプリング）
    # =========================================================
    # 学習用 NumPy 配列 (train_data.npy) の生成
    # =========================================================
    if args.balance:
        print("\n--- データバランシング (0.1刻みビン分割サンプリング) を開始 ---")
        bins = [[] for _ in range(10)]
        for state_key, (count, mean_val) in cumulative_data.items():
            win_rate = (mean_val + 1.0) / 2.0
            win_rate = np.clip(win_rate, 0.0, 1.0)
            bin_idx = min(int(win_rate * 10), 9)
            bins[bin_idx].append((state_key, float(mean_val)))

        sampled_dataset = []
        print(f"各ビンの登録局面数とサンプリング数 (上限: {MAX_SAMPLES_PER_BIN}件):")
        for b_idx in range(10):
            low = b_idx * 0.1
            high = (b_idx + 1) * 0.1
            total_in_bin = len(bins[b_idx])
            if total_in_bin <= MAX_SAMPLES_PER_BIN:
                selected = bins[b_idx]
            else:
                selected = random.sample(bins[b_idx], MAX_SAMPLES_PER_BIN)
            sampled_dataset.extend(selected)
            print(f"  Bin {b_idx} [{low:.1f} - {high:.1f}]: {total_in_bin:6d} 局面中 -> {len(selected):4d} 局面を採択")

        print(f"\n合計サンプリング局面数: {len(sampled_dataset)} 局面")
        random.shuffle(sampled_dataset)
        num_items = len(sampled_dataset)
        dataset_array = np.empty((num_items, 140), dtype=np.float32)
        for idx, (state_key, val) in enumerate(sampled_dataset):
            dataset_array[idx, :139] = list(state_key)
            dataset_array[idx, 139] = val
    else:
        print("\n--- 全局面データセットの生成 (過去の100万局・全件学習仕様) ---")
        num_items = len(cumulative_data)
        dataset_array = np.empty((num_items, 140), dtype=np.float32)
        keys = list(cumulative_data.keys())
        for idx, state_key in enumerate(keys):
            count, mean_val = cumulative_data.pop(state_key)
            state_vec = list(state_key)
            # 符号付き平方根（p=0.5）でスケーリングして0付近を引き伸ばす
            scaled_val = np.sign(mean_val) * (np.abs(mean_val) ** 0.5)
            dataset_array[idx, :139] = state_vec
            dataset_array[idx, 139] = scaled_val
            if (idx + 1) % 1000000 == 0:
                print(f"  - 配列化進行状況: {idx + 1} / {num_items} 局面完了")
        del keys
        del cumulative_data

    output_filename = "train_data.npy"
    np.save(output_filename, dataset_array)
    print(f"学習用データを保存しました: {output_filename} (Shape: {dataset_array.shape})")

    # 処理済みの output*.pkl をアーカイブフォルダに移動
    if pkl_files:
        if not os.path.exists(ARCHIVE_DIR):
            os.makedirs(ARCHIVE_DIR)
        for filepath in pkl_files:
            filename = os.path.basename(filepath)
            dest_path = os.path.join(ARCHIVE_DIR, filename)
            if os.path.exists(dest_path):
                os.remove(dest_path)
            shutil.move(filepath, dest_path)
        print(f"マージ済みの対戦ログ {len(pkl_files)} 件を {ARCHIVE_DIR}/ フォルダに退避しました。")

    # =========================================================
    # 対局ログ（match_log_*.pkl）の集計とCSV/JSON出力
    # =========================================================
    match_log_files = sorted(glob.glob("match_log_*.pkl"))
    if match_log_files:
        print("\n対局ログをマージしています...")
        all_records = []
        for mf in match_log_files:
            try:
                with open(mf, 'rb') as f:
                    records = pickle.load(f)
                    all_records.extend(records)
            except Exception as e:
                print(f"警告: {mf} の読み込みに失敗しました: {e}")

        if all_records:
            total_games = len(all_records)
            p1_wins = sum(1 for r in all_records if r['winner'] == 1)
            p2_wins = sum(1 for r in all_records if r['winner'] == 2)
            draws = sum(1 for r in all_records if r['winner'] not in (1, 2))
            short_games = sum(1 for r in all_records if r.get('is_short_game', False))
            avg_moves = float(np.mean([r['moves'] for r in all_records])) if total_games > 0 else 0.0

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

            # match_stats.json 保存
            stats_filename = "match_stats.json"
            with open(stats_filename, 'w', encoding='utf-8') as f:
                json.dump(match_stats, f, ensure_ascii=False, indent=2)
            if player_gen is not None:
                gen_stats_filename = f"match_stats_gen{player_gen}.json"
                with open(gen_stats_filename, 'w', encoding='utf-8') as f:
                    json.dump(match_stats, f, ensure_ascii=False, indent=2)
                print(f"対局統計サマリーを保存しました: {stats_filename} / {gen_stats_filename}")
            else:
                print(f"対局統計サマリーを保存しました: {stats_filename}")
            print(f"  - 総対戦数: {total_games}, 先手勝率: {match_stats['p1_win_rate']}%, 後手勝率: {match_stats['p2_win_rate']}%, 千日手率: {match_stats['draw_rate']}%, 平均手数: {match_stats['avg_moves']}, 即死(3-6手): {short_games}局")

            # match_log_15000games.csv 保存
            csv_filename = "match_log_15000games.csv"
            with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                header = ["game_id", "winner", "moves", "is_short_game"] + [f"move_{i+1}" for i in range(10)]
                writer.writerow(header)
                for idx, r in enumerate(all_records):
                    m10 = r.get('moves_10', [])
                    row = [idx + 1, r['winner'], r['moves'], r.get('is_short_game', False)]
                    row.extend(m10 + [""] * (10 - len(m10)))
                    writer.writerow(row)
            if player_gen is not None:
                gen_csv_filename = f"match_log_15000games_gen{player_gen}.csv"
                shutil.copy2(csv_filename, gen_csv_filename)
                print(f"全対局CSVを保存しました: {csv_filename} / {gen_csv_filename}")
            else:
                print(f"全対局CSVを保存しました: {csv_filename}")

            # アーカイブ移動
            for mf in match_log_files:
                dest_path = os.path.join(ARCHIVE_DIR, os.path.basename(mf))
                if os.path.exists(dest_path):
                    os.remove(dest_path)
                shutil.move(mf, dest_path)

    print("-" * 50)
    print("make_dataset.py の処理が正常に完了しました。")
    print("-" * 50)

if __name__ == "__main__":
    main()
