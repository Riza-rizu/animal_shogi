"""
シミュレーション結果（output*.pkl）をマージし、
学習用の NumPy 配列（train_data.npy）を生成するスクリプト。
キーを bytes 型に統一することで、メモリ消費を約1/15に圧縮し、
1,500万局面超でも安全・高速に全件学習データを作成する。
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
import gc

# 古いデータの影響力を落とす減衰率 (0.0 〜 1.0)
DECAY_FACTOR = 0.80
CUMULATIVE_FILE = "cumulative_dataset.pkl"
ARCHIVE_DIR = "processed_pkl"
MAX_SAMPLES_PER_BIN = 3000  # 各ビン（0.1刻み）あたりの最大サンプリング数 (合計最大30,000件)

def to_bytes_key(k):
    """
    盤面キー（intタプル、floatタプル、bytes）を安全に 139バイトの bytes 型に変換
    """
    if isinstance(k, bytes):
        return k
    try:
        return bytes(k)
    except TypeError:
        return bytes([int(x) for x in k])

def main():
    parser = argparse.ArgumentParser(description="対局ログのマージと学習データセット作成")
    parser.add_argument("--gen", type=int, default=None, help="現在作成中の目標世代番号（例: Gen4モデルを作成中なら --gen 4）")
    parser.add_argument("--balance", action="store_true", help="0.1刻みビン分割バランシングを行う（軽量実験用）")
    args = parser.parse_args()

    # 過去のテンポラリファイルが残っていたら掃除
    for old_temp in glob.glob("temp_chunk_*.pkl"):
        try:
            os.remove(old_temp)
        except Exception:
            pass

    # ディレクトリ内の output*.pkl ファイルを取得
    pkl_files = glob.glob("output*.pkl")
    cumulative_data = {}

    if not pkl_files:
        if not os.path.exists(CUMULATIVE_FILE):
            print("エラー: 新しい対戦ログ（output*.pkl）が見つかりません。")
            print("先にシミュレーションを実行してデータを生成してください。")
            sys.exit(1)
        else:
            print("新しい対戦ログはありません。既存の累計データベースを使用します。")
            with open(CUMULATIVE_FILE, 'rb') as f:
                cumulative_data = pickle.load(f)
            print(f"累計データベースをロードしました。現在の登録局面数: {len(cumulative_data)}")
            first_key = next(iter(cumulative_data)) if cumulative_data else None
            if first_key is not None and not isinstance(first_key, bytes):
                bytes_cumulative = {}
                while cumulative_data:
                    k, v = cumulative_data.popitem()
                    bytes_cumulative[to_bytes_key(k)] = v
                cumulative_data = bytes_cumulative
                del bytes_cumulative
                gc.collect()
    else:
        print(f"検出された新しいデータファイル数: {len(pkl_files)}")
        print("【ステップ1】 新しい対戦ログを集計しています (bytesキー・省メモリ高速集計)...")
        sys.stdout.flush()

        # 新しいデータファイルの内容を集計 (キーを bytes 型にしてメモリ消費を約2.5GBに抑制)
        new_data_stats = {}
        for filepath in pkl_files:
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                for state_key, values in data.items():
                    byte_key = to_bytes_key(state_key)
                    v_count = len(values)
                    v_sum = float(sum(values))
                    if byte_key in new_data_stats:
                        new_data_stats[byte_key][0] += v_count
                        new_data_stats[byte_key][1] += v_sum
                    else:
                        new_data_stats[byte_key] = [v_count, v_sum]
                del data
                gc.collect()
            except Exception as e:
                print(f"警告: {filepath} の読み込み中にエラーが発生しました: {e}")

        num_new = len(new_data_stats)
        print(f"新データ集計完了。ユニーク局面数: {num_new} (メモリ使用量: 軽微)")
        sys.stdout.flush()

        # 最新世代の対局評価値 S_n(x) [α=1] を配列化して保存
        print("最新世代の対局評価値 S_n(x) [α=1] を保存しています...")
        sys.stdout.flush()
        current_data_array = np.empty((num_new, 140), dtype=np.float32)
        for idx, (byte_key, (new_count, new_sum)) in enumerate(new_data_stats.items()):
            new_mean = new_sum / new_count
            scaled_sn = np.sign(new_mean) * (np.abs(new_mean) ** 0.5)
            current_data_array[idx, :139] = list(byte_key)
            current_data_array[idx, 139] = scaled_sn

        current_data_filename = "current_gen_data.npy"
        np.save(current_data_filename, current_data_array)
        print(f"最新対局結果データ (S_n(x), α=1) を保存しました: {current_data_filename} (Shape: {current_data_array.shape})")
        del current_data_array
        gc.collect()
        sys.stdout.flush()

        # 【ステップ2】 累計データベースのロードと bytes 化
        print("\n【ステップ2】 累計データベースをロードしています...")
        sys.stdout.flush()
        cumulative_data = {}
        if os.path.exists(CUMULATIVE_FILE):
            with open(CUMULATIVE_FILE, 'rb') as f:
                cumulative_data = pickle.load(f)
            print(f"累計データベースをロードしました。既存局面数: {len(cumulative_data)}")
        else:
            print("新規の累計データベースを構築します。")
        sys.stdout.flush()

        # 累計DBがタプルキーの場合は、省メモリ化のため bytes キーに一括移行 (60GB -> 4.5GB へ激減)
        first_key = next(iter(cumulative_data)) if cumulative_data else None
        if first_key is not None and not isinstance(first_key, bytes):
            print("累計DBのキーをタプル型から bytes 型に変換し、メモリを極小化しています...")
            sys.stdout.flush()
            bytes_cumulative = {}
            use_int_cast = False
            try:
                bytes(first_key)
            except TypeError:
                use_int_cast = True

            idx = 0
            num_total = len(cumulative_data)
            while cumulative_data:
                k, v = cumulative_data.popitem()
                if use_int_cast:
                    bk = bytes([int(x) for x in k])
                else:
                    try:
                        bk = bytes(k)
                    except TypeError:
                        bk = bytes([int(x) for x in k])
                bytes_cumulative[bk] = v
                idx += 1
                if idx % 3000000 == 0:
                    print(f"  - bytes変換進行状況: {idx} / {num_total} 局面完了")
                    sys.stdout.flush()

            cumulative_data = bytes_cumulative
            del bytes_cumulative
            gc.collect()
            print(f"キーの bytes 化完了。メモリ消費を劇的に圧縮しました (残存局面数: {len(cumulative_data)})")
            sys.stdout.flush()

        # 累計データベースの更新 (減衰付き加重平均)
        print("減衰付き加重平均を計算し、累計データベースを更新しています...")
        sys.stdout.flush()
        updated_count = 0
        new_added_count = 0

        while new_data_stats:
            byte_key, (new_count, new_sum) = new_data_stats.popitem()
            new_mean = new_sum / new_count
            if byte_key in cumulative_data:
                old_count, old_mean = cumulative_data[byte_key]
                decayed_old_count = old_count * DECAY_FACTOR
                merged_count = decayed_old_count + new_count
                merged_mean = (decayed_old_count * old_mean + new_sum) / merged_count
                cumulative_data[byte_key] = (merged_count, merged_mean)
                updated_count += 1
            else:
                cumulative_data[byte_key] = (float(new_count), float(new_mean))
                new_added_count += 1

        del new_data_stats
        gc.collect()
        print(f"データベース更新完了 (更新: {updated_count}件, 新規追加: {new_added_count}件, 総数: {len(cumulative_data)}件)")
        sys.stdout.flush()

        # 累計データベースの保存 (bytesキーで保存されるため次回以降も爆速・超軽量)
        print(f"累計データベースを保存しています: {CUMULATIVE_FILE} ...")
        sys.stdout.flush()
        with open(CUMULATIVE_FILE, 'wb') as f:
            pickle.dump(cumulative_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("累計データベースの保存が完了しました。")
        sys.stdout.flush()

    # =========================================================
    # 【ステップ3】 学習用 NumPy 配列 (train_data.npy) の生成
    # =========================================================
    if args.balance:
        print("\n--- データバランシング (0.1刻みビン分割サンプリング) を開始 ---")
        bins = [[] for _ in range(10)]
        for byte_key, (count, mean_val) in cumulative_data.items():
            win_rate = (mean_val + 1.0) / 2.0
            win_rate = np.clip(win_rate, 0.0, 1.0)
            bin_idx = min(int(win_rate * 10), 9)
            bins[bin_idx].append((byte_key, float(mean_val)))

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
        for idx, (byte_key, val) in enumerate(sampled_dataset):
            dataset_array[idx, :139] = list(byte_key)
            dataset_array[idx, 139] = val
    else:
        print("\n--- 全局面データセットの生成 (過去の100万局・全件学習仕様: popitem省メモリ) ---")
        num_items = len(cumulative_data)
        dataset_array = np.empty((num_items, 140), dtype=np.float32)
        idx = 0
        while cumulative_data:
            byte_key, (count, mean_val) = cumulative_data.popitem()
            state_vec = list(byte_key)
            # 符号付き平方根（p=0.5）でスケーリングして0付近を引き伸ばす
            scaled_val = np.sign(mean_val) * (np.abs(mean_val) ** 0.5)
            dataset_array[idx, :139] = state_vec
            dataset_array[idx, 139] = scaled_val
            idx += 1
            if idx % 1000000 == 0:
                print(f"  - 配列化進行状況: {idx} / {num_items} 局面完了")
                sys.stdout.flush()
        del cumulative_data
        gc.collect()

    output_filename = "train_data.npy"
    np.save(output_filename, dataset_array)
    print(f"学習用データを保存しました: {output_filename} (Shape: {dataset_array.shape})")
    del dataset_array
    gc.collect()
    sys.stdout.flush()

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
