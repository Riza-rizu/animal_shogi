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

# 古いデータの影響力を落とす減衰率 (0.0 〜 1.0)
DECAY_FACTOR = 0.80
CUMULATIVE_FILE = "cumulative_dataset.pkl"
ARCHIVE_DIR = "processed_pkl"
MAX_SAMPLES_PER_BIN = 1000  # 各ビン（0.1刻み）あたりの最大サンプリング数

def main():
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

    # =========================================================
    # 0.1刻みの勝率ビンによるデータバランシング（サンプリング）
    # =========================================================
    print("\n--- データバランシング (0.1刻みビン分割サンプリング) を開始 ---")
    
    # 10個のビンを準備 (0: [0.0, 0.1), 1: [0.1, 0.2), ..., 9: [0.9, 1.0])
    # 各ビンには (state_key, win_rate) を格納
    bins = [[] for _ in range(10)]
    
    for state_key, (count, mean_val) in cumulative_data.items():
        # mean_val (-1.0〜1.0) を勝率 (0.0〜1.0) に変換
        win_rate = (mean_val + 1.0) / 2.0
        win_rate = np.clip(win_rate, 0.0, 1.0)
        
        # ビンインデックスの決定 (0〜9)
        # 1.0 の場合はビン 9 に所属
        bin_idx = min(int(win_rate * 10), 9)
        bins[bin_idx].append((state_key, win_rate))

    sampled_dataset = []
    print(f"各ビンの登録局面数とサンプリング数 (上限: {MAX_SAMPLES_PER_BIN}件):")
    
    for b_idx in range(10):
        low = b_idx * 0.1
        high = (b_idx + 1) * 0.1
        total_in_bin = len(bins[b_idx])
        
        if total_in_bin <= MAX_SAMPLES_PER_BIN:
            selected = bins[b_idx]
        else:
            # 上限を超える場合はランダムにサンプリング（アンダーサンプリング）
            selected = random.sample(bins[b_idx], MAX_SAMPLES_PER_BIN)
            
        sampled_dataset.extend(selected)
        print(f"  Bin {b_idx} [{low:.1f} - {high:.1f}]: {total_in_bin:6d} 局面中 -> {len(selected):4d} 局面を採択")

    print(f"\n合計サンプリング局面数: {len(sampled_dataset)} 局面")

    # サンプルをシャッフル
    random.shuffle(sampled_dataset)

    # 学習用 NumPy 配列 (train_data.npy) の作成
    num_items = len(sampled_dataset)
    dataset_array = np.empty((num_items, 140), dtype=np.float32)

    for idx, (state_key, win_rate) in enumerate(sampled_dataset):
        dataset_array[idx, :139] = list(state_key)
        dataset_array[idx, 139] = win_rate

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

    print("-" * 50)
    print("make_dataset.py の処理が正常に完了しました。")
    print("-" * 50)

if __name__ == "__main__":
    main()
