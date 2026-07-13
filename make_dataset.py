"""
シミュレーション結果（output*.pkl）をマージし、
学習用の NumPy 配列（train_data.npy）を生成するスクリプト。
"""

import os
import sys
import glob
import pickle
import numpy as np
import shutil

# 古いデータの影響力を落とす減衰率 (0.0 〜 1.0)
# 0.8 の場合、過去の対戦の出現回数（信頼度）が 80% に減衰され、最新データが相対的に重視されます
DECAY_FACTOR = 0.8
CUMULATIVE_FILE = "cumulative_dataset.pkl"
ARCHIVE_DIR = "processed_pkl"

def main():
    # ディレクトリ内の output*.pkl ファイルを取得
    pkl_files = glob.glob("output*.pkl")
    
    # 累計データベースのロード (存在しない場合は新規作成)
    cumulative_data = {}
    db_reconstructed = False
    
    if os.path.exists(CUMULATIVE_FILE):
        try:
            with open(CUMULATIVE_FILE, 'rb') as f:
                cumulative_data = pickle.load(f)
            print(f"累計データベースをロードしました。現在の登録局面数: {len(cumulative_data)}")
        except Exception as e:
            print(f"警告: 既存 of {CUMULATIVE_FILE} の読み込みに失敗しました。新規作成します。エラー: {e}")
    elif os.path.exists("train_data.npy"):
        try:
            print("既存の train_data.npy から累計データベースを復元しています...")
            dataset = np.load("train_data.npy")
            # 過去の出現回数を一律で 10 回として復元します (新しいデータとのブレ防止の重み)
            DEFAULT_OLD_COUNT = 10.0
            for row in dataset:
                # 0〜138列目: 状態ベクトル
                state_vec = tuple(row[:139].tolist())
                # 139列目: 符号付き平方根でスケールされた評価値
                scaled_val = row[139]
                # スケールを元に戻す (逆関数: sign(y) * y^2)
                mean_val = np.sign(scaled_val) * (scaled_val ** 2)
                
                cumulative_data[state_vec] = (DEFAULT_OLD_COUNT, mean_val)
            print(f"既存の train_data.npy から {len(cumulative_data)} 局面を復元しました。")
            db_reconstructed = True
        except Exception as e:
            print(f"警告: train_data.npy からの復元に失敗しました: {e}")
    else:
        print(f"新規の累計データベースを構築します。")

    # 新しいデータファイルがない場合の処理
    if not pkl_files:
        if db_reconstructed:
            # データベースの復元のみが行われた場合、それを保存して正常終了する
            print(f"データベースの復元が完了したため、{CUMULATIVE_FILE} を保存します。")
            with open(CUMULATIVE_FILE, 'wb') as f:
                pickle.dump(cumulative_data, f)
            print("処理を正常終了します。")
            sys.exit(0)
        else:
            print("エラー: 新しい対戦ログ（output*.pkl）が見つかりません。")
            print("また、復元元となる train_data.npy も存在しませんでした。")
            print("先にシミュレーションを実行してデータを生成してください。")
            sys.exit(1)

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

    print(f"新データ集計完了。新規局面数: {len(new_data)}")
    print("減衰付き加重平均を計算し、累計データベースを更新しています...")

    # 累計データベースの更新
    updated_count = 0
    new_added_count = 0
    
    for state_key, new_vals in new_data.items():
        new_count = len(new_vals)
        new_mean = np.mean(new_vals)
        
        if state_key in cumulative_data:
            # 既存局面の場合：減衰付き加重平均を適用
            old_count, old_mean = cumulative_data[state_key]
            
            decayed_old_count = old_count * DECAY_FACTOR
            merged_count = decayed_old_count + new_count
            merged_mean = (decayed_old_count * old_mean + new_count * new_mean) / merged_count
            
            cumulative_data[state_key] = (merged_count, merged_mean)
            updated_count += 1
        else:
            # 新規局面の場合：そのまま追加
            cumulative_data[state_key] = (float(new_count), float(new_mean))
            new_added_count += 1

    print(f"データベース更新完了 (更新: {updated_count}件, 新規追加: {new_added_count}件, 総数: {len(cumulative_data)}件)")

    # 累計データベースの保存
    print(f"累計データベースを保存しています: {CUMULATIVE_FILE} ...")
    with open(CUMULATIVE_FILE, 'wb') as f:
        pickle.dump(cumulative_data, f)

    # 学習用 train_data.npy の生成
    print("学習用 NumPy 配列を生成しています...")
    dataset_list = []
    for state_key, (count, mean_val) in cumulative_data.items():
        state_vec = list(state_key)
        # 符号付き平方根（p=0.5）でスケーリングして0付近を引き伸ばす
        scaled_val = np.sign(mean_val) * (np.abs(mean_val) ** 0.5)
        row = state_vec + [scaled_val]
        dataset_list.append(row)

    dataset_array = np.array(dataset_list, dtype=np.float32)
    output_filename = "train_data.npy"
    np.save(output_filename, dataset_array)

    # 処理が成功したため、読み込んだ output*.pkl をアーカイブフォルダに移動 (二重マージ防止)
    if not os.path.exists(ARCHIVE_DIR):
        os.makedirs(ARCHIVE_DIR)
    
    for filepath in pkl_files:
        filename = os.path.basename(filepath)
        dest_path = os.path.join(ARCHIVE_DIR, filename)
        # 同名ファイルがある場合は上書き
        if os.path.exists(dest_path):
            os.remove(dest_path)
        shutil.move(filepath, dest_path)

    print("-" * 50)
    print(f"処理が完了しました。")
    print(f"出力ファイル: {output_filename}")
    print(f"  - 局面総数 (Shape): {dataset_array.shape[0]}")
    print(f"  - マージ済みの output*.pkl は {ARCHIVE_DIR}/ フォルダに退避しました。")
    print("-" * 50)

if __name__ == "__main__":
    main()
