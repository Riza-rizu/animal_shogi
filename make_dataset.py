"""
シミュレーション結果（output*.pkl）をマージし、
学習用の NumPy 配列（train_data.npy）を生成するスクリプト。
"""

import os
import sys
import glob
import pickle
import numpy as np

def main():
    # ディレクトリ内の output*.pkl ファイルを取得
    pkl_files = glob.glob("output*.pkl")
    
    if not pkl_files:
        print("エラー: output*.pkl ファイルが見つかりません。")
        print("先にシミュレーションを実行してデータを生成してください。")
        sys.exit(1)

    print(f"検出されたデータファイル数: {len(pkl_files)}")
    print("データファイルを読み込み、マージしています...")

    # 全てのデータを一つの辞書に統合（同一状態の評価値を集計）
    merged_data = {}
    for filepath in pkl_files:
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                for state_key, values in data.items():
                    if state_key not in merged_data:
                        merged_data[state_key] = []
                    merged_data[state_key].extend(values)
        except Exception as e:
            print(f"警告: {filepath} の読み込み中にエラーが発生しました: {e}")

    print(f"マージ完了。ユニークな状態（局面）数: {len(merged_data)}")
    print("各状態の評価値を平均化し、NumPy 配列を生成しています...")

    # 状態ベクトル(139次元)と平均評価値(1次元)を結合して140次元の行ベクトルを作成
    dataset_list = []
    for state_key, values in merged_data.items():
        # state_key は 139次元のタプル
        state_vec = list(state_key)
        # 評価値の平均を計算（-1.0 〜 1.0）
        mean_val = np.mean(values)
        # 符号付き平方根（p=0.5）でスケーリングして0付近を引き伸ばす
        scaled_val = np.sign(mean_val) * (np.abs(mean_val) ** 0.5)
        # 結合して 140次元 にする
        row = state_vec + [scaled_val]
        dataset_list.append(row)

    # NumPy 配列に変換 (形状: [N, 140])
    dataset_array = np.array(dataset_list, dtype=np.float32)

    # train_data.npy として保存
    output_filename = "train_data.npy"
    np.save(output_filename, dataset_array)

    print("-" * 50)
    print(f"処理が完了しました。")
    print(f"出力ファイル: {output_filename}")
    print(f"配列の形状 (Shape): {dataset_array.shape}")
    print(f"  - 0〜138列目: 状態ベクトル (139次元)")
    print(f"  - 139列目: 平均評価値 (1次元, -1.0〜1.0)")
    print("-" * 50)

if __name__ == "__main__":
    main()
