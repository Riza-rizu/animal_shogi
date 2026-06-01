import os
import glob
import pickle
import argparse
import numpy as np

def merge_and_distill(input_dir, output_path, n_min, m_min):
    # output*.pkl の探索
    search_path = os.path.join(input_dir, "output*.pkl")
    pkl_files = glob.glob(search_path)
    
    if not pkl_files:
        print(f"エラー: {input_dir} 内に output*.pkl ファイルが見つかりませんでした。")
        return
        
    print(f"発見したpklファイル数: {len(pkl_files)}")
    
    # 統合用辞書: { state_key: [vals...] }
    global_dict = {}
    
    # 1. 各ファイル読み込みと第1段階蒸留 (ファイル内の出現回数が n_min 以上)
    for pkl_file in pkl_files:
        filename = os.path.basename(pkl_file)
        with open(pkl_file, 'rb') as f:
            try:
                data = pickle.load(f)
            except Exception as e:
                print(f"警告: {filename} の読み込みに失敗しました。スキップします。エラー: {e}")
                continue
        
        # 第1段階のフィルタリング
        filtered_count = 0
        for state_key, vals in data.items():
            if len(vals) >= n_min:
                if state_key not in global_dict:
                    global_dict[state_key] = []
                global_dict[state_key].extend(vals)
                filtered_count += 1
                
        print(f"  {filename}: ユニーク状態数 {len(data)} -> 蒸留後 (n>={n_min}): {filtered_count}")
        
    # 2. 第2段階蒸留 (統合後の出現回数が m_min 以上)
    final_states = []
    final_values = []
    
    distilled_count = 0
    for state_key, vals in global_dict.items():
        if len(vals) >= m_min:
            # 評価値の平均値を算出
            mean_val = np.mean(vals)
            
            final_states.append(list(state_key))
            final_values.append(mean_val)
            distilled_count += 1
            
    print(f"統合完了: 全ユニーク状態数 {len(global_dict)} -> 最終蒸留後 (m>={m_min}): {distilled_count}")
    
    if distilled_count == 0:
        print("警告: 最終データが0件のため、保存をスキップします。")
        return
        
    # Numpy配列の作成
    # final_states: (N, 139)
    # final_values: (N,)
    X = np.array(final_states, dtype=np.float32)
    y = np.array(final_values, dtype=np.float32).reshape(-1, 1)
    
    # 結合して (N, 140) にする
    train_data = np.concatenate([X, y], axis=1)
    
    # 保存
    np.save(output_path, train_data)
    print(f"学習用データを保存しました: {output_path} (形状: {train_data.shape})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="複数プロセスのpklデータを統合・蒸留してtrain_data.npyを作成します。")
    parser.add_argument("--dir", type=str, default=".", help="pklファイルが保存されているディレクトリ (デフォルト: カレントディレクトリ)")
    parser.add_argument("--output", type=str, default="train_data.npy", help="出力先ファイルパス")
    parser.add_argument("-n", type=int, default=1, help="第1段階蒸留: 各プロセスでの最小出現回数 (デフォルト: 1)")
    parser.add_argument("-m", type=int, default=1, help="第2段階蒸留: 統合プロセス全体での最小出現回数 (デフォルト: 1)")
    
    args = parser.parse_args()
    merge_and_distill(args.dir, args.output, args.n, args.m)
