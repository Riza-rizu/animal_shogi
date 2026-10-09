"""
深さ2探索 (depth=2) パイプライン計算時間計測ベンチマークスクリプト
- 既存の Gen5 モデルを使用
- depth=2, beta=10.0 で 1,000 局の自己対戦シミュレーション
- データセット変換 -> モデル学習 (10 epochs) までの一連の所要時間を精密計測
- 局数別 (1万局, 5万局, 10万局, 20万局, 50万局, 100万局) の完了予測時間を自動算出
"""

import os
import sys
import time
import shutil
import glob
import pickle
import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

# TensorFlow 消音
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings("ignore")

repo_dir = os.path.dirname(os.path.abspath(__file__))
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)

from board import Board, state2vec
from player import ModelPlayer
from engine import GameEngine
from constants import PLAYER_1, PLAYER_2

def simulate_worker(worker_id, games_for_worker, model_path, beta, depth, seed):
    """並列ワーカー: depth=2 で自己対戦を行い、局面辞書と対局ログを返す"""
    import os, sys
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')
    import numpy as np
    np.random.seed(seed)

    player = ModelPlayer(PLAYER_1, model_path_or_model=model_path, beta=beta, depth=depth)
    
    local_data = {}  # state_bytes -> [score_sum, count]
    move_counts = []
    winners = []
    
    for g in range(games_for_worker):
        board = Board()
        board.setup()
        current_p = PLAYER_1
        history = []
        max_moves = 200
        winner = None

        for m in range(max_moves):
            s_vec = state2vec(board, current_p)
            history.append((s_vec, current_p))
            
            legal = GameEngine.get_legal_moves(board, current_p)
            if not legal:
                winner = PLAYER_2 if current_p == PLAYER_1 else PLAYER_1
                break
            
            move = player.select_move(board)
            if move is None:
                winner = PLAYER_2 if current_p == PLAYER_1 else PLAYER_1
                break
            
            board.apply_move(move, current_p)
            
            # トライ勝利
            if board.check_try(current_p):
                winner = current_p
                break
                
            current_p = PLAYER_2 if current_p == PLAYER_1 else PLAYER_1

        total_moves = len(history)
        move_counts.append(total_moves)
        winners.append(winner)

        # 報酬付与 (勝者: +1, 敗者: -1, 引分: 0)
        for s_vec, pid in history:
            k = bytes(int(x) for x in s_vec)
            if winner is None:
                reward = 0.0
            elif winner == pid:
                reward = 1.0
            else:
                reward = -1.0
            
            if k not in local_data:
                local_data[k] = [reward, 1]
            else:
                local_data[k][0] += reward
                local_data[k][1] += 1

    return local_data, move_counts, winners

def format_time(seconds):
    """秒数を読みやすい形式に変換"""
    if seconds < 60:
        return f"{seconds:.1f} 秒"
    elif seconds < 3600:
        m = int(seconds // 60)
        s = seconds % 60
        return f"{m} 分 {s:.1f} 秒"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h} 時間 {m} 分"

def main():
    parser = argparse.ArgumentParser(description="depth=2 ベンチマーク・所要時間計測")
    parser.add_argument("--model", type=str, default="experiments/depth0_pure_mse/gen5/model.keras",
                        help="基準モデル (デフォルト: 新Gen5)")
    parser.add_argument("--games", type=int, default=1000, help="テスト対局数 (デフォルト: 1000)")
    parser.add_argument("--workers", type=int, default=None, help="並列プロセス数 (デフォルト: CPUコア数)")
    parser.add_argument("--beta", type=float, default=10.0, help="逆温度 beta")
    parser.add_argument("--depth", type=int, default=2, help="探索深さ (デフォルト: 2)")
    args = parser.parse_args()

    num_workers = args.workers or max(1, multiprocessing.cpu_count() - 2)
    print("=" * 75)
    print("      【深さ2 (depth=2) 自己対戦 & 学習パイプライン ベンチマーク】")
    print("=" * 75)
    print(f"  * 基準モデル         : {args.model}")
    print(f"  * テスト対局数       : {args.games:,} 局")
    print(f"  * 探索深さ (depth)   : {args.depth} (Min-Max 2手読み)")
    print(f"  * 逆温度 (beta)      : {args.beta}")
    print(f"  * 使用プロセス並列数 : {num_workers} 並列")
    print("-" * 75)

    if not os.path.exists(args.model):
        print(f"[エラー] モデルファイルが見つかりません: {args.model}")
        sys.exit(1)

    # ----------------------------------------------------
    # フェーズ 1: 自己対戦シミュレーション
    # ----------------------------------------------------
    print(f"\n[フェーズ 1] 自己対戦シミュレーション開始 ({args.games:,} 局, {num_workers} 並列)...")
    start_sim_time = time.time()

    games_per_worker = [args.games // num_workers] * num_workers
    for i in range(args.games % num_workers):
        games_per_worker[i] += 1

    all_data = {}
    all_move_counts = []
    all_winners = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for wid in range(num_workers):
            n_g = games_per_worker[wid]
            seed = 10000 + wid
            futures.append(executor.submit(
                simulate_worker, wid, n_g, args.model, args.beta, args.depth, seed
            ))

        completed_games = 0
        for f in futures:
            w_data, w_moves, w_win = f.result()
            all_move_counts.extend(w_moves)
            all_winners.extend(w_win)
            for k, (s_sum, cnt) in w_data.items():
                if k not in all_data:
                    all_data[k] = [s_sum, cnt]
                else:
                    all_data[k][0] += s_sum
                    all_data[k][1] += cnt
            completed_games += len(w_moves)
            elapsed = time.time() - start_sim_time
            print(f"  >> 完了進捗: {completed_games:,} / {args.games:,} 局 (経過: {format_time(elapsed)})")

    sim_elapsed = time.time() - start_sim_time
    sec_per_game = sim_elapsed / args.games
    avg_moves = sum(all_move_counts) / len(all_move_counts)
    print(f"[フェーズ 1 完了] 所要時間: {format_time(sim_elapsed)}")
    print(f"  * 1局あたり所要時間 : {sec_per_game:.4f} 秒 (並列全体での実効スループット)")
    print(f"  * 1分あたり処理局数 : {60.0 / sec_per_game:.1f} 局/分")
    print(f"  * 平均手数           : {avg_moves:.2f} 手")
    print(f"  * 収集ユニーク局面数 : {len(all_data):,} 局面")

    # ----------------------------------------------------
    # フェーズ 2: データセット作成 (NumPy変換)
    # ----------------------------------------------------
    print("\n[フェーズ 2] 学習データセット作成 (NumPy配列化)...")
    start_data_time = time.time()
    
    rows = []
    for k, (s_sum, cnt) in all_data.items():
        state_vec = np.frombuffer(k, dtype=np.uint8).astype(np.float32)
        val = float(s_sum / cnt)
        rows.append(np.append(state_vec, val))
    
    train_data = np.array(rows, dtype=np.float32)
    bench_data_path = "bench_train_data.npy"
    np.save(bench_data_path, train_data)
    
    data_elapsed = time.time() - start_data_time
    print(f"[フェーズ 2 完了] 所要時間: {format_time(data_elapsed)} (データ件数: {len(train_data):,} 件)")

    # ----------------------------------------------------
    # フェーズ 3: モデル学習 (10エポック)
    # ----------------------------------------------------
    print("\n[フェーズ 3] モデル学習実行 (10 epochs, batch_size=256)...")
    start_train_time = time.time()
    
    from tensorflow import keras
    from sklearn.model_selection import train_test_split
    
    X = train_data[:, :139]
    y = train_data[:, 139]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = keras.Sequential([
        keras.layers.Input(shape=(139,), name="state_input"),
        keras.layers.Dense(512, activation="relu", name="dense"),
        keras.layers.Dense(256, activation="relu", name="dense_1"),
        keras.layers.Dense(1, activation="tanh", name="eval_value")
    ])
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=10,
        batch_size=256,
        verbose=0
    )
    
    train_elapsed = time.time() - start_train_time
    val_loss = history.history['val_loss'][-1]
    val_mae = history.history['val_mae'][-1]
    print(f"[フェーズ 3 完了] 所要時間: {format_time(train_elapsed)} (Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f})")

    # 一時ファイルの削除
    if os.path.exists(bench_data_path):
        os.remove(bench_data_path)

    total_pipeline_time = sim_elapsed + data_elapsed + train_elapsed

    # ----------------------------------------------------
    # 総合結果 & スケーリング予測レポート
    # ----------------------------------------------------
    print("\n" + "=" * 75)
    print("                    【ベンチマーク結果 & スケーリング予測】")
    print("=" * 75)
    print(f"■ 今回の計測実績 (テスト対局数: {args.games:,} 局 / {num_workers} 並列):")
    print(f"  1. 自己対戦シミュレーション : {format_time(sim_elapsed):>12s}  ({sim_elapsed/total_pipeline_time*100:5.1f}%)")
    print(f"  2. データセット作成         : {format_time(data_elapsed):>12s}  ({data_elapsed/total_pipeline_time*100:5.1f}%)")
    print(f"  3. モデル学習 (10 epochs)   : {format_time(train_elapsed):>12s}  ({train_elapsed/total_pipeline_time*100:5.1f}%)")
    print(f"  -------------------------------------------------------------")
    print(f"  ★ 合計所要時間             : {format_time(total_pipeline_time):>12s}")
    print(f"  ★ 1局あたり実効処理速度    : {sec_per_game:.4f} 秒/局")

    print("\n■ 対局数別の完了予測時間一覧 (同並列環境での推定):")
    print(f"  +----------------+---------------------+---------------------+")
    print(f"  |   総対局数     |  シミュレーション   |  パイプライン総時間 |")
    print(f"  +----------------+---------------------+---------------------+")
    
    target_games_list = [5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000]
    for tg in target_games_list:
        est_sim = sec_per_game * tg
        # データセット作成と学習時間はデータ量に応じて若干増加
        est_data_train = (data_elapsed + train_elapsed) * max(1.0, (tg / args.games) ** 0.5)
        est_total = est_sim + est_data_train
        print(f"  | {tg:>12,d} 局 | {format_time(est_sim):>19s} | {format_time(est_total):>19s} |")
    print(f"  +----------------+---------------------+---------------------+")
    print("=" * 75)

    # レポートファイル保存
    out_txt = "benchmark_depth2_result.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"depth=2 ベンチマーク結果 (計測対局数: {args.games}局, 並列数: {num_workers})\n")
        f.write(f"1局あたり実効秒数: {sec_per_game:.4f} 秒/局\n")
        f.write(f"1,000局総時間: {format_time(total_pipeline_time)}\n")
        f.write("\n対局数別予測:\n")
        for tg in target_games_list:
            est_sim = sec_per_game * tg
            f.write(f"  {tg:>10,d} 局: 約 {format_time(est_sim)}\n")
    print(f"\n[保存完了] 詳細結果を {out_txt} に保存しました。")

if __name__ == "__main__":
    main()
