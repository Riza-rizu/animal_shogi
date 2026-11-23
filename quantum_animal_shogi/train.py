"""
量子どうぶつしょうぎ 学習用スクリプト (MLP + MSE / tanh)
"""

import os
import sys
import json
import argparse
import numpy as np

# CPU実行を強制してGPUドライバエラーおよび無駄なメモリ確保を抑制
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from tensorflow import keras

def create_model():
    """
    全結合型ニューラルネットワーク (MLP: 512 -> 256 -> 1, tanh)
    """
    model = keras.Sequential([
        keras.layers.Input(shape=(139,), name="state_input"),
        keras.layers.Dense(512, activation="relu", name="dense"),
        keras.layers.Dense(256, activation="relu", name="dense_1"),
        keras.layers.Dense(1, activation="tanh", name="eval_value")
    ])

    model.compile(
        optimizer="adam",
        loss="mean_squared_error",
        metrics=["mae"]
    )
    return model


def get_data(data_path="train_data.npy", n_val=10000):
    print(f"データセットをロードしています: {data_path} ...")
    # メモリマップ (mmap_mode='r') でロード
    dataset = np.load(data_path, mmap_mode='r')
    num_samples = len(dataset)
    print(f"データセット総局面数: {num_samples:,}")

    # シャッフルしたインデックスを作成
    indices = np.random.permutation(num_samples)

    # 学習監視用（Validation）として極小の参照サンプル（最大1万局面）を抽出
    val_count = min(n_val, max(100, int(num_samples * 0.01)))
    val_idx = indices[:val_count]

    # 学習データ: 100% すべての局面を学習に投入（探索した局面を一切捨てない）
    X_train = np.array(dataset[indices, :139], dtype=np.float32)
    y_train = np.array(dataset[indices, 139], dtype=np.float32)

    X_val = np.array(dataset[val_idx, :139], dtype=np.float32)
    y_val = np.array(dataset[val_idx, 139], dtype=np.float32)

    # メモリマップおよびインデックス配列を即時解放
    del dataset
    del indices
    del val_idx
    import gc
    gc.collect()

    print(f"学習データ数: {X_train.shape[0]:,} (全データ 100% を投入)")
    print(f"学習監視用データ数: {X_val.shape[0]:,} (極小参照サンプル)")
    return X_train, y_train, X_val, y_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", action="store_true", help="初期モデル (Gen0) を作成して保存")
    parser.add_argument("--output", type=str, default="quantum_model.keras", help="保存先モデルファイル名")
    parser.add_argument("--data", type=str, default="train_data.npy", help="学習データファイル名")
    parser.add_argument("--epochs", type=int, default=10, help="エポック数")
    parser.add_argument("--batch_size", type=int, default=256, help="バッチサイズ")
    parser.add_argument("--n_val", type=int, default=10000, help="学習監視用サンプル数 (デフォルト: 10000)")
    args = parser.parse_args()

    if args.init:
        print("ランダム初期化された初期モデルを作成して保存します...")
        init_model = create_model()
        init_model.summary()
        init_model.save(args.output)
        print(f"初期モデルを保存しました: {args.output}")
        return

    # 1. データ取得 (100% の全データを学習に投入)
    if not os.path.exists(args.data):
        print(f"エラー: 学習データファイル '{args.data}' が見つかりません。")
        return

    X_train, y_train, X_val, y_val = get_data(args.data, n_val=args.n_val)

    # 2. モデル作成
    model = create_model()
    model.summary()

    # 3. 学習 (全局面 100% でパラメータ更新)
    print(f"学習を開始します (エポック数: {args.epochs}, バッチサイズ: {args.batch_size})...")
    history = model.fit(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_val, y_val),
        verbose=2
    )

    # 4. 評価 (監視用サンプルでの損失)
    test_loss, test_mae = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Loss (MSE) = {test_loss:.6f}")
    print(f"Validation MAE = {test_mae:.6f}")

    # 5. 保存
    model.save(args.output)
    print(f"モデルを保存しました: {args.output}")

    # 6. 学習履歴保存
    history_dict = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "test_loss": float(test_loss),
        "test_mae": float(test_mae),
        "loss": [float(v) for v in history.history.get("loss", [])],
        "mae": [float(v) for v in history.history.get("mae", [])],
        "val_loss": [float(v) for v in history.history.get("val_loss", [])],
        "val_mae": [float(v) for v in history.history.get("val_mae", [])]
    }
    with open("training_history.json", "w", encoding="utf-8") as f:
        json.dump(history_dict, f, indent=2)
    print("学習履歴を保存しました: training_history.json")


if __name__ == "__main__":
    main()
