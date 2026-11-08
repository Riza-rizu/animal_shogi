"""
量子どうぶつしょうぎ 学習用スクリプト (MLP + MSE / tanh)
"""

import os
import sys
import json
import argparse
import numpy as np
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


def get_data(data_path="train_data.npy"):
    print(f"データセットをロードしています: {data_path} ...")
    dataset = np.load(data_path)

    # 0〜138列目: 状態ベクトル (139次元)
    X = dataset[:, :139].astype(np.float32)
    # 139列目: 評価値 (-1.0〜1.0)
    y = dataset[:, 139].astype(np.float32)

    # シャッフルして 8:2 に分割
    indices = np.random.permutation(len(X))
    split_idx = int(len(X) * 0.8)
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(f"学習データ数: {X_train.shape[0]}")
    print(f"テストデータ数: {X_test.shape[0]}")
    return X_train, y_train, X_test, y_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", action="store_true", help="初期モデル (Gen0) を作成して保存")
    parser.add_argument("--output", type=str, default="quantum_model.keras", help="保存先モデルファイル名")
    parser.add_argument("--data", type=str, default="train_data.npy", help="学習データファイル名")
    parser.add_argument("--epochs", type=int, default=10, help="エポック数")
    parser.add_argument("--batch_size", type=int, default=256, help="バッチサイズ")
    args = parser.parse_args()

    if args.init:
        print("ランダム初期化された初期モデルを作成して保存します...")
        init_model = create_model()
        init_model.summary()
        init_model.save(args.output)
        print(f"初期モデルを保存しました: {args.output}")
        return

    # 1. データ取得
    if not os.path.exists(args.data):
        print(f"エラー: 学習データファイル '{args.data}' が見つかりません。")
        return

    X_train, y_train, X_test, y_test = get_data(args.data)

    # 2. モデル作成
    model = create_model()
    model.summary()

    # 3. 学習
    print(f"学習を開始します (エポック数: {args.epochs}, バッチサイズ: {args.batch_size})...")
    history = model.fit(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_test, y_test),
        verbose=2
    )

    # 4. 評価
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss (MSE) = {test_loss:.6f}")
    print(f"Test MAE = {test_mae:.6f}")

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
