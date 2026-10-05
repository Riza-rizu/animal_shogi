"""
学習用スクリプト (CNN + Binary Cross-Entropy + 100%全件学習)
"""

import os
import json
import numpy as np
from tensorflow import keras

from sklearn.model_selection import train_test_split

def get_data():
    print("データセットをロードしています: train_data.npy ...")
    dataset = np.load("train_data.npy")
    
    # 0〜138列目: 状態ベクトル (139次元)
    X = dataset[:, :139].astype(np.float32)
    # 139列目: 評価値 (-1.0〜1.0)
    y = dataset[:, 139].astype(np.float32)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"学習データ数: {X_train.shape[0]}")
    print(f"テストデータ数: {X_test.shape[0]}")
    return X_train, y_train, X_test, y_test

def create_model():
    """
    全結合型ニューラルネットワーク (MLP: 512 -> 256 -> 1)
    総パラメータ数: 203,265 (過去の正常進化実績構成)
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

def main():
    # 1. データの取得
    X_train, y_train, X_test, y_test = get_data()
    
    # 2. モデルの作成
    model_filename = "animal_shogi_model.keras"
    print("全結合モデル (512 -> 256 -> 1, MSE/tanh) を作成します...")
    model = create_model()
    model.summary()
    
    # 3. 学習 
    epochs = 10
    batch_size = 256
    print(f"学習を開始します (エポック数: {epochs}, バッチサイズ: {batch_size})...")
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=2
    )
    
    # 4. 評価
    print("テストデータで評価を行っています...")
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss (MSE) = {test_loss:.6f}")
    print(f"Test MAE = {test_mae:.6f}")
    
    # 5. 保存
    model.save(model_filename)
    print(f"モデルを保存しました: {model_filename}")
    
    # 6. 学習履歴ログの保存 (training_history.json)
    history_dict = {
        "epochs": epochs,
        "batch_size": batch_size,
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
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--init":
        print("ランダム初期化された初期モデル (Gen0) を作成して保存します...")
        init_model = create_model()
        init_model.summary()
        model_filename = "animal_shogi_model.keras"
        init_model.save(model_filename)
        print(f"初期モデルを保存しました: {model_filename}")
    else:
        main()
