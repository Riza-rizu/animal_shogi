"""
学習用スクリプト
"""

import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split

def get_data():
    print("データセットをロードしています...")
    dataset = np.load("train_data.npy")
    
    # 0〜138列目: 状態ベクトル (139次元)
    X = dataset[:, :139]
    # 139列目: 平均評価値 (1次元)
    y = dataset[:, 139]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"学習データ数: {X_train.shape[0]}")
    print(f"テストデータ数: {X_test.shape[0]}")
    return X_train, y_train, X_test, y_test

def create_model():
    # 入力層 139次元、出力層 1次元の回帰モデル
    model = keras.Sequential([
        keras.layers.Input(shape=(139,)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(64, activation="relu"),
        # 出力値を評価値の範囲（-1.0〜1.0）にするため tanh を使用
        keras.layers.Dense(1, activation="tanh")
    ])
    
    # 損失関数として平均二乗誤差 (MSE) 
    model.compile(
        optimizer="adam",
        loss="mean_squared_error",
        metrics=["mae"]  # 評価指標として平均絶対誤差 (MAE) 
    )
    return model

def main():
    # 1. データの取得
    X_train, y_train, X_test, y_test = get_data()
    
    # 2. モデルの作成
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
        validation_data=(X_test, y_test)
    )
    
    # 4. 評価
    print("テストデータで評価を行っています...")
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss (MSE) = {test_loss:.6f}")
    print(f"Test MAE = {test_mae:.6f}")
    
    # 5. 保存
    model_filename = "animal_shogi_model.keras"
    model.save(model_filename)
    print(f"モデルを保存しました: {model_filename}")

if __name__ == "__main__":
    main()
