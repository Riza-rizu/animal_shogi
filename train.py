"""
学習用スクリプト (CNN + Binary Cross-Entropy + 100%全件学習)
"""

import os
import numpy as np
from tensorflow import keras

def get_data():
    print("データセットをロードしています: train_data.npy ...")
    dataset = np.load("train_data.npy")
    
    # 0〜138列目: 状態ベクトル (139次元)
    X = dataset[:, :139].astype(np.float32)
    # 139列目: 勝率ラベル (0.0〜1.0)
    y = dataset[:, 139].astype(np.float32)
    
    # 100%全件を学習に投入（テストデータ除外による記憶欠落を防止）
    # データをシャッフル
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X_train = X[indices]
    y_train = y[indices]
    
    print(f"学習データ数: {X_train.shape[0]} (全件学習)")
    return X_train, y_train

def create_model():
    """
    盤面 (4, 3, 11) を CNN で空間認識し、持ち駒・手番 (7) と結合して勝率 (0〜1) を出力するモデル
    """
    inputs = keras.layers.Input(shape=(139,), name="state_input")
    
    # 盤面部分 (132次元) をスライスして (4, 3, 11) に Reshape
    board_vec = inputs[:, :132]
    board_tensor = keras.layers.Reshape((4, 3, 11))(board_vec)
    
    # 持ち駒・手番部分 (7次元)
    meta_vec = inputs[:, 132:]
    
    # 畳み込み層（3x3 フィルタで局所的な駒の利き・位置関係を抽出）
    x = keras.layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu")(board_tensor)
    x = keras.layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu")(x)
    x = keras.layers.Flatten()(x) # 4 * 3 * 64 = 768次元
    
    # 持ち駒・手番と結合
    concat = keras.layers.Concatenate()([x, meta_vec]) # 768 + 7 = 775次元
    
    # 全結合層
    h = keras.layers.Dense(256, activation="relu")(concat)
    h = keras.layers.Dense(128, activation="relu")(h)
    
    # 出力層: 勝率 (0.0〜1.0)
    outputs = keras.layers.Dense(1, activation="sigmoid", name="win_rate")(h)
    
    model = keras.models.Model(inputs=inputs, outputs=outputs)
    
    # 二値交差エントロピー損失 (BCE) で端点の妥協を排除
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["mae"]
    )
    return model

def main():
    # 1. データの取得
    X_train, y_train = get_data()
    
    # 2. モデルの準備 (前世代のモデルが存在すればロードして追加学習、なければスクラッチ作成)
    model_filename = "animal_shogi_model.keras"
    if os.path.exists(model_filename):
        print(f"既存のモデル（前世代）をロードして追加学習（ファインチューニング）します: {model_filename}")
        try:
            model = keras.models.load_model(model_filename)
            # 前世代の知識を急激に壊さないよう、学習率をやや控えめ (0.0005) に設定
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.0005),
                loss="binary_crossentropy",
                metrics=["mae"]
            )
        except Exception as e:
            print(f"警告: 既存モデルのロードに失敗しました ({e})。新規作成します。")
            model = create_model()
    else:
        print("新規モデルを作成します (スクラッチ学習)...")
        model = create_model()
        
    model.summary()
    
    # 3. 学習 
    epochs = 50
    batch_size = 256
    print(f"学習を開始します (エポック数: {epochs}, バッチサイズ: {batch_size})...")
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=2
    )
    
    # 4. 保存
    model.save(model_filename)
    print(f"モデルを保存しました: {model_filename}")

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
