"""
世代自己評価スクリプト: evaluate_generation.py
train_data.npy と学習済みモデルの評価指標および散布図 (scatter_evaluation.png) を出力
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Meiryo', 'Yu Gothic', 'Hiragino Sans', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def calculate_metrics(y_true, y_pred):
    mse = float(np.mean((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 1e-7 else 0.0
    mae = float(np.mean(np.abs(y_true - y_pred)))

    nonzero_mask = np.abs(y_true) > 0.05
    if np.sum(nonzero_mask) > 0:
        sign_match = (np.sign(y_true[nonzero_mask]) == np.sign(y_pred[nonzero_mask]))
        sign_acc = float(np.mean(sign_match) * 100.0)
    else:
        sign_acc = 100.0

    return {
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "sign_accuracy_percent": sign_acc
    }


def main():
    parser = argparse.ArgumentParser(description="モデル評価と散布図生成")
    parser.add_argument("--model", type=str, default="quantum_model.keras", help="評価対象モデル")
    parser.add_argument("--data", type=str, default="train_data.npy", help="評価用データセット")
    parser.add_argument("--gen", type=int, default=1, help="世代番号")
    args = parser.parse_args()

    if not os.path.exists(args.model) or not os.path.exists(args.data):
        print(f"エラー: モデル {args.model} またはデータ {args.data} が見つかりません。")
        return

    import tensorflow as tf
    model = tf.keras.models.load_model(args.model)

    data = np.load(args.data)
    X = data[:, :139]
    y_true = data[:, 139]

    # サンプリング (最大5,000件で散布図を描画)
    if len(X) > 5000:
        idx = np.random.choice(len(X), 5000, replace=False)
        X_sub, y_true_sub = X[idx], y_true[idx]
    else:
        X_sub, y_true_sub = X, y_true

    y_pred_sub = model.predict(X_sub, verbose=0).flatten()
    metrics = calculate_metrics(y_true_sub, y_pred_sub)

    print(f"評価メトリクス (Gen{args.gen}):")
    print(f"  MSE: {metrics['mse']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  R2 : {metrics['r2']:.4f}")
    print(f"  符号一致率: {metrics['sign_accuracy_percent']:.1f}%")

    with open("metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # 散布図プロット
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true_sub, y_pred_sub, alpha=0.3, s=15, c="royalblue")
    plt.plot([-1, 1], [-1, 1], "r--", label="理想線 (y = x)")
    plt.xlabel("実績勝率 / 目標評価値 (y_true)")
    plt.ylabel("モデル予測値 (y_pred)")
    plt.title(f"Gen{args.gen} 評価値散布図 (MSE: {metrics['mse']:.4f}, R2: {metrics['r2']:.4f})")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.legend()
    plt.tight_layout()
    plt.savefig("scatter_evaluation.png", dpi=150)
    plt.close()
    print("散布図を保存しました: scatter_evaluation.png")


if __name__ == "__main__":
    main()
