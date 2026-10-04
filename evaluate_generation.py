"""
世代自己評価スクリプト: evaluate_generation.py
- カレントディレクトリの train_data.npy と animal_shogi_model.keras を評価
- 散布図画像 scatter_evaluation.png (左: 散布図, 右: 指標表) を生成
- 指標数値 metrics.json を出力
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Meiryo', 'Yu Gothic', 'Hiragino Sans', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def sample_uniform_by_bins(X, y, n_bins=20, n_samples_per_bin=100, seed=42):
    """-1.0〜1.0 を均等サンプリング"""
    np.random.seed(seed)
    bin_edges = np.linspace(-1.0, 1.0, n_bins + 1)
    sampled_indices = []

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            in_bin = np.where((y >= low) & (y <= high))[0]
        else:
            in_bin = np.where((y >= low) & (y < high))[0]

        if len(in_bin) == 0:
            continue
        if len(in_bin) <= n_samples_per_bin:
            sampled_indices.extend(in_bin)
        else:
            chosen = np.random.choice(in_bin, size=n_samples_per_bin, replace=False)
            sampled_indices.extend(chosen)

    sampled_indices = np.array(sampled_indices)
    np.random.shuffle(sampled_indices)
    return sampled_indices


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

    extreme_mask = np.abs(y_true) >= 0.8
    if np.sum(extreme_mask) > 0:
        extreme_mae = float(np.mean(np.abs(y_true[extreme_mask] - y_pred[extreme_mask])))
        extreme_count = int(np.sum(extreme_mask))
    else:
        extreme_mae = 0.0
        extreme_count = 0

    return {
        "mse": round(mse, 6),
        "r2": round(r2, 4),
        "mae": round(mae, 4),
        "sign_acc": round(sign_acc, 2),
        "extreme_mae": round(extreme_mae, 4),
        "extreme_count": extreme_count,
        "total_count": len(y_true)
    }


def plot_scatter(gen_label, y_true, y_pred, metrics, output_filename):
    fig, (ax_plot, ax_text) = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={'width_ratios': [1.3, 1.0]})

    # 散布図
    ax_plot.scatter(y_true, y_pred, alpha=0.35, color='#1f77b4', edgecolors='none', s=25, label='サンプリング局面')
    ax_plot.plot([-1.05, 1.05], [-1.05, 1.05], color='#d62728', linestyle='--', linewidth=2, label='理想線 (y = x)')
    ax_plot.set_xlim([-1.05, 1.05])
    ax_plot.set_ylim([-1.05, 1.05])
    ax_plot.set_xlabel('教師データ正解評価値 (y_true)', fontsize=12, fontweight='bold')
    ax_plot.set_ylabel('モデル単体推論値 (y_pred)', fontsize=12, fontweight='bold')
    ax_plot.set_title(f'【{gen_label}】 自己フィッティング散布図', fontsize=14, fontweight='bold', pad=12)
    ax_plot.grid(True, linestyle=':', alpha=0.6)
    ax_plot.legend(loc='upper left', fontsize=10)
    ax_plot.axhline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    ax_plot.axvline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)

    # 定量指標表
    ax_text.axis('off')
    table_data = [
        ["評価指標", "測定値", "理想 / 目標"],
        ["MSE (平均二乗誤差)", f"{metrics['mse']:.5f}", "0.00000 (完全一致)"],
        ["R² (決定係数)", f"{metrics['r2']:.4f}", "1.0000 (完全相関)"],
        ["MAE (平均絶対誤差)", f"±{metrics['mae']:.4f}", "0.0000 (ズレなし)"],
        ["勝敗方向一致率", f"{metrics['sign_acc']:.2f} %", "100.0 % (符号完全一致)"],
        ["極値誤差 (|y|≧0.8)", f"±{metrics['extreme_mae']:.4f}", "0.0000 (自滅/詰み認識)"]
    ]

    table = ax_text.table(cellText=table_data, colWidths=[0.45, 0.28, 0.35], cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 2.2)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#2c3e50')
            cell.set_text_props(color='white', fontweight='bold')
        else:
            cell.set_facecolor('#f8f9fa' if row % 2 == 1 else '#ffffff')
            if col == 1:
                cell.set_text_props(fontweight='bold', color='#1a5276')

    ax_text.text(0.5, 0.90, f"自己評価精度レポート: {gen_label}", ha='center', va='center', fontsize=15, fontweight='bold', color='#2c3e50')
    ax_text.text(0.5, 0.15,
                 f"・検証サンプル数: {metrics['total_count']} 局面 (一様サンプリング)\n"
                 f"・極値局面数 (|y|≧0.8): {metrics['extreme_count']} 局面\n"
                 f"※ モデル推論は深さ0（探索なし単体推論）",
                 ha='center', va='center', fontsize=10, color='#555555',
                 bbox=dict(boxstyle='round,pad=0.6', facecolor='#f4f6f7', edgecolor='#bdc3c7'))

    plt.tight_layout()
    plt.savefig(output_filename, dpi=200, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="世代自己評価スクリプト")
    parser.add_argument("--system", type=str, required=True, choices=["depth0", "depth2"],
                        help="実験系統: depth0 (深さ0) または depth2 (深さ2 Min-Max)")
    parser.add_argument("--gen", type=int, required=True, help="世代番号 (例: 1, 2, ...)")
    parser.add_argument("--model", type=str, default="animal_shogi_model.keras", help="モデルパス")
    parser.add_argument("--data", type=str, default="train_data.npy", help="教師データパス")
    args = parser.parse_args()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf

    if not os.path.exists(args.model) or not os.path.exists(args.data):
        print(f"エラー: モデルまたはデータが見つかりません (model={args.model}, data={args.data})")
        sys.exit(1)

    system_label = "深さ0 Min-Maxなし" if args.system == "depth0" else "深さ2 Min-Maxあり"
    gen_label = f"Gen{args.gen} ({system_label})"
    print(f"自己評価を開始します: {gen_label}")
    dataset = np.load(args.data)
    X_all = dataset[:, :139].astype(np.float32)
    y_all = dataset[:, 139].astype(np.float32)

    sample_indices = sample_uniform_by_bins(X_all, y_all, n_bins=20, n_samples_per_bin=100)
    X_sample = X_all[sample_indices]
    y_sample = y_all[sample_indices]

    model = tf.keras.models.load_model(args.model)
    # モデル単体（深さ0）の素の推論値
    y_pred = model(X_sample, training=False).numpy().flatten()

    metrics = calculate_metrics(y_sample, y_pred)
    metrics["system"] = args.system
    metrics["generation"] = args.gen
    print(f"  MSE: {metrics['mse']:.5f} | R²: {metrics['r2']:.4f} | MAE: ±{metrics['mae']:.4f} | 符号一致: {metrics['sign_acc']:.1f}%")

    # JSON保存
    with open("metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("  [SAVED] metrics.json")

    # 散布図画像保存
    out_img = "scatter_evaluation.png"
    plot_scatter(gen_label, y_sample, y_pred, metrics, out_img)
    print(f"  [SAVED] {out_img}")


if __name__ == "__main__":
    main()
