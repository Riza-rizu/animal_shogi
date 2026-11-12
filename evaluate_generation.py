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


def plot_scatter(gen_label, y_true, y_pred, metrics, output_filename, is_sn=False):
    fig, (ax_plot, ax_text) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1.3, 1.0]})

    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    # 散布図
    ax_plot.scatter(y_true, y_pred, alpha=0.35, color='#1f77b4', edgecolors='none', s=25, label='Sampled Positions')
    ax_plot.plot([-1.05, 1.05], [-1.05, 1.05], color='#d62728', linestyle='--', linewidth=2, label='Ideal Line (y = x)')
    ax_plot.set_xlim([-1.05, 1.05])
    ax_plot.set_ylim([-1.05, 1.05])
    
    if is_sn:
        ax_plot.set_xlabel('Current Generation Match Value S_n(x) [α=1.0]', fontsize=12, fontweight='bold')
        ax_plot.set_ylabel('Model Direct Predicted Value (y_pred)', fontsize=12, fontweight='bold')
        ax_plot.set_title(f'[{gen_label}] Match Fitting: S_n(x) vs Model Prediction (α=1)', fontsize=13, fontweight='bold', pad=12)
    else:
        ax_plot.set_xlabel('Ground Truth Target Value (y_true)', fontsize=12, fontweight='bold')
        ax_plot.set_ylabel('Model Direct Predicted Value (y_pred)', fontsize=12, fontweight='bold')
        ax_plot.set_title(f'[{gen_label}] Self-Fitting Scatter Plot', fontsize=14, fontweight='bold', pad=12)

    ax_plot.grid(True, linestyle=':', alpha=0.6)
    ax_plot.legend(loc='upper left', fontsize=10)
    ax_plot.axhline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    ax_plot.axvline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)

    # 定量指標表
    ax_text.axis('off')
    table_data = [
        ["Metric", "Value", "Ideal / Target"],
        ["MSE (Mean Squared Error)", f"{metrics['mse']:.5f}", "0.00000 (Exact)"],
        ["R² (Determination Coeff)", f"{metrics['r2']:.4f}", "1.0000 (Perfect)"],
        ["MAE (Mean Absolute Error)", f"±{metrics['mae']:.4f}", "0.0000 (Zero error)"],
        ["Sign Accuracy (Win/Loss)", f"{metrics['sign_acc']:.2f}%", "100.0% (Exact match)"],
        ["Extreme Error (|y|>=0.8)", f"±{metrics['extreme_mae']:.4f}", "0.0000 (Blunder/Mate)"]
    ]

    table = ax_text.table(cellText=table_data, colWidths=[0.46, 0.24, 0.30], cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 2.1)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#2c3e50')
            cell.set_text_props(color='white', fontweight='bold')
        else:
            cell.set_facecolor('#f8f9fa' if row % 2 == 1 else '#ffffff')
            if col == 1:
                cell.set_text_props(fontweight='bold', color='#1a5276')

    report_title = f"Evaluation Report: {gen_label}" + (" [S_n(x), α=1]" if is_sn else "")
    ax_text.text(0.5, 0.90, report_title, ha='center', va='center', fontsize=14, fontweight='bold', color='#2c3e50')
    
    note_text = (
        f"• Ground truth: Pure current-generation self-play matches (S_n(x), α=1.0)\n"
        f"• Sample size: {metrics['total_count']} positions | Extreme (|y|>=0.8): {metrics['extreme_count']}\n"
        f"* Historical carryover excluded to verify genuine new learning"
    ) if is_sn else (
        f"• Sample size: {metrics['total_count']} positions (Uniform sampling)\n"
        f"• Extreme positions (|y|>=0.8): {metrics['extreme_count']} positions\n"
        f"* Note: Model inference evaluated at depth=0 (pure intuition)"
    )

    ax_text.text(0.5, 0.15, note_text,
                 ha='center', va='center', fontsize=10, color='#555555',
                 bbox=dict(boxstyle='round,pad=0.6', facecolor='#f4f6f7', edgecolor='#bdc3c7'))

    plt.tight_layout()
    plt.savefig(output_filename, dpi=200, bbox_inches='tight')
    plt.close()


def main():
    SYSTEM_DIR_MAP = {
        "depth0": "depth0_pure_mse",
        "depth2": "depth2_minmax_mse",
        "d0_rep": "d0_rep_pena_mse",
        "d0_rep_pena_mse": "d0_rep_pena_mse",
        "d0_six": "d0_six"
    }

    parser = argparse.ArgumentParser(description="世代自己評価スクリプト")
    parser.add_argument("--system", type=str, required=True, choices=["depth0", "depth2", "d0_rep", "d0_rep_pena_mse", "d0_six"],
                        help="実験系統: depth0, depth2, d0_rep, d0_six")
    parser.add_argument("--gen", type=int, required=True, help="世代番号 (例: 1, 2, ...)")
    parser.add_argument("--model", type=str, default="animal_shogi_model.keras", help="モデルパス")
    parser.add_argument("--data", type=str, default=None, help="教師データパス (未指定時は current_gen_data.npy 優先)")
    parser.add_argument("--archived", action="store_true", help="保存済みフォルダ(experiments/.../genN/)から自動ロードしてその中に再出力")
    parser.add_argument("--out_dir", type=str, default=None, help="出力先ディレクトリ")
    args = parser.parse_args()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf

    is_sn = False
    if args.archived:
        system_dir_name = SYSTEM_DIR_MAP[args.system]
        archived_dir = os.path.join("experiments", system_dir_name, f"gen{args.gen}")
        args.model = os.path.join(archived_dir, "model.keras")
        sn_path = os.path.join(archived_dir, "current_gen_data.npy")
        if (args.data is None or "current_gen_data" in args.data) and os.path.exists(sn_path):
            args.data = sn_path
            is_sn = True
        elif args.data is None:
            args.data = os.path.join(archived_dir, "train_data.npy")
        if args.out_dir is None:
            args.out_dir = archived_dir
        print(f"アーカイブ再評価モード: {archived_dir} のデータを参照します (データ: {os.path.basename(args.data)})")
    else:
        # 未指定の場合、最新対局結果 S_n(x) [α=1] を優先
        if args.data is None:
            if os.path.exists("current_gen_data.npy"):
                args.data = "current_gen_data.npy"
                is_sn = True
            else:
                args.data = "train_data.npy"
        elif "current_gen_data" in args.data:
            is_sn = True

    if not os.path.exists(args.model) or not os.path.exists(args.data):
        print(f"エラー: モデルまたはデータが見つかりません (model={args.model}, data={args.data})")
        sys.exit(1)

    if "d0_six" in args.system:
        system_label = "Depth 0, WarmStart Gen6+"
    elif "d0_rep" in args.system:
        system_label = "Depth 0, Rep Penalty"
    elif args.system == "depth0":
        system_label = "Depth 0, No Min-Max"
    else:
        system_label = "Depth 2, Min-Max"
    gen_label = f"Gen{args.gen} ({system_label})"
    mode_str = "S_n(x) [α=1.0: 最新対局のみ]" if is_sn else "Y_n(x) [累計DB]"
    print(f"自己評価を開始します: {gen_label} (評価対象: {mode_str})")
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
    metrics["is_sn_alpha1"] = is_sn
    print(f"  MSE: {metrics['mse']:.5f} | R²: {metrics['r2']:.4f} | MAE: ±{metrics['mae']:.4f} | 符号一致: {metrics['sign_acc']:.1f}%")

    out_dir = args.out_dir or "."
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, "metrics.json")
    out_img = os.path.join(out_dir, "scatter_evaluation.png")

    # JSON保存
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"  [SAVED] {out_json}")

    # 散布図描画
    plot_scatter(gen_label, y_sample, y_pred, metrics, out_img, is_sn=is_sn)
    print(f"  [SAVED] {out_img}")


if __name__ == "__main__":
    main()
