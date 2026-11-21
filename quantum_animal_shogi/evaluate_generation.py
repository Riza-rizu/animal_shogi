"""
世代自己評価スクリプト: evaluate_generation.py (量子どうぶつ将棋・第6期完全準拠版)
- train_data.npy と学習済みモデルを評価
- 散布図画像 scatter_evaluation.png (左: 散布図, 右: 指標表) を生成
- 指標数値 metrics.json を第6期と100%同一フォーマットで出力
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Meiryo', 'Yu Gothic', 'Hiragino Sans', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def sample_uniform_by_bins(X, y, n_bins=20, n_samples_per_bin=100, seed=42):
    """-1.0〜1.0 を均等サンプリング (第6期完全同一)"""
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
    """定量評価指標の算出 (第6期完全同一)"""
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
    """散布図＋指標表の2分割プロット (第6期完全同一フォーマット)"""
    fig, (ax_plot, ax_text) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1.3, 1.0]})

    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    # 左側: 散布図
    ax_plot.scatter(y_true, y_pred, alpha=0.35, color='#1f77b4', edgecolors='none', s=25, label='Sampled Positions')
    ax_plot.plot([-1.05, 1.05], [-1.05, 1.05], color='#d62728', linestyle='--', linewidth=2, label='Ideal Line (y = x)')
    ax_plot.set_xlim([-1.05, 1.05])
    ax_plot.set_ylim([-1.05, 1.05])

    if is_sn:
        ax_plot.set_xlabel('Current Generation Match Value S_n(x)', fontsize=12, fontweight='bold')
        ax_plot.set_ylabel('Model Direct Predicted Value (y_pred)', fontsize=12, fontweight='bold')
        ax_plot.set_title(f'[{gen_label}] Match Fitting: S_n(x) vs Model Prediction', fontsize=13, fontweight='bold', pad=12)
    else:
        ax_plot.set_xlabel('Ground Truth Target Value (y_true)', fontsize=12, fontweight='bold')
        ax_plot.set_ylabel('Model Direct Predicted Value (y_pred)', fontsize=12, fontweight='bold')
        ax_plot.set_title(f'[{gen_label}] Self-Fitting Scatter Plot', fontsize=14, fontweight='bold', pad=12)

    ax_plot.grid(True, linestyle=':', alpha=0.6)
    ax_plot.legend(loc='upper left', fontsize=10)
    ax_plot.axhline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    ax_plot.axvline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)

    # 右側: 定量指標表
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

    report_title = f"Evaluation Report: {gen_label}"
    ax_text.text(0.5, 0.90, report_title, ha='center', va='center', fontsize=14, fontweight='bold', color='#2c3e50')

    note_text = (
        f"• Sample size: {metrics['total_count']} positions (Uniform sampling)\n"
        f"• Extreme positions (|y|>=0.8): {metrics['extreme_count']} positions\n"
        f"• Time discount factor α = 0.95 | Repetition penalty γ = 0.6\n"
        f"* Note: Model inference evaluated at depth=0 (pure intuition)"
    )

    ax_text.text(0.5, 0.15, note_text,
                 ha='center', va='center', fontsize=10, color='#555555',
                 bbox=dict(boxstyle='round,pad=0.6', facecolor='#f4f6f7', edgecolor='#bdc3c7'))

    plt.tight_layout()
    plt.savefig(output_filename, dpi=200, bbox_inches='tight')
    plt.close()


def plot_moves_distribution(gen_num, out_dir="."):
    """手数の出現頻度棒グラフをプロットして保存"""
    player_gen = gen_num - 1
    moves = None

    # 1. match_replay npz からの読み込み (超高速)
    npz_candidates = [
        f"match_replay_gen{player_gen}.npz",
        "match_replay.npz",
        os.path.join(out_dir, f"match_replay_gen{player_gen}.npz"),
        os.path.join(out_dir, "match_replay.npz")
    ]
    for npz_path in npz_candidates:
        if os.path.exists(npz_path):
            try:
                data = np.load(npz_path)
                if 'games_ptr' in data:
                    moves = np.diff(data['games_ptr'])
                    break
            except Exception:
                pass

    # 2. CSV からの読み込み (フォールバック)
    if moves is None:
        csv_candidates = [
            f"match_summary_1million_games_gen{player_gen}.csv",
            "match_summary.csv",
            f"match_log_1million_games_gen{player_gen}.csv",
            os.path.join(out_dir, f"match_summary_1million_games_gen{player_gen}.csv"),
            os.path.join(out_dir, f"match_log_1million_games_gen{player_gen}.csv")
        ]
        import csv
        for csv_path in csv_candidates:
            if os.path.exists(csv_path):
                try:
                    moves_list = []
                    with open(csv_path, 'r', encoding='utf-8') as f:
                        reader = csv.reader(f)
                        header = next(reader)
                        col_idx = 2 if len(header) > 2 and header[2] == 'moves' else -1
                        if col_idx == -1:
                            for idx, h in enumerate(header):
                                if h == 'moves':
                                    col_idx = idx
                                    break
                        if col_idx != -1:
                            for row in reader:
                                if row and len(row) > col_idx:
                                    moves_list.append(int(row[col_idx]))
                    if moves_list:
                        moves = np.array(moves_list, dtype=np.int32)
                        break
                except Exception:
                    pass

    if moves is None or len(moves) == 0:
        print("  [SKIP] 手数分布データが見つからないため棒グラフをスキップします。")
        return

    avg_m = float(np.mean(moves))
    min_m = int(np.min(moves))
    max_m = int(np.max(moves))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    counts, _ = np.histogram(moves, bins=np.arange(min_m, max_m + 2) - 0.5)
    bin_centers = np.arange(min_m, max_m + 1)

    ax.bar(bin_centers, counts, width=0.8, color='#3498db', edgecolor='#2980b9', alpha=0.85, zorder=3)
    ax.axvline(avg_m, color='#e74c3c', linestyle='--', linewidth=2, zorder=4)

    ax.set_title(f'[Gen{gen_num}] Move Count Distribution (Self-Play)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Moves', fontsize=12, fontweight='bold')
    ax.set_ylabel('Game Count', fontsize=12, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)

    # 右上に Avg: ○○ moves を表示
    ax.text(0.96, 0.94, f'Avg: {avg_m:.1f} moves', transform=ax.transAxes,
            fontsize=13, fontweight='bold', color='#2c3e50',
            horizontalalignment='right', verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffffff', edgecolor='#bdc3c7', alpha=0.95))

    plt.tight_layout()
    out_img_gen = os.path.join(out_dir, f"moves_distribution_gen{gen_num}.png")
    out_img_common = os.path.join(out_dir, "moves_distribution.png")
    plt.savefig(out_img_gen, dpi=200)
    plt.savefig(out_img_common, dpi=200)
    plt.close()
    print(f"  [SAVED] {out_img_gen}")


def main():
    parser = argparse.ArgumentParser(description="世代自己評価スクリプト (第6期完全準拠)")
    parser.add_argument("--gen", type=int, required=True, help="世代番号 (例: 1, 2, ...)")
    parser.add_argument("--system", type=str, default="quantum_d0", help="実験系統名")
    parser.add_argument("--model", type=str, default=None, help="モデルパス")
    parser.add_argument("--data", type=str, default=None, help="教師データパス")
    parser.add_argument("--out_dir", type=str, default=".", help="出力先ディレクトリ")
    args = parser.parse_args()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf

    # モデルファイルの自動検出
    if args.model is None:
        candidates = [
            f"quantum_model_gen{args.gen}.keras",
            "quantum_model.keras",
            os.path.join(args.out_dir, f"quantum_model_gen{args.gen}.keras"),
            os.path.join(args.out_dir, "model.keras")
        ]
        for c in candidates:
            if os.path.exists(c):
                args.model = c
                break

    # データセットファイルの自動検出
    if args.data is None:
        candidates = [
            "train_data.npy",
            os.path.join(args.out_dir, "train_data.npy")
        ]
        for c in candidates:
            if os.path.exists(c):
                args.data = c
                break

    if not args.model or not os.path.exists(args.model):
        print(f"エラー: モデルファイルが見つかりません ({args.model})")
        sys.exit(1)
    if not args.data or not os.path.exists(args.data):
        print(f"エラー: データセットファイルが見つかりません ({args.data})")
        sys.exit(1)

    system_label = "Quantum Depth 0"
    gen_label = f"Gen{args.gen} ({system_label})"
    print(f"自己評価を開始します: {gen_label}")

    dataset = np.load(args.data)
    X_all = dataset[:, :139].astype(np.float32)
    y_all = dataset[:, 139].astype(np.float32)

    sample_indices = sample_uniform_by_bins(X_all, y_all, n_bins=20, n_samples_per_bin=100)
    X_sample = X_all[sample_indices]
    y_sample = y_all[sample_indices]

    model = tf.keras.models.load_model(args.model)
    y_pred = model(X_sample, training=False).numpy().flatten()

    metrics = calculate_metrics(y_sample, y_pred)
    metrics["system"] = args.system
    metrics["generation"] = args.gen
    metrics["is_sn_alpha1"] = False  # 割引率 alpha=0.95
    print(f"  MSE: {metrics['mse']:.5f} | R²: {metrics['r2']:.4f} | MAE: ±{metrics['mae']:.4f} | 符号一致: {metrics['sign_acc']:.1f}%")

    os.makedirs(args.out_dir, exist_ok=True)
    out_json = os.path.join(args.out_dir, "metrics.json")
    out_img = os.path.join(args.out_dir, "scatter_evaluation.png")

    # JSON保存 (第6期完全同一フォーマット)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"  [SAVED] {out_json}")

    # 散布図描画 (第6期完全同一フォーマット)
    plot_scatter(gen_label, y_sample, y_pred, metrics, out_img, is_sn=False)
    print(f"  [SAVED] {out_img}")

    # 手数分布棒グラフの描画
    plot_moves_distribution(args.gen, args.out_dir)

    # 情報エントロピー減衰曲線の解析・描画
    try:
        from analyze_entropy import calculate_entropy_trajectory, plot_single_entropy
        player_gen = args.gen - 1
        npz_candidates = [
            f"match_replay_gen{player_gen}.npz",
            "match_replay.npz",
            os.path.join(args.out_dir, f"match_replay_gen{player_gen}.npz"),
            os.path.join(args.out_dir, "match_replay.npz")
        ]
        target_npz = None
        for c in npz_candidates:
            if os.path.exists(c):
                target_npz = c
                break
        if target_npz:
            plies, mean_piece, mean_lion, ent_stats = calculate_entropy_trajectory(target_npz)
            out_ent_png = os.path.join(args.out_dir, f"entropy_decay_curve_gen{args.gen}.png")
            out_ent_json = os.path.join(args.out_dir, f"entropy_metrics_gen{args.gen}.json")
            plot_single_entropy(plies, mean_piece, mean_lion, f"Gen{args.gen}", out_ent_png)
            with open(out_ent_json, "w", encoding="utf-8") as f:
                json.dump(ent_stats, f, indent=2)
            print(f"  [SAVED] {out_ent_json}")
    except Exception as e:
        print(f"  [SKIP] エントロピー解析スキップ: {e}")


if __name__ == "__main__":
    main()
