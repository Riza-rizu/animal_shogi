"""
各世代モデルの散布図および評価指標出力スクリプト
- 各世代ごとに「左側: 散布図」「右側: 定量指標パネル」の独立した画像を生成
- 教師データから -1.0〜+1.0 の各領域を一様サンプリングして評価
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

# 日本語フォント設定（環境に応じてフォールバック）
plt.rcParams['font.sans-serif'] = ['Meiryo', 'Yu Gothic', 'Hiragino Sans', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def find_file_candidates(patterns):
    """指定されたパターン候補の中から存在する最初のファイルを返す"""
    for p in patterns:
        if os.path.exists(p):
            return p
    return None


def sample_uniform_by_bins(X, y, n_bins=20, n_samples_per_bin=100, seed=42):
    """
    評価値 y (-1.0〜1.0) を複数のビンに分割し、
    各領域から均等（一様）にサンプリングしたインデックスを返す
    """
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
    """学術的評価に必要な5大指標を算出"""
    # 1. MSE (平均二乗誤差)
    mse = np.mean((y_true - y_pred) ** 2)

    # 2. R^2 (決定係数)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-7 else 0.0

    # 3. MAE (平均絶対誤差)
    mae = np.mean(np.abs(y_true - y_pred))

    # 4. 符号一致率 (Sign Accuracy: 勝敗の向きが合っているか)
    # 0.05 以下の微差は除外または一致扱い
    nonzero_mask = np.abs(y_true) > 0.05
    if np.sum(nonzero_mask) > 0:
        sign_match = (np.sign(y_true[nonzero_mask]) == np.sign(y_pred[nonzero_mask]))
        sign_acc = np.mean(sign_match) * 100.0
    else:
        sign_acc = 100.0

    # 5. 極値局面 (|y| >= 0.8: 自滅・詰み寸前) での誤差
    extreme_mask = np.abs(y_true) >= 0.8
    if np.sum(extreme_mask) > 0:
        extreme_mae = np.mean(np.abs(y_true[extreme_mask] - y_pred[extreme_mask]))
        extreme_count = np.sum(extreme_mask)
    else:
        extreme_mae = 0.0
        extreme_count = 0

    return {
        "mse": mse,
        "r2": r2,
        "mae": mae,
        "sign_acc": sign_acc,
        "extreme_mae": extreme_mae,
        "extreme_count": extreme_count,
        "total_count": len(y_true)
    }


def plot_single_generation(gen_name, y_true, y_pred, metrics, output_filename):
    """1世代分の散布図（左）＋ 指標パネル（右）の画像を生成・保存"""
    fig, (ax_plot, ax_text) = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={'width_ratios': [1.3, 1.0]})

    # --- 左側: 散布図 ---
    ax_plot.scatter(y_true, y_pred, alpha=0.35, color='#1f77b4', edgecolors='none', s=25, label='サンプリング局面')
    # 理想線 y = x
    ax_plot.plot([-1.05, 1.05], [-1.05, 1.05], color='#d62728', linestyle='--', linewidth=2, label='理想線 (y = x)')

    ax_plot.set_xlim([-1.05, 1.05])
    ax_plot.set_ylim([-1.05, 1.05])
    ax_plot.set_xlabel('教師データ正解評価値 (y_true)', fontsize=12, fontweight='bold')
    ax_plot.set_ylabel('モデル単体推論値 (y_pred)', fontsize=12, fontweight='bold')
    ax_plot.set_title(f'【{gen_name}】 散布図 (評価値フィッティング)', fontsize=14, fontweight='bold', pad=12)
    ax_plot.grid(True, linestyle=':', alpha=0.6)
    ax_plot.legend(loc='upper left', fontsize=10)

    # ゼロ軸
    ax_plot.axhline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    ax_plot.axvline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)

    # --- 右側: 定量指標パネル ---
    ax_text.axis('off')

    table_data = [
        ["評価指標", "測定値", "理想 / 目標"],
        ["MSE (平均二乗誤差)", f"{metrics['mse']:.5f}", "0.00000 (完全一致)"],
        ["R² (決定係数)", f"{metrics['r2']:.4f}", "1.0000 (完全相関)"],
        ["MAE (平均絶対誤差)", f"±{metrics['mae']:.4f}", "0.0000 (ズレなし)"],
        ["勝敗方向一致率", f"{metrics['sign_acc']:.2f} %", "100.0 % (符号完全一致)"],
        ["極値誤差 (|y|≧0.8)", f"±{metrics['extreme_mae']:.4f}", "0.0000 (即死/詰み認識)"]
    ]

    table = ax_text.table(
        cellText=table_data,
        colWidths=[0.45, 0.28, 0.35],
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 2.2)

    # テーブルヘッダーとスタイリング
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#2c3e50')
            cell.set_text_props(color='white', fontweight='bold')
        else:
            if row % 2 == 1:
                cell.set_facecolor('#f8f9fa')
            else:
                cell.set_facecolor('#ffffff')
            if col == 1:
                cell.set_text_props(fontweight='bold', color='#1a5276')

    # タイトルと注釈テキスト
    ax_text.text(
        0.5, 0.90, f"定量精度レポート: {gen_name}",
        ha='center', va='center', fontsize=15, fontweight='bold', color='#2c3e50'
    )
    ax_text.text(
        0.5, 0.15,
        f"・検証サンプル数: {metrics['total_count']} 局面 (一様サンプリング)\n"
        f"・極値局面数 (|y|≧0.8): {metrics['extreme_count']} 局面\n"
        f"※ モデル推論は深さ0（探索なし単体推論）",
        ha='center', va='center', fontsize=10, color='#555555',
        bbox=dict(boxstyle='round,pad=0.6', facecolor='#f4f6f7', edgecolor='#bdc3c7')
    )

    plt.tight_layout()
    plt.savefig(output_filename, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] {output_filename}")


def load_dataset(data_path):
    """train_data.npy から X (139次元) と y (1次元) を読み込む"""
    print(f"  データセット読み込み中: {data_path} ...")
    dataset = np.load(data_path)
    X = dataset[:, :139]
    y = dataset[:, 139]
    print(f"  総件数: {len(y)} 局面")
    return X, y


def main():
    parser = argparse.ArgumentParser(description="各世代の散布図および指標レポート画像を個別生成")
    parser.add_argument("--generations", nargs="+", type=int, default=[3, 4, 5, 6, 7],
                        help="対象世代のリスト (例: --generations 3 4 5 6 7)")
    parser.add_argument("--samples_per_bin", type=int, default=100,
                        help="各ビンごとのサンプリング局面数 (デフォルト: 100件 x 20ビン = 2,000件)")
    args = parser.parse_args()

    # TensorFlow のインポートとGPU/ログ抑制
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf

    print("=" * 60)
    print("どうぶつしょうぎ 各世代散布図・精度評価スクリプト")
    print(f"対象世代: Gen {args.generations}")
    print("=" * 60)

    for gen in args.generations:
        gen_label = f"Gen{gen}"
        print(f"\n>>> 【{gen_label}】の処理を開始します...")

        # 1. モデルファイルの検索
        model_candidates = [
            f"animal_shogi_model_gen{gen}.keras",
            f"legacy_v1/animal_shogi_model_server_gen{gen}.keras",
            f"animal_shogi_model_local_gen{gen}.keras",
            f"legacy_v1/animal_shogi_model_gen{gen}.keras",
        ]
        # Gen7 の特別扱い
        if gen == 7:
            model_candidates.insert(0, "animal_shogi_model_gen7_depth2.keras")

        model_path = find_file_candidates(model_candidates)
        if not model_path:
            print(f"  [SKIP] {gen_label} のモデルファイルが見つかりません: {model_candidates[0]}")
            continue

        # 2. 教師データの検索
        data_candidates = [
            f"train_data_gen{gen}.npy",
            f"legacy_v1/train_data_gen{gen}.npy",
            "train_data.npy",
            "legacy_v1/train_data.npy"
        ]
        data_path = find_file_candidates(data_candidates)
        if not data_path:
            print(f"  [SKIP] {gen_label} の教師データが見つかりません。")
            continue

        # 3. データの読み込みと一様サンプリング
        X_all, y_all = load_dataset(data_path)
        sample_indices = sample_uniform_by_bins(
            X_all, y_all, n_bins=20, n_samples_per_bin=args.samples_per_bin, seed=42 + gen
        )
        X_test = X_all[sample_indices]
        y_true = y_all[sample_indices]

        # 4. モデルのロードと推論 (深さ0)
        print(f"  モデル読み込み中: {model_path} ...")
        model = tf.keras.models.load_model(model_path)
        y_pred = model(X_test, training=False).numpy().flatten()

        # 5. 指標の計算
        metrics = calculate_metrics(y_true, y_pred)
        print(f"  MSE: {metrics['mse']:.5f} | R²: {metrics['r2']:.4f} | MAE: ±{metrics['mae']:.4f} | 符号一致: {metrics['sign_acc']:.1f}%")

        # 6. 画像の生成・保存
        out_img_name = f"scatter_analysis_Gen{gen}.png"
        plot_single_generation(gen_label, y_true, y_pred, metrics, out_img_name)

    print("\n" + "=" * 60)
    print("すべての処理が完了しました！")
    print("=" * 60)


if __name__ == "__main__":
    main()
