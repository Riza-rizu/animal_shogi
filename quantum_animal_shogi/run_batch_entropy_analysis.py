"""
全世代（Gen1〜Gen8）一括エントロピー解析スクリプト: run_batch_entropy_analysis.py
- experiments/quantum_d0_notry 配下の全世代の match_replay_genX.npz を一括解析
- 各世代フォルダに entropy_decay_curve_genX.png と entropy_metrics_genX.json を出力
- 全世代を重ね合わせた比較グラフ entropy_comparison_all_gens.png を生成
- 各世代の比較指標テーブルを出力
"""

import os
import sys
import glob
import json
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Meiryo', 'Yu Gothic', 'Hiragino Sans', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from analyze_entropy import calculate_entropy_trajectory, plot_single_entropy


def main():
    import argparse
    parser = argparse.ArgumentParser(description="全世代一括情報エントロピー解析")
    parser.add_argument("--max_games", type=int, default=1000000, help="各世代で解析する最大対局数 (デフォルト: 1000000)")
    parser.add_argument("--base_dir", type=str, default="experiments/quantum_d0_notry", help="対象ディレクトリ")
    args = parser.parse_args()

    base_dir = args.base_dir
    max_games = args.max_games
    gen_dirs = sorted([d for d in glob.glob(os.path.join(base_dir, "gen*")) if os.path.isdir(d)],
                      key=lambda x: int(os.path.basename(x).replace("gen", "")))

    print("=" * 70)
    print("【前期データ（Gen1〜Gen8）一括情報エントロピー解析を開始】")
    print(f"対象ディレクトリ: {base_dir} ({len(gen_dirs)} 世代)")
    print(f"1世代あたりの解析対局数: {max_games:,} 局")
    print("=" * 70)

    comparison_results = []
    summary_table = []

    for g_dir in gen_dirs:
        gen_name = os.path.basename(g_dir)
        gen_num = int(gen_name.replace("gen", ""))
        npz_files = glob.glob(os.path.join(g_dir, "match_replay*.npz"))
        if not npz_files:
            continue

        npz_path = npz_files[0]
        player_gen = gen_num - 1
        label = f"Gen{gen_num} (Player: Gen{player_gen})"
        print(f"\n>>> [{gen_name}] 解析中 ({max_games:,} 局): {os.path.basename(npz_path)} ...", flush=True)

        plies, mean_piece, mean_lion, stats = calculate_entropy_trajectory(npz_path, max_games=max_games, max_ply=20)

        # 個別世代の保存
        out_png = os.path.join(g_dir, f"entropy_decay_curve_{gen_name}.png")
        out_json = os.path.join(g_dir, f"entropy_metrics_{gen_name}.json")
        plot_single_entropy(plies, mean_piece, mean_lion, f"Gen{gen_num}", out_png)

        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        comparison_results.append((f"Gen{gen_num}", plies, mean_piece, mean_lion))

        summary_table.append({
            "gen": f"Gen{gen_num}",
            "player": f"Gen{player_gen}",
            "half_life": stats["half_life_ply"],
            "h_ply1": stats["mean_piece_entropy_by_ply"][0],
            "h_ply5": stats["mean_piece_entropy_by_ply"][4],
            "h_ply10": stats["mean_piece_entropy_by_ply"][9],
            "h_ply15": stats["mean_piece_entropy_by_ply"][14],
            "h_final": stats["final_mean_entropy_bits"]
        })

    # 全世代比較プロットの作成
    print("\n" + "=" * 70)
    print("全世代比較グラフを作成中...")
    print("=" * 70)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # カラーマップ (世代が進むにつれて青->緑->橙->赤へグラデーション)
    cmap = plt.cm.plasma(np.linspace(0.1, 0.9, len(comparison_results)))

    # 左グラフ: 全駒平均エントロピーの比較
    for idx, (label, plies, mean_piece, _) in enumerate(comparison_results):
        ax1.plot(plies, mean_piece, marker='o', markersize=4, linewidth=2.0, color=cmap[idx], label=label)

    ax1.axhline(1.0, color='#7f8c8d', linestyle=':', label='Half Entropy (1.0 bit)')
    ax1.set_ylim(-0.05, 2.1)
    ax1.set_xlim(0.8, len(plies) + 0.2)
    ax1.set_xticks(range(1, 21))
    ax1.set_title('All-Piece Mean Entropy Decay by Generation', fontsize=13, fontweight='bold', pad=12)
    ax1.set_xlabel('Game Move (Ply)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Shannon Entropy H (bits)', fontsize=11, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(fontsize=9, loc='upper right')

    # 右グラフ: ライオン位置不確定性エントロピーの比較
    for idx, (label, plies, _, mean_lion) in enumerate(comparison_results):
        ax2.plot(plies, mean_lion, marker='s', markersize=4, linewidth=2.0, color=cmap[idx], label=label)

    ax2.axhline(1.0, color='#7f8c8d', linestyle=':', label='Half Entropy (1.0 bit)')
    ax2.set_ylim(-0.05, 2.1)
    ax2.set_xlim(0.8, len(plies) + 0.2)
    ax2.set_xticks(range(1, 21))
    ax2.set_title('Lion Uncertainty Entropy Decay by Generation', fontsize=13, fontweight='bold', pad=12)
    ax2.set_xlabel('Game Move (Ply)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Lion Entropy H_lion (bits)', fontsize=11, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(fontsize=9, loc='upper right')

    plt.suptitle('Evolution of Information Entropy Dynamics Across Generations (Gen1 - Gen8)', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()

    out_comp_png = os.path.join(base_dir, "entropy_comparison_all_gens.png")
    plt.savefig(out_comp_png, dpi=200)
    plt.close()
    print(f"  [SAVED] {out_comp_png}")

    # 集計サマリー表示
    print("\n" + "=" * 70)
    print("【全世代エントロピー推移サマリー】")
    print(f"{'世代':<6} | {'対局モデル':<10} | {'半減期(手)':<10} | {'Ply 1':<8} | {'Ply 5':<8} | {'Ply 10':<8} | {'Ply 15':<8}")
    print("-" * 70)
    for r in summary_table:
        print(f"{r['gen']:<6} | {r['player']:<10} | {str(r['half_life']) + ' 手':<10} | {r['h_ply1']:.3f}    | {r['h_ply5']:.3f}    | {r['h_ply10']:.3f}    | {r['h_ply15']:.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
