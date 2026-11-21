"""
量子どうぶつしょうぎ 情報エントロピー解析スクリプト: analyze_entropy.py
- match_replay_genX.npz の全手番確率分布テンソルから情報エントロピー（Shannon Entropy）を算出
- 1手ごとのエントロピー減衰曲線（状態収束ダイナミクス）をプロット
- 単一世代の解析、および複数世代（Gen1 vs Gen3 vs Gen5など）の比較プロットに対応
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Meiryo', 'Yu Gothic', 'Hiragino Sans', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def calculate_entropy_trajectory(npz_path, max_games=50000, max_ply=20, eps=1e-9):
    """
    match_replay npz ファイルから手番ごとの平均エントロピーを算出
    戻り値:
        plies: [1, 2, ..., max_ply]
        mean_piece_h: 全駒平均エントロピー (bits)
        mean_lion_h: ライオン位置の不確定性エントロピー (bits)
        stats: 要約統計辞書
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"ファイルが見つかりません: {npz_path}")

    data = np.load(npz_path)
    ptr = data['games_ptr']
    probs = data['probs']

    total_available_games = len(ptr) - 1
    n_games = min(max_games, total_available_games)

    ply_piece_h = [[] for _ in range(max_ply)]
    ply_lion_h = [[] for _ in range(max_ply)]

    for g in range(n_games):
        start = ptr[g]
        end = ptr[g + 1]
        g_probs = probs[start:end].astype(np.float32)
        p_safe = np.clip(g_probs, eps, 1.0)

        # 1. 全駒平均エントロピー (8駒×5種のShannonエントロピー平均)
        h_pieces = -np.sum(g_probs * np.log2(p_safe), axis=-1)  # (T, 8)
        mean_h = np.mean(h_pieces, axis=-1)  # (T,)

        # 2. ライオン位置エントロピー (P1: 0..3, P2: 4..7)
        p1_l = g_probs[:, :4, 0]
        p1_l_safe = np.clip(p1_l, eps, 1.0)
        h_p1 = -np.sum(p1_l * np.log2(p1_l_safe), axis=-1)

        p2_l = g_probs[:, 4:, 0]
        p2_l_safe = np.clip(p2_l, eps, 1.0)
        h_p2 = -np.sum(p2_l * np.log2(p2_l_safe), axis=-1)

        h_lion = 0.5 * (h_p1 + h_p2)

        for t in range(min(len(mean_h), max_ply)):
            ply_piece_h[t].append(mean_h[t])
            ply_lion_h[t].append(h_lion[t])

    plies = np.arange(1, max_ply + 1)
    mean_piece_vals = [float(np.mean(ply_piece_h[t])) if ply_piece_h[t] else 0.0 for t in range(max_ply)]
    mean_lion_vals = [float(np.mean(ply_lion_h[t])) if ply_lion_h[t] else 0.0 for t in range(max_ply)]

    # 半減手数の計算 (1.0 bit を下回る手数)
    half_life_ply = None
    for idx, v in enumerate(mean_piece_vals):
        if v <= 1.0:
            half_life_ply = idx + 1
            break

    stats = {
        "analyzed_games": n_games,
        "max_ply": max_ply,
        "initial_entropy_bits": round(mean_piece_vals[0], 4) if mean_piece_vals else 2.0,
        "half_life_ply": half_life_ply,
        "final_mean_entropy_bits": round(mean_piece_vals[-1], 4) if mean_piece_vals else 0.0,
        "mean_piece_entropy_by_ply": [round(v, 4) for v in mean_piece_vals],
        "lion_entropy_by_ply": [round(v, 4) for v in mean_lion_vals]
    }

    return plies, mean_piece_vals, mean_lion_vals, stats


def plot_single_entropy(plies, mean_piece, mean_lion, gen_label, out_png):
    """単一世代のエントロピー減衰曲線をプロット"""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(plies, mean_piece, marker='o', linewidth=2.5, color='#2980b9', label='Mean Piece Entropy (All Pieces)')
    ax.plot(plies, mean_lion, marker='s', linewidth=2.5, color='#e74c3c', linestyle='--', label='Lion Uncertainty Entropy')
    ax.axhline(1.0, color='#7f8c8d', linestyle=':', label='Half Entropy Threshold (1.0 bit)')

    ax.set_ylim(-0.05, 2.1)
    ax.set_xlim(0.8, len(plies) + 0.2)
    ax.set_xticks(plies)
    ax.set_title(f'[{gen_label}] Information Entropy Decay by Game Move (State Convergence)', fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Game Move (Ply)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Shannon Entropy H (bits)', fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=11, loc='upper right')

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"  [SAVED] {out_png}")


def plot_comparison_entropy(results_list, out_png):
    """複数世代（Gen1 vs Gen3 vs Gen5など）の比較曲線をプロット"""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#2980b9', '#27ae60', '#e67e22', '#8e44ad', '#d35400', '#c0392b']

    for idx, (label, plies, mean_piece, _) in enumerate(results_list):
        c = colors[idx % len(colors)]
        ax.plot(plies, mean_piece, marker='o', linewidth=2.2, color=c, label=f'{label} (Piece Entropy)')

    ax.axhline(1.0, color='#7f8c8d', linestyle=':', label='Half Entropy (1.0 bit)')
    ax.set_ylim(-0.05, 2.1)
    ax.set_xlim(0.8, len(results_list[0][1]) + 0.2)
    ax.set_xticks(results_list[0][1])
    ax.set_title('Cross-Generation Comparison: Information Entropy Decay Dynamics', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Game Move (Ply)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Shannon Entropy H (bits)', fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=11, loc='upper right')

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"  [SAVED] {out_png}")


def main():
    parser = argparse.ArgumentParser(description="量子どうぶつしょうぎ 情報エントロピー解析")
    parser.add_argument("--npz", type=str, default=None, help="match_replay.npz へのパス")
    parser.add_argument("--gen", type=int, default=None, help="世代番号 (例: 1)")
    parser.add_argument("--out_dir", type=str, default=".", help="保存先ディレクトリ")
    parser.add_argument("--max_games", type=int, default=50000, help="解析対局数")
    parser.add_argument("--compare", nargs="+", default=None, help="比較対象の npz ファイルパスリスト")
    parser.add_argument("--labels", nargs="+", default=None, help="比較用ラベルリスト (例: Gen1 Gen3 Gen5)")
    args = parser.parse_args()

    # 複数世代の比較モード
    if args.compare:
        labels = args.labels if args.labels and len(args.labels) == len(args.compare) else [f"Model {i+1}" for i in range(len(args.compare))]
        results_list = []
        for path, label in zip(args.compare, labels):
            print(f"解析中: {label} ({path})...")
            plies, mean_piece, mean_lion, stats = calculate_entropy_trajectory(path, max_games=args.max_games)
            results_list.append((label, plies, mean_piece, mean_lion))

        out_png = os.path.join(args.out_dir, "entropy_comparison.png")
        plot_comparison_entropy(results_list, out_png)
        return

    # 単一世代モード
    npz_path = args.npz
    if npz_path is None and args.gen is not None:
        player_gen = args.gen - 1
        candidates = [
            f"match_replay_gen{player_gen}.npz",
            "match_replay.npz",
            os.path.join(args.out_dir, f"match_replay_gen{player_gen}.npz"),
            os.path.join(args.out_dir, "match_replay.npz")
        ]
        for c in candidates:
            if os.path.exists(c):
                npz_path = c
                break

    if not npz_path or not os.path.exists(npz_path):
        print("エラー: match_replay.npz が見つかりません。")
        sys.exit(1)

    gen_label = f"Gen{args.gen}" if args.gen is not None else "Quantum Model"
    print(f"エントロピー解析を開始します: {gen_label} ({npz_path})")

    plies, mean_piece, mean_lion, stats = calculate_entropy_trajectory(npz_path, max_games=args.max_games)

    os.makedirs(args.out_dir, exist_ok=True)
    gen_suffix = f"_gen{args.gen}" if args.gen is not None else ""
    out_png = os.path.join(args.out_dir, f"entropy_decay_curve{gen_suffix}.png")
    out_json = os.path.join(args.out_dir, f"entropy_metrics{gen_suffix}.json")

    plot_single_entropy(plies, mean_piece, mean_lion, gen_label, out_png)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"  [SAVED] {out_json}")


if __name__ == "__main__":
    main()
