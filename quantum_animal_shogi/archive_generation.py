"""
世代アーカイブ管理スクリプト: archive_generation.py
各世代の成果物を系統別フォルダ (experiments/d0_six/genX) に自動退避・整理
"""

import os
import sys
import shutil
import json
import argparse

EXPERIMENT_ROOT = "experiments"
SYSTEM_DIR_NAME = "quantum_d0"


def main():
    parser = argparse.ArgumentParser(description="各世代の成果物を系統別フォルダにアーカイブ保存")
    parser.add_argument("--gen", type=int, required=True, help="世代番号 (0, 1, 2, ...)")
    parser.add_argument("--system", type=str, default="quantum_d0", help="実験系統名")
    args = parser.parse_args()

    target_dir = os.path.join(EXPERIMENT_ROOT, args.system, f"gen{args.gen}")
    os.makedirs(target_dir, exist_ok=True)

    print("=" * 60)
    print(f"【アーカイブ保存】 系統: {args.system} | 世代: Gen{args.gen}")
    print(f"保存先: {target_dir}")
    print("=" * 60)

    # Gen0 は初期モデルのみ
    if args.gen == 0:
        model_src = "quantum_model_gen0.keras"
        if not os.path.exists(model_src):
            model_src = "quantum_model.keras"
        if os.path.exists(model_src):
            dst_path = os.path.join(target_dir, "quantum_model_gen0.keras")
            shutil.copy2(model_src, dst_path)
            print(f"  [OK] {model_src} -> quantum_model_gen0.keras")
        print(f"Gen0 の初期モデルを {target_dir} に保存しました。")
        return

    player_gen = args.gen - 1

    # 保存対象ファイルリスト
    archive_tasks = [
        ([f"quantum_model_gen{args.gen}.keras", "quantum_model.keras"], f"quantum_model_gen{args.gen}.keras"),
        (["train_data.npy"], "train_data.npy"),
        (["cumulative_dataset.pkl"], "cumulative_dataset.pkl"),
        (sorted(glob.glob(f"match_log_*games_gen{player_gen}.csv") + glob.glob("match_log_*games.csv")), f"match_log_gen{player_gen}.csv"),
        ([f"match_stats_gen{player_gen}.json", "match_stats.json"], f"match_stats_gen{player_gen}.json"),
        (["scatter_evaluation.png"], "scatter_evaluation.png"),
        (["metrics.json"], "metrics.json"),
        (["training_history.json"], "training_history.json")
    ]

    copied_count = 0
    for src_candidates, dst_name in archive_tasks:
        found_src = None
        for cand in src_candidates:
            if cand and os.path.exists(cand):
                found_src = cand
                break
        if found_src:
            dst_path = os.path.join(target_dir, dst_name)
            shutil.copy2(found_src, dst_path)
            print(f"  [OK] {found_src:35s} -> {dst_name}")
            copied_count += 1

    print(f"\n合計 {copied_count} 個のファイルを {target_dir} に正常保存しました。")
    print("=" * 60)


if __name__ == "__main__":
    import glob
    main()
