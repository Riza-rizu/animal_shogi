"""
世代アーカイブ管理スクリプト: archive_generation.py
- カレントディレクトリに生成された各成果物を、系統・世代別フォルダに自動退避・整理
- 混ざらない安全なフォルダ構造を維持
- 対局結果ファイル（棋譜・統計）には対局主体の世代（前世代: Gen N-1）を明記
"""

import os
import sys
import shutil
import json
import argparse

EXPERIMENT_ROOT = "experiments"

SYSTEM_NAMES = {
    "depth0": "depth0_pure_mse",
    "depth2": "depth2_minmax_mse"
}


def main():
    parser = argparse.ArgumentParser(description="各世代の成果物を系統別フォルダにアーカイブ保存")
    parser.add_argument("--system", type=str, required=True, choices=["depth0", "depth2"],
                        help="実験系統: depth0 (深さ0) または depth2 (深さ2 Min-Max)")
    parser.add_argument("--gen", type=int, required=True, help="世代番号 (0, 1, 2, ...)")
    parser.add_argument("--init_folders_only", action="store_true", help="フォルダ構造の初期作成のみ行う")
    args = parser.parse_args()

    system_dir_name = SYSTEM_NAMES[args.system]
    target_dir = os.path.join(EXPERIMENT_ROOT, system_dir_name, f"gen{args.gen}")
    os.makedirs(target_dir, exist_ok=True)

    if args.init_folders_only:
        print(f"フォルダを作成しました: {target_dir}")
        return

    print("=" * 60)
    print(f"【アーカイブ保存】 系統: {system_dir_name} | 世代: Gen{args.gen}")
    print(f"保存先: {target_dir}")
    print("=" * 60)

    # Gen0 は初期モデルのみ
    if args.gen == 0:
        model_src = "animal_shogi_model.keras"
        if os.path.exists(model_src):
            dst_path = os.path.join(target_dir, "model.keras")
            shutil.copy2(model_src, dst_path)
            size_kb = os.path.getsize(dst_path) / 1024.0
            print(f"  [OK] {model_src:32s} -> model.keras ({size_kb:.1f} KB)")
        else:
            print(f"  [ERROR] {model_src} が見つかりません。")
        print(f"\nGen0 の初期モデルを {target_dir} に正常保存しました。")
        print("=" * 60)
        return

    player_gen = args.gen - 1

    # 保存対象ファイルリストの構築
    # (元ファイル候補リスト, 保存先ファイル名)
    archive_tasks = [
        (["animal_shogi_model.keras"], "model.keras"),
        (["train_data.npy"], "train_data.npy"),
        (["cumulative_dataset.pkl"], "cumulative_dataset.pkl"),
        ([f"match_log_15000games_gen{player_gen}.csv", "match_log_15000games.csv"], f"match_log_15000games_gen{player_gen}.csv"),
        ([f"match_stats_gen{player_gen}.json", "match_stats.json"], f"match_stats_gen{player_gen}.json"),
        (["scatter_evaluation.png"], "scatter_evaluation.png"),
        (["metrics.json"], "metrics.json"),
        (["training_history.json"], "training_history.json")
    ]

    copied_count = 0
    for src_candidates, dst_name in archive_tasks:
        found_src = None
        for cand in src_candidates:
            if os.path.exists(cand):
                found_src = cand
                break

        if found_src:
            dst_path = os.path.join(target_dir, dst_name)

            # match_stats.json の場合、内部の player_generation メタデータを補正して保存
            if dst_name.endswith(".json") and "match_stats" in dst_name:
                try:
                    with open(found_src, "r", encoding="utf-8") as f:
                        stats = json.load(f)
                    stats["player_generation"] = f"Gen{player_gen}"
                    stats["created_model_generation"] = f"Gen{args.gen}"
                    with open(dst_path, "w", encoding="utf-8") as f:
                        json.dump(stats, f, ensure_ascii=False, indent=2)
                    size_kb = os.path.getsize(dst_path) / 1024.0
                    print(f"  [OK] {found_src:32s} -> {dst_name:32s} ({size_kb:.1f} KB) [メタデータ付与]")
                    copied_count += 1
                    continue
                except Exception as e:
                    print(f"  [WARN] match_stats のメタデータ補正失敗、通常コピーします: {e}")

            shutil.copy2(found_src, dst_path)
            size_kb = os.path.getsize(dst_path) / 1024.0
            print(f"  [OK] {found_src:32s} -> {dst_name:32s} ({size_kb:.1f} KB)")
            copied_count += 1
        else:
            print(f"  [SKIP/MISSING] {src_candidates[0]} が見つかりません。")

    print(f"\n合計 {copied_count} 個のファイルを {target_dir} に正常保存しました。")
    print("=" * 60)


if __name__ == "__main__":
    main()
