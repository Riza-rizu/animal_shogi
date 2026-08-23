"""
どうぶつしょうぎ ローカルPC用強化学習自動ループ実行スクリプト
"""

import os
import sys
import subprocess
import argparse
import shutil

def main():
    parser = argparse.ArgumentParser(description="ローカルPC用 強化学習自動ループ実行スクリプト")
    parser.add_argument("--cycles", type=int, default=1, help="学習ループを回す回数 (世代数)")
    parser.add_argument("--n_files", type=int, default=4, help="生成する設定ファイル数 (並列プロセス数)")
    parser.add_argument("--games", type=int, default=1000, help="1プロセスあたりのゲーム数")
    parser.add_argument("--beta", type=float, default=10.0, help="逆温度 beta")
    parser.add_argument("--output_model", type=str, default=None, help="学習結果を別名で保存するファイル名 (指定しない場合は上書き)")
    args = parser.parse_args()

    python_exe = sys.executable
    original_model = "animal_shogi_model.keras"
    backup_model = original_model + ".tmp_bak"
    use_alias = args.output_model is not None

    print("========================================")
    print("強化学習ローカル自動ループを開始します")
    print(f"  総サイクル（世代数）: {args.cycles}")
    print(f"  並列プロセス数      : {args.n_files}")
    print(f"  プロセス毎のゲーム数: {args.games}")
    print(f"  逆温度 beta         : {args.beta}")
    if use_alias:
        print(f"  出力モデル名（別名）: {args.output_model} (元モデルは上書きしません)")
    else:
        print(f"  出力モデル名        : {original_model} (上書き保存)")
    print(f"  実行 Python パス    : {python_exe}")
    print("========================================")

    # 1. 実行前のオリジナルモデルの一時退避
    if use_alias:
        if os.path.exists(original_model):
            print(f"\n[INFO] 元のモデル {original_model} を一時退避しています...")
            shutil.copy2(original_model, backup_model)
        else:
            print(f"\n[WARNING] 元のモデル {original_model} が存在しないため、退避をスキップします。")

    try:
        for cycle in range(args.cycles):
            print(f"\n[CYCLE {cycle + 1} / {args.cycles}] 開始します")
            print("-" * 40)

            # (1) 設定ファイルの生成
            print("1/5. 設定ファイル (input*.yaml) を生成しています...")
            for i in range(args.n_files):
                seed_val = i + (cycle * args.n_files)
                with open(f"input{i:03d}.yaml", "w") as f:
                    f.write(f"seed: {seed_val}\n")
                    f.write(f"games: {args.games}\n")
                    f.write("p1: ai\n")
                    f.write("p2: ai\n")
                    f.write(f"model_path: {original_model}\n") # シミュレーションは常にオリジナルのパス名で動くようにします
                    f.write(f"beta: {args.beta}\n")

            # (2) 並列シミュレーションの実行
            print("2/5. 並列対戦シミュレーションを実行中...")
            cmd_sim = [python_exe, "run_parallel.py", "--num_processes", str(args.n_files)]
            res_sim = subprocess.run(cmd_sim)
            if res_sim.returncode != 0:
                print("[ERROR] シミュレーションの実行中にエラーが発生しました。処理を中断します。")
                sys.exit(res_sim.returncode)

            # (3) データセットの作成
            print("3/5. データのマージと学習用データセット (train_data.npy) の作成中...")
            cmd_dataset = [python_exe, "make_dataset.py"]
            res_dataset = subprocess.run(cmd_dataset)
            if res_dataset.returncode != 0:
                print("[ERROR] データセットの作成中にエラーが発生しました。処理を中断します。")
                sys.exit(res_dataset.returncode)

            # (4) モデルの学習
            print("4/5. モデルの学習 (パラメータ更新) を実行中...")
            cmd_train = [python_exe, "train.py"]
            res_train = subprocess.run(cmd_train)
            if res_train.returncode != 0:
                print("[ERROR] モデルの学習中にエラーが発生しました。処理を中断します。")
                sys.exit(res_train.returncode)

            # (5) 不要になった YAML 設定ファイルの削除 (クリーンアップ)
            print("5/5. 一時設定ファイル (input*.yaml) をクリーンアップ中...")
            for i in range(args.n_files):
                yaml_file = f"input{i:03d}.yaml"
                if os.path.exists(yaml_file):
                    try:
                        os.remove(yaml_file)
                    except Exception as e:
                        print(f"警告: {yaml_file} の削除に失敗しました: {e}")

            print(f"[CYCLE {cycle + 1} / {args.cycles}] 完了しました！")
            print("-" * 40)

        # 2. 実行後のリネームと復元処理
        if use_alias:
            if os.path.exists(original_model):
                print(f"\n[INFO] 新しく学習したモデルを別名 {args.output_model} に保存しています...")
                if os.path.exists(args.output_model):
                    os.remove(args.output_model)
                os.rename(original_model, args.output_model)
            
            if os.path.exists(backup_model):
                print(f"[INFO] 元のモデル {original_model} を元の位置に復元しています...")
                os.rename(backup_model, original_model)

        print("\n[COMPLETE] すべての学習サイクルが正常に完了しました！")

    except Exception as e:
        print(f"\n[ERROR] 例外エラーが発生しました: {e}")
        # エラー発生時も元のモデルを復元
        if use_alias and os.path.exists(backup_model):
            print(f"[INFO] エラー発生に伴い、一時退避していた元のモデル {original_model} を復元します...")
            if os.path.exists(original_model):
                os.remove(original_model)
            os.rename(backup_model, original_model)
        sys.exit(1)

if __name__ == "__main__":
    main()
