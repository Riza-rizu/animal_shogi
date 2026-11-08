"""
量子どうぶつしょうぎ 世代交代自動ループ実行スクリプト (run_loop.py)
"""

import os
import sys
import subprocess
import argparse

repo_dir = os.path.dirname(os.path.abspath(__file__))


def run_command(cmd):
    print(f"\n[実行] {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=repo_dir)
    if res.returncode != 0:
        print(f"エラー: コマンドが失敗しました (終了コード: {res.returncode})")
        sys.exit(res.returncode)


def main():
    parser = argparse.ArgumentParser(description="量子どうぶつしょうぎ 世代交代ループ")
    parser.add_argument("--start_gen", type=int, default=0, help="開始世代 (デフォルト: 0)")
    parser.add_argument("--num_gens", type=int, default=3, help="学習を進める世代数")
    parser.add_argument("--games_per_gen", type=int, default=200, help="1世代あたりの自己対局数")
    parser.add_argument("--epochs", type=int, default=10, help="学習エポック数")
    parser.add_argument("--eval_games", type=int, default=100, help="世代間評価対戦数")
    parser.add_argument("--beta", type=float, default=10.0, help="逆温度パラメータ")
    args = parser.parse_args()

    # Gen0 の存在確認
    gen0_model = os.path.join(repo_dir, "quantum_model_gen0.keras")
    if args.start_gen == 0 and not os.path.exists(gen0_model):
        print("初期モデル (quantum_model_gen0.keras) が存在しないため作成します...")
        run_command([sys.executable, "train.py", "--init", "--output", "quantum_model_gen0.keras"])

    for gen in range(args.start_gen, args.start_gen + args.num_gens):
        current_model = f"quantum_model_gen{gen}.keras"
        next_model = f"quantum_model_gen{gen + 1}.keras"
        data_file = f"train_data_gen{gen}.npy"

        print(f"\n=======================================================")
        print(f"★ 世代 {gen} -> 世代 {gen + 1} の学習サイクルを開始します")
        print(f"=======================================================")

        # 1. 自己対局によるデータ収集
        print(f"\n[ステップ1] 自己対局データ収集 ({args.games_per_gen} 局)...")
        run_command([
            sys.executable, "battle.py",
            "--m1", current_model,
            "--m2", current_model,
            "--games", str(args.games_per_gen),
            "--beta", str(args.beta),
            "--save_data", data_file
        ])

        # 2. 学習
        print(f"\n[ステップ2] モデルの学習 ({next_model})...")
        run_command([
            sys.executable, "train.py",
            "--data", data_file,
            "--output", next_model,
            "--epochs", str(args.epochs)
        ])

        # 3. 新旧モデル対決
        print(f"\n[ステップ3] 新旧モデル対決評価 ({next_model} vs {current_model}, {args.eval_games} 局)...")
        run_command([
            sys.executable, "battle.py",
            "--m1", next_model,
            "--m2", current_model,
            "--games", str(args.eval_games),
            "--beta", str(args.beta)
        ])

    print("\nすべての世代交代サイクルが完了しました！")


if __name__ == "__main__":
    main()
