"""
設定ファイル生成スクリプト: makeseed.py
- depth 0 (深さ0 Min-Maxなし) と depth 2 (深さ2 Min-Maxあり) を切り替え可能
"""

import argparse

def main():
    parser = argparse.ArgumentParser(description="シミュレーション用 YAML 設定ファイル群の生成")
    parser.add_argument("--depth", type=int, default=2, choices=[0, 2],
                        help="探索深さ: 0 (Min-Maxなし) または 2 (深さ2 Min-Max探索)")
    parser.add_argument("--n_files", type=int, default=15, help="生成するワーカー数")
    parser.add_argument("--games", type=int, default=1000, help="1ファイルあたりの試合数")
    parser.add_argument("--beta", type=float, default=10.0, help="逆温度 beta")
    parser.add_argument("--model", type=str, default="animal_shogi_model.keras", help="使用モデルパス")
    args = parser.parse_args()

    print(f"設定ファイルを生成します: ワーカー数={args.n_files}, 1ファイル={args.games}局 (合計={args.n_files * args.games}局), 探索深さ={args.depth}")

    for i in range(args.n_files):
        with open(f"input{i:03d}.yaml", "w", encoding="utf-8") as f:
            f.write(f"seed: {i}\n")
            f.write(f"games: {args.games}\n")
            f.write("p1: ai\n")
            f.write("p2: ai\n")
            f.write(f"model_path: {args.model}\n")
            f.write(f"beta: {args.beta}\n")
            f.write(f"depth: {args.depth}\n")

    with open("tasks.sh", "w", encoding="utf-8") as f:
        for i in range(args.n_files):
            f.write(f".venv/bin/python main.py input{i:03d}.yaml\n")

    print(f"tasks.sh および input000.yaml 〜 input{args.n_files - 1:03d}.yaml を正常に生成しました。")

if __name__ == "__main__":
    main()
