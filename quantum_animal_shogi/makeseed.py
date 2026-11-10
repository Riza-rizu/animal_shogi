"""
設定ファイル生成スクリプト: makeseed.py
研究室サーバー・並列実行用
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="シミュレーション用 YAML 設定ファイル群の生成")
    parser.add_argument("--depth", type=int, default=0, choices=[0, 2],
                        help="探索深さ: 0 (直接推論: 100万局) または 2 (Min-Max: 20万局)")
    parser.add_argument("--p1", type=str, default="ai", choices=["ai", "random"],
                        help="プレイヤー1の種別 (ai または random)")
    parser.add_argument("--p2", type=str, default="ai", choices=["ai", "random"],
                        help="プレイヤー2の種別 (ai または random)")
    parser.add_argument("--gamma", type=float, default=0.6,
                        help="千日手レピティション・ペナルティ減衰率 (デフォルト: 0.6)")
    parser.add_argument("--alpha", type=float, default=0.95,
                        help="終局手数に応じた時間割引率 (デフォルト: 0.95)")
    parser.add_argument("--n_files", type=int, default=50, help="生成するワーカー数 (デフォルト: 50)")
    parser.add_argument("--total_games", type=int, default=None,
                        help="総試合数 (未指定時は depth=0 で 100万局, depth=2 で 20万局)")
    parser.add_argument("--games", type=int, default=None, help="1ファイルあたりの試合数 (未指定時は total_games / n_files)")
    parser.add_argument("--beta", type=float, default=10.0, help="逆温度 beta")
    parser.add_argument("--model", type=str, default="quantum_model_gen0.keras", help="使用モデルパス")
    parser.add_argument("--python_bin", type=str, default="python", help="Python実行バイナリパス")
    args = parser.parse_args()

    # 総試合数の決定: depth=0 なら 100万局, depth=2 なら 20万局
    if args.total_games is not None:
        total_games = args.total_games
    elif args.games is not None:
        total_games = args.games * args.n_files
    else:
        total_games = 1_000_000 if args.depth == 0 else 200_000

    games_per_file = total_games // args.n_files

    print(f"設定ファイルを生成します: p1={args.p1}, p2={args.p2}, ワーカー数={args.n_files}, 1ファイル={games_per_file}局 (合計={total_games}局), 探索深さ={args.depth}, alpha={args.alpha}")

    for i in range(args.n_files):
        with open(f"input{i:03d}.yaml", "w", encoding="utf-8") as f:
            f.write(f"seed: {i}\n")
            f.write(f"games: {games_per_file}\n")
            f.write(f"p1: {args.p1}\n")
            f.write(f"p2: {args.p2}\n")
            f.write(f"model_path: {args.model}\n")
            f.write(f"beta: {args.beta}\n")
            f.write(f"depth: {args.depth}\n")
            f.write(f"gamma: {args.gamma}\n")
            f.write(f"alpha: {args.alpha}\n")

    with open("tasks.sh", "w", encoding="utf-8") as f:
        for i in range(args.n_files):
            f.write(f"{args.python_bin} main.py input{i:03d}.yaml\n")

    print(f"tasks.sh および input000.yaml 〜 input{args.n_files - 1:03d}.yaml を正常に生成しました。")


if __name__ == "__main__":
    main()
