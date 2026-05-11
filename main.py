"""
どうぶつしょうぎ メインエントリーポイント

使い方:
    python main.py              # 人間 vs 人間
    python main.py --p1 human --p2 random      # 人間 vs ランダムAI
    python main.py --p1 minimax --p2 random     # ミニマックスAI vs ランダムAI
"""

import argparse
import sys
import os

# animal_shogi ディレクトリをパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from constants import PLAYER_1, PLAYER_2
from engine import Engine
from player import HumanPlayer, RandomAIPlayer, MinimaxAIPlayer
from ui.console import display_board, display_result


def create_player(player_type: str, player_id: int):
    """プレイヤーの種類に応じてインスタンスを生成する"""
    if player_type == "human":
        return HumanPlayer(player_id)
    elif player_type == "random":
        return RandomAIPlayer(player_id)
    elif player_type == "minimax":
        return MinimaxAIPlayer(player_id, depth=3)
    else:
        print(f"不明なプレイヤー種類: {player_type}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="どうぶつしょうぎ")
    parser.add_argument("--p1", type=str, default="human",
                        choices=["human", "random", "minimax"],
                        help="先手のプレイヤー種類")
    parser.add_argument("--p2", type=str, default="human",
                        choices=["human", "random", "minimax"],
                        help="後手のプレイヤー種類")
    args = parser.parse_args()

    engine = Engine()
    players = {
        PLAYER_1: create_player(args.p1, PLAYER_1),
        PLAYER_2: create_player(args.p2, PLAYER_2),
    }

    print("=== どうぶつしょうぎ 開始 ===")
    print(f"先手: {args.p1}  /  後手: {args.p2}\n")

    while not engine.is_game_over():
        display_board(engine)

        current = engine.current_player
        legal_moves = engine.get_legal_moves()

        if not legal_moves:
            # 合法手がない場合（通常はどうぶつしょうぎでは起きにくい）
            print("合法手がありません。パスします。")
            engine.switch_turn()
            continue

        player = players[current]
        move = player.select_move(engine, legal_moves)

        turn_name = "先手" if current == PLAYER_1 else "後手"
        print(f"\n{turn_name} の手: {move}")

        engine.make_move(move)

    display_board(engine)
    display_result(engine)


if __name__ == "__main__":
    main()
