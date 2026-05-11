"""
どうぶつしょうぎ コンソールUI
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constants import COLS, ROWS, PLAYER_1, PLAYER_2, PIECE_DISPLAY
from engine import Engine


def display_board(engine):
    """盤面をコンソールに表示する"""
    board = engine.board

    # 後手の持ち駒
    p2_hand = board.hands[PLAYER_2]
    print(f"\n後手の持ち駒: {p2_hand if p2_hand else 'なし'}")

    # 盤面
    print("  +" + "----+" * COLS)
    for r in range(ROWS):
        row_str = f"{r} |"
        for c in range(COLS):
            piece = board.get(c, r)
            if piece:
                row_str += f"{PIECE_DISPLAY[piece]} |"
            else:
                row_str += " ・ |"
        print(row_str)
        print("  +" + "----+" * COLS)
    col_str = "   "
    for c in range(COLS):
        col_str += f" {c}   "
    print(col_str)

    # 先手の持ち駒
    p1_hand = board.hands[PLAYER_1]
    print(f"先手の持ち駒: {p1_hand if p1_hand else 'なし'}")

    turn = "先手" if engine.current_player == PLAYER_1 else "後手"
    print(f"\n--- {turn} の番です ---")


def display_result(engine):
    """ゲーム結果を表示する"""
    print("\n" + "=" * 30)
    print(f"  結果: {engine.get_result_text()}")
    print(f"  総手数: {len(engine.move_history)} 手")
    print("=" * 30)
