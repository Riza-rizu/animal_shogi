"""
どうぶつしょうぎ メイン実行ファイル
"""

from board import Board  # 最初に作成した盤面クラスを board.py として保存してください
from engine import GameEngine
from player import HumanPlayer, RandomPlayer
from constants import PLAYER_1, PLAYER_2, PIECE_DISPLAY, COLS, ROWS

def print_board(board):
    """盤面をコンソール上に綺麗にテキスト描画する関数"""
    print("\n--- 盤面 ---")
    print("  0   1   2  (col)")
    for r in range(ROWS):
        row_str = f"{r} "
        for c in range(COLS):
            piece = board.get(c, r)
            if piece:
                row_str += f"[{PIECE_DISPLAY[piece]}]"
            else:
                row_str += "[    ]"
        print(row_str)
    
    print(f"先手(1)持ち駒: {board.hands[PLAYER_1]}")
    print(f"後手(2)持ち駒: {board.hands[PLAYER_2]}")
    print("------------")

def main():
    # 盤面とプレイヤーの初期化
    board = Board()
    board.setup()
    
    # 1を「人間」、2を「ランダムAI」に設定（両方HumanPlayerにすれば対戦可能）
    player1 = HumanPlayer(PLAYER_1)
    player2 = RandomPlayer(PLAYER_2)
    
    current_player_id = PLAYER_1
    players = {PLAYER_1: player1, PLAYER_2: player2}

    print("どうぶつしょうぎ を開始します！")
    
    # ゲームループ
    while True:
        print_board(board)
        
        # 1. 現在のプレイヤーの合法手を取得
        legal_moves = GameEngine.get_legal_moves(board, current_player_id)
        
        if not legal_moves:
            # 動かせる手がなければ詰み（相手の勝ち）
            winner = PLAYER_2 if current_player_id == PLAYER_1 else PLAYER_1
            print(f"プレイヤー {current_player_id} に打てる手がありません。")
            print(f"👑 プレイヤー {winner} の勝ちです！")
            break
            
        # 2. プレイヤーに手を選択させる
        current_player = players[current_player_id]
        selected_move = current_player.choose_move(board, legal_moves)
        print(f"\n選択された手: {selected_move}")
        
        # 3. 盤面を動かす（内部で remove して set する処理が走る）
        GameEngine.make_move(board, selected_move, current_player_id)
        
        # 4. 勝敗判定
        winner = GameEngine.check_winner(board)
        if winner:
            print_board(board)
            print(f"👑 プレイヤー {winner} の勝ちです！ゲーム終了。")
            break
            
        # 5. ターン交代
        current_player_id = PLAYER_2 if current_player_id == PLAYER_1 else PLAYER_1

if __name__ == "__main__":
    main()

