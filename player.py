"""
どうぶつしょうぎ プレイヤークラス
"""

import random

class HumanPlayer:
    def __init__(self, player_id):
        self.player_id = player_id

    def choose_move(self, board, legal_moves):
        """人間がコンソール入力で手を選ぶ"""
        print("\n選択可能な手:")
        for idx, m in enumerate(legal_moves):
            print(f"[{idx}] {m}")

        while True:
            try:
                choice = int(input(f"プレイヤー {self.player_id} の番です。番号を入力してください: "))
                if 0 <= choice < len(legal_moves):
                    return legal_moves[choice]
            except ValueError:
                pass
            print("無効な数値です。もう一度入力してください。")


class RandomPlayer:
    def __init__(self, player_id):
        self.player_id = player_id

    def choose_move(self, board, legal_moves):
        """合法手の中から完全にランダムで選ぶAI"""
        return random.choice(legal_moves)


