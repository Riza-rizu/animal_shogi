"""
どうぶつしょうぎ プレイヤー定義
"""

import random
from engine import Engine, Move


class BasePlayer:
    """プレイヤーの基底クラス"""

    def __init__(self, player_id):
        self.player_id = player_id

    def select_move(self, engine, legal_moves):
        raise NotImplementedError


class HumanPlayer(BasePlayer):
    """人間プレイヤー（コンソール入力）"""

    def select_move(self, engine, legal_moves):
        print(f"\n合法手一覧（{len(legal_moves)} 手）:")
        for i, move in enumerate(legal_moves):
            print(f"  [{i}] {move}")
        while True:
            try:
                idx = int(input("手を選んでください（番号入力）: "))
                if 0 <= idx < len(legal_moves):
                    return legal_moves[idx]
                print("範囲外です。もう一度入力してください。")
            except ValueError:
                print("数字を入力してください。")


class RandomAIPlayer(BasePlayer):
    """ランダムに手を選ぶAI"""

    def select_move(self, engine, legal_moves):
        return random.choice(legal_moves)


class MinimaxAIPlayer(BasePlayer):
    """ミニマックス法によるAI（αβ枝刈り付き）"""

    def __init__(self, player_id, depth=3):
        super().__init__(player_id)
        self.depth = depth

    def select_move(self, engine, legal_moves):
        best_score = float("-inf")
        best_move = legal_moves[0]
        for move in legal_moves:
            sim = self._copy_engine(engine)
            sim.make_move(move)
            score = self._minimax(sim, self.depth - 1, False,
                                  float("-inf"), float("inf"))
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

    def _minimax(self, engine, depth, is_maximizing, alpha, beta):
        if engine.is_game_over() or depth == 0:
            return self._evaluate(engine)
        moves = engine.get_legal_moves()
        if not moves:
            return self._evaluate(engine)

        if is_maximizing:
            max_eval = float("-inf")
            for move in moves:
                sim = self._copy_engine(engine)
                sim.make_move(move)
                val = self._minimax(sim, depth - 1, False, alpha, beta)
                max_eval = max(max_eval, val)
                alpha = max(alpha, val)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float("inf")
            for move in moves:
                sim = self._copy_engine(engine)
                sim.make_move(move)
                val = self._minimax(sim, depth - 1, True, alpha, beta)
                min_eval = min(min_eval, val)
                beta = min(beta, val)
                if beta <= alpha:
                    break
            return min_eval

    def _evaluate(self, engine):
        """盤面の評価値（正=自分有利）"""
        from constants import (
            RESULT_PLAYER1, RESULT_PLAYER2, RESULT_DRAW,
            ROWS, COLS, get_owner,
        )
        if engine.result == RESULT_PLAYER1:
            return 1000 if self.player_id == 1 else -1000
        elif engine.result == RESULT_PLAYER2:
            return 1000 if self.player_id == 2 else -1000
        elif engine.result == RESULT_DRAW:
            return 0

        # 駒の価値
        PIECE_VALUES = {
            'l': 0, 'L': 0,         # ライオン（取ったら勝ち）
            'g': 3, 'G': 3,         # きりん
            'e': 3, 'E': 3,         # ぞう
            'c': 1, 'C': 1,         # ひよこ
            'h': 5, 'H': 5,         # にわとり
        }
        score = 0
        for r in range(ROWS):
            for c in range(COLS):
                piece = engine.board.get(c, r)
                if piece is None:
                    continue
                val = PIECE_VALUES.get(piece, 0)
                if get_owner(piece) == self.player_id:
                    score += val
                else:
                    score -= val
        # 持ち駒
        opp = 2 if self.player_id == 1 else 1
        for p in engine.board.hands.get(self.player_id, []):
            score += PIECE_VALUES.get(p, 0)
        for p in engine.board.hands.get(opp, []):
            score -= PIECE_VALUES.get(p, 0)
        return score

    def _copy_engine(self, engine):
        new_engine = Engine.__new__(Engine)
        new_engine.board = engine.board.copy()
        new_engine.current_player = engine.current_player
        new_engine.result = engine.result
        new_engine.move_history = list(engine.move_history)
        new_engine.position_history = dict(engine.position_history)
        return new_engine
