"""
どうぶつしょうぎ ゲームエンジン
"""

from constants import (
    COLS, ROWS, PLAYER_1, PLAYER_2,
    P1_LION, P2_LION, P1_PIECES, P2_PIECES,
    MOVE_VECTORS, PROMOTE_MAP, CAN_PROMOTE, get_owner,
    RESULT_NONE, RESULT_PLAYER1, RESULT_PLAYER2, RESULT_DRAW,
)
from board import Board


class Move:
    """一手を表すデータクラス"""

    def __init__(self, from_pos, to_pos, piece, promote=False, drop=False):
        self.from_pos = from_pos   # (col, row) or None
        self.to_pos = to_pos       # (col, row)
        self.piece = piece         # 駒文字 ('l', 'G', etc.)
        self.promote = promote
        self.drop = drop

    def __repr__(self):
        if self.drop:
            return f"Drop({self.piece} -> {self.to_pos})"
        promo = "+" if self.promote else ""
        return f"Move({self.piece}{promo}: {self.from_pos} -> {self.to_pos})"

    def __eq__(self, other):
        if not isinstance(other, Move):
            return False
        return (self.from_pos == other.from_pos and
                self.to_pos == other.to_pos and
                self.piece == other.piece and
                self.promote == other.promote and
                self.drop == other.drop)

    def __hash__(self):
        return hash((self.from_pos, self.to_pos, self.piece,
                      self.promote, self.drop))


class Engine:
    """ゲーム進行管理クラス"""

    def __init__(self):
        self.board = Board()
        self.board.setup()
        self.current_player = PLAYER_1
        self.result = RESULT_NONE
        self.move_history = []
        self.position_history = {}
        self._record_position()

    def opponent(self, player=None):
        if player is None:
            player = self.current_player
        return PLAYER_2 if player == PLAYER_1 else PLAYER_1

    def switch_turn(self):
        self.current_player = self.opponent()

    # ---- 合法手生成 ----

    def get_legal_moves(self, player=None):
        if player is None:
            player = self.current_player
        my_pieces = P1_PIECES if player == PLAYER_1 else P2_PIECES
        moves = []

        # (A) 盤上の駒を動かす
        for r in range(ROWS):
            for c in range(COLS):
                piece = self.board.get(c, r)
                if piece is None or piece not in my_pieces:
                    continue
                for dc, dr in MOVE_VECTORS[piece]:
                    nc, nr = c + dc, r + dr
                    if not self.board.in_bounds(nc, nr):
                        continue
                    target = self.board.get(nc, nr)
                    if target is not None and target in my_pieces:
                        continue
                    promote = False
                    if piece in CAN_PROMOTE:
                        if player == PLAYER_1 and nr == 0:
                            promote = True
                        elif player == PLAYER_2 and nr == ROWS - 1:
                            promote = True
                    moves.append(Move((c, r), (nc, nr), piece, promote=promote))

        # (B) 持ち駒を打つ
        for kind in self.board.get_hand_types(player):
            for r in range(ROWS):
                for c in range(COLS):
                    if self.board.get(c, r) is None:
                        moves.append(Move(None, (c, r), kind, drop=True))
        return moves

    # ---- 手の実行 ----

    def make_move(self, move):
        player = self.current_player

        if move.drop:
            tc, tr = move.to_pos
            self.board.remove_from_hand(player, move.piece)
            self.board.set(tc, tr, move.piece)
        else:
            fc, fr = move.from_pos
            tc, tr = move.to_pos
            piece = self.board.remove(fc, fr)
            captured = self.board.get(tc, tr)
            if captured:
                self.board.remove(tc, tr)
                self.board.add_to_hand(player, captured)
            if move.promote:
                piece = PROMOTE_MAP[piece]
            self.board.set(tc, tr, piece)

        self.move_history.append(move)
        self.result = self._check_result()

        if self.result == RESULT_NONE:
            self.switch_turn()
            self._record_position()
            if self._is_repetition():
                self.result = RESULT_DRAW
        return self.result

    # ---- 勝利判定 ----

    def _check_result(self):
        player = self.current_player
        opp = self.opponent(player)

        # ライオンキャッチ
        if self.board.find_lion(opp) is None:
            return RESULT_PLAYER1 if player == PLAYER_1 else RESULT_PLAYER2

        # トライルール
        lion_pos = self.board.find_lion(player)
        if lion_pos:
            _, lr = lion_pos
            target_row = 0 if player == PLAYER_1 else ROWS - 1
            if lr == target_row:
                if not self._is_lion_capturable(lion_pos, opp):
                    return RESULT_PLAYER1 if player == PLAYER_1 else RESULT_PLAYER2

        return RESULT_NONE

    def _is_lion_capturable(self, lion_pos, attacker):
        lc, lr = lion_pos
        atk_pieces = P1_PIECES if attacker == PLAYER_1 else P2_PIECES
        for r in range(ROWS):
            for c in range(COLS):
                piece = self.board.get(c, r)
                if piece is None or piece not in atk_pieces:
                    continue
                for dc, dr in MOVE_VECTORS[piece]:
                    if (c + dc, r + dr) == (lc, lr):
                        return True
        return False

    # ---- 千日手 ----

    def _record_position(self):
        key = (self.board.to_hashable(), self.current_player)
        self.position_history[key] = self.position_history.get(key, 0) + 1

    def _is_repetition(self):
        key = (self.board.to_hashable(), self.current_player)
        return self.position_history.get(key, 0) >= 3

    def is_game_over(self):
        return self.result != RESULT_NONE

    def get_result_text(self):
        if self.result == RESULT_PLAYER1:
            return "先手（PLAYER_1）の勝ち！"
        elif self.result == RESULT_PLAYER2:
            return "後手（PLAYER_2）の勝ち！"
        elif self.result == RESULT_DRAW:
            return "引き分け（千日手）"
        return "対局中"
