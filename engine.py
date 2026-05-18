"""
どうぶつしょうぎ ゲームエンジンクラス
"""

from constants import (
    COLS, ROWS, PLAYER_1, PLAYER_2,
    P1_LION, P2_LION, P1_CHICK, P2_CHICK,
    MOVE_VECTORS, CAN_PROMOTE, PROMOTE_MAP, get_owner
)

class Move:
    def __init__(self, from_pos, to_pos, piece, is_drop=False):
        self.from_pos = from_pos  # (col, row) または 打つ場合は None
        self.to_pos = to_pos      # (col, row)
        self.piece = piece        # 動かす/打つ駒
        self.is_drop = is_drop    # 持ち駒を打つ着手なら True

    def __str__(self):
        if self.is_drop:
            return f"打 {self.piece} -> ({self.to_pos[0]},{self.to_pos[1]})"
        return f"動 {self.piece}: ({self.from_pos[0]},{self.from_pos[1]}) -> ({self.to_pos[0]},{self.to_pos[1]})"


class GameEngine:
    @staticmethod
    def get_legal_moves(board, player):
        """現在のプレイヤーのすべての合法手をリストで返す"""
        moves = []

        # --- 1. 盤上の駒を動かす手 ---
        for r in range(ROWS):
            for c in range(COLS):
                piece = board.get(c, r)
                if piece and get_owner(piece) == player:
                    # その駒の移動可能ベクトルを取得
                    vectors = MOVE_VECTORS[piece]
                    for dc, dr in vectors:
                        nc, nr = c + dc, r + dr
                        if board.in_bounds(nc, nr):
                            target = board.get(nc, nr)
                            # 移動先が空、または敵の駒なら進める
                            if target is None or get_owner(target) != player:
                                moves.append(Move((c, r), (nc, nr), piece, is_drop=False))

        # --- 2. 持ち駒を打つ手 ---
        hand_pieces = board.get_hand_types(player)
        for piece in hand_pieces:
            for r in range(ROWS):
                for c in range(COLS):
                    if board.get(c, r) is None:  # 空いているマスなら打てる
                        moves.append(Move(None, (c, r), piece, is_drop=True))

        return moves

    @staticmethod
    def make_move(board, move, player):
        """1手進める（状態変化を盤面に適用する）"""
        captured_piece = None

        if move.is_drop:
            # 持ち駒から打つ
            board.remove_from_hand(player, move.piece)
            board.set(move.to_pos[0], move.to_pos[1], move.piece)
        else:
            # 盤上の移動
            # 移動先に敵がいれば取る
            target = board.get(move.to_pos[0], move.to_pos[1])
            if target is not None:
                captured_piece = board.remove(move.to_pos[0], move.to_pos[1])
                board.add_to_hand(player, captured_piece)

            # 自分の駒を元の位置から外し、新しい位置へセット
            piece = board.remove(move.from_pos[0], move.from_pos[1])

            # ひよこの成り判定
            if piece in CAN_PROMOTE:
                # 先手なら最奥（row=0）、後手なら最奥（row=3）に入ったら成る
                if (player == PLAYER_1 and move.to_pos[1] == 0) or (player == PLAYER_2 and move.to_pos[1] == ROWS - 1):
                    piece = PROMOTE_MAP[piece]

            board.set(move.to_pos[0], move.to_pos[1], piece)

        return captured_piece

    @staticmethod
    def check_winner(board):
        """勝敗を判定する。ゲーム継続なら None"""
        # 盤上のライオンを捜索
        p1_lion = board.find_lion(PLAYER_1)
        p2_lion = board.find_lion(PLAYER_2)

        if not p1_lion: return PLAYER_2  # 先手ライオンが取られた
        if not p2_lion: return PLAYER_1  # 後手ライオンが取られた

        # トライルールの簡易判定
        # ※本来は「次の手に取られないこと」が必要
        if p1_lion[1] == 0: return PLAYER_1
        if p2_lion[1] == ROWS - 1: return PLAYER_2

        return None

