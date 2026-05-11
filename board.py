"""
どうぶつしょうぎ 盤面クラス
駒は1文字の文字列で管理する（Pieceクラスは不要）。
"""

import numpy as np
from constants import (
    COLS, ROWS, PLAYER_1, PLAYER_2,
    P1_LION, P1_GIRAFFE, P1_ELEPHANT, P1_CHICK,
    P2_LION, P2_GIRAFFE, P2_ELEPHANT, P2_CHICK,
    PIECE_TO_INDEX, NUM_PIECE_STATES,
    CAPTURE_MAP, get_owner,
    HAND_TYPES_P1, HAND_TYPES_P2,
)


class Board:
    """3×4 の盤面と持ち駒を管理するクラス"""

    def __init__(self):
        # 盤面: grid[row][col] = 駒文字 or None
        self.grid = [[None] * COLS for _ in range(ROWS)]
        # 持ち駒: {PLAYER_1: ['c', ...], PLAYER_2: ['G', ...]}
        self.hands = {PLAYER_1: [], PLAYER_2: []}

    # ---- 初期配置 ----

    def setup(self):
        """どうぶつしょうぎの初期配置"""
        self.grid = [[None] * COLS for _ in range(ROWS)]
        # 後手（上側）
        self.grid[0][0] = P2_GIRAFFE
        self.grid[0][1] = P2_LION
        self.grid[0][2] = P2_ELEPHANT
        self.grid[1][1] = P2_CHICK
        # 先手（下側）
        self.grid[2][1] = P1_CHICK
        self.grid[3][0] = P1_ELEPHANT
        self.grid[3][1] = P1_LION
        self.grid[3][2] = P1_GIRAFFE
        self.hands = {PLAYER_1: [], PLAYER_2: []}

    # ---- 基本操作 ----

    def get(self, col, row):
        return self.grid[row][col]

    def set(self, col, row, piece):
        self.grid[row][col] = piece

    def remove(self, col, row):
        piece = self.grid[row][col]
        self.grid[row][col] = None
        return piece

    def in_bounds(self, col, row):
        return 0 <= col < COLS and 0 <= row < ROWS

    # ---- 持ち駒操作 ----

    def add_to_hand(self, player, captured_piece):
        """捕獲した駒を持ち駒に追加（成り戻し＋所有者変換）"""
        converted = CAPTURE_MAP[captured_piece]
        self.hands[player].append(converted)

    def remove_from_hand(self, player, piece):
        """持ち駒から1枚取り出す"""
        self.hands[player].remove(piece)
        return piece

    def get_hand_types(self, player):
        """持ち駒の種類一覧（重複なし）"""
        return list(set(self.hands[player]))

    # ---- ライオン探索 ----

    def find_lion(self, player):
        """指定プレイヤーのライオンの座標 (col, row) を返す"""
        target = P1_LION if player == PLAYER_1 else P2_LION
        for r in range(ROWS):
            for c in range(COLS):
                if self.grid[r][c] == target:
                    return (c, r)
        return None

    # ---- コピー ----

    def copy(self):
        new_board = Board()
        for r in range(ROWS):
            for c in range(COLS):
                new_board.grid[r][c] = self.grid[r][c]
        for p in [PLAYER_1, PLAYER_2]:
            new_board.hands[p] = list(self.hands[p])
        return new_board

    # ---- シリアライズ（千日手判定用） ----

    def to_hashable(self):
        grid_t = tuple(tuple(row) for row in self.grid)
        h1 = tuple(sorted(self.hands[PLAYER_1]))
        h2 = tuple(sorted(self.hands[PLAYER_2]))
        return (grid_t, h1, h2)

    # ---- テンソル変換（AI学習用） ----

    def to_tensor(self):
        """
        盤面をワンホットテンソルに変換する。

        Returns:
            np.ndarray of shape (ROWS, COLS, NUM_PIECE_STATES) = (4, 3, 11)
            各マスが長さ11のワンホットベクトル。
        """
        tensor = np.zeros((ROWS, COLS, NUM_PIECE_STATES), dtype=np.float32)
        for r in range(ROWS):
            for c in range(COLS):
                idx = PIECE_TO_INDEX[self.grid[r][c]]
                tensor[r, c, idx] = 1.0
        return tensor

    def hands_to_vector(self):
        """
        持ち駒をベクトルに変換する。

        Returns:
            np.ndarray of shape (6,)
            [先手きりん数, 先手ぞう数, 先手ひよこ数,
             後手きりん数, 後手ぞう数, 後手ひよこ数]
        """
        vec = np.zeros(6, dtype=np.float32)
        for i, pt in enumerate(HAND_TYPES_P1):
            vec[i] = self.hands[PLAYER_1].count(pt)
        for i, pt in enumerate(HAND_TYPES_P2):
            vec[3 + i] = self.hands[PLAYER_2].count(pt)
        return vec

    def to_state_vector(self, current_player):
        """
        盤面全体を1次元のベクトルに変換する（NN入力用）。

        Returns:
            np.ndarray of shape (139,)
            = 盤面(4*3*11=132) + 持ち駒(6) + 手番(1)
        """
        board_flat = self.to_tensor().flatten()          # 132
        hands_vec = self.hands_to_vector()               # 6
        turn = np.array([1.0 if current_player == PLAYER_2 else 0.0],
                        dtype=np.float32)                # 1
        return np.concatenate([board_flat, hands_vec, turn])
