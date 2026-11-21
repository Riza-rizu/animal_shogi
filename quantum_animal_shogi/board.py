"""
量子どうぶつしょうぎ 盤面クラス
"""

import numpy as np
from constants import (
    COLS, ROWS, PLAYER_1, PLAYER_2,
    PIECE_LION, PIECE_GIRAFFE, PIECE_ELEPHANT, PIECE_CHICK, PIECE_HEN,
    ALL_PERMUTATIONS, P1_INITIAL_POSITIONS, P2_INITIAL_POSITIONS,
    INDEX_EMPTY, P1_INDICES, P2_INDICES, NUM_PIECE_STATES
)


class QuantumBoard:
    """
    3×4の盤面、持ち駒、および各駒の重ね合わせ（順列候補）を管理するクラス。
    各駒は (owner, piece_id) で一意に識別される。
      - owner: 初期の所属プレイヤー (PLAYER_1 または PLAYER_2)
      - piece_id: 初期の駒ID (0, 1, 2, 3)
    """

    def __init__(self):
        # 盤面: grid[row][col] = (controller, owner, piece_id) or None
        # controller は現在その駒を操作しているプレイヤー（手駒から打った場合は打った側）
        self.grid = [[None] * COLS for _ in range(ROWS)]

        # 持ち駒: {PLAYER_1: [(owner, piece_id), ...], PLAYER_2: [...]}
        self.hands = {PLAYER_1: [], PLAYER_2: []}

        # 生き残っている順列インデックス (0..23)
        self.valid_perms = {
            PLAYER_1: list(range(24)),
            PLAYER_2: list(range(24))
        }

        # 成りフラグ: {(owner, piece_id): bool}
        self.promoted = {}
        for p in [PLAYER_1, PLAYER_2]:
            for pid in range(4):
                self.promoted[(p, pid)] = False

        self.last_player = None

    def setup(self):
        """初期配置の設定"""
        self.grid = [[None] * COLS for _ in range(ROWS)]
        self.hands = {PLAYER_1: [], PLAYER_2: []}
        self.valid_perms = {
            PLAYER_1: list(range(24)),
            PLAYER_2: list(range(24))
        }
        for p in [PLAYER_1, PLAYER_2]:
            for pid in range(4):
                self.promoted[(p, pid)] = False

        # 先手の初期配置
        for pid, (c, r) in P1_INITIAL_POSITIONS.items():
            self.grid[r][c] = (PLAYER_1, PLAYER_1, pid)

        # 後手の初期配置
        for pid, (c, r) in P2_INITIAL_POSITIONS.items():
            self.grid[r][c] = (PLAYER_2, PLAYER_2, pid)

    def get(self, col, row):
        return self.grid[row][col]

    def set(self, col, row, piece_info):
        self.grid[row][col] = piece_info

    def remove(self, col, row):
        piece_info = self.grid[row][col]
        self.grid[row][col] = None
        return piece_info

    def in_bounds(self, col, row):
        return 0 <= col < COLS and 0 <= row < ROWS

    # ---- 駒の候補・確率・判定 ----

    def get_piece_candidates(self, owner, piece_id):
        """その駒が現在取りうる駒種 (set) を返す（成りも考慮）"""
        perms = self.valid_perms[owner]
        if not perms:
            return set()

        is_prom = self.promoted.get((owner, piece_id), False)
        candidates = set()
        for idx in perms:
            base_piece = ALL_PERMUTATIONS[idx][piece_id]
            if base_piece == PIECE_CHICK and is_prom:
                candidates.add(PIECE_HEN)
            else:
                candidates.add(base_piece)
        return candidates

    def can_be_lion(self, owner, piece_id):
        """その駒がライオンであり得るか"""
        perms = self.valid_perms[owner]
        return any(ALL_PERMUTATIONS[idx][piece_id] == PIECE_LION for idx in perms)

    def is_determined_lion(self, owner, piece_id):
        """その駒が100%ライオンに確定しているか"""
        perms = self.valid_perms[owner]
        if not perms:
            return False
        return all(ALL_PERMUTATIONS[idx][piece_id] == PIECE_LION for idx in perms)

    def get_probabilities(self, owner, piece_id):
        """
        その駒が各駒種である確率辞書を返す。
        {'l': pl, 'g': pg, 'e': pe, 'c': pc, 'h': ph}
        """
        perms = self.valid_perms[owner]
        tot = len(perms)
        probs = {PIECE_LION: 0.0, PIECE_GIRAFFE: 0.0, PIECE_ELEPHANT: 0.0, PIECE_CHICK: 0.0, PIECE_HEN: 0.0}
        if tot == 0:
            return probs

        is_prom = self.promoted.get((owner, piece_id), False)
        for idx in perms:
            base_piece = ALL_PERMUTATIONS[idx][piece_id]
            if base_piece == PIECE_CHICK and is_prom:
                probs[PIECE_HEN] += 1.0
            else:
                probs[base_piece] += 1.0

        for k in probs:
            probs[k] /= tot
        return probs

    def get_all_piece_probs_array(self):
        """
        全8駒（先手4駒 + 後手4駒）の各駒種確率を (8, 5) の NumPy 配列で返す。
        駒種順: [LION, GIRAFFE, ELEPHANT, CHICK, HEN]
        インデックス 0..3: 先手 P1 (pid 0..3)
        インデックス 4..7: 後手 P2 (pid 0..3)
        """
        piece_order = [PIECE_LION, PIECE_GIRAFFE, PIECE_ELEPHANT, PIECE_CHICK, PIECE_HEN]
        res = np.zeros((8, 5), dtype=np.float32)
        row = 0
        for owner in [PLAYER_1, PLAYER_2]:
            for pid in range(4):
                p_dict = self.get_probabilities(owner, pid)
                for col, p_type in enumerate(piece_order):
                    res[row, col] = p_dict[p_type]
                row += 1
        return res

    def get_determined_lion_id(self, owner):
        """そのプレイヤーのライオンが確定している場合、その pid を返す。未確定なら None"""
        for pid in range(4):
            if self.is_determined_lion(owner, pid):
                return pid
        return None

    # ---- 持ち駒操作 ----

    def add_to_hand(self, capturing_player, captured_owner, captured_pid):
        """
        相手の駒を取ったときの処理:
        1. 取られた駒はライオンではないので、生き残り順列から除外
        2. 成りをリセットして持ち駒に追加
        """
        # 取られた駒からライオンを除外
        self.valid_perms[captured_owner] = [
            idx for idx in self.valid_perms[captured_owner]
            if ALL_PERMUTATIONS[idx][captured_pid] != PIECE_LION
        ]
        # 成りはリセット（ニワトリはヒヨコに戻る）
        self.promoted[(captured_owner, captured_pid)] = False

        # capturing_player の持ち駒リストに追加
        self.hands[capturing_player].append((captured_owner, captured_pid))

    def remove_from_hand(self, player, piece_tuple):
        self.hands[player].remove(piece_tuple)

    # ---- コピー ----

    def copy(self):
        new_b = QuantumBoard()
        for r in range(ROWS):
            for c in range(COLS):
                new_b.grid[r][c] = self.grid[r][c]
        for p in [PLAYER_1, PLAYER_2]:
            new_b.hands[p] = list(self.hands[p])
            new_b.valid_perms[p] = list(self.valid_perms[p])
        new_b.promoted = dict(self.promoted)
        new_b.last_player = self.last_player
        return new_b

    # ---- シリアライズ（千日手・レピティション判定用） ----

    def to_hashable(self):
        grid_t = tuple(tuple(row) for row in self.grid)
        h1 = tuple(sorted(self.hands[PLAYER_1]))
        h2 = tuple(sorted(self.hands[PLAYER_2]))
        p1_perms = tuple(sorted(self.valid_perms[PLAYER_1]))
        p2_perms = tuple(sorted(self.valid_perms[PLAYER_2]))
        prom_items = tuple(sorted((k, v) for k, v in self.promoted.items() if v))
        return (grid_t, h1, h2, p1_perms, p2_perms, prom_items)

    # ---- テンソル変換（AI学習入力用: 計139次元） ----

    def to_state_vector(self, current_player):
        """
        盤面(132) + 持ち駒(6) + 手番(1) = 139次元のベクトルを出力
        """
        # 1. 盤面テンソル (4, 3, 11)
        board_tensor = np.zeros((ROWS, COLS, NUM_PIECE_STATES), dtype=np.float32)
        for r in range(ROWS):
            for c in range(COLS):
                cell = self.grid[r][c]
                if cell is None:
                    board_tensor[r, c, INDEX_EMPTY] = 1.0
                else:
                    controller, owner, pid = cell
                    probs = self.get_probabilities(owner, pid)
                    idx_map = P1_INDICES if controller == PLAYER_1 else P2_INDICES
                    for piece_type, prob in probs.items():
                        if prob > 0:
                            idx = idx_map[piece_type]
                            board_tensor[r, c, idx] = prob
        board_flat = board_tensor.flatten()  # 4 * 3 * 11 = 132

        # 2. 持ち駒ベクトル (6次元: P1 [g, e, c], P2 [g, e, c] の期待枚数)
        hands_vec = np.zeros(6, dtype=np.float32)
        for owner, pid in self.hands[PLAYER_1]:
            probs = self.get_probabilities(owner, pid)
            hands_vec[0] += probs[PIECE_GIRAFFE]
            hands_vec[1] += probs[PIECE_ELEPHANT]
            hands_vec[2] += probs[PIECE_CHICK]

        for owner, pid in self.hands[PLAYER_2]:
            probs = self.get_probabilities(owner, pid)
            hands_vec[3] += probs[PIECE_GIRAFFE]
            hands_vec[4] += probs[PIECE_ELEPHANT]
            hands_vec[5] += probs[PIECE_CHICK]

        # 3. 手番 (1次元: P2手番なら 1.0, P1なら 0.0)
        turn = np.array([1.0 if current_player == PLAYER_2 else 0.0], dtype=np.float32)

        return np.concatenate([board_flat, hands_vec, turn])


def state2vec(board, current_player_id):
    """盤面と手番から139次元の状態ベクトルを返す"""
    return board.to_state_vector(current_player_id)
