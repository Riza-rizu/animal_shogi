"""
量子どうぶつしょうぎ 定数定義
"""

import itertools

# --- 盤面サイズ ---
COLS = 3
ROWS = 4

# --- プレイヤー ---
PLAYER_1 = 1   # 先手（下側・小文字）
PLAYER_2 = 2   # 後手（上側・大文字）

# --- 駒種定義 ---
PIECE_LION     = 'l'
PIECE_GIRAFFE  = 'g'
PIECE_ELEPHANT = 'e'
PIECE_CHICK    = 'c'
PIECE_HEN      = 'h'

BASE_PIECES = [PIECE_LION, PIECE_GIRAFFE, PIECE_ELEPHANT, PIECE_CHICK]

# 初期4駒の全順列（4! = 24通り）
# 各順列は tuple of 4 pieces (例: ('l', 'g', 'e', 'c'))
# インデックス 0..3 は初期配置の piece_id 0..3 に対応
ALL_PERMUTATIONS = list(itertools.permutations(BASE_PIECES))

# --- 初期配置の駒IDと座標 ---
# 先手: (0,3)=id0, (1,3)=id1, (2,3)=id2, (1,2)=id3
P1_INITIAL_POSITIONS = {
    0: (0, 3),
    1: (1, 3),
    2: (2, 3),
    3: (1, 2)
}
# 後手: (2,0)=id0, (1,0)=id1, (0,0)=id2, (1,1)=id3
P2_INITIAL_POSITIONS = {
    0: (2, 0),
    1: (1, 0),
    2: (0, 0),
    3: (1, 1)
}

# --- 移動ベクトル（駒ごと） ---
_LION_MOVES     = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
_GIRAFFE_MOVES  = [(0, -1), (0, 1), (-1, 0), (1, 0)]
_ELEPHANT_MOVES = [(-1, -1), (1, -1), (-1, 1), (1, 1)]
_CHICK_P1       = [(0, -1)]
_CHICK_P2       = [(0, 1)]
_HEN_P1         = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (0, 1)]
_HEN_P2         = [(-1, 1), (0, 1), (1, 1), (-1, 0), (1, 0), (0, -1)]

MOVE_VECTORS = {
    PLAYER_1: {
        PIECE_LION: _LION_MOVES,
        PIECE_GIRAFFE: _GIRAFFE_MOVES,
        PIECE_ELEPHANT: _ELEPHANT_MOVES,
        PIECE_CHICK: _CHICK_P1,
        PIECE_HEN: _HEN_P1
    },
    PLAYER_2: {
        PIECE_LION: _LION_MOVES,
        PIECE_GIRAFFE: _GIRAFFE_MOVES,
        PIECE_ELEPHANT: _ELEPHANT_MOVES,
        PIECE_CHICK: _CHICK_P2,
        PIECE_HEN: _HEN_P2
    }
}

# ベクトルから移動可能な駒種を逆引きするマップ
def _build_capable_map():
    capable = {PLAYER_1: {}, PLAYER_2: {}}
    for p in [PLAYER_1, PLAYER_2]:
        for piece, vectors in MOVE_VECTORS[p].items():
            for v in vectors:
                if v not in capable[p]:
                    capable[p][v] = set()
                capable[p][v].add(piece)
    return capable

CAPABLE_PIECES = _build_capable_map()

# --- テンソルインデックス（11状態）---
# 0: 空白, 1..5: P1(l, g, e, c, h), 6..10: P2(l, g, e, c, h)
INDEX_EMPTY = 0
P1_INDICES = {
    PIECE_LION: 1,
    PIECE_GIRAFFE: 2,
    PIECE_ELEPHANT: 3,
    PIECE_CHICK: 4,
    PIECE_HEN: 5
}
P2_INDICES = {
    PIECE_LION: 6,
    PIECE_GIRAFFE: 7,
    PIECE_ELEPHANT: 8,
    PIECE_CHICK: 9,
    PIECE_HEN: 10
}
NUM_PIECE_STATES = 11

# --- 勝敗結果 ---
RESULT_NONE    = 0
RESULT_PLAYER1 = 1
RESULT_PLAYER2 = 2
RESULT_DRAW    = 3
