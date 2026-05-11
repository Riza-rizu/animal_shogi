"""
どうぶつしょうぎ 定数定義
駒の表記: 先手=小文字, 後手=大文字
  先手: l(ライオン), g(きりん), e(ぞう), c(ひよこ), h(にわとり)
  後手: L(ライオン), G(きりん), E(ぞう), C(ひよこ), H(にわとり)
"""

# --- 盤面サイズ ---
COLS = 3
ROWS = 4

# --- プレイヤー ---
PLAYER_1 = 1   # 先手（小文字）
PLAYER_2 = 2   # 後手（大文字）

# --- 駒の定義（1文字） ---
# 先手（小文字）
P1_LION     = 'l'
P1_GIRAFFE  = 'g'
P1_ELEPHANT = 'e'
P1_CHICK    = 'c'
P1_HEN      = 'h'
# 後手（大文字）
P2_LION     = 'L'
P2_GIRAFFE  = 'G'
P2_ELEPHANT = 'E'
P2_CHICK    = 'C'
P2_HEN      = 'H'

# --- プレイヤー判定 ---
P1_PIECES = frozenset({P1_LION, P1_GIRAFFE, P1_ELEPHANT, P1_CHICK, P1_HEN})
P2_PIECES = frozenset({P2_LION, P2_GIRAFFE, P2_ELEPHANT, P2_CHICK, P2_HEN})

def get_owner(piece):
    """駒の所有者を返す"""
    if piece in P1_PIECES: return PLAYER_1
    if piece in P2_PIECES: return PLAYER_2
    return None

# --- テンソルインデックス（11状態）---
PIECE_TO_INDEX = {
    None:        0,   # 空白
    P1_LION:     1,
    P1_GIRAFFE:  2,
    P1_ELEPHANT: 3,
    P1_CHICK:    4,
    P1_HEN:      5,
    P2_LION:     6,
    P2_GIRAFFE:  7,
    P2_ELEPHANT: 8,
    P2_CHICK:    9,
    P2_HEN:     10,
}
NUM_PIECE_STATES = 11

# --- 表示名 ---
PIECE_DISPLAY = {
    P1_LION: ' ラ', P1_GIRAFFE: ' キ', P1_ELEPHANT: ' ゾ',
    P1_CHICK: ' ヒ', P1_HEN: ' ニ',
    P2_LION: 'vラ', P2_GIRAFFE: 'vキ', P2_ELEPHANT: 'vゾ',
    P2_CHICK: 'vヒ', P2_HEN: 'vニ',
}

# --- 移動ベクトル（駒ごとに定義、P2は上下反転済み）---
_LION_MOVES     = [(-1,-1),(0,-1),(1,-1),(-1,0),(1,0),(-1,1),(0,1),(1,1)]
_GIRAFFE_MOVES  = [(0,-1),(0,1),(-1,0),(1,0)]
_ELEPHANT_MOVES = [(-1,-1),(1,-1),(-1,1),(1,1)]
_CHICK_P1       = [(0,-1)]
_CHICK_P2       = [(0, 1)]
_HEN_P1         = [(-1,-1),(0,-1),(1,-1),(-1,0),(1,0),(0,1)]
_HEN_P2         = [(-1,1),(0,1),(1,1),(-1,0),(1,0),(0,-1)]

MOVE_VECTORS = {
    P1_LION: _LION_MOVES,     P2_LION: _LION_MOVES,
    P1_GIRAFFE: _GIRAFFE_MOVES, P2_GIRAFFE: _GIRAFFE_MOVES,
    P1_ELEPHANT: _ELEPHANT_MOVES, P2_ELEPHANT: _ELEPHANT_MOVES,
    P1_CHICK: _CHICK_P1,     P2_CHICK: _CHICK_P2,
    P1_HEN: _HEN_P1,         P2_HEN: _HEN_P2,
}

# --- 成り ---
PROMOTE_MAP = {P1_CHICK: P1_HEN, P2_CHICK: P2_HEN}
CAN_PROMOTE = frozenset(PROMOTE_MAP.keys())

# --- 捕獲時の変換（成り駒は戻して所有者変更）---
CAPTURE_MAP = {
    P1_LION: P2_LION, P1_GIRAFFE: P2_GIRAFFE, P1_ELEPHANT: P2_ELEPHANT,
    P1_CHICK: P2_CHICK, P1_HEN: P2_CHICK,
    P2_LION: P1_LION, P2_GIRAFFE: P1_GIRAFFE, P2_ELEPHANT: P1_ELEPHANT,
    P2_CHICK: P1_CHICK, P2_HEN: P1_CHICK,
}

# --- 持ち駒テンソル用（ライオンは取ったら即勝ちなので除外）---
HAND_TYPES_P1 = [P1_GIRAFFE, P1_ELEPHANT, P1_CHICK]
HAND_TYPES_P2 = [P2_GIRAFFE, P2_ELEPHANT, P2_CHICK]

# --- 勝敗結果 ---
RESULT_NONE    = 0
RESULT_PLAYER1 = 1
RESULT_PLAYER2 = 2
RESULT_DRAW    = 3
