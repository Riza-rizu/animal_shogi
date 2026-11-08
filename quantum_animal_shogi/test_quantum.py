"""
量子どうぶつしょうぎの動作検証テスト
"""

import sys
import os

repo_dir = os.path.dirname(os.path.abspath(__file__))
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)

from constants import (
    COLS, ROWS, PLAYER_1, PLAYER_2,
    PIECE_LION, PIECE_GIRAFFE, PIECE_ELEPHANT, PIECE_CHICK, PIECE_HEN
)
from board import QuantumBoard
from engine import GameEngine, Move

def test_initial_setup():
    print("--- テスト1: 初期配置と初期順列数 ---")
    board = QuantumBoard()
    board.setup()
    assert len(board.valid_perms[PLAYER_1]) == 24, "先手の順列は初期24通りであるべき"
    assert len(board.valid_perms[PLAYER_2]) == 24, "後手の順列は初期24通りであるべき"
    print("初期配置OK (順列数: 24)")

def test_move_and_collapse():
    print("\n--- テスト2: 移動による波束収縮（もつれ） ---")
    board = QuantumBoard()
    board.setup()

    # 先手の駒 pid=1 (col=1, row=3, 通常のライオン位置) を真後ろには引けないが、
    # もし (1, 3) から (1, 2) に前進した場合:
    # ベクトル (0, -1) は ライオン, きりん, ひよこ が可能（ぞうは不可）
    move = Move((1, 3), (1, 2), (PLAYER_1, 1), is_drop=False)
    
    # ただし (1, 2) には先手のひよこ位置 (pid=3) がいるので、合法手としては空いている場所か敵の場所
    # 初期配置で空いているのは (0, 1), (2, 1), (0, 2), (2, 2)
    # pid=0 (0, 3) を (0, 2) に前進させる手:
    move1 = Move((0, 3), (0, 2), (PLAYER_1, 0), is_drop=False)
    GameEngine.make_move(board, move1, PLAYER_1)

    # (0, -1) の動きができるのは l, g, c のみ (e は不可)
    cands = board.get_piece_candidates(PLAYER_1, 0)
    print(f"前進後の pid=0 の候補駒: {cands}")
    assert PIECE_ELEPHANT not in cands, "前進した駒からゾウの可能性が消滅するべき"
    assert len(board.valid_perms[PLAYER_1]) == 18, f"順列数は24から18に絞られるべき (実際: {len(board.valid_perms[PLAYER_1])})"
    print(f"生き残り順列数: {len(board.valid_perms[PLAYER_1])} (期待値: 18)")

def test_capture_excludes_lion():
    print("\n--- テスト3: 相手の駒を取るとライオンが除外される ---")
    board = QuantumBoard()
    board.setup()

    # 相手の後手 pid=3 (1, 1) を先手が取ったとする
    board.add_to_hand(PLAYER_1, PLAYER_2, 3)
    cands = board.get_piece_candidates(PLAYER_2, 3)
    print(f"取られた後手 pid=3 の候補駒: {cands}")
    assert PIECE_LION not in cands, "取られた駒からライオンが除外されるべき"
    assert not board.can_be_lion(PLAYER_2, 3), "can_be_lion が False になるべき"
    print("ライオン除外OK")

def test_state_vector():
    print("\n--- テスト4: 状態ベクトル化 (139次元) ---")
    board = QuantumBoard()
    board.setup()
    vec = board.to_state_vector(PLAYER_1)
    print(f"状態ベクトルの形状: {vec.shape}")
    assert vec.shape == (139,), f"139次元であるべき (実際: {vec.shape})"
    print("状態ベクトル次元OK")

def test_auto_try():
    print("\n--- テスト5: 自動トライ判定 ---")
    board = QuantumBoard()
    board.setup()

    # 先手の駒 pid=1 を敵陣最奥 (0, 0) にワープ配置（テスト用）
    board.set(0, 0, (PLAYER_1, PLAYER_1, 1))
    # 後手の駒をすべて取れる手がない位置（あるいは後手の手駒を空にする）
    # 後手の (0, 0) 周辺の駒を取り除く
    board.remove(1, 0)
    board.remove(2, 0)
    board.remove(0, 1) # 空き
    board.remove(1, 1)

    # 後手が (0, 0) を取れないかチェック
    can_capture = GameEngine.can_capture_target(board, PLAYER_2, (0, 0))
    print(f"後手が (0, 0) を取れるか: {can_capture}")
    winner = GameEngine.check_winner(board)
    print(f"勝者判定結果: {winner}")
    assert winner == PLAYER_1, "次手で取られないため先手のトライ勝ちになるべき"
    print("自動トライ判定OK")

def test_giraffe_backward_and_drop_rules():
    print("\n--- テスト6: キリンの後退と最奥打ち（案A） ---")
    board = QuantumBoard()
    board.setup()

    # 先手駒 pid=0 (0, 3) を (0, 2) に動かす（前進） -> {l, g, c}
    m1 = Move((0, 3), (0, 2), (PLAYER_1, 0), is_drop=False)
    GameEngine.make_move(board, m1, PLAYER_1)

    # (0, 2) から (0, 3) に後退させる手:
    # 後退 (0, 1) ができるのは {l, g}（キリンは後退可能、ヒヨコは不可）
    m2 = Move((0, 2), (0, 3), (PLAYER_1, 0), is_drop=False)
    GameEngine.make_move(board, m2, PLAYER_1)

    cands = board.get_piece_candidates(PLAYER_1, 0)
    print(f"前進後、後退させた pid=0 の候補駒: {cands}")
    assert PIECE_CHICK not in cands, "後退した駒からヒヨコが消滅するべき"
    assert PIECE_GIRAFFE in cands, "キリンは後退できるので候補に残るべき"
    assert PIECE_LION in cands, "ライオンは後退できるので候補に残るべき"
    print("キリン後退OK (ヒヨコ消滅、キリンとライオン残存)")

    # 案A: 持ち駒を最奥列 (row=0) に打つテスト
    # 先手に持ち駒を追加 (後手 pid=0 を取ったとする)
    board.add_to_hand(PLAYER_1, PLAYER_2, 0)
    # 最奥マス (0, 0) に打つ
    m_drop = Move(None, (0, 0), (PLAYER_2, 0), is_drop=True)
    GameEngine.make_move(board, m_drop, PLAYER_1)

    cands_dropped = board.get_piece_candidates(PLAYER_2, 0)
    print(f"最奥列に打った後の候補駒: {cands_dropped}")
    assert PIECE_CHICK not in cands_dropped, "最奥に打った駒からヒヨコが消滅するべき"
    print("最奥打ちOK (案A: ヒヨコ消滅)")


if __name__ == "__main__":
    test_initial_setup()
    test_move_and_collapse()
    test_capture_excludes_lion()
    test_state_vector()
    test_auto_try()
    test_giraffe_backward_and_drop_rules()
    print("\n全テスト成功！")

