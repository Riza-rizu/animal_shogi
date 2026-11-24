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
    print("\n--- テスト4: 状態ベクトル化 (187次元: 順列48次元含む) ---")
    board = QuantumBoard()
    board.setup()
    vec = board.to_state_vector(PLAYER_1)
    print(f"状態ベクトルの形状: {vec.shape}")
    assert vec.shape == (187,), f"187次元であるべき (実際: {vec.shape})"
    print("状態ベクトル次元OK")

def test_plan_a_official_try():
    print("\n--- テスト5: 公式トライルール（案A）とキャッチ最優先の検証 ---")
    board = QuantumBoard()
    board.setup()
    board.remove(0, 3)
    board.set(0, 1, (PLAYER_1, PLAYER_1, 0))
    m_p1 = Move((0, 1), (0, 0), (PLAYER_1, 0), is_drop=False)
    GameEngine.make_move(board, m_p1, PLAYER_1)

    assert board.promoted.get((PLAYER_1, 0)) == True, "最奥移動で promoted フラグが True になるべき"
    assert board.can_be_lion(PLAYER_1, 0), "最奥移動後も can_be_lion が True であるべき"

    # 1. 先手が最奥に侵入した瞬間は勝敗がつかず、後手に手番が渡る
    w_immediate = GameEngine.check_winner(board, last_player=PLAYER_1)
    print(f"1. 最奥到達直後の勝敗判定: {w_immediate} (期待値: None)")
    assert w_immediate is None, "最奥に入った瞬間は即時勝利せず、後手に手番を渡すべき（案A）"

    # 2. 【最重要】後手が先手のライオンをキャッチした場合、キャッチ勝ちが最優先される
    board_catch = board.copy()
    board_catch.valid_perms[PLAYER_1] = []  # 先手ライオン消滅（捕獲）
    w_catch, reason_catch = GameEngine.check_winner_detail(board_catch, last_player=PLAYER_2)
    print(f"2. 後手がキャッチした場合の判定: winner={w_catch}, reason={reason_catch} (期待値: 2, Catch)")
    assert w_catch == PLAYER_2 and reason_catch == "Catch", "相手がライオンを捕獲した場合はトライよりキャッチ勝ちが優先されるべき"

    # 3. 後手が最奥の駒を取った場合、トライは阻止され継続
    board_prevent = board.copy()
    m_take = Move((1, 0), (0, 0), (PLAYER_2, 1), is_drop=False)
    GameEngine.make_move(board_prevent, m_take, PLAYER_2)
    w_prevent = GameEngine.check_winner(board_prevent, last_player=PLAYER_2)
    print(f"3. 後手が最奥駒を取った場合の判定: {w_prevent} (期待値: None)")
    assert w_prevent is None, "最奥駒が取られたらトライ阻止でゲーム継続すべき"

    # 4. 後手が別の手を指してターン終了した場合、先手のトライ勝ちが成立
    board_success = board.copy()
    board_success.remove(1, 1)  # 空きマスにする
    m_ignore = Move((1, 0), (1, 1), (PLAYER_2, 1), is_drop=False)
    GameEngine.make_move(board_success, m_ignore, PLAYER_2)
    w_success, reason_success = GameEngine.check_winner_detail(board_success, last_player=PLAYER_2)
    print(f"4. 後手が最奥駒を取らず手番終了した場合の判定: winner={w_success}, reason={reason_success} (期待値: 1, Try)")
    assert w_success == PLAYER_1 and reason_success == "Try", "1手生き残ったため先手のトライ勝ちになるべき"
    print("公式トライ判定（案A・キャッチ最優先）の全挙動テストOK！")

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
    test_plan_a_official_try()
    test_giraffe_backward_and_drop_rules()
    print("\n全テスト成功！")

