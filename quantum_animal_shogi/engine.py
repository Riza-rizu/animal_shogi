"""
量子どうぶつしょうぎ ゲームエンジンクラス
"""

from constants import (
    COLS, ROWS, PLAYER_1, PLAYER_2,
    PIECE_LION, PIECE_GIRAFFE, PIECE_ELEPHANT, PIECE_CHICK, PIECE_HEN,
    ALL_PERMUTATIONS, CAPABLE_PIECES, MOVE_VECTORS
)


class Move:
    def __init__(self, from_pos, to_pos, piece_token, is_drop=False):
        """
        Args:
            from_pos: (col, row) または 打つの場合は None
            to_pos: (col, row)
            piece_token: (owner, piece_id)
            is_drop: 持ち駒から打つ着手なら True
        """
        self.from_pos = from_pos
        self.to_pos = to_pos
        self.piece_token = piece_token
        self.is_drop = is_drop

    def __str__(self):
        owner, pid = self.piece_token
        # 第6期までの棋譜分析スクリプトと100%互換にするため、通常初期位置の駒記号に対応
        # P1: 0->e, 1->l, 2->g, 3->c / P2: 0->E, 1->L, 2->G, 3->C
        p1_map = {0: 'e', 1: 'l', 2: 'g', 3: 'c'}
        p2_map = {0: 'E', 1: 'L', 2: 'G', 3: 'C'}
        p_str = p1_map.get(pid, f"p1_{pid}") if owner == 1 else p2_map.get(pid, f"p2_{pid}")
        if self.is_drop:
            return f"打 {p_str} -> ({self.to_pos[0]},{self.to_pos[1]})"
        return f"動 {p_str}: ({self.from_pos[0]},{self.from_pos[1]}) -> ({self.to_pos[0]},{self.to_pos[1]})"

    def __eq__(self, other):
        if not isinstance(other, Move):
            return False
        return (self.from_pos == other.from_pos and
                self.to_pos == other.to_pos and
                self.piece_token == other.piece_token and
                self.is_drop == other.is_drop)

    def __hash__(self):
        return hash((self.from_pos, self.to_pos, self.piece_token, self.is_drop))


class GameEngine:
    @staticmethod
    def get_legal_moves(board, player):
        """現在のプレイヤーのすべての合法手を返す"""
        moves = []

        # --- 1. 盤上の駒を動かす手 ---
        for r in range(ROWS):
            for c in range(COLS):
                cell = board.get(c, r)
                if cell is not None:
                    controller, owner, pid = cell
                    if controller == player:
                        # その駒が現在取りうる駒種を取得
                        candidates = board.get_piece_candidates(owner, pid)
                        # それらの候補駒が動けるベクトルをすべて集める
                        valid_vectors = set()
                        for p_type in candidates:
                            for vec in MOVE_VECTORS[player][p_type]:
                                valid_vectors.add(vec)

                        for dc, dr in valid_vectors:
                            nc, nr = c + dc, r + dr
                            if board.in_bounds(nc, nr):
                                target = board.get(nc, nr)
                                # 移動先が空、または敵の駒なら動かせる（味方の駒には乗れない）
                                if target is None or target[0] != player:
                                    moves.append(Move((c, r), (nc, nr), (owner, pid), is_drop=False))

        # --- 2. 持ち駒を打つ手 ---
        # 持ち駒の各駒について
        for piece_token in board.hands[player]:
            owner, pid = piece_token
            candidates = board.get_piece_candidates(owner, pid)

            for r in range(ROWS):
                for c in range(COLS):
                    if board.get(c, r) is None:  # 空きマスなら打てる
                        # 案A: 敵陣最奥列に打つ場合、ひよこ以外の候補（きりん・ぞう）が残っている必要がある
                        is_furthest = (player == PLAYER_1 and r == 0) or (player == PLAYER_2 and r == ROWS - 1)
                        if is_furthest:
                            has_non_chick = any(cp in [PIECE_GIRAFFE, PIECE_ELEPHANT] for cp in candidates)
                            if not has_non_chick:
                                continue  # ひよこしか残っていない駒は最奥に打てない

                        moves.append(Move(None, (c, r), piece_token, is_drop=True))

        return moves

    @staticmethod
    def make_move(board, move, player):
        """1手進める（盤面と順列の確率分布を更新）"""
        captured_token = None
        owner, pid = move.piece_token

        if move.is_drop:
            # 持ち駒から打つ
            board.remove_from_hand(player, move.piece_token)

            # 敵陣最奥に打った場合: 案Aによりひよこの可能性を消滅させる
            is_furthest = (player == PLAYER_1 and move.to_pos[1] == 0) or (player == PLAYER_2 and move.to_pos[1] == ROWS - 1)
            if is_furthest:
                board.valid_perms[owner] = [
                    idx for idx in board.valid_perms[owner]
                    if ALL_PERMUTATIONS[idx][pid] != PIECE_CHICK
                ]

            board.set(move.to_pos[0], move.to_pos[1], (player, owner, pid))

        else:
            # 盤上の移動
            dc = move.to_pos[0] - move.from_pos[0]
            dr = move.to_pos[1] - move.from_pos[1]
            vector = (dc, dr)

            # 移動ベクトルによる順列の絞り込み（波束の収縮）
            capable = CAPABLE_PIECES[player].get(vector, set())
            is_prom = board.promoted.get((owner, pid), False)

            # 生き残り順列のフィルタリング
            new_perms = []
            for idx in board.valid_perms[owner]:
                base_piece = ALL_PERMUTATIONS[idx][pid]
                actual_piece = PIECE_HEN if (base_piece == PIECE_CHICK and is_prom) else base_piece
                if actual_piece in capable:
                    new_perms.append(idx)
            board.valid_perms[owner] = new_perms

            # 移動先に相手の駒がいれば取る
            target = board.get(move.to_pos[0], move.to_pos[1])
            if target is not None:
                _, opp_owner, opp_pid = target
                captured_token = (opp_owner, opp_pid)
                board.remove(move.to_pos[0], move.to_pos[1])
                board.add_to_hand(player, opp_owner, opp_pid)

            # 自分の駒を元の位置から外し、新しい位置へ
            board.remove(move.from_pos[0], move.from_pos[1])

            # ひよこの成り判定（敵陣最奥列に入ったら自動昇格）
            furthest_row = 0 if player == PLAYER_1 else (ROWS - 1)
            if move.to_pos[1] == furthest_row:
                board.promoted[(owner, pid)] = True

            board.set(move.to_pos[0], move.to_pos[1], (player, owner, pid))

        board.last_player = player
        return captured_token

    @staticmethod
    def can_capture_target(board, player, target_pos):
        """指定プレイヤーが次の手番で target_pos (col, row) にある駒を取れるか判定"""
        legal_moves = GameEngine.get_legal_moves(board, player)
        for m in legal_moves:
            if m.to_pos == target_pos:
                return True
        return False

    @staticmethod
    def check_winner(board, last_player=None):
        """
        勝敗を判定する。継続なら None、勝者が決まれば PLAYER_1 または PLAYER_2 を返す。
        【正規ルール・案A準拠】
        1. キャッチ判定（最優先）:
           ライオンである可能性が残っている駒が相手陣営に1つもなければ相手の負け（直前手番側の勝ち）
        2. 公式トライ判定（相手の手番が終了した時点で判定）:
           直前手番 last_player が手を指し終えた時点で、相手 (3 - last_player) のライオン候補駒が
           敵陣最奥列に存在し続けている場合、相手がその駒を取らずにターンを終了したため、
           相手 (3 - last_player) のトライ勝ち！
        """
        # 1. キャッチ判定（最優先）
        p1_has_lion = any(board.can_be_lion(PLAYER_1, pid) for pid in range(4))
        p2_has_lion = any(board.can_be_lion(PLAYER_2, pid) for pid in range(4))

        if not p1_has_lion and not p2_has_lion:
            return None
        if not p1_has_lion:
            return PLAYER_2  # 先手のライオンが消滅（後手の勝ち）
        if not p2_has_lion:
            return PLAYER_1  # 後手のライオンが消滅（先手の勝ち）

        # 2. 公式トライ判定（相手の手番が終了した時点で生き残っているか判定）
        lp = last_player if last_player is not None else getattr(board, 'last_player', None)
        if lp is not None:
            waiting_player = 3 - lp
            target_row = 0 if waiting_player == PLAYER_1 else (ROWS - 1)
            for c in range(COLS):
                cell = board.get(c, target_row)
                if cell is not None:
                    ctrl, owner, pid = cell
                    if ctrl == waiting_player and board.can_be_lion(owner, pid):
                        return waiting_player

        return None

    @staticmethod
    def check_winner_detail(board, last_player=None):
        """
        勝敗と勝因をタプル (winner, reason) で返す。
        継続なら (None, None)
        reason: 'Catch', 'Try'
        """
        p1_has_lion = any(board.can_be_lion(PLAYER_1, pid) for pid in range(4))
        p2_has_lion = any(board.can_be_lion(PLAYER_2, pid) for pid in range(4))

        if not p1_has_lion:
            return PLAYER_2, "Catch"
        if not p2_has_lion:
            return PLAYER_1, "Catch"

        lp = last_player if last_player is not None else getattr(board, 'last_player', None)
        if lp is not None:
            waiting_player = 3 - lp
            target_row = 0 if waiting_player == PLAYER_1 else (ROWS - 1)
            for c in range(COLS):
                cell = board.get(c, target_row)
                if cell is not None:
                    ctrl, owner, pid = cell
                    if ctrl == waiting_player and board.can_be_lion(owner, pid):
                        return waiting_player, "Try"

        return None, None
