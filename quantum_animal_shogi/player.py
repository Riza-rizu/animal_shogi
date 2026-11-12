"""
量子どうぶつしょうぎ AI プレイヤーモジュール
(深さ0直接推論 / 深さ2 Min-Max探索 + レピティションペナルティ)
"""

import os
import sys
import warnings

# 不要な TensorFlow ログを抑制
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np

repo_dir = os.path.dirname(os.path.abspath(__file__))
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)

from constants import (
    PLAYER_1, PLAYER_2, RESULT_PLAYER1, RESULT_PLAYER2, RESULT_DRAW
)
from engine import GameEngine
from board import state2vec


class BasePlayer:
    def __init__(self, player_id):
        self.player_id = player_id

    def choose_move(self, board, legal_moves, *args, **kwargs):
        raise NotImplementedError


class HumanPlayer(BasePlayer):
    def choose_move(self, board, legal_moves, *args, **kwargs):
        print(f"\nプレイヤー {self.player_id} の手番です。")
        for i, move in enumerate(legal_moves):
            print(f"{i}: {move}")
        while True:
            try:
                choice = int(input("手を選択してください (番号): "))
                if 0 <= choice < len(legal_moves):
                    return legal_moves[choice]
            except ValueError:
                pass
            print("無効な入力です。")


class RandomPlayer(BasePlayer):
    def choose_move(self, board, legal_moves, *args, **kwargs):
        if not legal_moves:
            return None
        return legal_moves[np.random.choice(len(legal_moves))]


def load_keras_model_safe(model_path):
    """Kerasモデルを安全にロードする"""
    import tensorflow as tf
    import zipfile
    import json
    import tempfile

    try:
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        err_msg = str(e)
        if ("quantization_config" in err_msg or 
            "Unrecognized keyword arguments" in err_msg or 
            "unexpected keyword argument" in err_msg or 
            "input_axes" in err_msg):
            cleaned_path = None
            try:
                fd, cleaned_path = tempfile.mkstemp(suffix=".keras")
                os.close(fd)
                with zipfile.ZipFile(model_path, 'r') as zin:
                    with zipfile.ZipFile(cleaned_path, 'w') as zout:
                        for item in zin.infolist():
                            data = zin.read(item.filename)
                            if item.filename == "config.json":
                                config_dict = json.loads(data.decode('utf-8'))
                                def remove_keys(d):
                                    if isinstance(d, dict):
                                        d.pop('quantization_config', None)
                                        d.pop('input_axes', None)
                                        d.pop('output_axes', None)
                                        for k, v in list(d.items()):
                                            remove_keys(v)
                                    elif isinstance(d, list):
                                        for x in d:
                                            remove_keys(x)
                                remove_keys(config_dict)
                                data = json.dumps(config_dict).encode('utf-8')
                            zout.writestr(item, data)
                model = tf.keras.models.load_model(cleaned_path)
                os.remove(cleaned_path)
                return model
            except Exception as clean_err:
                if cleaned_path and os.path.exists(cleaned_path):
                    os.remove(cleaned_path)
                raise e
        else:
            raise e


class ModelPlayer(BasePlayer):
    def __init__(self, player_id, model_path_or_model="quantum_model.keras", beta=10.0, depth=0):
        """
        AIプレイヤー。
        
        Args:
            player_id: PLAYER_1 または PLAYER_2
            model_path_or_model: モデルパスまたはKerasモデルオブジェクト
            beta: 逆温度パラメータ（手選択のソフトマックス分布用）
            depth: 探索深さ（0: 直接推論, 2: 深さ2 Min-Max探索）
        """
        super().__init__(player_id)
        self.beta = beta
        self.depth = depth

        import tensorflow as tf
        self.tf = tf

        if isinstance(model_path_or_model, str):
            self.model = load_keras_model_safe(model_path_or_model)
        else:
            self.model = model_path_or_model

        # 出力層が tanh か sigmoid か判定
        self.is_tanh = False
        try:
            last_layer = self.model.layers[-1]
            act = getattr(last_layer, 'activation', None)
            act_name = getattr(act, '__name__', '')
            if 'tanh' in act_name.lower():
                self.is_tanh = True
        except Exception:
            pass

        if self.is_tanh:
            self.win_val = 1.0
            self.loss_val = -1.0
            self.neutral_val = 0.0
        else:
            self.win_val = 1.0
            self.loss_val = 0.0
            self.neutral_val = 0.5

    def _apply_repetition_penalty(self, board, legal_moves, evals, history_counts, gamma):
        """
        幾何減衰レピティションペナルティの適用:
        V_penalized = loss_val + (V0 - loss_val) * (gamma ** k)
        """
        if not history_counts or gamma is None:
            return evals

        opponent_id = 3 - self.player_id
        for m_idx, move in enumerate(legal_moves):
            # 即勝ち・即死手はペナルティ不要
            if evals[m_idx] >= self.win_val - 1e-5:
                continue
            if evals[m_idx] <= self.loss_val + 1e-5:
                continue

            next_b = board.copy()
            GameEngine.make_move(next_b, move, self.player_id)
            next_state_key = (next_b.to_hashable(), opponent_id)
            k = history_counts.get(next_state_key, 0)
            if k > 0:
                v0 = evals[m_idx]
                if self.is_tanh:
                    evals[m_idx] = -1.0 + (v0 + 1.0) * (gamma ** k)
                else:
                    evals[m_idx] = 0.0 + (v0 - 0.0) * (gamma ** k)
        return evals

    def choose_move(self, board, legal_moves, history_counts=None, gamma=0.6):
        if not legal_moves:
            return None

        opponent_id = 3 - self.player_id
        num_moves = len(legal_moves)
        move_evals = np.zeros(num_moves, dtype=np.float32)

        # 1手進めた盤面をシミュレーション
        next_boards = []
        immediate_win_indices = []

        for idx, move in enumerate(legal_moves):
            nb = board.copy()
            GameEngine.make_move(nb, move, self.player_id)
            winner = GameEngine.check_winner(nb)

            if winner == self.player_id:
                immediate_win_indices.append(idx)
                move_evals[idx] = self.win_val
            elif winner == opponent_id:
                move_evals[idx] = self.loss_val
            else:
                next_boards.append((idx, nb))

        # 即座に勝てる手があればその中から選択
        if immediate_win_indices:
            choice_idx = np.random.choice(immediate_win_indices)
            return legal_moves[choice_idx]

        # 評価値計算
        if next_boards:
            if self.depth == 0:
                # 深さ0: 相手視点の次盤面をバッチ推論
                vectors = [state2vec(nb, opponent_id) for _, nb in next_boards]
                batch_x = np.array(vectors, dtype=np.float32)
                # predict() はオーバーヘッドが極めて大きいため直接モデルを呼び出す
                preds = self.model(batch_x, training=False).numpy().flatten()

                for (orig_idx, _), pred in zip(next_boards, preds):
                    # 予測値は相手手番目線なので符号反転（または 1 - p）
                    if self.is_tanh:
                        move_evals[orig_idx] = -pred
                    else:
                        move_evals[orig_idx] = 1.0 - pred

            elif self.depth == 2:
                # 深さ2: 相手の各応答手のMinを探索
                for orig_idx, nb in next_boards:
                    opp_moves = GameEngine.get_legal_moves(nb, opponent_id)
                    if not opp_moves:
                        move_evals[orig_idx] = self.win_val
                        continue

                    opp_next_boards = []
                    opp_has_win = False
                    for om in opp_moves:
                        onb = nb.copy()
                        GameEngine.make_move(onb, om, opponent_id)
                        w = GameEngine.check_winner(onb)
                        if w == opponent_id:
                            opp_has_win = True
                            break
                        opp_next_boards.append(onb)

                    if opp_has_win:
                        move_evals[orig_idx] = self.loss_val
                        continue

                    # 相手の合法手のうち、自分にとって最悪の手（相手にとって最善）
                    if opp_next_boards:
                        opp_vecs = [state2vec(onb, self.player_id) for onb in opp_next_boards]
                        batch_opp = np.array(opp_vecs, dtype=np.float32)
                        opp_preds = self.model(batch_opp, training=False).numpy().flatten()
                        # opp_preds は自分手番目線
                        move_evals[orig_idx] = np.min(opp_preds)
                    else:
                        move_evals[orig_idx] = self.neutral_val

        # レピティションペナルティの適用
        move_evals = self._apply_repetition_penalty(board, legal_moves, move_evals, history_counts, gamma)

        # ソフトマックスによる着手選択
        if self.beta <= 0:
            probs = np.ones(num_moves) / num_moves
        else:
            scaled = move_evals * self.beta
            scaled -= np.max(scaled)
            exp_vals = np.exp(scaled)
            probs = exp_vals / np.sum(exp_vals)

        chosen_idx = np.random.choice(num_moves, p=probs)
        return legal_moves[chosen_idx]
