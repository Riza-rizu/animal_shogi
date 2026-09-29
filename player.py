"""
AI プレイヤーモジュール (深さ2のMin-Max探索 + CNN勝率モデル)
"""

import os
import sys
import warnings

# 不要な TensorFlow / Keras 警告ログを完全に抑制
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np

# プロジェクトルートのパス解決
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from constants import (
    PLAYER_1, PLAYER_2, RESULT_PLAYER1, RESULT_PLAYER2, RESULT_DRAW
)
from engine import GameEngine
from board import state2vec

class BasePlayer:
    def __init__(self, player_id):
        self.player_id = player_id

    def choose_move(self, board, legal_moves):
        raise NotImplementedError


class HumanPlayer(BasePlayer):
    def choose_move(self, board, legal_moves):
        print(f"プレイヤー {self.player_id} の手番です。")
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
    def choose_move(self, board, legal_moves):
        if not legal_moves:
            return None
        return legal_moves[np.random.choice(len(legal_moves))]


def load_keras_model_safe(model_path):
    """
    Keras モデルを安全にロードするヘルパー関数。
    """
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
            print(f"警告: モデル {model_path} の互換性エラーを検出しました（Kerasバージョン差）。自動修復してロードを試みます...")
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
                print(f"修復失敗: {clean_err}")
                if cleaned_path and os.path.exists(cleaned_path):
                    os.remove(cleaned_path)
                raise e
        else:
            raise e


class ModelPlayer:
    def __init__(self, player_id, model_path_or_model="animal_shogi_model.keras", beta=10.0):
        """
        深さ2のMin-Max探索とニューラルネットワークモデルを併用するAIプレイヤー。
        
        Args:
            player_id: プレイヤーのID（PLAYER_1 または PLAYER_2）
            model_path_or_model: 学習済みモデルのパス（str）またはロード済みのモデルオブジェクト
            beta: 逆温度パラメータ（デフォルト: 10.0）
        """
        self.player_id = player_id
        self.beta = beta
        
        # TensorFlow の起動ログ出力を抑制
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        
        import tensorflow as tf
        self.tf = tf
        self.np = np
        
        if isinstance(model_path_or_model, str):
            try:
                self.model = load_keras_model_safe(model_path_or_model)
            except Exception as e:
                print(f"エラー: モデル {model_path_or_model} のロードに失敗しました: {e}")
                raise e
        else:
            self.model = model_path_or_model

    def choose_move(self, board, legal_moves):
        """
        深さ2のMin-Max探索により、相手の反撃を織り込んだ勝率評価値で手を選択する
        """
        if not legal_moves:
            return None
        if len(legal_moves) == 1:
            return legal_moves[0]

        opponent_id = 3 - self.player_id
        evals = self.np.zeros(len(legal_moves), dtype=self.np.float32)

        # 探索用のデータコンテナ
        eval_determined = {}
        move_opp_states = {}
        all_batch_states = []
        batch_mapping = []  # 各バッチ要素がどの候補手 (m_idx) に対応するか

        for m_idx, move in enumerate(legal_moves):
            next_board = board.copy()
            GameEngine.make_move(next_board, move, self.player_id)
            
            # 1. 自分が即座に勝利できる手か？
            winner = GameEngine.check_winner(next_board)
            if winner == self.player_id:
                evals[m_idx] = 1.0  # 即勝ち (勝率 100%)
                eval_determined[m_idx] = 1.0
                continue

            # 2. 相手の手番を展開
            opp_legal_moves = GameEngine.get_legal_moves(next_board, opponent_id)
            if not opp_legal_moves:
                evals[m_idx] = 1.0  # 相手の手がない＝自分の勝ち
                eval_determined[m_idx] = 1.0
                continue

            # 相手の合法手の中に「相手の即勝ち（自ライオン捕獲など）」があるか？
            has_immediate_loss = False
            opp_next_states = []

            for opp_move in opp_legal_moves:
                opp_board = next_board.copy()
                GameEngine.make_move(opp_board, opp_move, opponent_id)
                opp_winner = GameEngine.check_winner(opp_board)
                if opp_winner == opponent_id:
                    # 相手が即勝ちできる手が存在する -> この候補手は絶対敗北 (0.0)
                    has_immediate_loss = True
                    break
                else:
                    # 相手の手の後の盤面（自分の手番としての盤面）
                    state_vec = state2vec(opp_board, self.player_id)
                    opp_next_states.append(state_vec)

            if has_immediate_loss:
                evals[m_idx] = 0.0  # 自滅手・相手の即勝ち (勝率 0%)
                eval_determined[m_idx] = 0.0
            else:
                move_opp_states[m_idx] = len(opp_next_states)
                for s_vec in opp_next_states:
                    batch_mapping.append(m_idx)
                    all_batch_states.append(s_vec)

        # 3. 未確定の局面を一括バッチ推論
        if all_batch_states:
            batch_array = self.np.array(all_batch_states, dtype=self.np.float32)
            # predict() はオーバーヘッドが大きいため直接モデルを呼び出す
            preds = self.model(batch_array, training=False).numpy().flatten()
            
            # 各候補手ごとに、相手の最善応手（自分にとって勝率が最小 = Min）を集計
            move_preds = {m_idx: [] for m_idx in move_opp_states}
            for m_idx, pred in zip(batch_mapping, preds):
                move_preds[m_idx].append(pred)

            for m_idx, pred_list in move_preds.items():
                if pred_list:
                    # 相手は自分にとって最も厳しい手を選ぶ (Min)
                    evals[m_idx] = float(min(pred_list))
                else:
                    evals[m_idx] = 0.5

        # 4. ソフトマックス関数を用いて選択確率を計算
        beta = self.beta
        best_idx = int(self.np.argmax(evals))

        # オーバーフロー防止のためのシフト
        shifted_evals = evals - evals[best_idx]
        exp_vals = self.np.exp(beta * shifted_evals)
        probs = exp_vals / self.np.sum(exp_vals)

        # 5. 確率分布に従って手を選択
        choice_idx = self.np.random.choice(len(legal_moves), p=probs)
        selected_move = legal_moves[choice_idx]

        return selected_move
