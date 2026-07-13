"""
どうぶつしょうぎ プレイヤークラス
"""

import random

class HumanPlayer:
    def __init__(self, player_id):
        self.player_id = player_id

    def choose_move(self, board, legal_moves):
        """人間がコンソール入力で手を選ぶ"""
        print("\n選択可能な手:")
        for idx, m in enumerate(legal_moves):
            print(f"[{idx}] {m}")

        while True:
            try:
                choice = int(input(f"プレイヤー {self.player_id} の番です。番号を入力してください: "))
                if 0 <= choice < len(legal_moves):
                    return legal_moves[choice]
            except ValueError:
                pass
            print("無効な数値です。もう一度入力してください。")


class RandomPlayer:
    def __init__(self, player_id):
        self.player_id = player_id

    def choose_move(self, board, legal_moves):
        """合法手の中から完全にランダムで選ぶAI"""
        return random.choice(legal_moves)


class ModelPlayer:
    def __init__(self, player_id, model_path_or_model="animal_shogi_model.keras", ratio=4.0):
        """
        ニューラルネットワークモデルを使用したAIプレイヤー。
        
        Args:
            player_id: プレイヤーのID（PLAYER_1 または PLAYER_2）
            model_path_or_model: 学習済みモデルのパス（str）またはロード済みのモデルオブジェクト
            ratio: 最善手と次善手の確率比率 R（デフォルト: 4.0 = 8:2）
        """
        self.player_id = player_id
        self.ratio = ratio
        
        import os
        # TensorFlow の起動ログ出力を抑制
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        
        import tensorflow as tf
        import numpy as np
        self.tf = tf
        self.np = np
        
        if isinstance(model_path_or_model, str):
            try:
                self.model = tf.keras.models.load_model(model_path_or_model)
                print(f"モデル {model_path_or_model} を正常にロードしました。")
            except Exception as e:
                print(f"エラー: モデル {model_path_or_model} のロードに失敗しました: {e}")
                raise e
        else:
            self.model = model_path_or_model

    def choose_move(self, board, legal_moves):
        """ニューラルネットワークモデルを用いて手を選択する"""
        if not legal_moves:
            return None
        if len(legal_moves) == 1:
            return legal_moves[0]

        # 1. 各合法手における遷移先の状態（局面）ベクトルを取得
        next_states = []
        immediate_win_indices = set()

        from engine import GameEngine
        from board import state2vec

        for idx, move in enumerate(legal_moves):
            next_board = board.copy()
            # 仮想的に手を指す
            GameEngine.make_move(next_board, move, self.player_id)
            # 即座に勝利できるか判定
            winner = GameEngine.check_winner(next_board)
            if winner == self.player_id:
                immediate_win_indices.add(idx)
                next_states.append(None)
            else:
                # 遷移先の状態ベクトルを相手手番として取得
                opponent_id = 3 - self.player_id
                next_state_vec = state2vec(next_board, opponent_id)
                next_states.append(next_state_vec)

        # 2. モデルによる一括評価（バッチ処理）
        evals = self.np.zeros(len(legal_moves), dtype=self.np.float32)
        batch_states = []
        batch_indices = []

        for idx, state_vec in enumerate(next_states):
            if idx in immediate_win_indices:
                evals[idx] = 1.0  # 即勝ちの手は評価値 1.0 (最大)
            else:
                batch_states.append(state_vec)
                batch_indices.append(idx)

        if batch_states:
            # 状態ベクトルを一括で NumPy 配列にする (形: [N, 139])
            batch_array = self.np.array(batch_states, dtype=self.np.float32)
            # predict() は遅いため、直接 model() を呼び出す
            predictions = self.model(batch_array, training=False).numpy().flatten()
            
            # モデルの出力（相手番の評価）を自分から見た評価に反転
            for idx, pred in zip(batch_indices, predictions):
                evals[idx] = -pred

        # 3. 合法手と評価値をソートして出力 (すべて調べて示す)
        print(f"\n--- プレイヤー {self.player_id} (AI) 合法手評価リスト ---")
        sorted_indices = self.np.argsort(evals)[::-1]
        for rank, idx in enumerate(sorted_indices):
            move = legal_moves[idx]
            val = evals[idx]
            print(f"  [{rank}] {move} -> 評価値: {val:.4f}")

        # 4. 最善手と次善手の差から逆温度 beta を計算
        best_idx = sorted_indices[0]
        second_best_idx = sorted_indices[1]
        e1 = evals[best_idx]
        e2 = evals[second_best_idx]
        diff = e1 - e2

        if diff > 1e-5:
            beta = self.np.log(self.ratio) / diff
        else:
            beta = 1.0  # 評価値が同じ場合は等確率に近い選択

        # 5. ソフトマックス関数を用いて選択確率を計算
        shifted_evals = evals - e1  # オーバーフロー防止
        exp_vals = self.np.exp(beta * shifted_evals)
        probs = exp_vals / self.np.sum(exp_vals)

        # 確率一覧を表示
        print(f"  (計算結果 - 逆温度 beta: {beta:.4f}, 比率 R: {self.ratio:.1f})")
        print("  選択確率:")
        for rank, idx in enumerate(sorted_indices):
            move = legal_moves[idx]
            p = probs[idx]
            print(f"    {move}: {p*100:.2f}%")

        # 6. 確率分布に従ってランダム選択
        choice_idx = self.np.random.choice(len(legal_moves), p=probs)
        selected_move = legal_moves[choice_idx]
        print(f"  AIが選択した手: {selected_move} (確率: {probs[choice_idx]*100:.2f}%)")

        return selected_move


