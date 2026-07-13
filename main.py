"""
どうぶつしょうぎ メイン実行ファイル
"""

import sys
import pickle
import os
import yaml
import numpy as np
from board import Board, state2vec
from engine import GameEngine
from player import HumanPlayer, RandomPlayer, ModelPlayer
from constants import PLAYER_1, PLAYER_2, PIECE_DISPLAY, COLS, ROWS, RESULT_DRAW

def print_board(board):
    """盤面をテキスト描画する関数"""
    print("\n--- 盤面 ---")
    print("  0   1   2  (col)")
    for r in range(ROWS):
        row_str = f"{r} "
        for c in range(COLS):
            piece = board.get(c, r)
            if piece:
                row_str += f"[{PIECE_DISPLAY[piece]}]"
            else:
                row_str += "[    ]"
        print(row_str)
    
    print(f"先手(1)持ち駒: {board.hands[PLAYER_1]}")
    print(f"後手(2)持ち駒: {board.hands[PLAYER_2]}")
    print("------------")

def run_simulation(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    seed = config.get('seed', 0)
    games = config.get('games', 1000)
    alpha = config.get('alpha', 0.95)  # 割引率（デフォルト値: 0.90）
    
    # プレイヤーのタイプと設定 (デフォルトは互換性のためにrandom)
    p1_type = config.get('p1', 'random')
    p2_type = config.get('p2', 'random')
    model_path = config.get('model_path', 'animal_shogi_model.keras')
    ratio = config.get('ratio', 4.0)
    
    # 乱数シードの設定
    import random
    random.seed(seed)
    np.random.seed(seed)
    
    # AIモデルの事前ロード（重複ロードを避けるため）
    model = None
    if p1_type == 'ai' or p2_type == 'ai':
        import tensorflow as tf
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        if os.path.exists(model_path):
            print(f"シミュレーション用にモデル {model_path} をロードしています...")
            model = tf.keras.models.load_model(model_path)
        else:
            print(f"エラー: 指定されたモデルファイルが見つかりません: {model_path}")
            sys.exit(1)
            
    # 出力ファイル名
    output_filename = f"output{seed:03d}.pkl"
    
    # 既存データのロード
    data_dict = {}
    if os.path.exists(output_filename):
        try:
            with open(output_filename, 'rb') as f:
                data_dict = pickle.load(f)
            print(f"既存の {output_filename} をロードしました。現在の状態数: {len(data_dict)}")
        except Exception as e:
            print(f"警告: 既存の {output_filename} の読み込みに失敗しました。新規作成します。エラー: {e}")
            
    print(f"シード値: {seed}, 試合数: {games}, 割引率(alpha): {alpha} でシミュレーションを開始します。")
    print(f"プレイヤー1: {p1_type}, プレイヤー2: {p2_type}")
    print(f"出力ファイル: {output_filename}")
    
    for g in range(games):
        board = Board()
        board.setup()
        
        # プレイヤーの作成
        if p1_type == 'ai':
            player1 = ModelPlayer(PLAYER_1, model_path_or_model=model, ratio=ratio)
        else:
            player1 = RandomPlayer(PLAYER_1)
            
        if p2_type == 'ai':
            player2 = ModelPlayer(PLAYER_2, model_path_or_model=model, ratio=ratio)
        else:
            player2 = RandomPlayer(PLAYER_2)
            
        players = {PLAYER_1: player1, PLAYER_2: player2}
        current_player_id = PLAYER_1
        
        # 1試合の履歴を記録するリスト: (state_vector, player_id)
        history = []
        
        # 千日手
        max_moves = 200
        winner = None
        
        for move_count in range(max_moves):
            # 現時点の状態ベクトルを記録
            state_vec = state2vec(board, current_player_id)
            history.append((state_vec, current_player_id))
            
            # 合法手の取得
            legal_moves = GameEngine.get_legal_moves(board, current_player_id)
            if not legal_moves:
                winner = PLAYER_2 if current_player_id == PLAYER_1 else PLAYER_1
                break
                
            selected_move = players[current_player_id].choose_move(board, legal_moves)
            GameEngine.make_move(board, selected_move, current_player_id)
            
            # 勝敗判定
            winner = GameEngine.check_winner(board)
            if winner:
                break
                
            current_player_id = PLAYER_2 if current_player_id == PLAYER_1 else PLAYER_1
        else:
            # 200手を超えたら引き分け
            winner = RESULT_DRAW
            
        # 評価値の計算とデータ蓄積（割引率 alpha を適用）
        # 終局に近い手ほど勝敗への影響が大きく、初期に近い手ほど小さくなるように割引く
        num_moves = len(history)
        for i, (state_vec, player_id) in enumerate(history):
            state_key = tuple(state_vec.tolist())
            
            # 終局（最後の局面）からの残り手数（距離）
            steps_from_end = num_moves - 1 - i
            discount = alpha ** steps_from_end
            
            if winner == RESULT_DRAW:
                val = 0.0
            elif winner == player_id:
                val = 1.0 * discount
            else:
                val = -1.0 * discount
                
            if state_key not in data_dict:
                data_dict[state_key] = []
            data_dict[state_key].append(val)
            
        # 1000試合ごとに進捗状況をコンソールに表示（ファイル保存はしない）
        if (g + 1) % 1000 == 0 or (g + 1) == games:
            print(f"進捗: {g + 1} / {games} 試合完了 (現在のユニーク状態数: {len(data_dict)})")
            
    # ループ完了後に、最後にまとめて1回だけファイルに保存する
    print(f"データをファイルに書き込んでいます: {output_filename} ...")
    with open(output_filename, 'wb') as f:
        pickle.dump(data_dict, f)
    print("シミュレーションおよびファイルの保存が完了しました。")

def main():
    import argparse
    import random
    
    # 互換性維持：引数が1つで YAML ファイルの場合はシミュレーションを実行
    if len(sys.argv) == 2 and (sys.argv[1].endswith('.yaml') or sys.argv[1].endswith('.yml')):
        yaml_path = sys.argv[1]
        if os.path.exists(yaml_path):
            run_simulation(yaml_path)
            return
        else:
            print(f"エラー: 指定されたファイルが見つかりません: {yaml_path}")
            sys.exit(1)
            
    # それ以外の場合は通常のコマンドライン引数解析
    parser = argparse.ArgumentParser(description="どうぶつしょうぎ 実行スクリプト")
    parser.add_argument("--p1", type=str, choices=["human", "random", "ai"], default="human",
                        help="先手のプレイヤー種別 (human, random, ai)")
    parser.add_argument("--p2", type=str, choices=["human", "random", "ai"], default="random",
                        help="後手のプレイヤー種別 (human, random, ai)")
    parser.add_argument("--ratio", type=float, default=4.0,
                        help="AIプレイヤーの最善手と次善手の確率比率 R (デフォルト: 4.0 = 8:2)")
    parser.add_argument("--model", type=str, default="animal_shogi_model.keras",
                        help="使用するモデルのパス (デフォルト: animal_shogi_model.keras)")
    
    args = parser.parse_args()
    
    # AIモデルの事前ロード（重複ロードを避けるため）
    model = None
    if args.p1 == 'ai' or args.p2 == 'ai':
        import tensorflow as tf
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        if os.path.exists(args.model):
            print(f"対戦用にモデル {args.model} をロードしています...")
            model = tf.keras.models.load_model(args.model)
        else:
            print(f"エラー: 指定されたモデルファイルが見つかりません: {args.model}")
            sys.exit(1)

    # 盤面の初期化
    board = Board()
    board.setup()
    
    # プレイヤー1（先手）の初期化
    if args.p1 == 'human':
        player1 = HumanPlayer(PLAYER_1)
    elif args.p1 == 'random':
        player1 = RandomPlayer(PLAYER_1)
    elif args.p1 == 'ai':
        player1 = ModelPlayer(PLAYER_1, model_path_or_model=model, ratio=args.ratio)
        
    # プレイヤー2（後手）の初期化
    if args.p2 == 'human':
        player2 = HumanPlayer(PLAYER_2)
    elif args.p2 == 'random':
        player2 = RandomPlayer(PLAYER_2)
    elif args.p2 == 'ai':
        player2 = ModelPlayer(PLAYER_2, model_path_or_model=model, ratio=args.ratio)
        
    players = {PLAYER_1: player1, PLAYER_2: player2}
    current_player_id = PLAYER_1

    print("\n" + "="*40)
    print("どうぶつしょうぎ 対戦を開始します！")
    print(f"先手(1): {args.p1}  vs  後手(2): {args.p2}")
    if model:
        print(f"使用モデル: {args.model} (確率比 R: {args.ratio})")
    print("="*40)
    
    # ゲームループ
    while True:
        print_board(board)
        
        # 1. 現在のプレイヤーの合法手を取得
        legal_moves = GameEngine.get_legal_moves(board, current_player_id)
        
        if not legal_moves:
            winner = PLAYER_2 if current_player_id == PLAYER_1 else PLAYER_1
            print(f"プレイヤー {current_player_id} に打てる手がありません。")
            print(f"【勝者】 プレイヤー {winner} の勝ちです！")
            break
            
        current_player = players[current_player_id]
        selected_move = current_player.choose_move(board, legal_moves)
        print(f"\n選択された手: {selected_move}")
        
        GameEngine.make_move(board, selected_move, current_player_id)
        
        winner = GameEngine.check_winner(board)
        if winner:
            print_board(board)
            print(f"【勝者】 プレイヤー {winner} の勝ちです！ゲーム終了。")
            break
            
        current_player_id = PLAYER_2 if current_player_id == PLAYER_1 else PLAYER_1

if __name__ == "__main__":
    main()

