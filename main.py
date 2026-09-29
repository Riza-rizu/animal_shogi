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
from player import HumanPlayer, RandomPlayer, ModelPlayer, load_keras_model_safe
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
    beta = config.get('beta', 10.0)
    
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
            model = load_keras_model_safe(model_path)
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
            player1 = ModelPlayer(PLAYER_1, model_path_or_model=model, beta=beta)
        else:
            player1 = RandomPlayer(PLAYER_1)
            
        if p2_type == 'ai':
            player2 = ModelPlayer(PLAYER_2, model_path_or_model=model, beta=beta)
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
    parser = argparse.ArgumentParser(description="どうぶつしょうぎ 実行・評価スクリプト")
    parser.add_argument("--p1", type=str, choices=["human", "random", "ai"], default="human",
                        help="先手のプレイヤー種別 (human, random, ai)")
    parser.add_argument("--p2", type=str, choices=["human", "random", "ai"], default="random",
                        help="後手のプレイヤー種別 (human, random, ai)")
    parser.add_argument("--beta", type=float, default=10.0,
                        help="AIプレイヤーの逆温度 beta (デフォルト: 10.0)")
    parser.add_argument("--model", type=str, default="animal_shogi_model.keras",
                        help="使用するモデルのパス (デフォルト: animal_shogi_model.keras)")
    parser.add_argument("--model1", type=str, default=None,
                        help="先手(PLAYER_1)用モデルのパス。指定しない場合は --model を使用")
    parser.add_argument("--model2", type=str, default=None,
                        help="後手(PLAYER_2)用モデルのパス。指定しない場合は --model を使用")
    parser.add_argument("--games", type=int, default=1,
                        help="対戦を行う回数。2回以上の場合はコンソール出力を抑制して勝率を自動集計します (デフォルト: 1)")
    
    args = parser.parse_args()
    
    # モデル1 & モデル2 のパス決定
    m1_path = args.model1 if args.model1 else args.model
    m2_path = args.model2 if args.model2 else args.model
    
    # AIモデルの事前ロード（重複ロードを避けるため）
    import tensorflow as tf
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    model1 = None
    if args.p1 == 'ai':
        if os.path.exists(m1_path):
            print(f"先手(1)用にモデル {m1_path} をロードしています...")
            model1 = load_keras_model_safe(m1_path)
        else:
            print(f"エラー: 指定されたモデルファイルが見つかりません: {m1_path}")
            sys.exit(1)
            
    model2 = None
    if args.p2 == 'ai':
        if os.path.exists(m2_path):
            print(f"後手(2)用にモデル {m2_path} をロードしています...")
            model2 = load_keras_model_safe(m2_path)
        else:
            print(f"エラー: 指定されたモデルファイルが見つかりません: {m2_path}")
            sys.exit(1)

    # プレイヤーの初期化
    from player import HumanPlayer, RandomPlayer, ModelPlayer
    
    if args.p1 == 'human':
        player1 = HumanPlayer(PLAYER_1)
    elif args.p1 == 'random':
        player1 = RandomPlayer(PLAYER_1)
    elif args.p1 == 'ai':
        player1 = ModelPlayer(PLAYER_1, model_path_or_model=model1, beta=args.beta)
        
    if args.p2 == 'human':
        player2 = HumanPlayer(PLAYER_2)
    elif args.p2 == 'random':
        player2 = RandomPlayer(PLAYER_2)
    elif args.p2 == 'ai':
        player2 = ModelPlayer(PLAYER_2, model_path_or_model=model2, beta=args.beta)
        
    players = {PLAYER_1: player1, PLAYER_2: player2}

    print("\n" + "="*40)
    print("どうぶつしょうぎ 対戦を開始します！")
    print(f"先手(1): {args.p1} " + (f"({m1_path})" if args.p1 == 'ai' else ""))
    print(f"後手(2): {args.p2} " + (f"({m2_path})" if args.p2 == 'ai' else ""))
    print(f"総対戦数: {args.games} 回")
    print("="*40)
    
    p1_wins = 0
    p2_wins = 0
    draws = 0
    
    # 複数対戦の場合は詳細ログを抑制する
    verbose = (args.games == 1)
    
    for g in range(args.games):
        if verbose:
            print(f"\n--- 第 {g+1} 局 ---")
        board = Board()
        board.setup()
        current_player_id = PLAYER_1
        
        # 千日手
        max_moves = 200
        winner = None
        
        for move_count in range(max_moves):
            if verbose:
                print_board(board)
            
            # 1. 現在のプレイヤーの合法手を取得
            legal_moves = GameEngine.get_legal_moves(board, current_player_id)
            
            if not legal_moves:
                winner = PLAYER_2 if current_player_id == PLAYER_1 else PLAYER_1
                if verbose:
                    print(f"プレイヤー {current_player_id} に打てる手がありません。")
                break
                
            current_player = players[current_player_id]
            
            if verbose:
                selected_move = current_player.choose_move(board, legal_moves)
                print(f"\n選択された手: {selected_move}")
            else:
                # バッチ実行時は評価値表示などの標準出力を一時的に捨てる
                import io
                import sys as sys_module
                original_stdout = sys_module.stdout
                sys_module.stdout = io.StringIO()
                try:
                    selected_move = current_player.choose_move(board, legal_moves)
                finally:
                    sys_module.stdout = original_stdout
            
            GameEngine.make_move(board, selected_move, current_player_id)
            
            winner = GameEngine.check_winner(board)
            if winner:
                break
                
            current_player_id = PLAYER_2 if current_player_id == PLAYER_1 else PLAYER_1
        else:
            winner = RESULT_DRAW
            
        if winner == PLAYER_1:
            p1_wins += 1
            if verbose:
                print_board(board)
                print("【勝者】 プレイヤー 1 の勝ちです！")
        elif winner == PLAYER_2:
            p2_wins += 1
            if verbose:
                print_board(board)
                print("【勝者】 プレイヤー 2 の勝ちです！")
        else:
            draws += 1
            if verbose:
                print_board(board)
                print("【結果】 引き分け（千日手）です。")
                
        # 複数ゲーム対戦時の進捗表示
        if not verbose and ((g + 1) % 50 == 0 or (g + 1) == args.games):
            print(f"進捗: {g + 1} / {args.games} 局完了...")

    # 最終的な勝率レポート
    print("\n" + "="*40)
    print(" 対戦結果サマリー ")
    print("="*40)
    print(f"総対戦数: {args.games} 局")
    print(f"先手 (1) 勝利: {p1_wins} 局 ({p1_wins / args.games * 100:.1f}%)")
    print(f"後手 (2) 勝利: {p2_wins} 局 ({p2_wins / args.games * 100:.1f}%)")
    print(f"引き分け (千日手): {draws} 局 ({draws / args.games * 100:.1f}%)")
    print("="*40)


if __name__ == "__main__":
    main()

