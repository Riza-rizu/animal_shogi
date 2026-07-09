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
from player import HumanPlayer, RandomPlayer
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
    alpha = config.get('alpha', 0.90)  # 割引率（デフォルト値: 0.90）
    
    # 乱数シードの設定
    import random
    random.seed(seed)
    np.random.seed(seed)
    
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
    print(f"出力ファイル: {output_filename}")
    
    for g in range(games):
        board = Board()
        board.setup()
        
        player1 = RandomPlayer(PLAYER_1)
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
            
        if (g + 1) % 100 == 0 or (g + 1) == games:
            with open(output_filename, 'wb') as f:
                pickle.dump(data_dict, f)
            print(f"進捗: {g + 1} / {games} 試合完了 (現在のユニーク状態数: {len(data_dict)})")
            
    print("シミュレーションが完了しました。")


def main():
    import random  # シミュレーション用と対戦用で利用するため
    
    # 引数チェック
    if len(sys.argv) > 1:
        yaml_path = sys.argv[1]
        if yaml_path.endswith('.yaml') or yaml_path.endswith('.yml'):
            if os.path.exists(yaml_path):
                run_simulation(yaml_path)
            else:
                print(f"エラー: 指定されたファイルが見つかりません: {yaml_path}")
                sys.exit(1)
        else:
            print("エラー: 引数にはYAMLファイルを指定してください。")
            sys.exit(1)
    else:
        # 既存の対戦モード
        board = Board()
        board.setup()
        
        player1 = RandomPlayer(PLAYER_1)
        player2 = RandomPlayer(PLAYER_2)
        
        current_player_id = PLAYER_1
        players = {PLAYER_1: player1, PLAYER_2: player2}

        print("どうぶつしょうぎ を開始します！")
        
        # ゲームループ
        while True:
            print_board(board)
            
            # 1. 現在のプレイヤーの合法手を取得
            legal_moves = GameEngine.get_legal_moves(board, current_player_id)
            
            if not legal_moves:
                winner = PLAYER_2 if current_player_id == PLAYER_1 else PLAYER_1
                print(f"プレイヤー {current_player_id} に打てる手がありません。")
                print(f"👑 プレイヤー {winner} の勝ちです！")
                break
                
            current_player = players[current_player_id]
            selected_move = current_player.choose_move(board, legal_moves)
            print(f"\n選択された手: {selected_move}")
            
            GameEngine.make_move(board, selected_move, current_player_id)
            
            winner = GameEngine.check_winner(board)
            if winner:
                print_board(board)
                print(f"👑 プレイヤー {winner} の勝ちです！ゲーム終了。")
                break
                
            current_player_id = PLAYER_2 if current_player_id == PLAYER_1 else PLAYER_1

if __name__ == "__main__":
    main()

