"""
第9期 Gen5 vs 第8期 Gen5 直接対決検証スクリプト
- 第9期 Gen5 (187次元入力・順列特徴量あり)
- 第8期 Gen5 (139次元入力・順列特徴量なしベースライン)

【対戦形式 (第7期・第8期クロス対決と同一仕様)】
- 総対局数: 1,000 局
  - ブロック 1: 第9期 Gen5 (先手) vs 第8期 Gen5 (後手) 500 局
  - ブロック 2: 第8期 Gen5 (先手) vs 第9期 Gen5 (後手) 500 局
- パラメータ: depth=0, beta=10.0, gamma=0.6, max_moves=200
- 成果物出力:
  - CSV: experiments/cross_phase_matches/match_log_Phase9Gen5_vs_Phase8Gen5_1000games.csv
  - MD:  experiments/cross_phase_matches/match_report_Phase9Gen5_vs_Phase8Gen5.md
"""

import os
import sys
import time
import json
import csv
from collections import Counter
import numpy as np

repo_dir = os.path.dirname(os.path.abspath(__file__))
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)

from constants import PLAYER_1, PLAYER_2, RESULT_DRAW
from board import QuantumBoard
from engine import GameEngine
from player import ModelPlayer, load_keras_model_safe


def run_match_block(p1_player, p2_player, p1_label, p2_label, num_games=500, start_id=1, gamma=0.6):
    records = []
    print(f"  >> 対戦開始: 先手 {p1_label} vs 後手 {p2_label} ({num_games} 局) ...")
    t0 = time.time()

    for g in range(num_games):
        game_id = start_id + g
        board = QuantumBoard()
        board.setup()
        current_p = PLAYER_1

        state_counts = Counter()
        init_key = (board.to_hashable(), current_p)
        state_counts[init_key] = 1

        game_moves = 0
        max_moves = 200
        winner = None
        win_reason = "Draw_MaxMoves"
        move_history_str = []

        p1_collapse_ply = -1
        p2_collapse_ply = -1

        while game_moves < max_moves:
            if p1_collapse_ply == -1 and board.get_determined_lion_id(PLAYER_1) is not None:
                p1_collapse_ply = game_moves
            if p2_collapse_ply == -1 and board.get_determined_lion_id(PLAYER_2) is not None:
                p2_collapse_ply = game_moves

            legal_moves = GameEngine.get_legal_moves(board, current_p)
            if not legal_moves:
                winner = 3 - current_p
                win_reason = "NoLegalMoves"
                break

            p_obj = p1_player if current_p == PLAYER_1 else p2_player
            move = p_obj.choose_move(board, legal_moves, history_counts=state_counts, gamma=gamma)

            move_history_str.append(str(move))
            GameEngine.make_move(board, move, current_p)
            game_moves += 1

            next_state_key = (board.to_hashable(), 3 - current_p)
            state_counts[next_state_key] += 1

            winner, reason = GameEngine.check_winner_detail(board, last_player=current_p)
            if winner is not None:
                win_reason = reason
                break

            current_p = 3 - current_p

        if p1_collapse_ply == -1 and board.get_determined_lion_id(PLAYER_1) is not None:
            p1_collapse_ply = game_moves
        if p2_collapse_ply == -1 and board.get_determined_lion_id(PLAYER_2) is not None:
            p2_collapse_ply = game_moves

        if winner is None:
            winner = RESULT_DRAW
            win_reason = "Draw_MaxMoves"

        if winner == PLAYER_1:
            winner_model = p1_label
        elif winner == PLAYER_2:
            winner_model = p2_label
        else:
            winner_model = "Draw"

        rec = {
            "game_id": game_id,
            "p1_model": p1_label,
            "p2_model": p2_label,
            "winner_player": winner,
            "winner_model": winner_model,
            "win_reason": win_reason,
            "moves": game_moves,
            "is_short_game": (game_moves <= 6 and winner != RESULT_DRAW),
            "p1_collapse_ply": p1_collapse_ply if p1_collapse_ply != -1 else game_moves,
            "p2_collapse_ply": p2_collapse_ply if p2_collapse_ply != -1 else game_moves,
            "move_1": move_history_str[0] if len(move_history_str) > 0 else "",
            "move_2": move_history_str[1] if len(move_history_str) > 1 else "",
            "move_3": move_history_str[2] if len(move_history_str) > 2 else "",
            "move_4": move_history_str[3] if len(move_history_str) > 3 else ""
        }
        records.append(rec)

        if (g + 1) % 100 == 0 or (g + 1) == num_games:
            elapsed = time.time() - t0
            print(f"    [{p1_label} vs {p2_label}] {g + 1:3d}/{num_games} 局完了 ({elapsed:.1f} 秒)")

    return records


def main():
    m8_path = "experiments/quantum_d0_noperm8/gen5/quantum_model_gen5.keras"
    m9_path = "experiments/quantum_d0_perm9/gen5/quantum_model_gen5.keras"
    out_dir = "experiments/cross_phase_matches"

    if not os.path.exists(m8_path):
        print(f"エラー: 第8期モデルが見つかりません: {m8_path}")
        sys.exit(1)
    if not os.path.exists(m9_path):
        print(f"エラー: 第9期モデルが見つかりません: {m9_path}")
        sys.exit(1)

    print("=" * 70)
    print("【世紀の頂上決戦】第9期 Gen5 (187次元) vs 第8期 Gen5 (139次元)")
    print(f"第8期モデル: {m8_path}")
    print(f"第9期モデル: {m9_path}")
    print("設定: depth=0, beta=10.0, gamma=0.6, 先後入れ替え各500局 (計1,000局)")
    print("=" * 70)

    print("モデルをロード中...")
    m8 = load_keras_model_safe(m8_path)
    m9 = load_keras_model_safe(m9_path)

    label_9 = "Phase9_Gen5"
    label_8 = "Phase8_Gen5"

    p1_m9 = ModelPlayer(PLAYER_1, m9, beta=10.0, depth=0)
    p2_m8 = ModelPlayer(PLAYER_2, m8, beta=10.0, depth=0)

    p1_m8 = ModelPlayer(PLAYER_1, m8, beta=10.0, depth=0)
    p2_m9 = ModelPlayer(PLAYER_2, m9, beta=10.0, depth=0)

    # ブロック 1: 第9期 (先手) vs 第8期 (後手) 500局
    print("\n--- [ブロック 1/2] 第9期 Gen5 (先手) vs 第8期 Gen5 (後手) (500局) ---")
    rec1 = run_match_block(p1_m9, p2_m8, label_9, label_8, num_games=500, start_id=1)

    # ブロック 2: 第8期 (先手) vs 第9期 (後手) 500局
    print("\n--- [ブロック 2/2] 第8期 Gen5 (先手) vs 第9期 Gen5 (後手) (500局) ---")
    rec2 = run_match_block(p1_m8, p2_m9, label_8, label_9, num_games=500, start_id=501)

    all_records = rec1 + rec2
    total_games = len(all_records)

    # 集計
    m9_wins = sum(1 for r in all_records if r["winner_model"] == label_9)
    m8_wins = sum(1 for r in all_records if r["winner_model"] == label_8)
    draws = sum(1 for r in all_records if r["winner_model"] == "Draw")

    # 手番別
    b1_m9_wins = sum(1 for r in rec1 if r["winner_model"] == label_9)
    b1_m8_wins = sum(1 for r in rec1 if r["winner_model"] == label_8)
    b1_draws = sum(1 for r in rec1 if r["winner_model"] == "Draw")

    b2_m8_wins = sum(1 for r in rec2 if r["winner_model"] == label_8)
    b2_m9_wins = sum(1 for r in rec2 if r["winner_model"] == label_9)
    b2_draws = sum(1 for r in rec2 if r["winner_model"] == "Draw")

    # 決着理由
    try_total = sum(1 for r in all_records if r["win_reason"] == "Try")
    try_m9_win = sum(1 for r in all_records if r["win_reason"] == "Try" and r["winner_model"] == label_9)
    try_m8_win = sum(1 for r in all_records if r["win_reason"] == "Try" and r["winner_model"] == label_8)

    ext_total = sum(1 for r in all_records if r["win_reason"] in ["Catch", "Extinction"])
    ext_m9_win = sum(1 for r in all_records if r["win_reason"] in ["Catch", "Extinction"] and r["winner_model"] == label_9)
    ext_m8_win = sum(1 for r in all_records if r["win_reason"] in ["Catch", "Extinction"] and r["winner_model"] == label_8)

    all_moves = [r["moves"] for r in all_records]
    avg_moves = np.mean(all_moves)
    max_moves = max(all_moves)
    min_moves = min(all_moves)

    short_games = [r for r in all_records if r["is_short_game"]]
    short_count = len(short_games)
    short_m9_losses = sum(1 for r in short_games if r["winner_model"] == label_8)
    short_m8_losses = sum(1 for r in short_games if r["winner_model"] == label_9)

    m9_collapses = []
    m8_collapses = []
    for r in rec1:
        m9_collapses.append(r["p1_collapse_ply"])
        m8_collapses.append(r["p2_collapse_ply"])
    for r in rec2:
        m8_collapses.append(r["p1_collapse_ply"])
        m9_collapses.append(r["p2_collapse_ply"])

    avg_m9_collapse = np.mean(m9_collapses)
    avg_m8_collapse = np.mean(m8_collapses)

    m9_openings = Counter(r["move_1"] for r in rec1)
    m8_openings = Counter(r["move_1"] for r in rec2)

    # 画面レポート
    print("\n" + "=" * 70)
    print("■ 対戦結果サマリー (1,000 局)")
    print("=" * 70)
    print(f"  ・第9期 Gen5 勝利 : {m9_wins:4d} 局 ({m9_wins / total_games * 100:.2f} %)")
    print(f"  ・第8期 Gen5 勝利 : {m8_wins:4d} 局 ({m8_wins / total_games * 100:.2f} %)")
    print(f"  ・引き分け         : {draws:4d} 局 ({draws / total_games * 100:.2f} %)")
    print(f"  ・手番別勝率:")
    print(f"     第9期 Gen5 [先手]: {b1_m9_wins:3d}/500 ({b1_m9_wins / 500 * 100:.1f}%) | [後手]: {b2_m9_wins:3d}/500 ({b2_m9_wins / 500 * 100:.1f}%)")
    print(f"     第8期 Gen5 [先手]: {b2_m8_wins:3d}/500 ({b2_m8_wins / 500 * 100:.1f}%) | [後手]: {b1_m8_wins:3d}/500 ({b1_m8_wins / 500 * 100:.1f}%)")
    print(f"  ・Try 決着数      : 計 {try_total} 局 (第9期: {try_m9_win} 局, 第8期: {try_m8_win} 局)")
    print(f"  ・Catch/Ext 決着数 : 計 {ext_total} 局 (第9期: {ext_m9_win} 局, 第8期: {ext_m8_win} 局)")
    print(f"  ・平均手数         : {avg_moves:.2f} 手 (最長 {max_moves} 手, 最短 {min_moves} 手)")
    print(f"  ・短手数(3-6手)決着: {short_count} 局 (第9期ポカ負け: {short_m9_losses} 局, 第8期ポカ負け: {short_m8_losses} 局)")
    print(f"  ・ライオン確定手数 : 第9期平均 {avg_m9_collapse:.2f} 手目 vs 第8期平均 {avg_m8_collapse:.2f} 手目")

    # CSV・MD 保存
    os.makedirs(out_dir, exist_ok=True)
    csv_fn = os.path.join(out_dir, "match_log_Phase9Gen5_vs_Phase8Gen5_1000games.csv")
    with open(csv_fn, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_records[0].keys()))
        writer.writeheader()
        writer.writerows(all_records)

    md_fn = os.path.join(out_dir, "match_report_Phase9Gen5_vs_Phase8Gen5.md")
    with open(md_fn, "w", encoding="utf-8") as f:
        f.write("# 第9期 Gen5 vs 第8期 Gen5 直接対決検証レポート (1,000局)\n\n")
        f.write(f"- 実施日時: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("- 対局条件: depth=0, beta=10.0, gamma=0.6, 先後入れ替え各500局 (計1,000局)\n")
        f.write(f"- 第9期モデル: `{m9_path}` (187次元入力・順列特徴量あり)\n")
        f.write(f"- 第8期モデル: `{m8_path}` (139次元入力・ベースライン)\n\n")

        f.write("## 1. 総合対局成績\n\n")
        f.write("| モデル | 総勝利数 | 勝率 | 先手時勝率 | 後手時勝率 |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        f.write(f"| **第9期 Gen5 (187次元)** | **{m9_wins}** | **{m9_wins / total_games * 100:.2f}%** | {b1_m9_wins / 500 * 100:.2f}% ({b1_m9_wins}/500) | {b2_m9_wins / 500 * 100:.2f}% ({b2_m9_wins}/500) |\n")
        f.write(f"| **第8期 Gen5 (139次元)** | {m8_wins} | {m8_wins / total_games * 100:.2f}% | {b2_m8_wins / 500 * 100:.2f}% ({b2_m8_wins}/500) | {b1_m8_wins / 500 * 100:.2f}% ({b1_m8_wins}/500) |\n")
        f.write(f"| 引き分け | {draws} | {draws / total_games * 100:.2f}% | - | - |\n\n")

        f.write("## 2. 決着事由（実数カウント）\n\n")
        f.write("| 決着事由 | 総対局数 | 第9期 Gen5 勝利数 | 第8期 Gen5 勝利数 |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        f.write(f"| **Try（トライ到達）** | **{try_total}** 局 | **{try_m9_win}** 局 | **{try_m8_win}** 局 |\n")
        f.write(f"| **Catch/Extinction（捕獲・絶滅）** | **{ext_total}** 局 | **{ext_m9_win}** 局 | **{ext_m8_win}** 局 |\n\n")

        f.write("## 3. 量子将棋特性・手数・ポカ分析\n\n")
        f.write(f"- **第9期 Gen5 ライオン平均確定手数**: `{avg_m9_collapse:.2f}` 手目\n")
        f.write(f"- **第8期 Gen5 ライオン平均確定手数**: `{avg_m8_collapse:.2f}` 手目\n")
        f.write(f"- **平均手数**: `{avg_moves:.2f}` 手 (最長 `{max_moves}` 手 / 最短 `{min_moves}` 手)\n")
        f.write(f"- **短手数（3〜6手）決着数**: `{short_count}` 局 (第9期ポカ負け: `{short_m9_losses}` 局, 第8期ポカ負け: `{short_m8_losses}` 局)\n\n")

        f.write("## 4. 序盤定跡（初手の選択傾向 TOP 3）\n\n")
        f.write("### 第9期 Gen5 先手時\n")
        for mv, cnt in m9_openings.most_common(3):
            f.write(f"- `{mv}`: {cnt} 回 ({cnt / 500 * 100:.1f}%)\n")
        f.write("\n### 第8期 Gen5 先手時\n")
        for mv, cnt in m8_openings.most_common(3):
            f.write(f"- `{mv}`: {cnt} 回 ({cnt / 500 * 100:.1f}%)\n")

    print(f"\n  -> CSVログ保存完了: {csv_fn}")
    print(f"  -> レポート保存完了: {md_fn}")
    print("=" * 70)


if __name__ == "__main__":
    main()
