"""
Gen4 vs Gen3 直接対決スクリプト: match_gen4_vs_gen3.py
先後入れ替え各500局（計1,000局）で直接対戦し、以下の重要指標を網羅計測・集計・レポート出力します：
1. 全体勝率および先手/後手別の勝率（手番バイアスの分離）
2. 【量子将棋特有】各モデルのライオン確定（波束収縮）手数の比較
3. 決着事由の対局数（特に Try で決着した対局の実数、Extinction 実数など）
4. 手数分布（平均、最長、最短）および短手数（3〜6手）ポカ即死数
5. 初手・2手目の定跡着手傾向（頻出上位手）
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


def to_bytes_key(vec_139):
    return vec_139.astype(np.float32).tobytes()


def run_match_block(p1_player, p2_player, p1_label, p2_label, num_games=500, start_id=1, gamma=0.6, beta=10.0):
    records = []
    print(f"\n>> 対戦開始: 先手 {p1_label} vs 後手 {p2_label} ({num_games} 局) ...")
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
            # ライオン確定タイミングの検知
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

            # レピティション管理
            next_state_key = (board.to_hashable(), 3 - current_p)
            state_counts[next_state_key] += 1

            winner, reason = GameEngine.check_winner_detail(board, last_player=current_p)
            if winner is not None:
                win_reason = reason
                break

            current_p = 3 - current_p

        # 終局時のライオン確定チェック（確定していなければ最終手数）
        if p1_collapse_ply == -1 and board.get_determined_lion_id(PLAYER_1) is not None:
            p1_collapse_ply = game_moves
        if p2_collapse_ply == -1 and board.get_determined_lion_id(PLAYER_2) is not None:
            p2_collapse_ply = game_moves

        if winner is None:
            winner = RESULT_DRAW
            win_reason = "Draw_MaxMoves"

        # 勝者ラベル
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
            print(f"  [{p1_label} vs {p2_label}] {g + 1}/{num_games} 局完了 ({elapsed:.1f} 秒)")

    return records


def main():
    print("=" * 70)
    print("★ 量子どうぶつしょうぎ 直接対決検証: Gen4 vs Gen3 (計 1,000 局)")
    print("=" * 70)

    # モデルパスの探索（ローカルフォルダまたは experiments 内）
    gen4_candidates = [
        "experiments/quantum_d0/gen4/quantum_model_gen4.keras",
        "quantum_model_gen4.keras",
        "../experiments/quantum_d0/gen4/quantum_model_gen4.keras"
    ]
    gen3_candidates = [
        "experiments/quantum_d0/gen3/quantum_model_gen3.keras",
        "quantum_model_gen3.keras",
        "../experiments/quantum_d0/gen3/quantum_model_gen3.keras"
    ]

    path_gen4 = next((p for p in gen4_candidates if os.path.exists(p)), None)
    path_gen3 = next((p for p in gen3_candidates if os.path.exists(p)), None)

    if not path_gen4 or not path_gen3:
        print(f"エラー: モデルファイルが見つかりません。")
        print(f"  Gen4 パス候補: {gen4_candidates} (見つかった: {path_gen4})")
        print(f"  Gen3 パス候補: {gen3_candidates} (見つかった: {path_gen3})")
        sys.exit(1)

    print(f"Gen4 モデル: {path_gen4}")
    print(f"Gen3 モデル: {path_gen3}")

    print("モデルをメモリにロード中...")
    m4 = load_keras_model_safe(path_gen4)
    m3 = load_keras_model_safe(path_gen3)

    # プレイヤーインスタンス作成 (depth=0, beta=10.0)
    p1_gen4 = ModelPlayer(PLAYER_1, m4, beta=10.0, depth=0)
    p2_gen3 = ModelPlayer(PLAYER_2, m3, beta=10.0, depth=0)

    p1_gen3 = ModelPlayer(PLAYER_1, m3, beta=10.0, depth=0)
    p2_gen4 = ModelPlayer(PLAYER_2, m4, beta=10.0, depth=0)

    # ブロック1: 先手 Gen4 vs 後手 Gen3 (500局)
    records_block1 = run_match_block(p1_gen4, p2_gen3, "Gen4", "Gen3", num_games=500, start_id=1)

    # ブロック2: 先手 Gen3 vs 後手 Gen4 (500局)
    records_block2 = run_match_block(p1_gen3, p2_gen4, "Gen3", "Gen4", num_games=500, start_id=501)

    all_records = records_block1 + records_block2
    total_games = len(all_records)

    # =========================================================================
    # 集計処理
    # =========================================================================
    # 1. 勝敗
    g4_wins = sum(1 for r in all_records if r["winner_model"] == "Gen4")
    g3_wins = sum(1 for r in all_records if r["winner_model"] == "Gen3")
    draws = sum(1 for r in all_records if r["winner_model"] == "Draw")

    # 手番別
    b1_g4_wins = sum(1 for r in records_block1 if r["winner_model"] == "Gen4")
    b1_g3_wins = sum(1 for r in records_block1 if r["winner_model"] == "Gen3")
    b1_draws = sum(1 for r in records_block1 if r["winner_model"] == "Draw")

    b2_g4_wins = sum(1 for r in records_block2 if r["winner_model"] == "Gen4")
    b2_g3_wins = sum(1 for r in records_block2 if r["winner_model"] == "Gen3")
    b2_draws = sum(1 for r in records_block2 if r["winner_model"] == "Draw")

    # 2. 決着事由（Try 対局数・Extinction 対局数の実数カウント）
    try_games_total = sum(1 for r in all_records if r["win_reason"] == "Try")
    try_games_g4_win = sum(1 for r in all_records if r["win_reason"] == "Try" and r["winner_model"] == "Gen4")
    try_games_g3_win = sum(1 for r in all_records if r["win_reason"] == "Try" and r["winner_model"] == "Gen3")

    ext_games_total = sum(1 for r in all_records if r["win_reason"] == "Extinction")
    ext_games_g4_win = sum(1 for r in all_records if r["win_reason"] == "Extinction" and r["winner_model"] == "Gen4")
    ext_games_g3_win = sum(1 for r in all_records if r["win_reason"] == "Extinction" and r["winner_model"] == "Gen3")

    other_reason_counts = Counter(r["win_reason"] for r in all_records)

    # 3. 手数統計
    all_moves = [r["moves"] for r in all_records]
    avg_moves = np.mean(all_moves)
    max_moves = max(all_moves)
    min_moves = min(all_moves)

    # 短手数（3〜6手）ポカ即死
    short_games = [r for r in all_records if r["is_short_game"]]
    short_game_count = len(short_games)
    short_g4_losses = sum(1 for r in short_games if r["winner_model"] == "Gen3")  # Gen4のポカ負け
    short_g3_losses = sum(1 for r in short_games if r["winner_model"] == "Gen4")  # Gen3のポカ負け

    # 4. ライオン確定（波束収縮）手数
    # Gen4 のライオン確定手数（先手時 p1_collapse, 後手時 p2_collapse）
    g4_lion_collapses = []
    g3_lion_collapses = []

    for r in records_block1:  # P1=Gen4, P2=Gen3
        g4_lion_collapses.append(r["p1_collapse_ply"])
        g3_lion_collapses.append(r["p2_collapse_ply"])

    for r in records_block2:  # P1=Gen3, P2=Gen4
        g3_lion_collapses.append(r["p1_collapse_ply"])
        g4_lion_collapses.append(r["p2_collapse_ply"])

    avg_g4_lion_collapse = np.mean(g4_lion_collapses)
    avg_g3_lion_collapse = np.mean(g3_lion_collapses)

    # 5. 初手・2手目の頻出定跡
    g4_openings = Counter(r["move_1"] for r in records_block1)
    g3_openings = Counter(r["move_1"] for r in records_block2)

    # =========================================================================
    # レポート表示
    # =========================================================================
    print("\n" + "=" * 70)
    print("【Gen4 vs Gen3 直接対決 総合検証結果レポート】")
    print("=" * 70)

    print(f"\n■ ① 全体勝敗と勝率 (計 {total_games} 局)")
    print(f"  ・Gen4 勝利数: {g4_wins:4d} 局 ({g4_wins / total_games * 100:5.2f} %)")
    print(f"  ・Gen3 勝利数: {g3_wins:4d} 局 ({g3_wins / total_games * 100:5.2f} %)")
    print(f"  ・引き分け   : {draws:4d} 局 ({draws / total_games * 100:5.2f} %)")

    print(f"\n■ ② 手番別勝率（手番バイアスの分離）")
    print(f"  【第1部: 先手 Gen4 vs 後手 Gen3 (500局)】")
    print(f"    - 先手 Gen4 勝利: {b1_g4_wins:3d} 局 ({b1_g4_wins / 500 * 100:5.2f} %)")
    print(f"    - 後手 Gen3 勝利: {b1_g3_wins:3d} 局 ({b1_g3_wins / 500 * 100:5.2f} %)")
    print(f"    - 引き分け      : {b1_draws:3d} 局")
    print(f"  【第2部: 先手 Gen3 vs 後手 Gen4 (500局)】")
    print(f"    - 後手 Gen4 勝利: {b2_g4_wins:3d} 局 ({b2_g4_wins / 500 * 100:5.2f} %)")
    print(f"    - 先手 Gen3 勝利: {b2_g3_wins:3d} 局 ({b2_g3_wins / 500 * 100:5.2f} %)")
    print(f"    - 引き分け      : {b2_draws:3d} 局")

    print(f"\n■ ③ 決着事由の内訳（対局実数カウント）")
    print(f"  ・Try で決着した対局数         : 計 {try_games_total:3d} 局")
    print(f"      └ うち Gen4 の Try 勝ち    : {try_games_g4_win:3d} 局")
    print(f"      └ うち Gen3 の Try 勝ち    : {try_games_g3_win:3d} 局")
    print(f"  ・Extinction (捕獲) 決着対局数 : 計 {ext_games_total:3d} 局")
    print(f"      └ うち Gen4 の捕獲勝ち     : {ext_games_g4_win:3d} 局")
    print(f"      └ うち Gen3 の捕獲勝ち     : {ext_games_g3_win:3d} 局")
    for r_name, r_cnt in other_reason_counts.items():
        if r_name not in ["Try", "Extinction"]:
            print(f"  ・{r_name:18s}        : 計 {r_cnt:3d} 局")

    print(f"\n■ ④ 【量子将棋特有】ライオン確定（観測崩壊）タイミング")
    print(f"  ・Gen4 ライオン平均確定手数    : {avg_g4_lion_collapse:.2f} 手目")
    print(f"  ・Gen3 ライオン平均確定手数    : {avg_g3_lion_collapse:.2f} 手目")
    if avg_g4_lion_collapse > avg_g3_lion_collapse:
        print(f"  >> 考察: Gen4 は自陣ライオンを Gen3 より平均 {avg_g4_lion_collapse - avg_g3_lion_collapse:.2f} 手長く隠蔽できています。")
    else:
        print(f"  >> 考察: Gen3 の方が自陣ライオンの隠蔽維持手数が長いです。")

    print(f"\n■ ⑤ 手数分布および短手数ポカ即死数")
    print(f"  ・平均手数: {avg_moves:.2f} 手 (最長: {max_moves} 手, 最短: {min_moves} 手)")
    print(f"  ・3〜6手 短手数決着対局数: 計 {short_game_count:3d} 局")
    print(f"      └ Gen4 のポカ即死負け: {short_g4_losses:3d} 局")
    print(f"      └ Gen3 のポカ即死負け: {short_g3_losses:3d} 局")

    print(f"\n■ ⑥ 序盤定跡（初手の選択傾向 上位3手）")
    print(f"  【Gen4 が先手の初手】")
    for mv, cnt in g4_openings.most_common(3):
        print(f"    - {mv:30s}: {cnt:3d} 回 ({cnt / 500 * 100:5.1f} %)")
    print(f"  【Gen3 が先手の初手】")
    for mv, cnt in g3_openings.most_common(3):
        print(f"    - {mv:30s}: {cnt:3d} 回 ({cnt / 500 * 100:5.1f} %)")

    print("=" * 70)

    # =========================================================================
    # ファイル保存 (CSV および Markdown レポート)
    # =========================================================================
    out_dir = "experiments/quantum_d0/cross_matches"
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "match_log_gen4_vs_gen3_1000games.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_records[0].keys()))
        writer.writeheader()
        writer.writerows(all_records)
    print(f"\n対局ログCSVを保存しました: {csv_path}")

    md_path = os.path.join(out_dir, "match_report_gen4_vs_gen3.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Gen4 vs Gen3 直接対決検証レポート (1,000局)\n\n")
        f.write(f"- 実施日時: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- 条件: depth=0, beta=10.0, gamma=0.6, 先後入れ替え各500局\n\n")

        f.write("## 1. 総合対局成績\n\n")
        f.write(f"| モデル | 総勝利数 | 勝率 | 先手時勝率 | 後手時勝率 |\n")
        f.write(f"| :--- | :---: | :---: | :---: | :---: |\n")
        f.write(f"| **Gen4** | **{g4_wins}** | **{g4_wins / total_games * 100:.2f}%** | {b1_g4_wins / 500 * 100:.2f}% ({b1_g4_wins}/500) | {b2_g4_wins / 500 * 100:.2f}% ({b2_g4_wins}/500) |\n")
        f.write(f"| Gen3 | {g3_wins} | {g3_wins / total_games * 100:.2f}% | {b2_g3_wins / 500 * 100:.2f}% ({b2_g3_wins}/500) | {b1_g3_wins / 500 * 100:.2f}% ({b1_g3_wins}/500) |\n")
        f.write(f"| 引き分け | {draws} | {draws / total_games * 100:.2f}% | - | - |\n\n")

        f.write("## 2. 決着事由（実数カウント）\n\n")
        f.write(f"| 決着事由 | 総対局数 | Gen4 勝利数 | Gen3 勝利数 |\n")
        f.write(f"| :--- | :---: | :---: | :---: |\n")
        f.write(f"| **Try（トライ到達）** | **{try_games_total}** 局 | **{try_games_g4_win}** 局 | **{try_games_g3_win}** 局 |\n")
        f.write(f"| **Extinction（捕獲・絶滅）** | **{ext_games_total}** 局 | **{ext_games_g4_win}** 局 | **{ext_games_g3_win}** 局 |\n")
        for r_name, r_cnt in other_reason_counts.items():
            if r_name not in ["Try", "Extinction"]:
                f.write(f"| {r_name} | {r_cnt} 局 | - | - |\n")
        f.write("\n")

        f.write("## 3. 量子将棋特性・手数・ポカ分析\n\n")
        f.write(f"- **Gen4 ライオン平均確定手数**: `{avg_g4_lion_collapse:.2f}` 手目\n")
        f.write(f"- **Gen3 ライオン平均確定手数**: `{avg_g3_lion_collapse:.2f}` 手目\n")
        f.write(f"- **平均手数**: `{avg_moves:.2f}` 手 (最長 `{max_moves}` 手 / 最短 `{min_moves}` 手)\n")
        f.write(f"- **短手数（3〜6手）決着数**: `{short_game_count}` 局 (Gen4ポカ負け: `{short_g4_losses}` 局, Gen3ポカ負け: `{short_g3_losses}` 局)\n\n")

    print(f"詳細レポートMarkdownを保存しました: {md_path}")


if __name__ == "__main__":
    main()
