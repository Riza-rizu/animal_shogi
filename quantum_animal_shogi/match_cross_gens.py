"""
任意世代間 直接対戦検証スクリプト: match_cross_gens.py
第3世代〜第6世代における未完了の5対戦（または任意の2世代）を一括自動対戦・集計・レポート出力します。

【対象対戦（計5対戦、各1,000局）】
1. Gen5 vs Gen3
2. Gen5 vs Gen4
3. Gen6 vs Gen3
4. Gen6 vs Gen4
5. Gen6 vs Gen5

【計測・出力指標（Gen4 vs Gen3 と同一仕様）】
1. 全体勝敗・勝率および先手/後手別勝率
2. 決着事由の実数（特に Try で決着した対局実数、Extinction 実数）
3. 【量子将棋特有】各モデルのライオン確定（波束収縮）手数の比較
4. 手数分布（平均・最長・最短）および短手数（3〜6手）ポカ即死数
5. 序盤定跡（初手の選択傾向 TOP 3）

成果物はすべて experiments/quantum_d0/cross_matches/ に保存されます。
"""

import os
import sys
import time
import json
import csv
import argparse
from collections import Counter
import numpy as np

repo_dir = os.path.dirname(os.path.abspath(__file__))
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)

from constants import PLAYER_1, PLAYER_2, RESULT_DRAW
from board import QuantumBoard
from engine import GameEngine
from player import ModelPlayer, load_keras_model_safe


def run_match_block(p1_player, p2_player, p1_label, p2_label, num_games=500, start_id=1, gamma=0.6, beta=10.0):
    records = []
    print(f"  >> 対戦ブロック開始: 先手 {p1_label} vs 後手 {p2_label} ({num_games} 局) ...")
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


def get_model_path(gen):
    candidates = [
        f"experiments/quantum_d0/gen{gen}/quantum_model_gen{gen}.keras",
        f"quantum_model_gen{gen}.keras",
        f"../experiments/quantum_d0/gen{gen}/quantum_model_gen{gen}.keras"
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def run_head_to_head(gen_a, gen_b, loaded_models, out_dir="experiments/quantum_d0/cross_matches", num_games_per_side=500):
    label_a = f"Gen{gen_a}"
    label_b = f"Gen{gen_b}"
    print("\n" + "=" * 70)
    print(f"【対決開始】 {label_a} vs {label_b} (計 {num_games_per_side * 2} 局)")
    print("=" * 70)

    m_a = loaded_models[gen_a]
    m_b = loaded_models[gen_b]

    p1_a = ModelPlayer(PLAYER_1, m_a, beta=10.0, depth=0)
    p2_b = ModelPlayer(PLAYER_2, m_b, beta=10.0, depth=0)

    p1_b = ModelPlayer(PLAYER_1, m_b, beta=10.0, depth=0)
    p2_a = ModelPlayer(PLAYER_2, m_a, beta=10.0, depth=0)

    # 先後入れ替え対局
    rec1 = run_match_block(p1_a, p2_b, label_a, label_b, num_games=num_games_per_side, start_id=1)
    rec2 = run_match_block(p1_b, p2_a, label_b, label_a, num_games=num_games_per_side, start_id=num_games_per_side + 1)
    all_records = rec1 + rec2
    total_games = len(all_records)

    # 集計
    a_wins = sum(1 for r in all_records if r["winner_model"] == label_a)
    b_wins = sum(1 for r in all_records if r["winner_model"] == label_b)
    draws = sum(1 for r in all_records if r["winner_model"] == "Draw")

    b1_a_wins = sum(1 for r in rec1 if r["winner_model"] == label_a)
    b1_b_wins = sum(1 for r in rec1 if r["winner_model"] == label_b)
    b1_draws = sum(1 for r in rec1 if r["winner_model"] == "Draw")

    b2_a_wins = sum(1 for r in rec2 if r["winner_model"] == label_a)
    b2_b_wins = sum(1 for r in rec2 if r["winner_model"] == label_b)
    b2_draws = sum(1 for r in rec2 if r["winner_model"] == "Draw")

    try_total = sum(1 for r in all_records if r["win_reason"] == "Try")
    try_a_win = sum(1 for r in all_records if r["win_reason"] == "Try" and r["winner_model"] == label_a)
    try_b_win = sum(1 for r in all_records if r["win_reason"] == "Try" and r["winner_model"] == label_b)

    ext_total = sum(1 for r in all_records if r["win_reason"] == "Extinction")
    ext_a_win = sum(1 for r in all_records if r["win_reason"] == "Extinction" and r["winner_model"] == label_a)
    ext_b_win = sum(1 for r in all_records if r["win_reason"] == "Extinction" and r["winner_model"] == label_b)

    other_counts = Counter(r["win_reason"] for r in all_records)

    all_moves = [r["moves"] for r in all_records]
    avg_moves = np.mean(all_moves)
    max_moves = max(all_moves)
    min_moves = min(all_moves)

    short_games = [r for r in all_records if r["is_short_game"]]
    short_game_count = len(short_games)
    short_a_losses = sum(1 for r in short_games if r["winner_model"] == label_b)
    short_b_losses = sum(1 for r in short_games if r["winner_model"] == label_a)

    a_lion_collapses = []
    b_lion_collapses = []
    for r in rec1:
        a_lion_collapses.append(r["p1_collapse_ply"])
        b_lion_collapses.append(r["p2_collapse_ply"])
    for r in rec2:
        b_lion_collapses.append(r["p1_collapse_ply"])
        a_lion_collapses.append(r["p2_collapse_ply"])

    avg_a_collapse = np.mean(a_lion_collapses)
    avg_b_collapse = np.mean(b_lion_collapses)

    a_openings = Counter(r["move_1"] for r in rec1)
    b_openings = Counter(r["move_1"] for r in rec2)

    # 画面表示
    print(f"\n■ {label_a} vs {label_b} 結果サマリー")
    print(f"  ・{label_a} 勝利: {a_wins} 局 ({a_wins / total_games * 100:.2f} %)")
    print(f"  ・{label_b} 勝利: {b_wins} 局 ({b_wins / total_games * 100:.2f} %)")
    print(f"  ・引き分け : {draws} 局")
    print(f"  ・手番別   : 先手{label_a}勝率 {b1_a_wins/num_games_per_side*100:.1f}%, 後手{label_a}勝率 {b2_a_wins/num_games_per_side*100:.1f}%")
    print(f"  ・Try決着数: 計 {try_total} 局 ({label_a} Try勝: {try_a_win} 局, {label_b} Try勝: {try_b_win} 局)")
    print(f"  ・Ext決着数: 計 {ext_total} 局 ({label_a} 捕獲勝: {ext_a_win} 局, {label_b} 捕獲勝: {ext_b_win} 局)")
    print(f"  ・平均手数 : {avg_moves:.1f} 手 (最長 {max_moves} 手, 最短 {min_moves} 手)")
    print(f"  ・自滅即死 : 3〜6手決着 {short_game_count} 局 ({label_a}ポカ負け: {short_a_losses}, {label_b}ポカ負け: {short_b_losses})")
    print(f"  ・ライオン確定手数: {label_a}平均 {avg_a_collapse:.2f}手目 vs {label_b}平均 {avg_b_collapse:.2f}手目")

    # 保存
    os.makedirs(out_dir, exist_ok=True)
    csv_fn = os.path.join(out_dir, f"match_log_gen{gen_a}_vs_gen{gen_b}_1000games.csv")
    with open(csv_fn, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_records[0].keys()))
        writer.writeheader()
        writer.writerows(all_records)

    md_fn = os.path.join(out_dir, f"match_report_gen{gen_a}_vs_gen{gen_b}.md")
    with open(md_fn, "w", encoding="utf-8") as f:
        f.write(f"# {label_a} vs {label_b} 直接対決検証レポート (1,000局)\n\n")
        f.write(f"- 実施日時: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- 条件: depth=0, beta=10.0, gamma=0.6, 先後入れ替え各{num_games_per_side}局\n\n")

        f.write("## 1. 総合対局成績\n\n")
        f.write(f"| モデル | 総勝利数 | 勝率 | 先手時勝率 | 後手時勝率 |\n")
        f.write(f"| :--- | :---: | :---: | :---: | :---: |\n")
        f.write(f"| **{label_a}** | **{a_wins}** | **{a_wins / total_games * 100:.2f}%** | {b1_a_wins / num_games_per_side * 100:.2f}% ({b1_a_wins}/{num_games_per_side}) | {b2_a_wins / num_games_per_side * 100:.2f}% ({b2_a_wins}/{num_games_per_side}) |\n")
        f.write(f"| **{label_b}** | {b_wins} | {b_wins / total_games * 100:.2f}% | {b2_b_wins / num_games_per_side * 100:.2f}% ({b2_b_wins}/{num_games_per_side}) | {b1_b_wins / num_games_per_side * 100:.2f}% ({b1_b_wins}/{num_games_per_side}) |\n")
        f.write(f"| 引き分け | {draws} | {draws / total_games * 100:.2f}% | - | - |\n\n")

        f.write("## 2. 決着事由（実数カウント）\n\n")
        f.write(f"| 決着事由 | 総対局数 | {label_a} 勝利数 | {label_b} 勝利数 |\n")
        f.write(f"| :--- | :---: | :---: | :---: |\n")
        f.write(f"| **Try（トライ到達）** | **{try_total}** 局 | **{try_a_win}** 局 | **{try_b_win}** 局 |\n")
        f.write(f"| **Extinction（捕獲・絶滅）** | **{ext_total}** 局 | **{ext_a_win}** 局 | **{ext_b_win}** 局 |\n")
        for r_name, r_cnt in other_counts.items():
            if r_name not in ["Try", "Extinction"]:
                f.write(f"| {r_name} | {r_cnt} 局 | - | - |\n")
        f.write("\n")

        f.write("## 3. 量子将棋特性・手数・ポカ分析\n\n")
        f.write(f"- **{label_a} ライオン平均確定手数**: `{avg_a_collapse:.2f}` 手目\n")
        f.write(f"- **{label_b} ライオン平均確定手数**: `{avg_b_collapse:.2f}` 手目\n")
        f.write(f"- **平均手数**: `{avg_moves:.2f}` 手 (最長 `{max_moves}` 手 / 最短 `{min_moves}` 手)\n")
        f.write(f"- **短手数（3〜6手）決着数**: `{short_game_count}` 局 ({label_a}ポカ負け: `{short_a_losses}` 局, {label_b}ポカ負け: `{short_b_losses}` 局)\n\n")

        f.write("## 4. 序盤定跡（初手の選択傾向 上位3手）\n\n")
        f.write(f"### {label_a} 先手時\n")
        for mv, cnt in a_openings.most_common(3):
            f.write(f"- `{mv}`: {cnt} 回 ({cnt / num_games_per_side * 100:.1f}%)\n")
        f.write(f"\n### {label_b} 先手時\n")
        for mv, cnt in b_openings.most_common(3):
            f.write(f"- `{mv}`: {cnt} 回 ({cnt / num_games_per_side * 100:.1f}%)\n")

    print(f"  -> 保存完了: {md_fn}")
    return {
        "gen_a": gen_a,
        "gen_b": gen_b,
        "a_wins": a_wins,
        "b_wins": b_wins,
        "draws": draws,
        "try_total": try_total,
        "ext_total": ext_total,
        "avg_moves": avg_moves
    }


def main():
    parser = argparse.ArgumentParser(description="世代間 直接対戦検証スクリプト")
    parser.add_argument("--genA", type=int, default=None, help="対戦世代A (例: 5)")
    parser.add_argument("--genB", type=int, default=None, help="対戦世代B (例: 4)")
    parser.add_argument("--all_remaining", action="store_true", default=True,
                        help="3vs4 以外の残り5対戦 (5vs3, 5vs4, 6vs3, 6vs4, 6vs5) を一括実行")
    args = parser.parse_args()

    # 実行対象のカード決定
    if args.genA is not None and args.genB is not None:
        matchup_list = [(args.genA, args.genB)]
    else:
        # 残り5対戦
        matchup_list = [
            (5, 3),
            (5, 4),
            (6, 3),
            (6, 4),
            (6, 5)
        ]

    # 必要な全世代モデルをあらかじめ一括ロード
    required_gens = set()
    for ga, gb in matchup_list:
        required_gens.add(ga)
        required_gens.add(gb)

    loaded_models = {}
    print("=" * 70)
    print("対戦に必要なニューラルネットワークモデルをロードしています...")
    for g in sorted(required_gens):
        p = get_model_path(g)
        if not p:
            print(f"エラー: Gen{g} のモデルが見つかりません。")
            sys.exit(1)
        print(f"  [Load] Gen{g}: {p}")
        loaded_models[g] = load_keras_model_safe(p)
    print("=" * 70)

    results = []
    t_start = time.time()
    for ga, gb in matchup_list:
        res = run_head_to_head(ga, gb, loaded_models)
        results.append(res)

    t_total = time.time() - t_start

    # 全体まとめテーブルの表示・出力
    if len(results) > 1:
        print("\n" + "=" * 70)
        print("★【全5対戦 完了サマリー表】")
        print("=" * 70)
        print(f"{'対戦カード':<16} | {'勝利数 (勝率)':<22} | {'相手勝利数 (勝率)':<22} | {'引分':<4} | {'Try数':<5} | {'平均手数':<6}")
        print("-" * 75)
        for r in results:
            ga, gb = r["gen_a"], r["gen_b"]
            card = f"Gen{ga} vs Gen{gb}"
            a_stat = f"{r['a_wins']} ({r['a_wins']/1000*100:.1f}%)"
            b_stat = f"{r['b_wins']} ({r['b_wins']/1000*100:.1f}%)"
            print(f"{card:<16} | {a_stat:<22} | {b_stat:<22} | {r['draws']:<4} | {r['try_total']:<5} | {r['avg_moves']:<6.1f}")
        print("=" * 70)
        print(f"総所要時間: {t_total/60:.1f} 分")

        # 総合サマリー Markdown の保存
        sum_md = "experiments/quantum_d0/cross_matches/cross_matches_summary_gen3_to_gen6.md"
        with open(sum_md, "w", encoding="utf-8") as f:
            f.write("# 第3世代〜第6世代 世代間直接対決 総合サマリー\n\n")
            f.write(f"- 実施日時: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"- 条件: depth=0, beta=10.0, 各1,000局（先後500局ずつ）\n\n")
            f.write("| 対戦カード | 世代A勝数 (勝率) | 世代B勝数 (勝率) | 引き分け | Try決着数 | 平均手数 |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
            # 既に対戦済みの 4vs3 も含めて総合表にする
            f.write("| Gen4 vs Gen3 (先行実施済) | 663 (66.3%) | 337 (33.7%) | 0 | 0 | 25.4 |\n")
            for r in results:
                ga, gb = r["gen_a"], r["gen_b"]
                f.write(f"| **Gen{ga} vs Gen{gb}** | **{r['a_wins']} ({r['a_wins']/1000*100:.1f}%)** | {r['b_wins']} ({r['b_wins']/1000*100:.1f}%) | {r['draws']} | {r['try_total']} 局 | {r['avg_moves']:.1f} 手 |\n")
        print(f"総合サマリー表を保存しました: {sum_md}")


if __name__ == "__main__":
    main()
