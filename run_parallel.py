"""
どうぶつしょうぎ シミュレーション並列実行スクリプト
"""

import os
import sys
import glob
import subprocess
import argparse
from multiprocessing import Pool

def run_simulation(yaml_path):
    print(f"[START] {yaml_path} のシミュレーションを開始します。")
    
    # 呼び出し元の Python インタプリタをそのまま使用して main.py を実行
    cmd = [sys.executable, "main.py", yaml_path]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    
    if result.returncode == 0:
        print(f"[SUCCESS] {yaml_path} のシミュレーションが完了しました。")
    else:
        print(f"[FAILED] {yaml_path} の実行中にエラーが発生しました。\nStdout:\n{result.stdout}\nStderr:\n{result.stderr}")
        
    return yaml_path, result.returncode

def main():
    parser = argparse.ArgumentParser(description="どうぶつしょうぎのシミュレーションを並列で実行します。")
    parser.add_argument(
        "--num_processes", 
        type=int, 
        default=os.cpu_count(),
        help="並列実行するプロセス数（デフォルト: システムの最大CPUコア数）"
    )
    args = parser.parse_args()

    # ディレクトリ内の input*.yaml ファイルを取得してソート
    yaml_files = sorted(glob.glob("input*.yaml"))
    
    if not yaml_files:
        print("エラー: input*.yaml ファイルが見つかりません。")
        print("先に makeseed.py を実行して設定ファイルを生成してください。")
        sys.exit(1)

    print(f"検出された設定ファイル数: {len(yaml_files)}")
    print(f"使用プロセス数（並列度）: {args.num_processes}")
    print("並列実行を開始します...")
    print("-" * 50)

    # プロセスプールを使用して並列実行
    with Pool(processes=args.num_processes) as pool:
        results = pool.map(run_simulation, yaml_files)

    print("-" * 50)
    print("すべての実行が完了しました。結果サマリー:")
    
    success_count = 0
    for yaml_path, code in results:
        status = "成功" if code == 0 else "失敗"
        if code == 0:
            success_count += 1
        print(f"  - {yaml_path}: {status}")
        
    print(f"\n合計: {success_count} / {len(yaml_files)} 件成功")

if __name__ == "__main__":
    main()
