#!/bin/bash
#SBATCH --job-name=finish_gen1
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=job_finish_gen1_%j.log
#SBATCH --error=job_finish_gen1_%j.log

# ==============================================================================
# Gen1 残り工程（学習・評価・アーカイブ）のSlurmバッチ実行スクリプト
# 100万局の自己対局データ（train_data.npy等）は既に完成しているため、
# 計算ノード上で安全に学習・評価・アーカイブのみを実行します。
# 使い方: sbatch finish_gen1.sh
# ==============================================================================

echo "=========================================================="
echo "★ 計算ノード上で Gen1 残り工程（学習・評価・アーカイブ）を開始します"
echo "実行ノード: $(hostname)"
echo "ジョブID:   $SLURM_JOB_ID"
echo "開始時刻:   $(date)"
echo "=========================================================="

cd ~/github/animal_shogi/quantum_animal_shogi
source ../.venv/bin/activate
set -e

# 1. Gen1 モデルの学習 (batch_size 256, max_samples 200万)
echo -e "\n[ステップ 1/3] train.py で Gen1 モデルの学習を実行中..."
python -u train.py --data train_data.npy --output quantum_model_gen1.keras --epochs 6 --batch_size 256 --max_samples 2000000

# 2. 評価散布図と指標の作成（約10秒）
echo -e "\n[ステップ 2/3] evaluate_generation.py で自己評価を実行中..."
python -u evaluate_generation.py --model quantum_model_gen1.keras --gen 1

# 3. 成果物を experiments/quantum_d0/gen1/ へアーカイブ保存
echo -e "\n[ステップ 3/3] archive_generation.py で Gen1 フォルダへ保存中..."
python -u archive_generation.py --gen 1

echo "=========================================================="
echo "★ Gen1 が完全に完了しました！: $(date)"
echo "=========================================================="
