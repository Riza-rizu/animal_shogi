#!/bin/bash
#SBATCH --job-name=relearn_gen6
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --output=job_relearn_gen6_%j.log
#SBATCH --error=job_relearn_gen6_%j.log

# ==============================================================================
# 第9期 Gen 6 高品質再学習バッチスクリプト (relearn_gen6.sh)
# - visits >= 0.70 プルーニング適用
# - 300万件上限撤廃 (全件学習)
# - Gen 5 の累積辞書 + Gen 6 の100万局対局データで再構築
# ==============================================================================

cd ~/github/animal_shogi/quantum_animal_shogi
source ../.venv/bin/activate

echo "=========================================================="
echo "★ [Gen 6] 高品質再学習パイプライン開始: $(date)"
echo "実行ノード: $(hostname)"
echo "ジョブID:   $SLURM_JOB_ID"
echo "=========================================================="

echo ">>> [ステップ 1/4] make_dataset.py (プルーニング th=0.70 & 全件 train_data.npy 構築)..."
python -u make_dataset.py --gen 6

echo ">>> [ステップ 2/4] train.py (全件学習・エポック6)..."
python -u train.py --data train_data.npy --output quantum_model_gen6.keras --epochs 6 --batch_size 256 --max_samples 0

echo ">>> [ステップ 3/4] evaluate_generation.py (モデル自己評価)..."
python -u evaluate_generation.py --model quantum_model_gen6.keras --gen 6

echo ">>> [ステップ 4/4] archive_generation.py (成果物のアーカイブ)..."
python -u archive_generation.py --gen 6

echo ">>> 一時ファイルのクリーンアップ..."
rm -f input*.yaml tasks.sh

echo "=========================================================="
echo "★ [Gen 6] 高品質再学習が完全完了しました！: $(date)"
echo "=========================================================="
