#!/bin/bash
#SBATCH --job-name=quantum_d0_multigen
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=72:00:00
#SBATCH --output=job_multigen_%j.log
#SBATCH --error=job_multigen_%j.log

# ==============================================================================
# 量子どうぶつしょうぎ 連続世代 Slurm バッチ投入スクリプト (submit_multigen.sh)
# 使い方: sbatch submit_multigen.sh <開始世代> <終了世代>
# 例:     sbatch submit_multigen.sh 1 5
# ==============================================================================

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "エラー: 開始世代と終了世代を指定してください"
    echo "例: sbatch submit_multigen.sh 1 5"
    exit 1
fi

START_GEN=$1
END_GEN=$2

echo "=========================================================="
echo "★ 計算ノード上で Gen${START_GEN} から Gen${END_GEN} までの連続学習を開始します"
echo "実行ノード: $(hostname)"
echo "ジョブID:   $SLURM_JOB_ID"
echo "開始時刻:   $(date)"
echo "=========================================================="

cd ~/github/animal_shogi/quantum_animal_shogi
source ../.venv/bin/activate

for GEN in $(seq $START_GEN $END_GEN); do
    echo ""
    echo "=========================================================="
    echo ">>> Gen${GEN} パイプライン開始: $(date)"
    echo "=========================================================="
    bash auto_pipeline.sh $GEN
    echo ">>> Gen${GEN} パイプライン完了: $(date)"
done

echo "=========================================================="
echo "★ 全指定世代 (Gen${START_GEN}〜Gen${END_GEN}) の連続学習が完全完了しました！: $(date)"
echo "=========================================================="
