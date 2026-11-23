#!/bin/bash
#SBATCH --job-name=quantum_d0
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=job_%x_gen%a_%j.log
#SBATCH --error=job_%x_gen%a_%j.log

# ==============================================================================
# 量子どうぶつしょうぎ Slurm ジョブ投入スクリプト (submit_job.sh)
# 使い方: sbatch submit_job.sh <世代番号>
# 例:     sbatch submit_job.sh 3
# ==============================================================================

if [ -z "$1" ]; then
    echo "エラー: 世代番号を指定してください (例: sbatch submit_job.sh 3)"
    exit 1
fi

GEN=$1

echo "=========================================================="
echo "★ 計算ノード上で Gen${GEN} の全自動学習パイプラインを開始します"
echo "実行ノード: $(hostname)"
echo "ジョブID:   $SLURM_JOB_ID"
echo "開始時刻:   $(date)"
echo "=========================================================="

# 念のため作業ディレクトリへ移動
cd ~/github/animal_shogi/quantum_animal_shogi

# 仮想環境のアクティベート
source ../.venv/bin/activate

# パイプライン本体の実行
bash auto_pipeline.sh $GEN

echo "=========================================================="
echo "★ Gen${GEN} ジョブ完全完了時刻: $(date)"
echo "=========================================================="
