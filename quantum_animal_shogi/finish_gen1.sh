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
# 純粋な「新ルール Gen1」データ再構築・100%全データ学習・評価スクリプト
# 100万局の自己対局データ（processed_pkl/）から純粋な新ルールGen0データセットを生成し、
# 100% 全ての局面を用いて Gen1 モデルを学習・評価・アーカイブします。
# 使い方: sbatch finish_gen1.sh
# ==============================================================================

set -e

echo "=========================================================="
echo "★ 純粋な新ルール Gen1 のデータ再構築・学習・評価を開始します"
echo "実行ノード: $(hostname)"
echo "ジョブID:   $SLURM_JOB_ID"
echo "開始時刻:   $(date)"
echo "=========================================================="

cd ~/github/animal_shogi/quantum_animal_shogi
source ../.venv/bin/activate

# 1. 過去の汚染ファイル（古いnotry時代の累計データや肥大化したデータ）を削除
echo -e "\n[前処理] 過去の不要な残骸ファイルをクリーンアップ中..."
rm -f cumulative_dataset.pkl train_data.npy
rm -rf experiments/quantum_d0/gen1

# 2. processed_pkl/ から、今回計算した純粋な新ルールGen0の対局ログを復元
if [ -d "processed_pkl" ] && [ "$(ls -A processed_pkl 2>/dev/null)" ]; then
    echo "processed_pkl/ から純粋な新ルールGen0の対局ログを復元します..."
    mv processed_pkl/* .
    rmdir processed_pkl
fi

# 3. 純粋な新ルールGen0のデータセット作成 (make_dataset.py)
echo -e "\n[ステップ 1/4] make_dataset.py で純粋なGen0データセットを構築中..."
python -u make_dataset.py --gen 1

# 4. 全局面（100%）を用いたモデル学習 (train.py)
echo -e "\n[ステップ 2/4] train.py で全データ(100%)を用いたGen1モデル学習を実行中..."
python -u train.py --data train_data.npy --output quantum_model_gen1.keras --epochs 6 --batch_size 256

# 5. 自己評価 (evaluate_generation.py)
echo -e "\n[ステップ 3/4] evaluate_generation.py で自己評価を実行中..."
python -u evaluate_generation.py --model quantum_model_gen1.keras --gen 1

# 6. 成果物のアーカイブ保存 (archive_generation.py)
echo -e "\n[ステップ 4/4] archive_generation.py で Gen1 フォルダへ保存中..."
python -u archive_generation.py --gen 1

echo "=========================================================="
echo "★ 純粋な新ルール Gen1 が完全に完了しました！: $(date)"
echo "=========================================================="
