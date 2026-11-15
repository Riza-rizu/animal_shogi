#!/bin/bash
#SBATCH --job-name=finish_gen7
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=job_finish_gen7_%j.log
#SBATCH --error=job_finish_gen7_%j.log

echo "=========================================================="
echo "★ Gen7 残り工程（学習・評価・アーカイブ）の再開実行"
echo "ノード: $(hostname) | 開始時刻: $(date)"
echo "=========================================================="

cd ~/github/animal_shogi/quantum_animal_shogi
source ../.venv/bin/activate

# 1. Gen7 モデルの深層学習（約15〜20分）
echo "\n[ステップ 1/3] train.py で Gen7 モデルの学習を実行中..."
python -u train.py --data train_data.npy --output quantum_model_gen7.keras

# 2. 評価散布図と指標の作成（約10秒）
echo "\n[ステップ 2/3] evaluate_generation.py で自己評価を実行中..."
python -u evaluate_generation.py --model quantum_model_gen7.keras --gen 7

# 3. 成果物を experiments/quantum_d0/gen7/ へアーカイブ保存
echo "\n[ステップ 3/3] archive_generation.py で Gen7 フォルダへ保存中..."
python -u archive_generation.py --gen 7

echo "=========================================================="
echo "★ Gen7 が完全に完了しました！: $(date)"
echo "=========================================================="
