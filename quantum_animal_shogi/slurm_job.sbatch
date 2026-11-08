#!/bin/bash
#SBATCH --job-name=quantum_shogi
#SBATCH --output=slurm_%j.log
#SBATCH --error=slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00

# ==============================================================================
# 研究室サーバー用 Slurm バッチ実行スクリプト (sbatch slurm_job.sbatch)
# 第3期・第4期と完全に同一の手順・形式で大規模自己対局と学習を実行します。
# ==============================================================================

set -e

# Python 仮想環境のアクティベート (サーバーのパスに合わせて適宜変更)
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "../.venv" ]; then
    source ../.venv/bin/activate
fi

PYTHON=python

# 開始世代と終了世代の設定 (例: Gen1 から Gen5 まで)
START_GEN=1
END_GEN=5

# 探索深さ (0: 直接推論 = 100万局, 2: Min-Max = 20万局)
DEPTH=0

# ワーカー数 (サーバーのCPUコア数やスレッド数に合わせて設定)
N_FILES=50

# 1世代あたりの総試合数 (第3・4期公式仕様: depth 0 なら 100万局, depth 2 なら 20万局)
if [ "$DEPTH" -eq 0 ]; then
    TOTAL_GAMES=1000000
else
    TOTAL_GAMES=200000
fi

BETA=10.0
GAMMA=0.6

echo "======================================================================"
echo "量子どうぶつしょうぎ 大規模サーバー学習ジョブ開始"
echo "世代範囲: Gen${START_GEN} -> Gen${END_GEN}"
echo "探索深さ: depth=${DEPTH} (1世代あたり総対局数: ${TOTAL_GAMES} 局)"
echo "ワーカー数: ${N_FILES} (1ワーカーあたり: $((TOTAL_GAMES / N_FILES)) 局)"
echo "======================================================================"

# Gen0 が無ければ作成
if [ ! -f "quantum_model_gen0.keras" ]; then
    echo "[初期化] 初期モデル quantum_model_gen0.keras を生成します..."
    $PYTHON train.py --init --output quantum_model_gen0.keras
    $PYTHON archive_generation.py --gen 0
fi

for GEN in $(seq $START_GEN $END_GEN); do
    PREV_GEN=$((GEN - 1))
    PREV_MODEL="quantum_model_gen${PREV_GEN}.keras"
    CURR_MODEL="quantum_model_gen${GEN}.keras"

    echo ""
    echo "======================================================================"
    echo "★ [Gen${GEN} サイクル開始] 基準モデル: ${PREV_MODEL} -> 新規モデル: ${CURR_MODEL}"
    echo "======================================================================"

    # 1. 設定ファイル群 (input*.yaml, tasks.sh) の生成
    echo "[1/5] makeseed.py で設定ファイルを生成中..."
    $PYTHON makeseed.py \
        --n_files $N_FILES \
        --total_games $TOTAL_GAMES \
        --depth $DEPTH \
        --beta $BETA \
        --gamma $GAMMA \
        --model $PREV_MODEL \
        --python_bin $PYTHON

    # 2. 並列シミュレーションの実行 (CPUコア数に合わせて並列実行)
    echo "[2/5] 並列自己対局シミュレーションを実行中..."
    if command -v xargs &> /dev/null; then
        cat tasks.sh | xargs -I {} -P $N_FILES bash -c "{}"
    else
        bash tasks.sh
    fi

    # 3. データの集約と train_data.npy の作成
    echo "[3/5] make_dataset.py でログを集約中..."
    $PYTHON make_dataset.py --gen $GEN

    # 4. モデルの学習 (パラメータ更新)
    echo "[4/5] train.py でニューラルネットワークを学習中..."
    $PYTHON train.py --data train_data.npy --output $CURR_MODEL --epochs 10 --batch_size 256

    # 5. 評価・散布図生成とアーカイブ
    echo "[5/5] evaluate_generation.py & archive_generation.py で保存中..."
    $PYTHON evaluate_generation.py --model $CURR_MODEL --gen $GEN
    $PYTHON archive_generation.py --gen $GEN

    # 一時設定ファイルのクリーンアップ
    rm -f input*.yaml tasks.sh

    echo "★ Gen${GEN} の学習とアーカイブ保存が完了しました！"
done

echo ""
echo "======================================================================"
echo "全世代の学習ジョブが正常に完了しました！"
echo "成果物は experiments/d0_six/ に世代別に保存されています。"
echo "======================================================================"
