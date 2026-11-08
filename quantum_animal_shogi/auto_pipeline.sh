#!/bin/bash
# ==============================================================================
# 量子どうぶつしょうぎ 1世代一括全自動実行スクリプト (auto_pipeline.sh)
# 使い方: bash auto_pipeline.sh <世代番号>
# 例:     bash auto_pipeline.sh 1
# ==============================================================================

if [ -z "$1" ]; then
    echo "エラー: 世代番号を指定してください (例: bash auto_pipeline.sh 1)"
    exit 1
fi

GEN=$1
PREV_GEN=$((GEN - 1))
PREV_MODEL="quantum_model_gen${PREV_GEN}.keras"

echo "=========================================================="
echo "★ Gen${GEN} の全自動学習パイプラインを開始します"
echo "基準モデル: ${PREV_MODEL}"
echo "=========================================================="

# Gen0 が無ければ自動的にランダム初期化して作成
if [ ! -f "quantum_model_gen0.keras" ]; then
    echo "Gen0 が見つからないため、初期モデルを自動作成します..."
    python train.py --init --output quantum_model_gen0.keras
fi

# -------------------------------------------------------------
# ③ 対局設定の生成と並列シミュレーション (100万局: 15ワーカー × 各66667局)
# -------------------------------------------------------------
echo "[ステップ 1/4] makeseed.py で設定ファイル群を生成中 (計 1,000,005 局)..."
python makeseed.py --depth 0 --n_files 15 --games 66667 --model $PREV_MODEL

echo "[ステップ 2/4] 並列シミュレーションを実行中..."
# 15並列で実行し、すべての対局が完了するまで待機
cat tasks.sh | xargs -I {} -P 15 bash -c "{}"

echo ">> 100万局の自己対局がすべて完了しました！"

# -------------------------------------------------------------
# ④ データ集約・学習・評価・アーカイブ
# -------------------------------------------------------------
echo "[ステップ 3/4] データ集約・学習・評価・アーカイブを実行中..."
python -u make_dataset.py --gen $GEN && \
python -u train.py --data train_data.npy --output quantum_model_gen${GEN}.keras && \
python -u evaluate_generation.py --model quantum_model_gen${GEN}.keras --gen $GEN && \
python -u archive_generation.py --gen $GEN

# 一時設定ファイルのクリーンアップ
rm -f input*.yaml tasks.sh

echo "=========================================================="
echo "★ Gen${GEN} の全パイプラインが完全完了しました！"
echo "成果物は experiments/d0_six/gen${GEN}/ に保存されています。"
echo "=========================================================="
