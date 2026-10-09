"""
各世代の散布図における回帰直線の傾き (スロープ) ・ばらつき (標準偏差) 分析スクリプト
- S_n(x) [生対局実績] のばらつき (std_true)
- y_pred [モデル推論値] のばらつき (std_pred)
- 回帰直線の傾き a (y = a * x + b)
- 相関係数 r, 決定係数 R^2
を完全集計・比較
"""

import os
import sys
import numpy as np

# TensorFlow 消音
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf

repo_dir = os.path.dirname(os.path.abspath(__file__))

def analyze_generation_slope(gen, base_dir=None):
    if base_dir is None:
        base_dir = os.path.join(repo_dir, "experiments", "depth0_pure_mse")
    gen_dir = os.path.join(base_dir, f"gen{gen}")
    model_path = os.path.join(gen_dir, "model.keras")
    
    sn_path = os.path.join(gen_dir, "current_gen_data.npy")
    train_path = os.path.join(gen_dir, "train_data.npy")
    
    data_path = None
    data_type = ""
    if os.path.exists(sn_path):
        data_path = sn_path
        data_type = "S_n(x) (α=1)"
    elif os.path.exists(train_path):
        data_path = train_path
        data_type = "Y_n(x) (累計)"
    else:
        return None

    if not os.path.exists(model_path):
        return None

    data = np.load(data_path)
    X = data[:, :139].astype(np.float32)
    y_true = data[:, 139].astype(np.float32)

    # 評価スクリプトと同様に均等サンプリング (3,000件)
    n_samples = min(3000, len(y_true))
    np.random.seed(42)
    indices = np.random.choice(len(y_true), n_samples, replace=False)
    X_sample = X[indices]
    y_sample = y_true[indices]

    model = tf.keras.models.load_model(model_path)
    y_pred = model(X_sample, training=False).numpy().flatten()

    # 線形回帰 y_pred = a * y_true + b
    a, b = np.polyfit(y_sample, y_pred, 1)
    corr = np.corrcoef(y_sample, y_pred)[0, 1]
    r2 = corr ** 2

    # ばらつき (標準偏差)
    std_true = float(np.std(y_sample))
    std_pred = float(np.std(y_pred))
    mean_true = float(np.mean(y_sample))
    mean_pred = float(np.mean(y_pred))

    return {
        "gen": gen,
        "data_type": data_type,
        "std_true": std_true,
        "std_pred": std_pred,
        "slope_a": float(a),
        "intercept_b": float(b),
        "corr_r": float(corr),
        "r2": float(r2),
        "n_samples": n_samples
    }

def main():
    print("=" * 85)
    print("      【散布図 傾き・ばらつき (標準偏差) 世代別総合分析レポート】")
    print("=" * 85)
    print("※ 数学的関係式: 傾き a = 相関係数 r × (std_pred / std_true)")
    print("※ 理想値: 傾き a = 1.000, 相関 r = 1.000, std比 = 1.000\n")

    results = []
    for g in range(1, 8):
        res = analyze_generation_slope(g)
        if res:
            results.append(res)

    print("+------+-----------------+-----------------+-----------------+-----------+-----------+------------+")
    print("| 世代 | S_n(x)ばらつき  | y_predばらつき  | ばらつき比      | 傾き (a)  | 切片 (b)  | 相関係数(r)|")
    print("|      | (真値の標準偏差)| (推論の標準偏差)| (pred / true)   |           |           |            |")
    print("+------+-----------------+-----------------+-----------------+-----------+-----------+------------+")
    for r in results:
        ratio = r['std_pred'] / r['std_true']
        print(f"| Gen{r['gen']} |     {r['std_true']:6.4f}      |     {r['std_pred']:6.4f}      |     {ratio:6.4f}      |  {r['slope_a']:6.4f}   |  {r['intercept_b']:6.4f}   |   {r['corr_r']:6.4f}   |")
    print("+------+-----------------+-----------------+-----------------+-----------+-----------+------------+")

    print("\n■ 数値の読み解きポイント:")
    print("  1. 『y_predばらつき (std_pred)』が小さいほど、モデルが推論を0付近に縮退（平坦化）させていることを示す。")
    print("  2. 『ばらつき比 (pred / true)』が 1.0 より大幅に小さい世代（例: Gen2）ほど、運による振れ幅にモデルが追いついていない。")
    print("  3. 傾き a は 『相関係数 r × ばらつき比』で完全に決定される。")
    print("=" * 85)

if __name__ == "__main__":
    main()
