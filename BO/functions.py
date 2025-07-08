import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import GPy
from sklearn.preprocessing import StandardScaler
from scipy.stats import ncx2
from scipy.stats import norm
from tqdm import trange, tqdm
import warnings


warnings.filterwarnings("ignore")

def l2_lcb_exact(means: np.ndarray,
                 variances: np.ndarray,
                 target: np.ndarray,
                 p: float = 0.05,
                 eps: float = 1e-8) -> np.ndarray:
    """
    L2-Lower Confidence Bound for multi-output GP via exact non-central chi-square.
    """
    k = means.shape[1]
    gamma2 = np.maximum(variances.mean(axis=1), eps)
    diff2 = np.sum((means - target)**2, axis=1)
    # print(diff2)
    lam = diff2 / gamma2
    q = ncx2.ppf(1 - p, df=k, nc=lam)
    # print(f"p: {p}, k: {k}, gamma2: {gamma2.mean()}, diff2: {diff2.mean()}, lam: {lam.mean()}, q: {q.mean()}")
    return - q * gamma2

def l2_lcb_per_dim(means: np.ndarray,
                   variances: np.ndarray,
                   target: np.ndarray,
                   p: float = 0.05,
                   eps: float = 1e-8) -> np.ndarray:

    # 1) prevent zero variances
    vars_safe = np.maximum(variances, eps)
    # 2) compute non-centrality λ for each candidate
    lam = np.sum((means - target)**2 / vars_safe, axis=1)
    # 3) inverse survival function: P(X ≥ q) = p for χ²ₖ(λ)
    q   = ncx2.isf(p, df=means.shape[1], nc=lam)
    # 4) score = −q (we pick argmax scores ⇒ minimize q)
    # print(variances)
    return -q

def l2_lcb_per_dim_euclid(
    means: np.ndarray,      # shape = (n_cand, n_obj)
    variances: np.ndarray,  # shape = (n_cand, n_obj)
    target: np.ndarray,     # shape = (n_obj,)
    p: float = 0.05,
    eps: float = 1e-8
) -> np.ndarray:
    """
    各候補点 i, 各目的関数 j について
      • lam_ij = (μ_ij – target_j)^2 / var_ij
      • q_ij   = χ²₁(non-central λ=lam_ij) の上側 p% 点
      • d2_ij  = q_ij * var_ij  （「ズレ²」の worst‐case 上限推定）
      • d_ij   = sqrt(d2_ij)    （「ズレ」の worst‐case 上限推定）
    → 最後に各 i についてベクトル d_i を L₂ ノルムでまとめ、
      小さいほど良いスコア −dist_i を返す。
    """
    # 1) ゼロ分散回避
    vars_safe = np.maximum(variances, eps)
    #   → 分散が 0 に近いとき数値が爆発するのを防ぐ

    # 2) non-centrality λ の計算
    lam = (means - target[np.newaxis, :])**2 / vars_safe
    #   lam_ij = ((μ_ij – target_j)^2) / var_ij

    # 3) df=1 の非心χ² 上側 p% 点 q_ij を求める
    q = ncx2.isf(p, df=1, nc=lam)
    #   → P(X ≥ q_ij) = p を満たす閾値 q_ij

    # 4) 「ズレ²」の worst‐case 上限推定 d2_ij = q_ij * var_ij
    d2 = q * vars_safe

    # 5) 「ズレ」の worst‐case 上限推定 d_ij = sqrt(d2_ij)
    d = np.sqrt(d2)

    # 6) 各候補 i のベクトル d_i の L2 ノルム dist_i を計算
    dist = np.linalg.norm(d, axis=1)
    #   → 次元ごとの worst‐case 距離を合成して「全体の worst‐case 距離」を得る
    # print(dist)
    # 7) 獲得関数用に、小さい距離を良しとするようマイナス符号をつけて返す
    return -dist


def l2_ei(means: np.ndarray,
          variances: np.ndarray,
          target: np.ndarray,
          y_min: float,
          eps: float = 1e-8) -> np.ndarray:
    """
    L2-Expected Improvement for multi-output GP via non-central chi-square.
    """
    k = means.shape[1]
    gamma2 = np.maximum(variances.mean(axis=1), eps)
    diff2 = np.sum((means - target)**2, axis=1)
    nc = diff2 / gamma2
    x = y_min / gamma2
    t1 = y_min * ncx2.cdf(x, df=k, nc=nc)
    t2 = gamma2 * (
        k   * ncx2.cdf(x, df=k+2, nc=nc)
      + nc  * ncx2.cdf(x, df=k+4, nc=nc)
    )
    return np.maximum(0, t1 - t2)




def fit_multi_gpy(X: np.ndarray, Y: np.ndarray):
    """各出力次元ごとに GPy の GPRegression をフィッティングしてリストで返す。"""
    models = []
    for i in range(Y.shape[1]):
        kern = GPy.kern.Exponential(input_dim=X.shape[1], ARD=False)
        m = GPy.models.GPRegression(X, Y[:, [i]], kern)
        # m.Gaussian_noise.variance = 1e-6
        # m.Gaussian_noise.variance.fix()
        m.optimize(messages=False,max_iters=5)
        models.append(m)
        # print(m)
    return models

def predict_multi_gpy(models, X: np.ndarray):
    """リスト化した GPRegression モデル群で予測し、平均と分散の配列を返す。"""
    mus, vars_ = [], []
    for m in models:
        mu, var = m.predict(X)
        mus.append(mu.flatten())
        vars_.append(var.flatten())
    return np.column_stack(mus), np.column_stack(vars_)

def export_sorted_distances_to_csv(Y: np.ndarray,
                                   target: np.ndarray,
                                   filename: str = "/Users/macmini/Documents/distances_to_target.csv"):
    """
    全候補点 Y と target のユークリッド距離を計算し、
    距離が小さい順にソートした結果を CSV に書き出す。
    """
    dists = np.linalg.norm(Y - target.reshape(1, -1), axis=1)
    order = np.argsort(dists)
    df_out = pd.DataFrame({
        "index": order,
        "distance": dists[order]
    })
    df_out.to_csv(filename, index=False)
    print(f"Sorted distances saved to {filename}")
