import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import GPy
from sklearn.preprocessing import StandardScaler
from scipy.stats import ncx2
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
    lam = diff2 / gamma2
    q = ncx2.ppf(1 - p, df=k, nc=lam)
    q = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
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
    return -q


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

def compare_all_methods(df, X_cols, Y_cols, target,
                        methods=('lcb','ei','rand'),
                        init_size=5, trials=50, p=0.05, budget=None):
    """
    1回のBOサイクルで以下すべてを同時に記録：
      - df_hits:  各手法の反復ごとの5%発見割合
      - df_regret:各手法・各目的関数の反復ごとの累積後悔
      - df_picks: 各手法・各試行の反復ごとの選択インデックス
      - df_Y:     各手法・各目的関数の反復ごとの実際のY値
      - df_acq:   各手法の反復ごとの選択された獲得関数値の平均
    """
    X_all = df[X_cols].values
    Y_all = df[Y_cols].values
    N     = len(df)
    if budget is None:
        budget = N - init_size
    n_obj = len(Y_cols)
    top_n = int(np.ceil(N * 0.05))

    # “真”の top-5% インデックス
    scaler_g  = StandardScaler().fit(Y_all)
    Y_s_all   = scaler_g.transform(Y_all)
    t_s_all   = scaler_g.transform(target.reshape(1,-1)).flatten()
    top_idx   = set(np.argsort(np.linalg.norm(Y_s_all - t_s_all, axis=1))[:top_n])

    # 結果格納用
    hit_hist    = {m: np.zeros((trials, budget))         for m in methods}
    regret_hist = {m: np.zeros((trials, budget, n_obj))  for m in methods}
    picks_idx   = {m: np.zeros((trials, budget), dtype=int) for m in methods}
    picks_Y     = {m: np.zeros((trials, budget, n_obj))  for m in methods}
    acq_hist    = {m: np.zeros((trials, budget))         for m in methods}

    # trials × iterations ループ
    for run in trange(trials, desc="Trials"):
        rng  = np.random.RandomState(run+1)
        perm = rng.permutation(N)
        train_base = perm[:init_size].tolist()
        cand_base  = perm[init_size:].tolist()

        for m in methods:
            train_idx = train_base.copy()
            cand_idx  = cand_base.copy()
            cum_hits  = sum(i in top_idx for i in train_idx)
            cum_reg   = np.zeros(n_obj)

            # 初期モデル (LCB/EI の場合)
            if m in ('lcb','ei'):
                X_tr   = X_all[train_idx]
                Y_tr   = Y_all[train_idx]
                sc     = StandardScaler().fit(Y_tr)
                Y_tr_s = sc.transform(Y_tr)
                t_s    = sc.transform(target.reshape(1,-1)).flatten()
                models = fit_multi_gpy(X_tr, Y_tr_s)

            for it in range(budget):
                X_cd = X_all[cand_idx]

                # ——— 獲得関数の計算 ———
                if m == 'lcb':
                    mu, var = predict_multi_gpy(models, X_cd)
                    scores  = l2_lcb_per_dim(mu, var, t_s, p)
                elif m == 'ei':
                    mu, var = predict_multi_gpy(models, X_cd)
                    y_min   = np.min(np.sum((Y_tr_s - t_s)**2, axis=1))
                    scores  = l2_ei(mu, var, t_s, y_min)
                else:  # rand
                    scores = None

                # ——— pick と獲得関数値の記録 ———
                if scores is not None:
                    pick = np.argmax(scores)
                    acq_hist[m][run, it] = scores[pick]
                else:
                    pick = rng.randint(len(cand_idx))
                    acq_hist[m][run, it] = np.nan

                idx_pick = cand_idx.pop(pick)
                train_idx.append(idx_pick)

                # 1) 5%ヒット率
                if idx_pick in top_idx:
                    cum_hits += 1
                hit_hist[m][run, it] = cum_hits / top_n

                # 2) 累積後悔
                y_new = Y_all[idx_pick]
                cum_reg += np.abs(y_new - target)
                regret_hist[m][run, it, :] = cum_reg

                # 3) 選択インデックス・Y値
                picks_idx[m][run, it] = idx_pick
                picks_Y[m][run, it, :] = y_new

                # モデル更新 (LCB/EI)
                if m in ('lcb','ei'):
                    X_tr   = X_all[train_idx]
                    Y_tr   = Y_all[train_idx]
                    sc     = StandardScaler().fit(Y_tr)
                    Y_tr_s = sc.transform(Y_tr)
                    t_s    = sc.transform(target.reshape(1,-1)).flatten()
                    for obj_idx, mdl in enumerate(models):
                        # Y_tr_s[:, [obj_idx]] の shape は (n_train,1)
                        mdl.set_XY(X_tr, Y_tr_s[:, [obj_idx]])
                        mdl.optimize(messages=False, max_iters=5)
                        # print(mdl)
                        

    # DataFrame 化
    df_hits = pd.DataFrame(
        {m: hit_hist[m].mean(axis=0) for m in methods},
        index=np.arange(1, budget+1))
    df_hits.index.name = 'iteration'

    df_regret = {
        m: pd.DataFrame(
             regret_hist[m].mean(axis=0),
             columns=Y_cols,
             index=np.arange(1, budget+1)
           )
        for m in methods
    }

    df_picks = {
        m: pd.DataFrame(
             picks_idx[m],
             index=[f"run{r+1}" for r in range(trials)],
             columns=[f"iter{t+1}" for t in range(budget)]
           )
        for m in methods
    }

    df_Y = {
        m: {
            col: pd.DataFrame(
                     picks_Y[m][:, :, i],
                     index=[f"run{r+1}" for r in range(trials)],
                     columns=[f"iter{t+1}" for t in range(budget)]
                 )
            for i, col in enumerate(Y_cols)
        }
        for m in methods
    }

    df_acq = pd.DataFrame(
        {m: acq_hist[m].mean(axis=0) for m in methods},
        index=np.arange(1, budget+1))
    df_acq.index.name = 'iteration'

    return df_hits, df_regret, df_picks, df_Y, df_acq

# --- プロット関数 ---

def plot_hits(df_hits):
    plt.figure(figsize=(8,5))
    for m in df_hits.columns:
        plt.plot(df_hits.index, df_hits[m], label=m.upper())
    plt.xlabel("Iteration")
    plt.ylabel("Fraction of Top-5% Found")
    plt.title("5% Discovery Fraction")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_avg_target_regret_per_objective(df_regret: dict,
                                         methods: list,
                                         Y_cols: list):
    """
    df_regret: compare_all_methods が返す
               { method: DataFrame(iteration × objectives) }
    methods:   ['lcb','ei','rand'] など
    Y_cols:    目的関数カラムのリスト
    """
    for col in Y_cols:
        plt.figure(figsize=(8,5))
        for m in methods:
            df_m = df_regret[m]
            # 平均ターゲット後悔 = 累積後悔 / iteration
            avg_regret = df_m[col] / df_m.index
            plt.plot(df_m.index, avg_regret, label=m.upper())
        plt.title(f"Average Target Regret per Iteration ({col})")
        plt.xlabel("Iteration")
        plt.ylabel("Mean |y - y*|")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def plot_acquisition(df_acq):
    """
    df_acq: compare_all_methods が返す
            DataFrame(iteration × methods) — 各反復で選ばれた獲得関数値の平均
    """
    plt.figure(figsize=(8,5))
    for m in df_acq.columns:
        plt.plot(df_acq.index, df_acq[m], label=m.upper())
    plt.xlabel("Iteration") 
    plt.ylabel("Acquisition Value")
    plt.title("Mean Chosen Acquisition Value per Iteration")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

