"""Explanation reliability (paper eqs. 42-45): deletion AUC, remove-and-retrain, bootstrap stability and
SHAP-LIME agreement."""
from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau, spearmanr

from ..models.zoo import Model


def deletion_reference(X_train: np.ndarray, cfg: dict, seed: int = 0) -> np.ndarray:
    """What a "deleted" feature is replaced with in eq. 42 (cfg.reliability.deletion_baseline):
      background (default) - M = cfg.reliability.deletion_background random training accounts, shape (M, d);
      median               - the training median of each feature, shape (d,)."""
    rc = cfg.get("reliability", {})
    mode = rc.get("deletion_baseline", "background")
    X_train = np.asarray(X_train, dtype=float)
    if mode == "median":
        return np.nan_to_num(np.nanmedian(X_train, axis=0))
    if mode == "background":
        M = min(int(rc.get("deletion_background", 20)), len(X_train))
        return X_train[np.random.RandomState(seed).choice(len(X_train), M, replace=False)]
    raise ValueError(f"unknown reliability.deletion_baseline: {mode}")


def deletion_auc(model: Model, X: np.ndarray, rankings: np.ndarray, baseline: np.ndarray, K: int) -> np.ndarray:
    """Eq. 42. For each account, delete its top-k features (k = 0..K, order given by `rankings`, most important
    first) and average the predicted probability of the originally predicted class, normalised by the probability
    at k = 0. Lower = more faithful explanation.

    `baseline` (from deletion_reference) defines deletion:
      shape (M, d) - background mode: the deleted features take the values of each of M background accounts (all
                     deleted features from the same account), and the probability is averaged over the M draws;
      shape (d,)   - median mode: the deleted features take the training median."""
    X = np.asarray(X, dtype=float)
    B = np.asarray(baseline, dtype=float)
    B = B[None, :] if B.ndim == 1 else B
    n, d = X.shape
    M = B.shape[0]
    K = min(K, d)
    p0 = model.predict_proba(X)
    pred = p0.argmax(axis=1)
    base = p0[np.arange(n), pred]
    acc = np.ones(n)
    Xk = np.repeat(X[:, None, :], M, axis=1)  # (n, M, d): one copy of each account per background draw
    rows, draws = np.arange(n)[:, None], np.arange(M)[None, :]
    pred_rep = np.repeat(pred, M)
    for k in range(1, K + 1):
        cols = rankings[:, k - 1][:, None]
        Xk[rows, draws, cols] = B[draws, cols]
        p = model.predict_proba(Xk.reshape(n * M, d))[np.arange(n * M), pred_rep].reshape(n, M).mean(axis=1)
        acc += p / np.maximum(base, 1e-12)
    return acc / (K + 1)


def rank_by_attribution(attr: np.ndarray, towards_prediction: np.ndarray | None = None) -> np.ndarray:
    """Feature order per account. If `towards_prediction` (+1 bot / -1 human) is given, features are ranked by
    their signed support for the predicted class; otherwise by |attribution|."""
    if towards_prediction is None:
        score = np.abs(attr)
    else:
        score = attr * np.asarray(towards_prediction)[:, None]
    return np.argsort(-score, axis=1)


def random_rankings(n: int, d: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return np.array([rng.permutation(d) for _ in range(n)])


def stability_index(shap_runs: list[np.ndarray]) -> np.ndarray:
    """Eq. 44. Mean pairwise Spearman correlation of each account's SHAP vector across bootstrap models."""
    B = len(shap_runs)
    n = shap_runs[0].shape[0]
    out = np.zeros(n)
    pairs = [(a, b) for a in range(B) for b in range(a + 1, B)]
    for i in range(n):
        vals = []
        for a, b in pairs:
            r = spearmanr(shap_runs[a][i], shap_runs[b][i]).statistic
            vals.append(0.0 if np.isnan(r) else r)
        out[i] = np.mean(vals) if vals else np.nan
    return out


def agreement(shap_attr: np.ndarray, lime_attr: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Eq. 45 top-k overlap (SLA) and Kendall tau between |SHAP| and |LIME| rankings, per account."""
    n = shap_attr.shape[0]
    sla, tau = np.zeros(n), np.zeros(n)
    for i in range(n):
        a, b = np.abs(shap_attr[i]), np.abs(lime_attr[i])
        ta, tb = set(np.argsort(-a)[:k]), set(np.argsort(-b)[:k])
        sla[i] = len(ta & tb) / k
        t = kendalltau(a, b).statistic
        tau[i] = 0.0 if np.isnan(t) else t
    return sla, tau
