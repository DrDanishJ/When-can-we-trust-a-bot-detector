"""Extensions of the shortcut audit (paper Sections 3.5 and 5.3).

- artifact_sets: the registry set A and two wider sets that also contain features whose values can carry the
  collection date indirectly (A+counts: raw cumulative counts; A+counts+period: also writing-style features
  that can describe the period in which each group was collected).
- joint_permutation: dependence of a *fixed* fitted model on a feature set, measured by permuting all features
  of the set jointly (rows shuffled together, so the correlation inside the set is kept) on the test split.
  Unlike the attribution share (SDR), this does not depend on how TreeSHAP divides credit among correlated
  features.
- age_matched_sample: Cresci-17 humans and bots matched on account age at crawl time (coarsened exact
  matching on quantile bins of age at crawl, equal numbers of humans and bots per bin).
- dedup: Cresci-15 bots are the Cresci-17 fake_followers group; cross-dataset tests remove that overlap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.registry import artifact_set
from ..metrics import classification_metrics

COUNTS = ["statuses_count", "favourites_count", "followers_count", "friends_count", "listed_count"]
PERIOD = ["emoji_rate", "avg_urls", "capitalization_rate"]


def artifact_sets(cfg) -> dict[str, list[str]]:
    A = artifact_set(cfg)
    return {"A": A, "A+counts": A + COUNTS, "A+counts+period": A + COUNTS + PERIOD}


def joint_permutation(model, X: np.ndarray, y: np.ndarray, features: list[str], subset: list[str],
                      n_rep: int, seed: int) -> dict:
    cols = [i for i, f in enumerate(features) if f in set(subset)]
    base = classification_metrics(y, model.predict_proba(X)[:, 1])
    rng = np.random.default_rng(seed)
    d_f1, d_auc = [], []
    for _ in range(n_rep):
        Xp = X.copy()
        perm = rng.permutation(len(X))
        Xp[:, cols] = X[perm][:, cols]
        m = classification_metrics(y, model.predict_proba(Xp)[:, 1])
        d_f1.append(m["f1"] - base["f1"]); d_auc.append(m["auc"] - base["auc"])
    return {"n_permuted": len(cols), "delta_F1": float(np.mean(d_f1)), "delta_AUC": float(np.mean(d_auc))}


def age_matched_sample(df: pd.DataFrame, age: pd.Series, n_bins: int, seed: int) -> pd.DataFrame:
    """Coarsened exact matching: quantile bins of age at crawl; per bin keep min(n_human, n_bot) of each class."""
    d = df.assign(_age=age.to_numpy()).dropna(subset=["_age"])
    d["_bin"] = pd.qcut(d["_age"], n_bins, labels=False, duplicates="drop")
    rng = np.random.default_rng(seed)
    keep = []
    for _, g in d.groupby("_bin"):
        h, b = g.index[g["label"] == 0], g.index[g["label"] == 1]
        k = min(len(h), len(b))
        if k:
            keep += list(rng.choice(h, k, replace=False)) + list(rng.choice(b, k, replace=False))
    return d.loc[sorted(keep)].drop(columns=["_bin"])
