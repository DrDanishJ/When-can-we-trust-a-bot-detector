"""Checks of the Mondrian conformal layer (paper Section 5.4).

coverage_resampling: keeps each fitted model fixed and repeatedly re-partitions the pooled calibration + test
accounts into a calibration and a test part of the original sizes (stratified by class). Split conformal
guarantees E[coverage] >= 1 - alpha over such partitions, so the mean over many partitions tests the
implementation, while a single partition (as in Table 11) varies around it.
set_type_accuracy: accuracy of the point prediction (argmax) within each prediction-set type.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..uq.conformal import MondrianConformal, set_type


def coverage_resampling(p_cal, y_cal, p_te, y_te, alphas, n_rep, seed):
    P = np.vstack([p_cal, p_te]); Y = np.concatenate([y_cal, y_te]).astype(int)
    rng = np.random.default_rng(seed)
    idx_c = {c: np.where(Y == c)[0] for c in (0, 1)}
    ncal = {c: int((y_cal == c).sum()) for c in (0, 1)}
    rows = []
    for r in range(n_rep):
        cal, te = [], []
        for c in (0, 1):
            perm = rng.permutation(idx_c[c])
            cal.append(perm[:ncal[c]]); te.append(perm[ncal[c]:])
        cal, te = np.concatenate(cal), np.concatenate(te)
        for a in alphas:
            sets = MondrianConformal().fit(P[cal], Y[cal], a).predict_set(P[te])
            cov = sets[np.arange(len(te)), Y[te]]
            rows.append({"rep": r, "alpha": a,
                         "coverage_human": cov[Y[te] == 0].mean(), "coverage_bot": cov[Y[te] == 1].mean()})
    return pd.DataFrame(rows)


def set_type_accuracy(p_cal, y_cal, p_te, y_te, alpha):
    cp = MondrianConformal().fit(p_cal, y_cal, alpha)
    st = set_type(cp.predict_set(p_te))
    pred = p_te.argmax(1); correct = pred == np.asarray(y_te).astype(int)
    out = []
    for t in ("bot", "human", "ambiguous", "empty"):
        m = st == t
        out.append({"set_type": t, "n": int(m.sum()), "share": float(m.mean()),
                    "accuracy": float(correct[m].mean()) if m.any() else np.nan,
                    "mean_max_p": float(p_te[m].max(1).mean()) if m.any() else np.nan})
    return pd.DataFrame(out), cp.q
