"""Classification and calibration metrics."""
from __future__ import annotations

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score


def classification_metrics(y, p_bot, threshold: float = 0.5) -> dict:
    y = np.asarray(y).astype(int)
    p_bot = np.asarray(p_bot, dtype=float)
    pred = (p_bot >= threshold).astype(int)
    out = {
        "acc": accuracy_score(y, pred),
        "prec": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "auc": roc_auc_score(y, p_bot) if len(np.unique(y)) == 2 else np.nan,
        "brier": brier_score_loss(y, p_bot) if len(np.unique(y)) == 2 else np.nan,
        "ece": expected_calibration_error(y, p_bot),
    }
    return out


def expected_calibration_error(y, p_bot, bins: int = 10) -> float:
    """ECE of the predicted class probability (confidence) against accuracy."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p_bot, dtype=float)
    conf = np.maximum(p, 1 - p)
    correct = ((p >= 0.5).astype(int) == y).astype(float)
    edges = np.linspace(0.5, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf <= hi if hi == 1.0 else conf < hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def reliability_curve(y, p_bot, bins: int = 10):
    y = np.asarray(y).astype(int)
    p = np.asarray(p_bot, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    mids, frac, count = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p <= hi) if hi == 1 else (p < hi))
        if m.any():
            mids.append(p[m].mean())
            frac.append(y[m].mean())
            count.append(int(m.sum()))
    return np.array(mids), np.array(frac), np.array(count)


def mcnemar_p(y, pred_a, pred_b) -> float:
    """Exact McNemar test on two classifiers' test-set predictions."""
    y = np.asarray(y)
    a_ok = np.asarray(pred_a) == y
    b_ok = np.asarray(pred_b) == y
    n01 = int((a_ok & ~b_ok).sum())
    n10 = int((~a_ok & b_ok).sum())
    if n01 + n10 == 0:
        return 1.0
    return float(binomtest(n01, n01 + n10, 0.5).pvalue)
