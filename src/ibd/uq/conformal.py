"""Class-conditional (Mondrian) split conformal prediction for binary bot detection (paper eqs. 39-41)."""
from __future__ import annotations

import numpy as np


class MondrianConformal:
    """Nonconformity s(x, y) = 1 - p_y(x). Thresholds are computed per class on the calibration split, which
    gives coverage >= 1 - alpha separately for humans and for bots."""

    def fit(self, p_cal: np.ndarray, y_cal: np.ndarray, alpha: float):
        p_cal, y_cal = np.asarray(p_cal), np.asarray(y_cal).astype(int)
        self.alpha = alpha
        self.cal_scores = {}
        self.q = np.zeros(2)
        for c in (0, 1):
            s = np.sort(1.0 - p_cal[y_cal == c, c])
            self.cal_scores[c] = s
            n = len(s)
            if n == 0:
                self.q[c] = 1.0
                continue
            rank = int(np.ceil((n + 1) * (1 - alpha)))
            # with too few calibration points the finite-sample quantile is +inf: always include the label
            self.q[c] = 1.0 if rank > n else s[rank - 1]
        return self

    def scores(self, p: np.ndarray) -> np.ndarray:
        return 1.0 - np.asarray(p)

    def predict_set(self, p: np.ndarray) -> np.ndarray:
        """Boolean (n, 2): column c is True when label c is in the prediction set (eq. 40)."""
        return self.scores(p) <= self.q[None, :]

    def pvalues(self, p: np.ndarray) -> np.ndarray:
        """Conformal p-value of each label (eq. 41), shape (n, 2)."""
        s = self.scores(p)
        out = np.zeros_like(s, dtype=float)
        for c in (0, 1):
            cal = self.cal_scores[c]
            n = len(cal)
            # number of calibration scores >= s
            ge = n - np.searchsorted(cal, s[:, c], side="left")
            out[:, c] = (ge + 1) / (n + 1)
        return out


def set_summary(sets: np.ndarray, y: np.ndarray) -> dict:
    y = np.asarray(y).astype(int)
    size = sets.sum(axis=1)
    covered = sets[np.arange(len(y)), y]
    return {
        "coverage_bot": float(covered[y == 1].mean()) if (y == 1).any() else np.nan,
        "coverage_human": float(covered[y == 0].mean()) if (y == 0).any() else np.nan,
        "coverage": float(covered.mean()),
        "mean_set_size": float(size.mean()),
        "singleton_rate": float((size == 1).mean()),
        "ambiguous_rate": float((size == 2).mean()),
        "empty_rate": float((size == 0).mean()),
    }


def set_type(sets: np.ndarray) -> np.ndarray:
    """'bot', 'human', 'ambiguous' or 'empty' per account."""
    size = sets.sum(axis=1)
    out = np.where(size == 2, "ambiguous", np.where(size == 0, "empty", np.where(sets[:, 1], "bot", "human")))
    return out
