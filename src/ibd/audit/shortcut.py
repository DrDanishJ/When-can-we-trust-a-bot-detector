"""Shortcut audit (paper Section 4.6): Shortcut Dependence Ratio, retraining without artifact-prone features,
cross-dataset transfer and feature-category ablation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.registry import artifact_set
from ..features.store import load_features, usable_features
from ..metrics import classification_metrics
from ..models.zoo import fit_model
from ..splits import make_splits
from ..utils import seed_everything


def sdr(shap_vals: np.ndarray, features: list[str], A: list[str]) -> float:
    """Eq. 38: share of total mean |SHAP| that falls on the artifact-prone set A."""
    m = np.abs(shap_vals).mean(axis=0)
    mask = np.array([f in set(A) for f in features])
    return float(m[mask].sum() / max(m.sum(), 1e-12))


def eval_in_dataset(cfg, df, features, seed, name=None):
    """Train on the train split, evaluate on the test split; returns (metrics, model)."""
    name = name or cfg["models"]["explain_model"]
    seed_everything(seed)
    s = make_splits(df, seed, cfg)
    X, y = df[features].to_numpy(float), df["label"].to_numpy(int)
    m = fit_model(name, seed, cfg, X[s["train"]], y[s["train"]], X[s["select"]], y[s["select"]], features)
    return classification_metrics(y[s["test"]], m.predict_proba(X[s["test"]])[:, 1]), m


def common_features(cfg, a: str, b: str, drop_artifacts: bool = False) -> list[str]:
    fa = usable_features(load_features(cfg, a))
    fb = set(usable_features(load_features(cfg, b)))
    feats = [f for f in fa if f in fb]
    if drop_artifacts:
        A = set(artifact_set(cfg))
        feats = [f for f in feats if f not in A]
    return feats


def test_frame(cfg, dataset: str, seed: int) -> pd.DataFrame:
    """Accounts used as the test population for `dataset`: its test split if it has one, else everything."""
    df = load_features(cfg, dataset)
    if dataset in cfg["datasets"]["train_test"]:
        return df.iloc[make_splits(df, seed, cfg)["test"]]
    return df


def cross_eval(cfg, train_ds: str, test_ds: str, seed: int, features: list[str], name=None):
    name = name or cfg["models"]["explain_model"]
    seed_everything(seed)
    tr = load_features(cfg, train_ds)
    s = make_splits(tr, seed, cfg)
    X, y = tr[features].to_numpy(float), tr["label"].to_numpy(int)
    m = fit_model(name, seed, cfg, X[s["train"]], y[s["train"]], X[s["select"]], y[s["select"]], features)
    te = test_frame(cfg, test_ds, seed)
    p = m.predict_proba(te[features].to_numpy(float))[:, 1]
    return classification_metrics(te["label"].to_numpy(int), p), m, te, p
