"""Classifier zoo. Every model is wrapped in `Model`, which owns its preprocessing so callers always pass raw
feature matrices (with NaNs) and get calibrated-or-not probabilities back.

Leakage rules enforced here:
  * imputation/scaling are fitted on the training split only;
  * SMOTE is applied to the training split only (inside fit);
  * MLP uses an internal validation split of the *training* data for early stopping;
  * TabNet early-stops on the model-selection split passed explicitly as `X_sel`, never on the test set.
"""
from __future__ import annotations

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from ..utils import LOG

TREE_MODELS = {"rf", "et", "dt", "xgboost", "lightgbm", "catboost"}
SCALED_MODELS = {"svm", "mlp", "tabnet"}


def _make(name: str, seed: int, cfg: dict):
    """Build an unfitted classifier. Defaults below; cfg.models.params.<name> overrides any of them."""
    kw = dict((cfg or {}).get("models", {}).get("params", {}).get(name, {}) or {})
    est = _default(name, seed)
    if kw:
        est.set_params(**kw)
    return est


def _default(name: str, seed: int):
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed)
    if name == "et":
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(n_estimators=300, n_jobs=-1, random_state=seed)
    if name == "dt":
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(random_state=seed)
    if name == "svm":
        from sklearn.svm import SVC
        return SVC(probability=True, random_state=seed)
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6, eval_metric="logloss",
                             random_state=seed, n_jobs=-1, verbosity=0)
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=seed, verbose=-1, n_jobs=-1)
    if name == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=500, random_seed=seed, verbose=0, thread_count=-1)
    if name == "mlp":
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier(hidden_layer_sizes=(256, 128, 64, 32), early_stopping=True, validation_fraction=0.1,
                             n_iter_no_change=15, max_iter=400, random_state=seed)
    if name == "ebm":
        from interpret.glassbox import ExplainableBoostingClassifier
        return ExplainableBoostingClassifier(random_state=seed, interactions=10, outer_bags=8, max_rounds=5000)
    if name == "tabnet":
        from pytorch_tabnet.tab_model import TabNetClassifier  # optional: pip install pytorch-tabnet torch
        return TabNetClassifier(seed=seed, verbose=0)
    raise ValueError(f"unknown classifier {name}")


def available(names: list[str]) -> list[str]:
    out = []
    for n in names:
        try:
            _make(n, 0, {})
            out.append(n)
        except ImportError:
            LOG.warning("classifier %s skipped (library not installed)", n)
    return out


class Model:
    """Preprocessing + classifier. `predict_proba` takes raw features (NaNs allowed)."""

    def __init__(self, name: str, seed: int, cfg: dict):
        self.name, self.seed, self.cfg = name, seed, cfg
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        self.scaler = StandardScaler() if name in SCALED_MODELS else None
        self.clf = _make(name, seed, cfg)
        self.feature_names: list[str] = []

    # preprocessing ------------------------------------------------------------------------------------------
    def transform(self, X) -> np.ndarray:
        Z = self.imputer.transform(np.asarray(X, dtype=float))
        if self.scaler is not None:
            Z = self.scaler.transform(Z)
        return Z.astype(np.float32 if self.name == "tabnet" else np.float64)

    def fit(self, X, y, X_sel=None, y_sel=None, feature_names=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)
        self.feature_names = list(feature_names) if feature_names is not None else [f"f{i}" for i in range(X.shape[1])]
        self.imputer.fit(X)
        Z = self.imputer.transform(X)
        if self.scaler is not None:
            Z = self.scaler.fit_transform(Z)
        rng = np.random.RandomState(self.seed)
        if self.cfg.get("models", {}).get("smote", True):
            Z, y = _smote(Z, y, self.seed)
        if self.name == "svm" and len(y) > self.cfg["models"].get("svm_max_train", 20000):
            idx = rng.choice(len(y), self.cfg["models"]["svm_max_train"], replace=False)
            Z, y = Z[idx], y[idx]
        if self.name == "tabnet":
            Zs = self.transform(X_sel) if X_sel is not None else None
            kw = dict(max_epochs=self.cfg["models"].get("tabnet_max_epochs", 200), patience=20,
                      batch_size=256, virtual_batch_size=128)
            if Zs is not None:
                kw.update(eval_set=[(Zs, np.asarray(y_sel).astype(int))], eval_name=["select"], eval_metric=["logloss"])
            self.clf.fit(Z.astype(np.float32), y, **kw)
        else:
            self.clf.fit(Z, y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.clf.predict_proba(self.transform(X))

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    @property
    def is_tree(self) -> bool:
        return self.name in TREE_MODELS


def _smote(Z, y, seed):
    counts = np.bincount(y, minlength=2)
    if counts.min() < 2 or counts.min() == counts.max():
        return Z, y
    from imblearn.over_sampling import SMOTE
    k = int(min(5, counts.min() - 1))
    return SMOTE(random_state=seed, k_neighbors=k).fit_resample(Z, y)


def fit_model(name, seed, cfg, X_tr, y_tr, X_sel=None, y_sel=None, feature_names=None) -> Model:
    return Model(name, seed, cfg).fit(X_tr, y_tr, X_sel, y_sel, feature_names)
