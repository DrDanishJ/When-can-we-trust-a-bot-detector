"""Train-once, reuse-everywhere: fitted models and their splits are cached on disk.

`get_run(cfg, dataset, seed)` returns a `Run` with the feature frame, the feature list, the split indices and
the fitted explain-model (cfg.models.explain_model). Scripts 03-08 all build on these runs, so the expensive
training happens once in 02_train_eval.py (or on first use).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .features.store import load_features, usable_features
from .models.zoo import Model, fit_model
from .splits import make_splits
from .utils import LOG, seed_everything


@dataclass
class Run:
    dataset: str
    seed: int
    df: pd.DataFrame
    features: list[str]
    splits: dict[str, np.ndarray]
    model: Model
    extra: dict = field(default_factory=dict)

    def X(self, part: str) -> np.ndarray:
        return self.df.iloc[self.splits[part]][self.features].to_numpy(dtype=float)

    def y(self, part: str) -> np.ndarray:
        return self.df.iloc[self.splits[part]]["label"].to_numpy(dtype=int)


def model_path(cfg: dict, dataset: str, seed: int, name: str, tag: str = "") -> Path:
    p = Path(cfg["paths"]["cache"]) / "models" / f"{dataset}_{name}_seed{seed}{tag}.joblib"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def train(cfg, df, features, splits, name, seed):
    seed_everything(seed)
    X = df[features].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)
    return fit_model(name, seed, cfg, X[splits["train"]], y[splits["train"]],
                     X[splits["select"]], y[splits["select"]], feature_names=features)


def get_run(cfg: dict, dataset: str, seed: int, name: str | None = None, features: list[str] | None = None,
            tag: str = "", retrain: bool = False) -> Run:
    name = name or cfg["models"]["explain_model"]
    df = load_features(cfg, dataset)
    feats = features or usable_features(df)
    splits = make_splits(df, seed, cfg)
    path = model_path(cfg, dataset, seed, name, tag)
    if path.exists() and not retrain:
        blob = joblib.load(path)
        if blob["features"] == feats:
            return Run(dataset, seed, df, feats, splits, blob["model"])
        LOG.info("feature list changed for %s, retraining", path.name)
    model = train(cfg, df, feats, splits, name, seed)
    joblib.dump({"model": model, "features": feats}, path)
    return Run(dataset, seed, df, feats, splits, model)
