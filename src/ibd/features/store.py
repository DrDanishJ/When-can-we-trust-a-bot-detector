"""Build-once, read-many feature cache (one parquet file per dataset in paths.cache)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..io import loaders
from ..utils import LOG, timer
from .build import build_features
from .registry import FEATURE_NAMES


def _english_only(tweets: pd.DataFrame) -> pd.DataFrame:
    try:
        from langdetect import DetectorFactory, detect
        DetectorFactory.seed = 0
    except ImportError as e:
        raise ImportError("data.english_only=true needs `pip install langdetect`") from e

    def is_en(t):
        try:
            return detect(t) == "en"
        except Exception:
            return False
    keep = tweets["text"].map(is_en)
    LOG.info("  english_only: kept %.1f%% of tweets", 100 * keep.mean())
    return tweets[keep]


def cache_path(cfg: dict, dataset: str) -> Path:
    return Path(cfg["paths"]["cache"]) / f"features_{dataset}.parquet"


def build_dataset(cfg: dict, dataset: str) -> pd.DataFrame:
    with timer(f"load {dataset}"):
        if dataset == "lsabre":
            tu, tt = loaders.load_twibot20(cfg)
            users, tweets = loaders.load_lsabre(cfg, tu, tt)
        else:
            users, tweets = loaders.LOADERS[dataset](cfg)
    users = users.reset_index(drop=True)
    if cfg["data"].get("english_only"):
        tweets = _english_only(tweets)
    with timer(f"features {dataset} ({len(users)} accounts, {len(tweets)} tweets)"):
        feats = build_features(users, tweets, cfg)
    out = cache_path(cfg, dataset)
    out.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(out, index=False)
    LOG.info("wrote %s", out)
    return feats


def load_features(cfg: dict, dataset: str) -> pd.DataFrame:
    p = cache_path(cfg, dataset)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found - run scripts/01_build_features.py first")
    return pd.read_parquet(p)


def availability_table(cfg: dict, datasets: list[str]) -> pd.DataFrame:
    """One row per registered feature: name, category, source, new_in_ibd, artifact_prone, description, and one 0/1
    column per dataset (1 = usable there, see usable_features). Datasets without a feature cache are skipped."""
    from .registry import table
    out = table()[["name", "category", "source", "new_in_ibd", "artifact_prone", "description"]].copy()
    for d in datasets:
        if not cache_path(cfg, d).exists():
            LOG.warning("availability: no feature cache for %s, column skipped", d)
            continue
        usable = set(usable_features(load_features(cfg, d)))
        out[d] = out["name"].isin(usable).astype(int)
    return out


def usable_features(df: pd.DataFrame, candidates=None, max_missing: float = 0.9) -> list[str]:
    """Features with some variation and less than `max_missing` missing values in this frame."""
    out = []
    for f in candidates or FEATURE_NAMES:
        col = df[f]
        if col.isna().mean() > max_missing or col.nunique(dropna=True) <= 1:
            continue
        out.append(f)
    return out
