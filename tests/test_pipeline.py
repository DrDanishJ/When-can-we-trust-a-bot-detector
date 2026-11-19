"""Fast unit tests (no real data needed):  pytest -q"""

import numpy as np
import pandas as pd
import pytest

from ibd.config import load_config
from ibd.features import registry
from ibd.features.build import FEATURE_NAMES, build_features
from ibd.io.loaders import load_fox8
from ibd.models.zoo import fit_model
from ibd.reliability.measures import agreement, deletion_auc, deletion_reference, rank_by_attribution
from ibd.splits import make_splits
from ibd.uq.conformal import MondrianConformal, set_summary
from ibd.utils import parse_time


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ibd")
    from make_fox8_fixture import main as make_fixture
    path = tmp / "fox8.ndjson"
    make_fixture(str(path), n_per_class=25)
    return load_config(overrides=[f"paths.fox8={path}", f"paths.cache={tmp / 'cache'}",
                                  "features.n_jobs=1", "models.smote=false"])


def test_parse_time_formats():
    v = parse_time(["Tue Mar 26 14:46:37 +0000 2013", "2015-02-14 10:54:49", "1183552203000L", "12/6/2013 18:38",
                    "", None, "nan"])
    assert v[0] == pytest.approx(1364309197)
    assert v[1] == pytest.approx(1423911289)
    assert v[2] == pytest.approx(1183552203)
    assert not np.isnan(v[3])
    assert v[4:].isna().all()


def test_registry_covers_features():
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))
    assert set(registry.artifact_set()) <= set(FEATURE_NAMES)
    assert all(f.category in registry.CATEGORIES for f in registry.FEATURES)
    assert all(f.source for f in registry.FEATURES)
    assert all(f.new_in_ibd == f.source.startswith("IBD") for f in registry.FEATURES)


def test_fox8_loader_and_features(cfg):
    users, tweets = load_fox8(cfg)
    assert len(users) == 50 and set(users["label"]) == {0, 1}
    assert tweets.groupby("account_id").size().max() <= cfg["data"]["max_tweets_per_account"]
    feats = build_features(users, tweets, cfg)
    assert list(feats.columns[-len(FEATURE_NAMES):]) == FEATURE_NAMES
    # fox8 has tweet timestamps, so timing features must exist; it has no P_updated, so growth features are NaN
    assert feats["inter_tweet_time"].notna().mean() > 0.9
    assert feats["followers_growth"].isna().all()
    assert not np.isinf(feats[FEATURE_NAMES].to_numpy(dtype=float)).any()


def test_splits_are_disjoint():
    df = pd.DataFrame({"label": np.r_[np.zeros(60), np.ones(40)].astype(int), "dataset": "cresci15", "split": ""})
    s = make_splits(df, 0, load_config())
    parts = [set(v) for v in s.values()]
    assert sum(len(p) for p in parts) == len(df)
    assert all(not (a & b) for i, a in enumerate(parts) for b in parts[i + 1:])


def test_conformal_coverage():
    rng = np.random.RandomState(0)
    cov = []
    for _ in range(100):
        y = rng.randint(0, 2, 800)
        p1 = np.clip(0.5 + 0.3 * (2 * y - 1) + rng.normal(0, 0.3, 800), 0.01, 0.99)
        P = np.c_[1 - p1, p1]
        cp = MondrianConformal().fit(P[:400], y[:400], 0.1)
        s = set_summary(cp.predict_set(P[400:]), y[400:])
        cov.append([s["coverage_bot"], s["coverage_human"]])
    assert np.mean(cov, axis=0).min() >= 0.88
    # p-values are valid probabilities
    pv = cp.pvalues(P[400:])
    assert ((pv > 0) & (pv <= 1)).all()


def test_model_and_reliability(cfg):
    users, tweets = load_fox8(cfg)
    df = build_features(users, tweets, cfg)
    feats = [f for f in FEATURE_NAMES if df[f].notna().any() and df[f].nunique() > 1]
    X, y = df[feats].to_numpy(float), df["label"].to_numpy(int)
    m = fit_model("lightgbm", 0, cfg, X, y, feature_names=feats)
    p = m.predict_proba(X)
    assert p.shape == (len(X), 2)
    attr = np.random.RandomState(0).randn(len(X), len(feats))
    d = deletion_auc(m, X, rank_by_attribution(attr), np.nanmedian(X, axis=0), 5)
    assert d.shape == (len(X),) and np.isfinite(d).all()
    # median mode (1-D baseline) is background mode with a single background row
    d1 = deletion_auc(m, X, rank_by_attribution(attr), np.nanmedian(X, axis=0)[None, :], 5)
    assert np.allclose(d, d1)
    # background mode: if every background account equals the explained account, deletion changes nothing
    x0 = X[:1]
    d0 = deletion_auc(m, x0, rank_by_attribution(attr[:1]), np.repeat(x0, 4, axis=0), 5)
    assert np.allclose(d0, 1.0)
    ref = deletion_reference(X, {"reliability": {"deletion_baseline": "background", "deletion_background": 3}})
    assert ref.shape == (3, X.shape[1])
    sla, tau = agreement(attr, attr, 5)
    assert np.allclose(sla, 1.0)
