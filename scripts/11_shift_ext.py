"""Step 11 - paper Section 5.6 (Table 13 reference row): distribution shift, recalibration and LSABRE.

  tables/ext_shift_reference.csv   in-dataset DAUC / SLA of the Cresci-17 model trained on the features it shares
                                   with TwiBot-20 or fox8-23 (the reference the shifted Cresci-17 rows need)
  tables/ext_recalibration.csv     conformal layer recalibrated on n labelled target accounts (n in N_TARGET,
                                   10 draws per seed): per-class coverage, ambiguous rate, singleton accuracy
  tables/ext_lsabre_text_only.csv  TwiBot-20 model trained only on tweet-derived features, tested in-dataset and on
                                   every LSABRE variant (bots of the variant + the same 543 TwiBot-20 test humans)
"""
import _bootstrap  # noqa: F401
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from ibd.audit.shortcut import common_features, cross_eval
from ibd.config import cli
from ibd.features.registry import BY_NAME
from ibd.features.store import load_features, usable_features
from ibd.metrics import classification_metrics
from ibd.splits import make_splits
from ibd.uq.conformal import MondrianConformal, set_summary
from ibd.utils import LOG, mean_std, write_table

spec = importlib.util.spec_from_file_location("s07", Path(__file__).with_name("07_shift_llm.py"))
s07 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s07)

N_TARGET = [50, 100, 200, 500]
TWEET_ENGAGEMENT = ["reply_diversity", "avg_engagement_received", "retweet_ratio", "reply_ratio", "avg_hashtags",
                    "avg_mentions", "avg_urls", "hashtag_diversity"]
TEXT_CATS = {"content", "linguistic", "readability", "sentiment"}


def text_features(feats):
    return [f for f in feats if (BY_NAME[f].category in TEXT_CATS and f != "n_tweets_collected")
            or f in TWEET_ENGAGEMENT]


def target_pools(cfg, ds, seed):
    """(calibration pool, test frame) for a target dataset."""
    df = load_features(cfg, ds)
    if ds in cfg["datasets"]["train_test"]:
        s = make_splits(df, seed, cfg)
        return df.iloc[s["calib"]], df.iloc[s["test"]]
    rng = np.random.default_rng(500 + seed)
    idx = rng.permutation(len(df))
    return df.iloc[idx[:max(N_TARGET)]], df.iloc[idx[max(N_TARGET):]]


def main():
    cfg, _ = cli(__doc__)
    alpha = cfg["conformal"]["alphas"][0]
    seeds = cfg["models"]["seeds"]

    # ---- (1) in-dataset reference for the Cresci-17 model on shared features
    rows = []
    for other in ("twibot20", "fox8"):
        feats = common_features(cfg, "cresci17", other)
        lst = []
        for seed in seeds:
            _, model, te, _ = cross_eval(cfg, "cresci17", "cresci17", seed, feats)
            lst.append(s07.evaluate(cfg, model, load_features(cfg, "cresci17"), feats, seed, te, alpha, True))
        d = pd.DataFrame(lst)
        rows.append({"pair": f"cresci17 -> cresci17 (features shared with {other})", "n_features": len(feats),
                     **{c: mean_std(d[c]) for c in ["f1", "recall", "auc", "ambiguous_rate", "coverage", "DAUC", "SLA"]}})
        LOG.info("reference with %s done", other)
    t = pd.DataFrame(rows); write_table(t, cfg, "ext_shift_reference"); print(t.to_string(index=False))

    # ---- (2) recalibration on a small labelled target sample
    rows = []
    for a, b in [("cresci17", "twibot20"), ("cresci17", "fox8"), ("twibot20", "cresci17"), ("twibot20", "fox8")]:
        feats = common_features(cfg, a, b)
        for seed in seeds:
            _, model, _, _ = cross_eval(cfg, a, b, seed, feats)
            src = load_features(cfg, a); s = make_splits(src, seed, cfg)
            pcal_src = model.predict_proba(src.iloc[s["calib"]][feats].to_numpy(float))
            ycal_src = src.iloc[s["calib"]]["label"].to_numpy(int)
            pool, te = target_pools(cfg, b, seed)
            pt, yt = model.predict_proba(te[feats].to_numpy(float)), te["label"].to_numpy(int)
            ok = pt.argmax(1) == yt

            def record(n, draw, sets):
                r = set_summary(sets, yt); single = sets.sum(1) == 1
                rows.append({"train": a, "test": b, "n_target": n, "seed": seed, "draw": draw,
                             "coverage_bot": r["coverage_bot"], "coverage_human": r["coverage_human"],
                             "ambiguous_rate": r["ambiguous_rate"], "empty_rate": r["empty_rate"],
                             "singleton_rate": r["singleton_rate"],
                             "acc_singleton": ok[single].mean() if single.any() else np.nan})
            record(0, 0, MondrianConformal().fit(pcal_src, ycal_src, alpha).predict_set(pt))
            pp, yp = model.predict_proba(pool[feats].to_numpy(float)), pool["label"].to_numpy(int)
            rng = np.random.default_rng(800 + seed)
            for n in N_TARGET:
                for draw in range(10):
                    idx = np.concatenate([rng.choice(np.where(yp == c)[0], min(n // 2, (yp == c).sum()), replace=False)
                                          for c in (0, 1)])
                    record(n, draw, MondrianConformal().fit(pp[idx], yp[idx], alpha).predict_set(pt))
        LOG.info("recalibration %s -> %s done", a, b)
    d = pd.DataFrame(rows)
    write_table(d, cfg, "ext_recalibration_per_draw")
    out = d.groupby(["train", "test", "n_target"]).agg(
        {c: mean_std for c in ["coverage_bot", "coverage_human", "ambiguous_rate", "empty_rate", "acc_singleton"]}).reset_index()
    write_table(out, cfg, "ext_recalibration"); print(out.to_string(index=False))

    # ---- (3) tweet-only TwiBot-20 model on LSABRE
    tw = load_features(cfg, "twibot20")
    feats = [f for f in text_features(usable_features(tw)) if f in set(usable_features(load_features(cfg, "lsabre")))]
    LOG.info("tweet-only features (%d): %s", len(feats), feats)
    per = {}
    for seed in seeds:
        for tag, fs in (("all features", common_features(cfg, "twibot20", "lsabre")), ("tweet-only", feats)):
            _, model, te, _ = cross_eval(cfg, "twibot20", "lsabre", seed, fs)
            s = make_splits(tw, seed, cfg)
            cp = MondrianConformal().fit(model.predict_proba(tw.iloc[s["calib"]][fs].to_numpy(float)),
                                         tw.iloc[s["calib"]]["label"].to_numpy(int), alpha)
            ind = tw.iloc[s["test"]]
            parts = {"TwiBot-20 test (in-dataset)": ind}
            humans = te[te["group"] == "human"]
            for g in te["group"].unique():
                if g != "human":
                    parts[g] = pd.concat([te[te["group"] == g], humans])
            for g, fr in parts.items():
                p, y = model.predict_proba(fr[fs].to_numpy(float)), fr["label"].to_numpy(int)
                m = classification_metrics(y, p[:, 1]); ss = set_summary(cp.predict_set(p), y)
                per.setdefault((tag, g), []).append({"f1": m["f1"], "recall": m["recall"], "auc": m["auc"],
                                                     "ambiguous_rate": ss["ambiguous_rate"],
                                                     "coverage_bot": ss["coverage_bot"], "n_features": len(fs)})
        LOG.info("lsabre seed %d done", seed)
    rows = [{"model": k[0], "test": k[1], **{c: mean_std(pd.DataFrame(v)[c]) for c in v[0]}} for k, v in per.items()]
    t = pd.DataFrame(rows); write_table(t, cfg, "ext_lsabre_text_only"); print(t.to_string(index=False))


if __name__ == "__main__":
    main()
