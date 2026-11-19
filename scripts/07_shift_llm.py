"""Step 7 - paper Section 6.6 (Table 15): distribution shift and LLM-powered bots.

For every (train, test) pair in cfg.shift.pairs the explain model is trained on the train dataset using only
features available in both datasets, calibrated on the train dataset's calibration split, and applied to the
test population (its test split, or all accounts for fox8 / lsabre). For lsabre one row is written per rewrite
variant (each variant's bots + the TwiBot-20 test humans), plus the same bots with their original tweets.

  table15_shift.csv   F1, recall on bots, ambiguous-set rate, DAUC and SLA per pair
"""
import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from ibd.audit.shortcut import common_features, cross_eval
from ibd.config import cli
from ibd.explain.core import lime_weights, shap_values
from ibd.features.store import load_features
from ibd.metrics import classification_metrics
from ibd.reliability.measures import agreement, deletion_auc, deletion_reference, rank_by_attribution
from ibd.splits import make_splits
from ibd.uq.conformal import MondrianConformal, set_summary
from ibd.utils import LOG, mean_std, write_table


def evaluate(cfg, model, train_df, feats, seed, te, alpha, with_lime):
    s = make_splits(train_df, seed, cfg)
    Xtr = train_df.iloc[s["train"]][feats].to_numpy(float)
    Xcal = train_df.iloc[s["calib"]][feats].to_numpy(float)
    ycal = train_df.iloc[s["calib"]]["label"].to_numpy(int)
    X, y = te[feats].to_numpy(float), te["label"].to_numpy(int)
    p = model.predict_proba(X)
    met = classification_metrics(y, p[:, 1])
    sets = MondrianConformal().fit(model.predict_proba(Xcal), ycal, alpha).predict_set(p)
    out = {**met, **set_summary(sets, y)}
    rc = cfg["reliability"]
    n = min(len(X), rc["dauc_accounts"])
    idx = np.random.RandomState(seed).choice(len(X), n, replace=False)
    sv = shap_values(model, X[idx], Xtr, cfg)
    ref = deletion_reference(Xtr, cfg, seed)
    sign = np.where(p[idx].argmax(1) == 1, 1.0, -1.0)
    out["DAUC"] = float(deletion_auc(model, X[idx], rank_by_attribution(sv, sign), ref, rc["dauc_steps"]).mean())
    if with_lime:
        m = min(n, cfg["explain"]["lime_samples"])
        W = lime_weights(model, Xtr, X[idx[:m]], cfg, seed)
        out["SLA"] = float(agreement(sv[:m], W, rc["sla_k"])[0].mean())
    return out


def main():
    cfg, args = cli(__doc__, lambda ap: ap.add_argument("--no-lime", action="store_true"))
    alpha = cfg["conformal"]["alphas"][0]
    rows = []
    for train_ds, test_ds in cfg["shift"]["pairs"]:
        try:
            feats = common_features(cfg, train_ds, test_ds)
        except FileNotFoundError as e:
            LOG.warning("skipping %s -> %s: %s", train_ds, test_ds, e)
            continue
        per = {}
        for seed in cfg["models"]["seeds"]:
            _, model, te, _ = cross_eval(cfg, train_ds, test_ds, seed, feats)
            train_df = load_features(cfg, train_ds)
            if test_ds == "lsabre":
                humans = te[te["group"] == "human"]
                parts = {g: pd.concat([te[te["group"] == g], humans]) for g in te["group"].unique() if g != "human"}
            else:
                parts = {"": te}
            for g, frame in parts.items():
                r = evaluate(cfg, model, train_df, feats, seed, frame, alpha, with_lime=not args.no_lime)
                per.setdefault(g, []).append(r)
        for g, lst in per.items():
            df = pd.DataFrame(lst)
            label = f"{train_ds} -> {test_ds}" + (f" [{g}]" if g else "")
            cols = ["f1", "recall", "auc", "ambiguous_rate", "coverage", "DAUC"] + (["SLA"] if "SLA" in df else [])
            rows.append({"pair": label, "n_features": len(feats), **{c: mean_std(df[c]) for c in cols}})
        LOG.info("%s -> %s done", train_ds, test_ds)
    out = pd.DataFrame(rows)
    write_table(out, cfg, "table15_shift")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
