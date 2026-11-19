"""Step 2 - paper Table 9: every classifier on every train/test dataset, mean ± std over seeds.

Outputs
  results/tables/table09_classification.csv     mean ± std per dataset x classifier
  results/tables/table09_per_seed.csv           raw per-seed metrics
  results/tables/table09_mcnemar.csv            exact McNemar test of each model against the explain model
The explain model (cfg.models.explain_model) is cached for steps 3-8.
"""
import _bootstrap  # noqa: F401

import joblib
import numpy as np
import pandas as pd

from ibd.config import cli
from ibd.features.store import load_features, usable_features
from ibd.metrics import classification_metrics, mcnemar_p
from ibd.models.zoo import available
from ibd.runs import model_path, train
from ibd.splits import make_splits
from ibd.utils import LOG, mean_std, timer, write_table

METRICS = ["acc", "prec", "recall", "f1", "auc", "brier", "ece"]


def main():
    cfg, _ = cli(__doc__)
    names = available(cfg["models"]["classifiers"])
    ref = cfg["models"]["explain_model"]
    rows, tests = [], []
    for ds in cfg["datasets"]["train_test"]:
        df = load_features(cfg, ds)
        feats = usable_features(df)
        for seed in cfg["models"]["seeds"]:
            s = make_splits(df, seed, cfg)
            X, y = df[feats].to_numpy(float), df["label"].to_numpy(int)
            preds = {}
            for name in names:
                with timer(f"{ds} seed {seed} {name}"):
                    try:
                        m = train(cfg, df, feats, s, name, seed)
                    except Exception as e:  # keep going if one optional model fails
                        LOG.error("%s failed on %s: %s", name, ds, e)
                        continue
                p = m.predict_proba(X[s["test"]])[:, 1]
                preds[name] = (p >= 0.5).astype(int)
                rows.append({"dataset": ds, "seed": seed, "classifier": name,
                             **classification_metrics(y[s["test"]], p)})
                if name == ref:
                    joblib.dump({"model": m, "features": feats}, model_path(cfg, ds, seed, name))
            for name, pr in preds.items():
                if name != ref and ref in preds:
                    tests.append({"dataset": ds, "seed": seed, "classifier": name,
                                  "p_vs_" + ref: mcnemar_p(y[s["test"]], preds[ref], pr)})
    per_seed = pd.DataFrame(rows)
    write_table(per_seed, cfg, "table09_per_seed")
    summary = (per_seed.groupby(["dataset", "classifier"])
               .agg({m: mean_std for m in METRICS}).reset_index())
    write_table(summary, cfg, "table09_classification")
    if tests:
        t = pd.DataFrame(tests)
        col = "p_vs_" + ref
        t = t.groupby(["dataset", "classifier"])[col].agg(
            median_p="median", seeds_significant=lambda v: int((np.asarray(v) < 0.05).sum())).reset_index()
        write_table(t, cfg, "table09_mcnemar")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
