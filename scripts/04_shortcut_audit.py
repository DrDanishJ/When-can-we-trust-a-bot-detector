"""Step 4 - paper Section 6.3 (Tables 10-12): shortcut audit.

  table10_sdr.csv            SDR (eq. 38) and F1 with / without the artifact-prone set A, per dataset
  table11_cross_dataset.csv  F1 when training on one dataset and testing on another (with A / without A / without
                             the features new in IBD)
  table12_ablation.csv       change in F1 (and F1) when each feature category, A, or the features new in IBD are
                             removed; plus a model using only the features new in IBD
"New in IBD" = registry new_in_ibd (source 'IBD' or 'IBD variant of ...').
"""
import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from ibd.audit.shortcut import common_features, cross_eval, eval_in_dataset, sdr
from ibd.config import cli
from ibd.explain.core import shap_values
from ibd.features.registry import artifact_set, by_category, new_in_ibd_set
from ibd.runs import get_run
from ibd.utils import LOG, mean_std, write_table


def main():
    cfg, _ = cli(__doc__)
    A = artifact_set(cfg)
    NEW = new_in_ibd_set()
    seeds = cfg["models"]["seeds"]
    t10, t11, t12 = [], [], []
    for ds in cfg["datasets"]["train_test"]:
        sdrs, f1_with, f1_without = [], [], []
        for seed in seeds:
            run = get_run(cfg, ds, seed)
            sv = shap_values(run.model, run.X("test")[: cfg["explain"]["shap_max_samples"]], run.X("train"), cfg)
            sdrs.append(sdr(sv, run.features, A))
            f1_with.append(eval_in_dataset(cfg, run.df, run.features, seed)[0]["f1"])
            no_a = [f for f in run.features if f not in set(A)]
            f1_without.append(eval_in_dataset(cfg, run.df, no_a, seed)[0]["f1"])
        present = [f for f in run.features if f in set(A)]
        t10.append({"dataset": ds, "artifact_features_present": len(present), "SDR": mean_std(sdrs),
                    "F1_with_A": mean_std(f1_with), "F1_without_A": mean_std(f1_without),
                    "delta_F1": mean_std(np.array(f1_without) - np.array(f1_with)),
                    "A_present": ", ".join(present)})
        # category ablation (+ the features introduced or modified in IBD: removed, or used alone)
        cats = by_category(run.features)
        base = np.array(f1_with)
        new_present = [f for f in run.features if f in set(NEW)]
        removals = {c: fs for c, fs in cats.items() if fs}
        removals["artifact-prone (A)"] = present
        removals["new IBD features removed"] = new_present
        removals["new IBD features only"] = [f for f in run.features if f not in set(new_present)]
        for cat, fs in removals.items():
            keep = [f for f in run.features if f not in set(fs)]
            if not keep:
                continue
            f1s = np.array([eval_in_dataset(cfg, run.df, keep, s)[0]["f1"] for s in seeds])
            t12.append({"dataset": ds, "removed": cat, "n_removed": len(fs), "n_used": len(keep),
                        "delta_F1": mean_std(f1s - base), "F1": mean_std(f1s)})
        LOG.info("%s audited", ds)
    # cross-dataset matrix
    tt = cfg["datasets"]["train_test"]
    for a in tt:
        for b in tt:
            row = {"train": a, "test": b}
            variants = {"with_A": common_features(cfg, a, b),
                        "without_A": common_features(cfg, a, b, drop_artifacts=True),
                        "without_new": [f for f in common_features(cfg, a, b) if f not in set(NEW)]}
            for tag, feats in variants.items():
                f1 = [cross_eval(cfg, a, b, s, feats)[0]["f1"] for s in seeds]
                row[f"F1_{tag}"] = mean_std(f1)
                row[f"n_features_{tag}"] = len(feats)
            t11.append(row)
    write_table(pd.DataFrame(t10), cfg, "table10_sdr")
    write_table(pd.DataFrame(t11), cfg, "table11_cross_dataset")
    write_table(pd.DataFrame(t12), cfg, "table12_ablation")
    print(pd.DataFrame(t10).drop(columns="A_present").to_string(index=False))
    print(pd.DataFrame(t11).to_string(index=False))


if __name__ == "__main__":
    main()
