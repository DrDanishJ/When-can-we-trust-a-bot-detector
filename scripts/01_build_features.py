"""Step 1 - load every dataset, cap tweets, compute all features, write data/cache/features_<dataset>.parquet.

Also writes results/tables/feature_registry.csv (paper Tables 7-8: feature, category, source, new_in_ibd, artifact tag),
results/tables/feature_availability.csv (registry fields + 0/1 usable per dataset; all cached datasets)
and results/tables/dataset_summary.csv (paper Table 5).

    python scripts/01_build_features.py                       # all datasets in the config
    python scripts/01_build_features.py --datasets cresci15   # just one
"""
import _bootstrap  # noqa: F401

import pandas as pd

from ibd.config import cli
from ibd.features import registry
from ibd.features.store import availability_table, build_dataset, usable_features
from ibd.utils import LOG, write_table


def main():
    cfg, args = cli(__doc__, lambda ap: ap.add_argument("--datasets", nargs="*", default=None))
    datasets = args.datasets or cfg["datasets"]["train_test"] + cfg["datasets"]["test_only"]
    write_table(registry.table(), cfg, "feature_registry")
    rows = []
    for d in datasets:
        try:
            df = build_dataset(cfg, d)
        except FileNotFoundError as e:
            LOG.warning("skipping %s: %s", d, e)
            continue
        feats = usable_features(df)
        rows.append({"dataset": d, "accounts": len(df), "bots": int(df["label"].sum()),
                     "humans": int((df["label"] == 0).sum()), "usable_features": len(feats),
                     "groups": ", ".join(f"{g} ({n})" for g, n in df["group"].value_counts().items())})
        missing = [f for f in registry.FEATURE_NAMES if f not in feats]
        LOG.info("%s: %d accounts, %d usable features; not available: %s", d, len(df), len(feats), missing)
    write_table(pd.DataFrame(rows), cfg, "dataset_summary")
    all_ds = cfg["datasets"]["train_test"] + cfg["datasets"]["test_only"]
    write_table(availability_table(cfg, all_ds), cfg, "feature_availability")


if __name__ == "__main__":
    main()
