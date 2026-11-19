"""Step 10 - paper Sections 3.5 and 5.3 (Tables 8-9): extended shortcut audit.

  tables/ext_audit_sets.csv          per dataset and artifact set: SDR (attribution share), joint-permutation
                                     dependence of the fitted model, in-dataset F1 after retraining without the set
  tables/ext_cross_dataset_dedup.csv cross-dataset F1 with the Cresci-15/Cresci-17 overlap removed, for each set
                                     and without the features new in IBD
  tables/ext_overlap.csv             account overlap between Cresci-15 and Cresci-17
  tables/ext_age_matched.csv         Cresci-17 matched on account age at crawl: sample, single-feature AUCs, F1
"""
import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ibd.audit.shortcut import eval_in_dataset, sdr
from ibd.config import cli
from ibd.explain.core import shap_values
from ibd.features.store import load_features, usable_features
from ibd.metrics import classification_metrics
from ibd.models.zoo import fit_model
from ibd.features.registry import new_in_ibd_set
from ibd.extended.audit_ext import COUNTS, age_matched_sample, artifact_sets, joint_permutation
from ibd.extended.crawl_age import cresci17_crawl_age
from ibd.runs import get_run
from ibd.splits import make_splits
from ibd.utils import LOG, mean_std, seed_everything, write_table

N_PERM = 20


def _fit(cfg, df, feats, seed):
    seed_everything(seed)
    s = make_splits(df, seed, cfg)
    X, y = df[feats].to_numpy(float), df["label"].to_numpy(int)
    return fit_model(cfg["models"]["explain_model"], seed, cfg, X[s["train"]], y[s["train"]],
                     X[s["select"]], y[s["select"]], feats), s


def main():
    cfg, _ = cli(__doc__)
    seeds = cfg["models"]["seeds"]
    SETS = artifact_sets(cfg)
    tt = cfg["datasets"]["train_test"]
    frames = {ds: load_features(cfg, ds) for ds in tt}

    # ---- overlap
    ids15 = set(frames["cresci15"]["account_id"].astype(str))
    c17 = frames["cresci17"]
    ov = c17[c17["account_id"].astype(str).isin(ids15)]
    write_table(ov.groupby(["group", "label"]).size().rename("n_in_cresci15").reset_index(), cfg, "ext_overlap")
    LOG.info("overlap Cresci-15/Cresci-17: %d accounts", len(ov))

    # ---- per-set audit on the fitted models
    rows = []
    for ds in tt:
        per = {k: {"sdr": [], "perm_f1": [], "perm_auc": [], "rm_f1": [], "perm_rest_f1": [], "perm_rest_auc": []}
               for k in SETS}
        for seed in seeds:
            run = get_run(cfg, ds, seed)
            Xte, yte = run.X("test"), run.y("test")
            sv = shap_values(run.model, Xte[: cfg["explain"]["shap_max_samples"]], run.X("train"), cfg)
            base_f1 = eval_in_dataset(cfg, run.df, run.features, seed)[0]["f1"]
            for k, S in SETS.items():
                present = [f for f in run.features if f in set(S)]
                rest = [f for f in run.features if f not in set(S)]
                per[k]["sdr"].append(sdr(sv, run.features, S))
                jp = joint_permutation(run.model, Xte, yte, run.features, present, N_PERM, 100 + seed)
                jr = joint_permutation(run.model, Xte, yte, run.features, rest, N_PERM, 200 + seed)
                per[k]["perm_f1"].append(jp["delta_F1"]); per[k]["perm_auc"].append(jp["delta_AUC"])
                per[k]["perm_rest_f1"].append(jr["delta_F1"]); per[k]["perm_rest_auc"].append(jr["delta_AUC"])
                per[k]["rm_f1"].append(eval_in_dataset(cfg, run.df, rest, seed)[0]["f1"] - base_f1)
                per[k]["n"] = len(present)
        for k, v in per.items():
            rows.append({"dataset": ds, "set": k, "n_features": v["n"], "SDR": mean_std(v["sdr"]),
                         "perm_delta_F1": mean_std(v["perm_f1"]), "perm_delta_AUC": mean_std(v["perm_auc"]),
                         "perm_rest_delta_F1": mean_std(v["perm_rest_f1"]),
                         "perm_rest_delta_AUC": mean_std(v["perm_rest_auc"]),
                         "retrain_delta_F1": mean_std(v["rm_f1"])})
        LOG.info("%s: set audit done", ds)
    t = pd.DataFrame(rows); write_table(t, cfg, "ext_audit_sets"); print(t.to_string(index=False))

    # ---- cross-dataset transfer without the Cresci-15/17 overlap
    c17_no_ff = c17[c17["group"] != "fake_followers"].reset_index(drop=True)
    rows = []
    for a in tt:
        for b in tt:
            if a == b:
                continue
            tr = c17_no_ff if (a == "cresci17" and b == "cresci15") else frames[a]
            te_full = frames[b]
            note = "Cresci-17 trained without fake_followers" if tr is c17_no_ff else ""
            fa, fb = usable_features(tr), set(usable_features(te_full))
            common = [f for f in fa if f in fb]
            row = {"train": a, "test": b, "note": note}
            for k, S in {"with_A": [], **{f"without_{n}": S for n, S in SETS.items()},
                         "without_new": new_in_ibd_set()}.items():
                feats = [f for f in common if f not in set(S)]
                f1s = []
                for seed in seeds:
                    m, _ = _fit(cfg, tr, feats, seed)
                    te = te_full.iloc[make_splits(te_full, seed, cfg)["test"]]
                    if a == "cresci15" and b == "cresci17":
                        te = te[te["group"] != "fake_followers"]
                        row["note"] = "Cresci-17 test without fake_followers"
                    f1s.append(classification_metrics(te["label"].to_numpy(int),
                                                      m.predict_proba(te[feats].to_numpy(float))[:, 1])["f1"])
                row[f"F1_{k}"] = mean_std(f1s); row[f"n_{k}"] = len(feats)
            rows.append(row)
            LOG.info("cross %s -> %s done", a, b)
    t = pd.DataFrame(rows); write_table(t, cfg, "ext_cross_dataset_dedup"); print(t.to_string(index=False))

    # ---- Cresci-17 matched on account age at crawl
    age = cresci17_crawl_age(cfg["paths"]["cresci17"]).set_index("account_id")["age_at_crawl_days"]
    df = c17.copy()
    a_all = df["account_id"].astype(str).map(age)
    rows, single = [], []
    feats_all = usable_features(df)
    for seed in seeds:
        mdf = age_matched_sample(df, a_all, cfg.get("audit", {}).get("age_bins", 20), 300 + seed).reset_index(drop=True)
        y = mdf["label"].to_numpy(int)
        r = {"seed": seed, "n_matched": len(mdf), "n_bot": int(y.sum()),
             "AUC_age_at_crawl": roc_auc_score(y, mdf["_age"]),
             "bot_groups": "; ".join(f"{g}={n}" for g, n in mdf[mdf.label == 1]["group"].value_counts().items())}
        for f in COUNTS + ["account_age_days", "favorites_over_time"]:
            if f in mdf:
                v = mdf[f].fillna(mdf[f].median())
                single.append({"seed": seed, "feature": f, "AUC_matched": roc_auc_score(y, v),
                               "AUC_unmatched": roc_auc_score(df["label"], df[f].fillna(df[f].median()))})
        mfeats = usable_features(mdf)
        for k, S in {"all": [], **{f"without_{n}": S for n, S in SETS.items()}}.items():
            feats = [f for f in mfeats if f not in set(S)]
            m, s = _fit(cfg, mdf, feats, seed)
            met = classification_metrics(y[s["test"]], m.predict_proba(mdf[feats].to_numpy(float)[s["test"]])[:, 1])
            r[f"F1_{k}"], r[f"AUC_{k}"] = met["f1"], met["auc"]
        rows.append(r)
    t = pd.DataFrame(rows)
    write_table(t, cfg, "ext_age_matched_per_seed")
    summ = {c: mean_std(t[c]) for c in t.columns if c not in ("seed", "bot_groups")}
    write_table(pd.DataFrame([summ]), cfg, "ext_age_matched")
    sg = pd.DataFrame(single).groupby("feature")[["AUC_matched", "AUC_unmatched"]].agg(["mean", "std"]).reset_index()
    sg.columns = ["_".join(c).strip("_") for c in sg.columns]
    write_table(sg, cfg, "ext_age_matched_single_auc")
    print(pd.DataFrame([summ]).T.to_string()); print(t["bot_groups"].iloc[0]); print(sg.to_string(index=False))


if __name__ == "__main__":
    main()
