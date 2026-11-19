"""Step 6 - paper Section 6.5 (Table 14, Fig. 12): explanation reliability, for cfg.reliability.datasets.

Accounts are grouped by what the model and the conformal layer say about them:
  confident-correct (singleton set, correct), ambiguous ({bot, human} set), misclassified, all.
  table14_reliability_<ds>.csv           DAUC (SHAP vs random ranking), ESI, SLA and Kendall tau per group,
                                         mean ± std over cfg.models.seeds
  table14_reliability_<ds>_per_seed.csv  the same per seed
  table14_roar_<ds>.csv                  F1 after removing the k most important (SHAP) vs k random features,
                                         mean ± std over seeds (per seed in table14_roar_<ds>_per_seed.csv)
  figures/fig12_roar_<ds>.png
Runs every seed in cfg.models.seeds (these tests retrain many models). --suffix appends to every output name.
"""
import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from ibd import plots
from ibd.audit.shortcut import eval_in_dataset
from ibd.config import cli
from ibd.explain.core import lime_weights, shap_values
from ibd.models.zoo import fit_model
from ibd.reliability.measures import (agreement, deletion_auc, deletion_reference, random_rankings, rank_by_attribution,
                                      stability_index)
from ibd.runs import get_run
from ibd.uq.conformal import MondrianConformal, set_type
from ibd.utils import LOG, mean_std, write_table


def groups(run, alpha):
    p_te = run.model.predict_proba(run.X("test"))
    sets = MondrianConformal().fit(run.model.predict_proba(run.X("calib")), run.y("calib"), alpha).predict_set(p_te)
    kind = set_type(sets)
    wrong = p_te.argmax(1) != run.y("test")
    return {"confident-correct": (kind != "ambiguous") & ~wrong, "ambiguous": kind == "ambiguous",
            "misclassified": wrong, "all": np.ones(len(kind), bool)}


def reliability_one(cfg, ds, seed):
    """Table 14 rows and ROAR curve for one dataset and seed."""
    rc = cfg["reliability"]
    run = get_run(cfg, ds, seed)
    X_tr, X_te = run.X("train"), run.X("test")
    ref = deletion_reference(X_tr, cfg, seed)  # what "deleting" a feature means in eq. 42
    g = groups(run, cfg["conformal"]["alphas"][0])
    rng = np.random.RandomState(seed)
    n = len(X_te)
    idx = np.sort(rng.choice(n, min(n, rc["dauc_accounts"]), replace=False))
    sv = shap_values(run.model, X_te[idx], X_tr, cfg)
    pred = run.model.predict_proba(X_te[idx]).argmax(1)
    sign = np.where(pred == 1, 1.0, -1.0)
    dauc_shap = deletion_auc(run.model, X_te[idx], rank_by_attribution(sv, sign), ref, rc["dauc_steps"])
    dauc_rand = deletion_auc(run.model, X_te[idx], random_rankings(len(idx), X_te.shape[1], seed), ref,
                             rc["dauc_steps"])
    # ESI: bootstrap models (seed 0 keeps the bootstrap seeds 1000 + b of the single-seed version)
    esi_idx = idx[: rc["esi_accounts"]]
    y_tr = run.y("train")
    runs_shap = []
    for b in range(rc["esi_bootstrap"]):
        bs = 1000 + 100 * seed + b
        bi = np.random.RandomState(bs).choice(len(y_tr), len(y_tr), replace=True)
        mb = fit_model(run.model.name, bs, cfg, X_tr[bi], y_tr[bi], feature_names=run.features)
        runs_shap.append(shap_values(mb, X_te[esi_idx], X_tr, cfg))
    esi = stability_index(runs_shap)
    # SLA: LIME on a subsample
    lime_idx = idx[: cfg["explain"]["lime_samples"]]
    W = lime_weights(run.model, X_tr, X_te[lime_idx], cfg, seed)
    sla, tau = agreement(sv[: len(lime_idx)], W, rc["sla_k"])
    rows = []
    for name, mask in g.items():
        m_d = mask[idx]
        m_e = mask[esi_idx]
        m_l = mask[lime_idx]
        rows.append({
            "seed": seed, "group": name, "n_accounts": int(mask.sum()),
            "DAUC_shap": float(np.mean(dauc_shap[m_d])) if m_d.any() else np.nan,
            "DAUC_random": float(np.mean(dauc_rand[m_d])) if m_d.any() else np.nan,
            "ESI": float(np.mean(esi[m_e])) if m_e.any() else np.nan,
            f"SLA_k{rc['sla_k']}": float(np.mean(sla[m_l])) if m_l.any() else np.nan,
            "kendall_tau_shap_lime": float(np.mean(tau[m_l])) if m_l.any() else np.nan,
            "n_dauc": int(m_d.sum()), "n_esi": int(m_e.sum()), "n_lime": int(m_l.sum()),
        })
    # ROAR
    order = np.argsort(-np.abs(sv).mean(0))
    feats = run.features
    roar_rows = []
    for k in rc["roar_k"]:
        keep = [feats[i] for i in range(len(feats)) if i not in set(order[:k])]
        top = eval_in_dataset(cfg, run.df, keep, seed)[0]["f1"]
        vals = []
        for r in range(rc["roar_random_repeats"]):
            drop = set(np.random.RandomState(r).choice(len(feats), k, replace=False)) if k else set()
            vals.append(eval_in_dataset(cfg, run.df, [feats[i] for i in range(len(feats)) if i not in drop], seed)[0]["f1"])
        roar_rows.append({"seed": seed, "k_removed": k, "F1_remove_top_shap": top, "F1_remove_random": float(np.mean(vals))})
    return rows, roar_rows


def main():
    cfg, args = cli(__doc__, lambda ap: ap.add_argument("--suffix", default="", help="appended to every output name"))
    rc, sfx = cfg["reliability"], args.suffix
    for ds in rc["datasets"]:
        rows, roar_rows = [], []
        for seed in cfg["models"]["seeds"]:
            r, rr = reliability_one(cfg, ds, seed)
            rows += r
            roar_rows += rr
            LOG.info("%s seed %s reliability done", ds, seed)
        per = pd.DataFrame(rows)
        write_table(per, cfg, f"table14_reliability_{ds}_per_seed{sfx}")
        metrics = [c for c in per.columns if c not in ("seed", "group")]
        summary = per.groupby("group", sort=False).agg({c: mean_std for c in metrics}).reset_index()
        write_table(summary, cfg, f"table14_reliability_{ds}{sfx}")
        print(summary.to_string(index=False))
        roar = pd.DataFrame(roar_rows)
        write_table(roar, cfg, f"table14_roar_{ds}_per_seed{sfx}")
        rs = roar.groupby("k_removed").agg({c: mean_std for c in ["F1_remove_top_shap", "F1_remove_random"]}).reset_index()
        write_table(rs, cfg, f"table14_roar_{ds}{sfx}")
        g = roar.groupby("k_removed")
        m, s = g.mean(numeric_only=True), g.std(numeric_only=True).fillna(0.0)
        plots.lines(cfg, f"fig12_roar_{ds}{sfx}",
                    {"remove top-k SHAP features": (m.index.to_numpy(), m["F1_remove_top_shap"].to_numpy(),
                                                    s["F1_remove_top_shap"].to_numpy()),
                     "remove k random features": (m.index.to_numpy(), m["F1_remove_random"].to_numpy(),
                                                  s["F1_remove_random"].to_numpy())},
                    "features removed and model retrained (k)", "test F1 (mean ± std over seeds)", title=f"{ds}: ROAR")
        LOG.info("%s reliability done", ds)


if __name__ == "__main__":
    main()
