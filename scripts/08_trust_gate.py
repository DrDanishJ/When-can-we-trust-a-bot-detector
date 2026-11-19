"""Step 8 - paper Section 6.7 (Fig. 13): the trust gate (eqs. 46-47).

Every threshold is set on the calibration split; the test split is only scored.

Four selective-prediction policies, each a one-parameter family indexed by a level r in [0, 1]. At level r every
threshold of the policy is the calibration quantile that keeps a fraction r of calibration accounts on that
criterion alone:
  confidence only          max p >= t_conf(r)
  singleton only           p2 <= a(r) < p1 (conformal set is a singleton at significance a(r); p1 >= p2 are the
                           two conformal p-values)
  singleton + DAUC <= tau  singleton at a(r) AND DAUC <= tau(r)
  confidence + DAUC <= tau max p >= t_conf(r) AND DAUC <= tau(r)
For each target coverage q, r is chosen so that calibration coverage reaches q, and the same thresholds are applied
to the test split. Curves show test coverage against test accuracy. AUC is the trapezoid area under each curve
over the coverage it reaches; AUC_normalised divides by that range; AUC_common is the mean accuracy over the
coverage range reached by all four policies (conformal p-values are discrete, so not every policy reaches every
coverage).

The default operating point of the paper (singleton at cfg.trust_gate.alpha AND DAUC <= tau, tau = median DAUC of
correctly classified calibration accounts) is reported with confidence-only gating at the same calibration
coverage.

  trust_gate_summary.csv    default operating point, per seed (mean ± std)
  trust_gate_auc.csv        area under each policy's accuracy-coverage curve (mean ± std)
  trust_gate_curve.csv      accuracy-coverage curves per seed and policy
  table_dauc_diagnostic.csv diagnostic only: per seed and test DAUC decile, accuracy, mean max probability, share
                            of accounts whose top-1 |SHAP| feature is `verified`, the most common top-1 feature,
                            Spearman(DAUC, max probability)
DAUC uses cfg.reliability.deletion_baseline (background by default). Options: --suffix appends to every output
name; --no-artifacts uses the explain model trained without the artifact-prone features (Table 10 "without A").
  figures/fig13_trust_gate.png
"""
import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ibd import plots
from ibd.config import cli
from ibd.explain.core import shap_values
from ibd.features.registry import artifact_set
from ibd.features.store import load_features, usable_features
from ibd.reliability.measures import deletion_auc, deletion_reference, rank_by_attribution
from ibd.runs import get_run
from ibd.uq.conformal import MondrianConformal
from ibd.utils import mean_std, write_table

POLICIES = ["confidence only", "singleton only", "singleton + DAUC <= tau", "confidence + DAUC <= tau"]
LEVELS = np.linspace(0.0, 1.0, 1001)


def dauc_for(run, X, med, K, cfg, return_shap=False):
    sv = shap_values(run.model, X, run.X("train"), cfg)
    sign = np.where(run.model.predict_proba(X).argmax(1) == 1, 1.0, -1.0)
    d = deletion_auc(run.model, X, rank_by_attribution(sv, sign), med, K)
    return (d, sv) if return_shap else d


def dauc_diagnostic(seed, dauc, p, correct, sv, features, top_feature="verified"):
    """Diagnostic only (does not affect the gate): how DAUC relates to confidence, accuracy and the top-1 |SHAP|
    feature on the test split. One row per DAUC decile."""
    rho, pval = spearmanr(dauc, p.max(1))
    decile = pd.qcut(dauc, 10, labels=False, duplicates="drop")
    top1 = np.array(features)[np.abs(sv).argmax(1)]
    top_is = top1 == top_feature  # all False when the model does not use top_feature (e.g. without A)
    rows = []
    for k in np.unique(decile):
        m = decile == k
        rows.append({"seed": seed, "dauc_decile": int(k) + 1, "dauc_min": float(dauc[m].min()),
                     "dauc_max": float(dauc[m].max()), "n": int(m.sum()), "accuracy": float(correct[m].mean()),
                     "mean_max_prob": float(p[m].max(1).mean()), f"share_top1_{top_feature}": float(top_is[m].mean()),
                     "top1_most_common": pd.Series(top1[m]).mode()[0],
                     "share_top1_most_common": float((top1[m] == pd.Series(top1[m]).mode()[0]).mean()),
                     "spearman_dauc_maxprob": float(rho), "spearman_p": float(pval)})
    return rows


def selective(correct, keep):
    return (float(keep.mean()), float(correct[keep].mean()) if keep.any() else np.nan)


class Policies:
    """Thresholds come from the calibration arrays only; `keep` applies them to any split."""

    def __init__(self, conf_cal, p2_cal, dauc_cal):
        self.conf_cal, self.p2_cal, self.dauc_cal = conf_cal, p2_cal, dauc_cal

    def thresholds(self, r):
        return {"t_conf": np.quantile(self.conf_cal, 1 - r), "a": np.quantile(self.p2_cal, r),
                "tau": np.quantile(self.dauc_cal, r)}

    @staticmethod
    def keep(policy, t, conf, p1, p2, dauc):
        single = (p2 <= t["a"]) & (p1 > t["a"])
        by_conf = conf >= t["t_conf"]
        faithful = dauc <= t["tau"]
        return {"confidence only": by_conf, "singleton only": single,
                "singleton + DAUC <= tau": single & faithful, "confidence + DAUC <= tau": by_conf & faithful}[policy]


def main():
    def extra(ap):
        ap.add_argument("--no-artifacts", action="store_true",
                        help="use the explain model trained without the artifact-prone features (Table 10 'without A')")
        ap.add_argument("--suffix", default="", help="appended to every output name, e.g. _median or _noA")
    cfg, args = cli(__doc__, extra)
    sfx = args.suffix
    ds, alpha = cfg["trust_gate"]["dataset"], cfg["trust_gate"]["alpha"]
    feats = None
    if args.no_artifacts:
        A = set(artifact_set(cfg))
        feats = [f for f in usable_features(load_features(cfg, ds)) if f not in A]
    K = cfg["reliability"]["dauc_steps"]
    targets = np.linspace(0.05, 1.0, 20)
    summary, curve_rows, auc_rows, diag_rows = [], [], [], []
    for seed in cfg["models"]["seeds"]:
        run = get_run(cfg, ds, seed, features=feats, tag="_noA" if args.no_artifacts else "")
        med = deletion_reference(run.X("train"), cfg, seed)  # what "deleting" a feature means in eq. 42
        Xc, yc, Xt, yt = run.X("calib"), run.y("calib"), run.X("test"), run.y("test")
        pc, pt = run.model.predict_proba(Xc), run.model.predict_proba(Xt)
        cp = MondrianConformal().fit(pc, yc, alpha)
        d_cal = dauc_for(run, Xc, med, K, cfg)
        d_te, sv_te = dauc_for(run, Xt, med, K, cfg, return_shap=True)
        pv_cal, pv_te = np.sort(cp.pvalues(pc), axis=1), np.sort(cp.pvalues(pt), axis=1)
        cal = dict(conf=pc.max(1), p1=pv_cal[:, 1], p2=pv_cal[:, 0], dauc=d_cal)
        te = dict(conf=pt.max(1), p1=pv_te[:, 1], p2=pv_te[:, 0], dauc=d_te)
        correct = pt.argmax(1) == yt
        diag_rows += dauc_diagnostic(seed, d_te, pt, correct, sv_te, run.features)
        pol = Policies(cal["conf"], cal["p2"], cal["dauc"])

        # default operating point (paper): singleton at alpha AND DAUC <= tau
        tau = float(np.median(d_cal[pc.argmax(1) == yc]))
        tg_cal = (cp.predict_set(pc).sum(1) == 1) & (d_cal <= tau)
        tg = (cp.predict_set(pt).sum(1) == 1) & (d_te <= tau)
        cov, acc = selective(correct, tg)
        t_conf = np.quantile(cal["conf"], 1 - tg_cal.mean())  # confidence threshold with the same calib coverage
        cov_c, acc_c = selective(correct, te["conf"] >= t_conf)
        summary.append({"seed": seed, "tau": tau, "calib_coverage_TG": float(tg_cal.mean()),
                        "coverage_TG": cov, "acc_TG": acc, "coverage_conf": cov_c, "acc_conf": acc_c,
                        "acc_no_gate": float(correct.mean()),
                        "errors_referred": float((~correct & ~tg).sum() / max((~correct).sum(), 1))})

        # accuracy-coverage curves
        curves = {}
        for policy in POLICIES:
            cal_cov = np.array([Policies.keep(policy, pol.thresholds(r), **cal).mean() for r in LEVELS])
            pts = []
            for q in targets:
                r = LEVELS[np.argmax(cal_cov >= q - 1e-9)] if (cal_cov >= q - 1e-9).any() else 1.0
                keep = Policies.keep(policy, pol.thresholds(r), **te)
                c, a = selective(correct, keep)
                pts.append((c, a))
                curve_rows.append({"seed": seed, "policy": policy, "target": q, "level": r,
                                   "calib_coverage": float(Policies.keep(policy, pol.thresholds(r), **cal).mean()),
                                   "coverage": c, "accuracy": a})
            pts = sorted(p for p in pts if not np.isnan(p[1]))
            curves[policy] = (np.array([p[0] for p in pts]), np.array([p[1] for p in pts]))
        # conformal p-values are discrete, so policies reach different coverage ranges; AUC_common integrates
        # every curve over the range all four reach, divided by its width (= mean accuracy over that range)
        lo = max(x.min() for x, _ in curves.values())
        hi = min(x.max() for x, _ in curves.values())
        grid = np.linspace(lo, hi, 200)
        for policy, (x, y) in curves.items():
            auc_rows.append({"seed": seed, "policy": policy, "AUC": float(np.trapezoid(y, x)),
                             "coverage_min": float(x.min()), "coverage_max": float(x.max()),
                             "AUC_normalised": float(np.trapezoid(y, x) / max(x.max() - x.min(), 1e-12)),
                             "common_lo": float(lo), "common_hi": float(hi),
                             "AUC_common": float(np.trapezoid(np.interp(grid, x, y), grid) / max(hi - lo, 1e-12))
                             if hi > lo else np.nan})

    s = pd.DataFrame(summary)
    write_table(pd.DataFrame([{c: mean_std(s[c]) for c in s.columns if c != "seed"}]), cfg, "trust_gate_summary" + sfx)
    a = pd.DataFrame(auc_rows)
    auc = a.groupby("policy", sort=False).agg({c: mean_std for c in ["AUC", "AUC_normalised", "coverage_min",
                                                                     "coverage_max", "AUC_common", "common_lo",
                                                                     "common_hi"]}).reset_index()
    write_table(auc, cfg, "trust_gate_auc" + sfx)
    write_table(pd.DataFrame(diag_rows), cfg, "table_dauc_diagnostic" + sfx)
    cr = pd.DataFrame(curve_rows)
    write_table(cr, cfg, "trust_gate_curve" + sfx)
    g = cr.groupby(["policy", "target"], sort=False).mean(numeric_only=True).reset_index()
    plots.lines(cfg, "fig13_trust_gate" + sfx,
                {p: (g[g.policy == p]["coverage"].to_numpy(), g[g.policy == p]["accuracy"].to_numpy(), None)
                 for p in POLICIES},
                "test coverage (share of accounts decided automatically)", "test accuracy on retained accounts",
                title=f"{ds}: selective accuracy, thresholds set on calib")
    print(s.to_string(index=False))
    print(auc.to_string(index=False))


if __name__ == "__main__":
    main()
