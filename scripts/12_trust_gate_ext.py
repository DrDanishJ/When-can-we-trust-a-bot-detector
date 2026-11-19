"""Step 12 - paper Section 5.7: what DAUC says about correctness on TwiBot-20.

Per seed, on the TwiBot-20 test split (alpha = cfg.trust_gate.alpha, background DAUC as in the paper):
  tables/ext_tg_baselines.csv       accuracy of all singletons, of the trust gate (singleton AND DAUC <= tau), of the
                                    reverse gate (singleton AND DAUC > tau) and of random halves of the singletons
  tables/ext_tg_correctness_auc.csv AUROC for predicting correctness from confidence, from -DAUC (low DAUC = more
                                    faithful), and from both (logistic regression fitted on calib, scored on test)
  tables/ext_tg_verified.csv        accuracy and mean max probability by DAUC half within verified / unverified
                                    accounts, mean DAUC and
                                    share of accounts whose top-1 |SHAP| feature is `verified`
"""
import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from ibd.config import cli
from ibd.explain.core import shap_values
from ibd.reliability.measures import deletion_auc, deletion_reference, rank_by_attribution
from ibd.runs import get_run
from ibd.uq.conformal import MondrianConformal
from ibd.utils import mean_std, write_table


def dauc(run, X, ref, K, cfg):
    sv = shap_values(run.model, X, run.X("train"), cfg)
    sign = np.where(run.model.predict_proba(X).argmax(1) == 1, 1.0, -1.0)
    return deletion_auc(run.model, X, rank_by_attribution(sv, sign), ref, K), sv


def main():
    cfg, _ = cli(__doc__)
    ds, alpha = cfg["trust_gate"]["dataset"], cfg["trust_gate"]["alpha"]
    K = cfg["reliability"]["dauc_steps"]
    base, cor, ver = [], [], []
    for seed in cfg["models"]["seeds"]:
        run = get_run(cfg, ds, seed)
        ref = deletion_reference(run.X("train"), cfg, seed)
        Xc, yc, Xt, yt = run.X("calib"), run.y("calib"), run.X("test"), run.y("test")
        pc, pt = run.model.predict_proba(Xc), run.model.predict_proba(Xt)
        cp = MondrianConformal().fit(pc, yc, alpha)
        dc, _ = dauc(run, Xc, ref, K, cfg)
        dt, sv = dauc(run, Xt, ref, K, cfg)
        ok_c, ok_t = pc.argmax(1) == yc, pt.argmax(1) == yt
        tau = float(np.median(dc[ok_c]))
        single = cp.predict_set(pt).sum(1) == 1
        rng = np.random.default_rng(900 + seed)
        rand = [ok_t[rng.choice(np.where(single)[0], (single & (dt <= tau)).sum(), replace=False)].mean()
                for _ in range(1000)]
        base.append({"seed": seed, "tau": tau, "acc_all": ok_t.mean(), "coverage_singleton": single.mean(),
                     "acc_singleton": ok_t[single].mean(),
                     "coverage_TG": (single & (dt <= tau)).mean(), "acc_TG": ok_t[single & (dt <= tau)].mean(),
                     "coverage_reverse": (single & (dt > tau)).mean(), "acc_reverse": ok_t[single & (dt > tau)].mean(),
                     "acc_random_same_size_mean": float(np.mean(rand)),
                     "acc_random_p025": float(np.quantile(rand, .025)), "acc_random_p975": float(np.quantile(rand, .975))})
        # correctness prediction
        Fc = np.c_[pc.max(1), dc]; Ft = np.c_[pt.max(1), dt]
        lr = LogisticRegression().fit(Fc, ok_c)
        cor.append({"seed": seed, "AUROC_confidence": roc_auc_score(ok_t, pt.max(1)),
                    "AUROC_minus_DAUC": roc_auc_score(ok_t, -dt), "AUROC_DAUC": roc_auc_score(ok_t, dt),
                    "AUROC_confidence_plus_DAUC": roc_auc_score(ok_t, lr.predict_proba(Ft)[:, 1]),
                    "coef_DAUC_sign": float(np.sign(lr.coef_[0][1]))})
        # verified
        fi = run.features.index("verified")
        v = Xt[:, fi]
        top1 = np.abs(sv).argmax(1) == fi
        for val in (0, 1):
            m = v == val
            lo, hi = m & (dt <= np.median(dt)), m & (dt > np.median(dt))
            ver.append({"seed": seed, "verified": val, "n": int(m.sum()), "share_bot_label": yt[m].mean(),
                        "accuracy": ok_t[m].mean(), "mean_DAUC": dt[m].mean(), "share_top1_verified": top1[m].mean(),
                        "share_in_low_DAUC_half": (m & (dt <= np.median(dt))).sum() / m.sum(),
                        "acc_low_DAUC_half": ok_t[lo].mean() if lo.any() else np.nan,
                        "acc_high_DAUC_half": ok_t[hi].mean() if hi.any() else np.nan,
                        "maxp_low_DAUC_half": pt[lo].max(1).mean() if lo.any() else np.nan,
                        "maxp_high_DAUC_half": pt[hi].max(1).mean() if hi.any() else np.nan})
    for name, rows, keys in (("ext_tg_baselines", base, []), ("ext_tg_correctness_auc", cor, []),
                             ("ext_tg_verified", ver, ["verified"])):
        d = pd.DataFrame(rows)
        write_table(d, cfg, name + "_per_seed")
        if keys:
            out = d.groupby(keys).agg({c: mean_std for c in d.columns if c not in keys + ["seed"]}).reset_index()
        else:
            out = pd.DataFrame([{c: mean_std(d[c]) for c in d.columns if c != "seed"}])
        write_table(out, cfg, name)
        print(out.T.to_string())


if __name__ == "__main__":
    main()
