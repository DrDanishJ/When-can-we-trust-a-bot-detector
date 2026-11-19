"""Step 5 - paper Section 6.4 (Table 13, Figs. 10-11): conformal uncertainty.

  table13_conformal.csv              coverage per class, mean set size, singleton / ambiguous rate per alpha
  table13_calibration.csv            Brier score and ECE of the explain model
  figures/fig10_reliability.png      reliability diagram per dataset (seed 0)
  figures/fig11_pvalue_shap_<ds>.png KernelSHAP on the second-largest conformal p-value (ambiguous accounts)
  tables/pvalue_shap_<ds>.csv        the values behind Fig. 11
"""
import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from ibd import plots
from ibd.config import cli
from ibd.explain.core import kernel_shap_fn
from ibd.metrics import classification_metrics, reliability_curve
from ibd.runs import get_run
from ibd.uq.conformal import MondrianConformal, set_summary, set_type
from ibd.utils import LOG, mean_std, write_table


def main():
    cfg, _ = cli(__doc__)
    ccfg = cfg["conformal"]
    rows, calib_rows, curves = [], [], {}
    for ds in cfg["datasets"]["train_test"]:
        stats = {a: [] for a in ccfg["alphas"]}
        cal_stats = []
        for seed in cfg["models"]["seeds"]:
            run = get_run(cfg, ds, seed)
            p_cal, p_te = run.model.predict_proba(run.X("calib")), run.model.predict_proba(run.X("test"))
            for a in ccfg["alphas"]:
                cp = MondrianConformal().fit(p_cal, run.y("calib"), a)
                stats[a].append(set_summary(cp.predict_set(p_te), run.y("test")))
            met = classification_metrics(run.y("test"), p_te[:, 1])
            cal_stats.append(met)
            if seed == cfg["models"]["seeds"][0]:
                mp, fp, _ = reliability_curve(run.y("test"), p_te[:, 1], ccfg["calibration_bins"])
                curves[ds] = (mp, fp, met["ece"])
                first = run
        for a, lst in stats.items():
            df = pd.DataFrame(lst)
            rows.append({"dataset": ds, "alpha": a, **{c: mean_std(df[c]) for c in df.columns}})
        cdf = pd.DataFrame(cal_stats)
        calib_rows.append({"dataset": ds, "brier": mean_std(cdf["brier"]), "ece": mean_std(cdf["ece"])})

        # KernelSHAP on the second-largest conformal p-value for ambiguous accounts (seed 0, first alpha).
        # A set is ambiguous exactly when this p-value exceeds alpha, so features with positive SHAP push the
        # account towards ambiguity.
        run, a = first, ccfg["alphas"][0]
        cp = MondrianConformal().fit(run.model.predict_proba(run.X("calib")), run.y("calib"), a)
        p_te = run.model.predict_proba(run.X("test"))
        amb = np.where(set_type(cp.predict_set(p_te)) == "ambiguous")[0][: ccfg["pvalue_shap_accounts"]]
        if len(amb) == 0:
            LOG.warning("%s: no ambiguous accounts at alpha=%s; skipping Fig. 11", ds, a)
            continue
        Zbg, Z = run.model.transform(run.X("train")), run.model.transform(run.X("test")[amb])
        k = cfg["explain"]["kernel_shap_background"]
        clf = run.model.clf
        second_p = lambda A_: np.sort(cp.pvalues(clf.predict_proba(np.asarray(A_, dtype=Z.dtype))), axis=1)[:, -2]  # noqa: E731
        phi = kernel_shap_fn(second_p, Z, Zbg, k)
        tbl = pd.DataFrame({"feature": run.features, "mean_shap_p2": phi.mean(0), "mean_abs_shap_p2": np.abs(phi).mean(0)})
        write_table(tbl.sort_values("mean_shap_p2", key=np.abs, ascending=False), cfg, f"pvalue_shap_{ds}")
        plots.signed_bar(cfg, f"fig11_pvalue_shap_{ds}", run.features, tbl["mean_shap_p2"].to_numpy(),
                         "mean SHAP on second-largest p-value  (+ = towards ambiguity)",
                         title=f"{ds}: sources of ambiguity ({len(amb)} accounts, alpha={a})",
                         top=cfg["explain"]["top_features_plot"], legend=("towards ambiguity", "towards a singleton"))
    write_table(pd.DataFrame(rows), cfg, "table13_conformal")
    write_table(pd.DataFrame(calib_rows), cfg, "table13_calibration")
    plots.reliability_diagram(cfg, "fig10_reliability", curves)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
