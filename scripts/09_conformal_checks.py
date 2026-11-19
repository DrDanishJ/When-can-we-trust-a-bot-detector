"""Step 9 - paper Section 5.4: check of the conformal layer (coverage resampling, accuracy by prediction-set type).

  tables/ext_coverage_resampling.csv   mean per-class coverage over n_rep re-partitions of calib+test, per seed
  tables/ext_coverage_per_seed.csv     per-seed coverage on the original split, calibration sizes, thresholds
  tables/ext_set_type_accuracy.csv     accuracy and mean max probability per set type (original split)
"""
import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from ibd.config import cli
from ibd.extended.conformal_checks import coverage_resampling, set_type_accuracy
from ibd.runs import get_run
from ibd.uq.conformal import MondrianConformal, set_summary
from ibd.utils import mean_std, write_table

N_REP = 1000


def main():
    cfg, _ = cli(__doc__)
    alphas = cfg["conformal"]["alphas"]
    res, per_seed, acc = [], [], []
    for ds in cfg["datasets"]["train_test"]:
        for seed in cfg["models"]["seeds"]:
            run = get_run(cfg, ds, seed)
            p_cal, p_te = run.model.predict_proba(run.X("calib")), run.model.predict_proba(run.X("test"))
            y_cal, y_te = run.y("calib"), run.y("test")
            r = coverage_resampling(p_cal, y_cal, p_te, y_te, alphas, N_REP, 7000 + seed)
            for a, g in r.groupby("alpha"):
                res.append({"dataset": ds, "seed": seed, "alpha": a,
                            "mean_cov_human": g.coverage_human.mean(), "mean_cov_bot": g.coverage_bot.mean(),
                            "p05_cov_bot": g.coverage_bot.quantile(.05), "p95_cov_bot": g.coverage_bot.quantile(.95),
                            "p05_cov_human": g.coverage_human.quantile(.05), "p95_cov_human": g.coverage_human.quantile(.95)})
            for a in alphas:
                cp = MondrianConformal().fit(p_cal, y_cal, a)
                s = set_summary(cp.predict_set(p_te), y_te)
                per_seed.append({"dataset": ds, "seed": seed, "alpha": a, "n_cal_human": int((y_cal == 0).sum()),
                                 "n_cal_bot": int((y_cal == 1).sum()), "q_human": cp.q[0], "q_bot": cp.q[1],
                                 "coverage_human": s["coverage_human"], "coverage_bot": s["coverage_bot"],
                                 "n_ties_at_q_human": int((np.isclose(cp.cal_scores[0], cp.q[0], rtol=0, atol=1e-12)).sum()),
                                 "n_ties_at_q_bot": int((np.isclose(cp.cal_scores[1], cp.q[1], rtol=0, atol=1e-12)).sum())})
                t, _ = set_type_accuracy(p_cal, y_cal, p_te, y_te, a)
                t.insert(0, "alpha", a); t.insert(0, "seed", seed); t.insert(0, "dataset", ds)
                acc.append(t)
    res = pd.DataFrame(res)
    summ = res.groupby(["dataset", "alpha"]).agg(
        mean_cov_human=("mean_cov_human", "mean"), mean_cov_bot=("mean_cov_bot", "mean"),
        min_seed_mean_human=("mean_cov_human", "min"), min_seed_mean_bot=("mean_cov_bot", "min")).reset_index()
    write_table(res, cfg, "ext_coverage_resampling_per_seed")
    write_table(summ, cfg, "ext_coverage_resampling")
    write_table(pd.DataFrame(per_seed), cfg, "ext_coverage_per_seed")
    acc = pd.concat(acc)
    write_table(acc, cfg, "ext_set_type_accuracy_per_seed")
    agg = acc.groupby(["dataset", "alpha", "set_type"]).agg(
        n=("n", "mean"), share=("share", "mean"), accuracy=("accuracy", "mean"),
        accuracy_sd=("accuracy", "std"), mean_max_p=("mean_max_p", "mean")).reset_index()
    write_table(agg, cfg, "ext_set_type_accuracy")
    pd.set_option("display.width", 200)
    print(summ.to_string(index=False)); print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
