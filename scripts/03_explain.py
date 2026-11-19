"""Step 3 - paper Section 6.2 (Figs. 4-9): SHAP, LIME and feature importance for the explain model.

Outputs per dataset (standard shap / lime library plots, first-seed explain model)
  figures/fig_shap_<ds>.png                   SHAP summary (beeswarm), top 20 features
  figures/fig_lime_<ds>_<kind>.html/.png/.pdf full LIME view for a confident-correct, an ambiguous and a
                                              misclassified account (HTML rendered with headless Chrome, 2x)
  figures/fig_lime_bar_<ds>_<kind>.png        LIME weights only (lime's as_pyplot_figure)
  figures/fig_force_<ds>_<kind>.png           SHAP force plot (probability output) for the same accounts
  figures/fig_waterfall_<ds>_<kind>.png       SHAP waterfall (probability output) for the same accounts
  figures/fig_fi_<ds>.png                     gain-based feature importance
  tables/shap_importance_<ds>.csv             mean |SHAP| per feature and seed
  tables/table_fi_vs_shap.csv                 Kendall tau between FI and mean |SHAP| rankings
"""
import _bootstrap  # noqa: F401

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from scipy.stats import kendalltau

from ibd import plots
from ibd.config import cli
from ibd.explain.core import _bot_class, feature_importance, shap_values
from ibd.runs import get_run
from ibd.uq.conformal import MondrianConformal, set_type
from ibd.utils import LOG, results_dir, write_table

# The shap / lime plots are drawn in matplotlib's default style (as in docs/figure_style/): the project style set
# in ibd.plots (serif font, grid, muted spines) would otherwise leak into them.
LIBRARY_STYLE = {k: v for k, v in mpl.rcParamsDefault.items() if k not in ("backend", "backend_fallback", "interactive")}
LIBRARY_STYLE["axes.grid"] = False


def save_current(cfg, name, fig=None):
    """Save the current (or given) matplotlib figure as PNG and PDF at 300 dpi, then close it."""
    fig = fig or plt.gcf()
    d = results_dir(cfg, "figures")
    for ext in ("png", "pdf"):
        fig.savefig(d / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    LOG.info("wrote %s.png/.pdf", d / name)


def probability_shap(model, X_train, X, cfg, seed=0, n_background=200):
    """Bot-class SHAP values and base value in probability space, for the force and waterfall plots.
    Tree models: interventional TreeSHAP on 200 random training rows; others: KernelSHAP on p(bot) as in
    shap_values(), which is already in probability space."""
    Ztr = model.transform(X_train)
    if model.is_tree:
        bg = Ztr[np.random.RandomState(seed).choice(len(Ztr), min(n_background, len(Ztr)), replace=False)]
        # an explicit masker keeps all background rows (a plain array is cut to 100 by shap)
        explainer = shap.TreeExplainer(model.clf, data=shap.maskers.Independent(bg, max_samples=len(bg)),
                                       model_output="probability",
                                       feature_perturbation="interventional")
        sv = _bot_class(explainer.shap_values(model.transform(X)))
    else:
        k = cfg["explain"]["kernel_shap_background"]
        bg = shap.kmeans(Ztr, min(k, len(Ztr))) if len(Ztr) > k else Ztr
        explainer = shap.KernelExplainer(lambda A: model.clf.predict_proba(A)[:, 1], bg)
        sv = np.asarray(explainer.shap_values(model.transform(X), nsamples="auto", silent=True))
    ev = np.atleast_1d(np.asarray(explainer.expected_value, dtype=float))
    return sv, float(ev[1] if len(ev) == 2 else ev[0])


def pick_accounts(run, alpha):
    """Indices (into the test split) of a confident-correct, an ambiguous and a misclassified account."""
    m = run.model
    p_te, p_cal = m.predict_proba(run.X("test")), m.predict_proba(run.X("calib"))
    sets = MondrianConformal().fit(p_cal, run.y("calib"), alpha).predict_set(p_te)
    kinds = set_type(sets)
    y = run.y("test")
    pred = p_te.argmax(1)
    conf = p_te.max(1)
    out = {}
    ok = np.where((pred == y) & (kinds != "ambiguous"))[0]
    if len(ok):
        out["confident"] = ok[np.argmax(conf[ok])]
    amb = np.where(kinds == "ambiguous")[0]
    if len(amb):
        out["ambiguous"] = amb[np.argmin(np.abs(p_te[amb, 1] - 0.5))]
    wrong = np.where(pred != y)[0]
    if len(wrong):
        out["misclassified"] = wrong[np.argmax(conf[wrong])]
    return out


def render_lime_html(browser, html_path, cfg, name):
    """Render a saved LIME HTML view (probabilities, weights, feature-value table) to PNG (2x scale) and PDF,
    cropped to the explanation."""
    d = results_dir(cfg, "figures")
    page = browser.new_page(viewport={"width": 1400, "height": 900}, device_scale_factor=2)
    page.goto(html_path.resolve().as_uri())
    # shrink the container to its content and show the whole feature table (LIME caps it at 300 px with scrollbars)
    page.add_style_tag(content="body{margin:0;background:white} .lime.top_div{display:inline-flex;flex-wrap:nowrap}"
                               " .lime.table_div,.lime.text_div{max-height:none!important;overflow:visible!important}"
                               " .lime.table_div td,.lime.table_div th{padding:1px 8px}")
    page.wait_for_selector(".lime.top_div svg")
    el = page.locator(".lime.top_div").first
    el.screenshot(path=str(d / f"{name}.png"))
    box = el.bounding_box()
    page.emulate_media(media="screen")
    page.pdf(path=str(d / f"{name}.pdf"), width=f"{int(np.ceil(box['x'] + box['width']))}px",
             height=f"{int(np.ceil(box['y'] + box['height']))}px", print_background=True, page_ranges="1")
    page.close()
    LOG.info("wrote %s.png/.pdf", d / name)


def main():
    cfg, _ = cli(__doc__)
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", headless=True)  # the locally installed Chrome
        explain_all(cfg, browser)
        browser.close()


def explain_all(cfg, browser):
    ecfg = cfg["explain"]
    tau_rows = []
    for ds in cfg["datasets"]["train_test"]:
        per_seed, fi_seed = [], []
        for seed in cfg["models"]["seeds"]:
            run = get_run(cfg, ds, seed)
            Xte = run.X("test")[: ecfg["shap_max_samples"]]
            sv = shap_values(run.model, Xte, run.X("train"), cfg)
            per_seed.append(np.abs(sv).mean(axis=0))
            fi_seed.append(feature_importance(run.model))
            if seed == cfg["models"]["seeds"][0]:
                first, sv_first, X_first = run, sv, Xte
        feats = first.features
        S = np.vstack(per_seed)
        mean = S.mean(0)
        tbl = pd.DataFrame(S.T, columns=[f"seed{s}" for s in cfg["models"]["seeds"]])
        tbl.insert(0, "feature", feats)
        tbl["mean_abs_shap"] = mean
        write_table(tbl.sort_values("mean_abs_shap", ascending=False), cfg, f"shap_importance_{ds}")
        with plt.rc_context(LIBRARY_STYLE):
            shap.summary_plot(sv_first, X_first, feature_names=feats, max_display=ecfg["top_features_plot"],
                              show=False)
            save_current(cfg, f"fig_shap_{ds}")
        FI = np.vstack(fi_seed).mean(0)
        if not np.isnan(FI).all():
            plots.importance_bar(cfg, f"fig_fi_{ds}", feats, FI, top=ecfg["top_features_plot"], title=f"{ds}: feature importance")
            tau_rows.append({"dataset": ds, "kendall_tau_fi_vs_shap": kendalltau(FI, mean).statistic})
        # LIME, force and waterfall plots for three chosen accounts (first-seed model)
        picks = pick_accounts(first, cfg["conformal"]["alphas"][0])
        if picks:
            m = first.model
            X_tr = first.X("train")
            rows = np.array(list(picks.values()))
            x_raw = first.X("test")[rows]
            x_show = x_raw.round(2)  # raw feature values shown in the force and waterfall plots
            sv, ev = probability_shap(m, X_tr, x_raw, cfg)  # f(x) and E[f(X)] are probabilities
            explanation = shap.Explanation(values=sv, base_values=np.full(len(rows), ev), data=x_show,
                                           feature_names=feats)
            # LIME works in the model's preprocessed space (as lime_weights does): NaNs imputed, scaled if needed
            lime_exp = LimeTabularExplainer(m.transform(X_tr), feature_names=feats, class_names=["human", "bot"],
                                            discretize_continuous=True, random_state=0, mode="classification")
            fig_dir = results_dir(cfg, "figures")
            for j, kind in enumerate(picks):
                exp = lime_exp.explain_instance(m.transform(x_raw[j:j + 1])[0], m.clf.predict_proba,
                                                num_features=10, num_samples=ecfg["lime_num_samples"])
                html = fig_dir / f"fig_lime_{ds}_{kind}.html"
                exp.save_to_file(str(html))
                render_lime_html(browser, html, cfg, f"fig_lime_{ds}_{kind}")
                with plt.rc_context(LIBRARY_STYLE):
                    save_current(cfg, f"fig_lime_bar_{ds}_{kind}", exp.as_pyplot_figure())
                    shap.force_plot(ev, sv[j], x_show[j], feature_names=feats, matplotlib=True, show=False)
                    save_current(cfg, f"fig_force_{ds}_{kind}")
                    shap.plots.waterfall(explanation[j], max_display=10, show=False)
                    save_current(cfg, f"fig_waterfall_{ds}_{kind}")
        LOG.info("%s explained", ds)
    if tau_rows:
        write_table(pd.DataFrame(tau_rows), cfg, "table_fi_vs_shap")


if __name__ == "__main__":
    main()
