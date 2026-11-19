"""Paper figures (PNG + PDF). One palette for the whole paper, validated for colour-vision deficiency:
slot 1 blue, slot 2 orange, slot 3 aqua (aqua is always paired with a distinct line style and a legend)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .features.registry import BY_NAME  # noqa: E402
from .utils import LOG, results_dir  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "figure.dpi": 150,
})


def save(fig, cfg, name):
    d = results_dir(cfg, "figures")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(d / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    LOG.info("wrote %s.png/.pdf", d / name)


def shap_bar(cfg, name, features, mean_abs, lo=None, hi=None, top=20, title=""):
    """Mean |SHAP| per feature with bootstrap 95% interval, coloured by tag (behavioural vs artifact-prone)."""
    order = np.argsort(mean_abs)[::-1][:top][::-1]
    f = [features[i] for i in order]
    v = np.asarray(mean_abs)[order]
    colors = [ORANGE if BY_NAME[x].artifact_prone else BLUE for x in f]
    fig, ax = plt.subplots(figsize=(5.2, 0.24 * len(f) + 0.9))
    xerr = None
    if lo is not None and hi is not None:
        lo_ = np.clip(np.asarray(lo)[order], 0, None)  # |SHAP| is non-negative
        xerr = np.vstack([v - lo_, np.asarray(hi)[order] - v]).clip(min=0)
    ax.barh(range(len(f)), v, color=colors, height=0.7, xerr=xerr,
            error_kw=dict(ecolor=MUTED, lw=0.8, capsize=2))
    ax.set_yticks(range(len(f)), f)
    ax.set_xlabel("mean |SHAP value| (bot class)")
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE, label="behavioural"), Patch(color=ORANGE, label="artifact-prone")],
              loc="lower right")
    if title:
        ax.set_title(title, loc="left", color=INK)
    save(fig, cfg, name)


def signed_bar(cfg, name, labels, values, xlabel, title="", top=20, legend=("towards bot", "towards human")):
    """Signed contributions (LIME weights, SHAP of p-values): orange = positive, blue = negative; `legend` names
    the two directions."""
    order = np.argsort(np.abs(values))[::-1][:top][::-1]
    lab = [labels[i] for i in order]
    v = np.asarray(values)[order]
    fig, ax = plt.subplots(figsize=(5.2, 0.24 * len(lab) + 0.9))
    ax.barh(range(len(lab)), v, color=[ORANGE if x > 0 else BLUE for x in v], height=0.7)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(range(len(lab)), lab)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=ORANGE, label=legend[0]), Patch(color=BLUE, label=legend[1])],
              loc="lower right")
    if title:
        ax.set_title(title, loc="left", color=INK)
    save(fig, cfg, name)


def importance_bar(cfg, name, features, imp, top=20, title=""):
    order = np.argsort(imp)[::-1][:top][::-1]
    f = [features[i] for i in order]
    v = np.asarray(imp)[order]
    fig, ax = plt.subplots(figsize=(5.2, 0.24 * len(f) + 0.9))
    ax.barh(range(len(f)), v, color=[ORANGE if BY_NAME[x].artifact_prone else BLUE for x in f], height=0.7)
    ax.set_yticks(range(len(f)), f)
    ax.set_xlabel("gain-based importance")
    ax.grid(axis="y", visible=False)
    if title:
        ax.set_title(title, loc="left", color=INK)
    save(fig, cfg, name)


def reliability_diagram(cfg, name, curves: dict):
    """curves: {label: (mean_pred, frac_pos, ece)}; one panel per dataset."""
    n = len(curves)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 2.7), squeeze=False)
    for ax, (lab, (mp, fp, ece)) in zip(axes[0], curves.items()):
        ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls="--")
        ax.plot(mp, fp, color=BLUE, lw=2, marker="o", ms=4)
        ax.set_title(f"{lab}  (ECE {ece:.3f})", loc="left", fontsize=9)
        ax.set_xlabel("predicted bot probability")
        ax.set_xlim(0, 1), ax.set_ylim(0, 1)
    axes[0][0].set_ylabel("observed bot fraction")
    save(fig, cfg, name)


def lines(cfg, name, series: dict, xlabel, ylabel, title=""):
    """series: {label: (x, y, yerr or None)}; up to four series (blue, orange, aqua, grey with distinct dashes)."""
    styles = [(BLUE, "-", "o"), (ORANGE, "--", "s"), (AQUA, ":", "^"), (MUTED, "-.", "D")]
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for (lab, (x, y, err)), (c, ls, mk) in zip(series.items(), styles):
        ax.plot(x, y, color=c, ls=ls, lw=2, marker=mk, ms=4, label=lab)
        if err is not None:
            ax.fill_between(x, np.asarray(y) - err, np.asarray(y) + err, color=c, alpha=0.15, lw=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", fontsize=8)
    if title:
        ax.set_title(title, loc="left", color=INK)
    save(fig, cfg, name)
