"""SHAP, LIME and feature importance for any `Model`.

All attributions are for the bot class (label 1) and are returned as dense arrays aligned with
`model.feature_names`. Tree models use exact TreeSHAP; other models use KernelSHAP in the model's
preprocessed feature space with a k-means background.
"""
from __future__ import annotations

import numpy as np

from ..models.zoo import Model


def _bot_class(values) -> np.ndarray:
    v = values.values if hasattr(values, "values") and not isinstance(values, np.ndarray) else values
    if isinstance(v, list):
        v = v[1] if len(v) == 2 else v[0]
    v = np.asarray(v)
    if v.ndim == 3:
        v = v[:, :, 1] if v.shape[2] == 2 else v[:, :, 0]
    return v


def shap_values(model: Model, X, X_background=None, cfg: dict | None = None, max_samples: int | None = None,
                seed: int = 0) -> np.ndarray:
    import shap
    Z = model.transform(X)
    if model.is_tree:
        explainer = shap.TreeExplainer(model.clf)
        return _bot_class(explainer.shap_values(Z, check_additivity=False))
    cfg = cfg or {}
    k = cfg.get("explain", {}).get("kernel_shap_background", 50)
    Zb = model.transform(X_background if X_background is not None else X)
    bg = shap.kmeans(Zb, min(k, len(Zb))) if len(Zb) > k else Zb
    f = lambda A: model.clf.predict_proba(np.asarray(A, dtype=Z.dtype))[:, 1]  # noqa: E731
    explainer = shap.KernelExplainer(f, bg)
    return np.asarray(explainer.shap_values(Z, nsamples="auto", silent=True))


def kernel_shap_fn(fn, Z, Z_background, k: int = 50) -> np.ndarray:
    """KernelSHAP for an arbitrary function of the preprocessed features (used for conformal p-values)."""
    import shap
    bg = shap.kmeans(Z_background, min(k, len(Z_background))) if len(Z_background) > k else Z_background
    explainer = shap.KernelExplainer(fn, bg)
    return np.asarray(explainer.shap_values(Z, nsamples="auto", silent=True))


def lime_weights(model: Model, X_train, X, cfg: dict, seed: int = 0) -> np.ndarray:
    """LIME weight of every feature for every row of X (bot class). Shape (n, d)."""
    from lime.lime_tabular import LimeTabularExplainer
    Ztr = model.transform(X_train)
    Z = model.transform(X)
    d = Z.shape[1]
    explainer = LimeTabularExplainer(Ztr, feature_names=model.feature_names, class_names=["human", "bot"],
                                     discretize_continuous=True, random_state=seed, mode="classification")
    out = np.zeros((len(Z), d))
    for i, z in enumerate(Z):
        exp = explainer.explain_instance(z, model.clf.predict_proba, num_features=d, labels=(1,),
                                         num_samples=cfg["explain"]["lime_num_samples"])
        for j, w in exp.as_map()[1]:
            out[i, j] = w
    return out


def feature_importance(model: Model) -> np.ndarray:
    """Gain-based importance for boosted trees, impurity importance for forests, |coef|-free fallback = NaN."""
    clf, d = model.clf, len(model.feature_names)
    try:
        if model.name == "lightgbm":
            return np.asarray(clf.booster_.feature_importance(importance_type="gain"), dtype=float)
        if model.name == "xgboost":
            score = clf.get_booster().get_score(importance_type="gain")
            return np.array([score.get(f"f{i}", 0.0) for i in range(d)], dtype=float)
        if model.name == "catboost":
            return np.asarray(clf.get_feature_importance(), dtype=float)
        if hasattr(clf, "feature_importances_"):
            return np.asarray(clf.feature_importances_, dtype=float)
        if model.name == "ebm":
            return np.asarray(clf.term_importances()[:d], dtype=float)
    except Exception:
        pass
    return np.full(d, np.nan)
