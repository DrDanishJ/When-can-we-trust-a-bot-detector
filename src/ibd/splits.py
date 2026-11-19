"""Train / model-selection / calibration / test splits.

Cresci-15 and Cresci-17: stratified random split per seed (sizes from cfg['split']).
TwiBot-20: official train / test; the official dev split is halved (stratified, per seed) into
model-selection and calibration parts. LSABRE bots are removed from TwiBot-20 training so that the
rewritten-tweet test set never contains accounts the model was trained on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def make_splits(df: pd.DataFrame, seed: int, cfg: dict) -> dict[str, np.ndarray]:
    y = df["label"].to_numpy()
    idx = np.arange(len(df))
    if df["dataset"].iloc[0] == "twibot20" and (df["split"] != "").any():
        tr = idx[df["split"].to_numpy() == "train"]
        dev = idx[df["split"].to_numpy() == "dev"]
        te = idx[df["split"].to_numpy() == "test"]
        if "in_lsabre" in df.columns:
            tr = tr[df["in_lsabre"].to_numpy()[tr] == 0]
        sel, cal = train_test_split(dev, test_size=0.5, stratify=y[dev], random_state=seed)
        return {"train": tr, "select": np.sort(sel), "calib": np.sort(cal), "test": te}
    s = cfg["split"]
    rest, te = train_test_split(idx, test_size=s["test_size"], stratify=y, random_state=seed)
    holdout = s["select_size"] + s["calib_size"]
    tr, hold = train_test_split(rest, test_size=holdout / (1 - s["test_size"]), stratify=y[rest], random_state=seed)
    sel, cal = train_test_split(hold, test_size=s["calib_size"] / holdout, stratify=y[hold], random_state=seed)
    return {"train": np.sort(tr), "select": np.sort(sel), "calib": np.sort(cal), "test": np.sort(te)}
