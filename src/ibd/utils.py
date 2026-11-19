"""Small shared helpers: logging, seeding, robust timestamp parsing, result writing."""
from __future__ import annotations

import logging
import os
import random
import re
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger("ibd")
if not LOG.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
    LOG.addHandler(h)
    LOG.setLevel(logging.INFO)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


@contextmanager
def timer(msg: str):
    t0 = time.time()
    LOG.info("%s ...", msg)
    yield
    LOG.info("%s done in %.1fs", msg, time.time() - t0)


_MS = re.compile(r"^\s*(\d{12,14})L?\s*$")
_TWITTER_FMT = "%a %b %d %H:%M:%S %z %Y"


def parse_time(values) -> pd.Series:
    """Parse the mixture of timestamp formats found in the Cresci, TwiBot-20 and fox8 files.

    Handles 'Tue Mar 26 14:46:37 +0000 2013', '2015-02-14 10:54:49', '1183552203000L' (ms since epoch)
    and '12/6/2013 18:38'. Returns UNIX seconds (float, UTC); unparseable values become NaN.
    """
    s = pd.Series(values, dtype="object").astype("string").str.strip()
    out = pd.Series(np.nan, index=s.index, dtype="float64")
    ms = s.str.extract(_MS)[0]
    is_ms = ms.notna()
    out[is_ms] = ms[is_ms].astype("float64") / 1000.0
    rest = ~is_ms & s.notna() & (s != "") & (s.str.lower() != "nan") & (s.str.lower() != "none")
    if rest.any():
        tw = pd.to_datetime(s[rest], format=_TWITTER_FMT, errors="coerce", utc=True)
        left = tw.isna()
        if left.any():
            tw[left] = pd.to_datetime(s[rest][left], errors="coerce", utc=True, format="mixed")
        out[rest] = (tw - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()
    return out


def to_bool(values) -> pd.Series:
    """Map the many truthy encodings ('True ', 1, '1', 'true', NaN) to 0/1 integers."""
    s = pd.Series(values, dtype="object").astype("string").str.strip().str.lower()
    return s.isin(["1", "1.0", "true", "t", "yes"]).astype("int8")


def to_num(values) -> pd.Series:
    s = pd.Series(values, dtype="object").astype("string").str.strip()
    return pd.to_numeric(s, errors="coerce")


def results_dir(cfg: dict, *parts: str) -> Path:
    p = Path(cfg["paths"]["results"]).joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_table(df: pd.DataFrame, cfg: dict, name: str, floatfmt: str = "%.3f") -> Path:
    """Write a results table as CSV and as a LaTeX tabular (for pasting into the paper)."""
    d = results_dir(cfg, "tables")
    df.to_csv(d / f"{name}.csv", index=False)
    try:
        (d / f"{name}.tex").write_text(df.to_latex(index=False, float_format=lambda x: floatfmt % x))
    except Exception:  # to_latex needs jinja2 in some pandas versions
        pass
    LOG.info("wrote %s", d / f"{name}.csv")
    return d / f"{name}.csv"


def mean_std(values, fmt: str = "{:.3f} ± {:.3f}") -> str:
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return "-"
    return fmt.format(v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0)
