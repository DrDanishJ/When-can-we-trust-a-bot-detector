"""Account age at crawl time for Cresci-17 (paper Section 3.5): crawled_at - created_at, read
from the users.csv files only (tweets are not needed)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..utils import parse_time

DAY = 86400.0


def cresci17_crawl_age(root: str | Path) -> pd.DataFrame:
    rows = []
    for folder in sorted(Path(root).glob("*.csv")):
        if not folder.is_dir():
            continue
        for f in folder.glob("*.csv"):
            if "user" not in f.name.lower():
                continue
            u = pd.read_csv(f, dtype=str, on_bad_lines="skip", encoding_errors="replace")
            u = u.rename(columns={"id": "account_id"})
            out = pd.DataFrame({"account_id": u["account_id"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True),
                                "group_raw": folder.name.replace(".csv", "")})
            for c in ("created_at", "crawled_at", "updated"):
                out[c] = parse_time(u[c]).astype("float64") if c in u.columns else np.nan
            # fake_followers have no crawled_at; their 'updated' field is the snapshot date (6 Dec 2013)
            out["crawl_source"] = np.where(out["crawled_at"].notna(), "crawled_at", "updated")
            out["crawled_at"] = out["crawled_at"].fillna(out["updated"])
            rows.append(out)
    df = pd.concat(rows, ignore_index=True).drop_duplicates("account_id")
    df["age_at_crawl_days"] = (df["crawled_at"] - df["created_at"]) / DAY
    return df
