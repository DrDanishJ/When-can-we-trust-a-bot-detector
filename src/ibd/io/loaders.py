"""Loaders for Cresci-15, Cresci-17, TwiBot-20, fox8-23 and the LSABRE rewritten TwiBot-20 set.

Each `load_<dataset>(cfg)` returns `(users, tweets)` in the schema of `ibd.io.schema`. Tweets are capped to the
`data.max_tweets_per_account` most recent tweets per account while streaming, so the 1 GB Cresci-17 tweet files
never have to fit in memory.
"""
from __future__ import annotations

import ast
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils import LOG, parse_time, to_bool, to_num
from .schema import BOOL_USER_COLUMNS, COUNT_USER_COLUMNS, TEXT_USER_COLUMNS, TWEET_COLUMNS, USER_COLUMNS

_CSV_KW = dict(dtype=str, on_bad_lines="skip", encoding="utf-8", encoding_errors="replace", keep_default_na=True)


# ----------------------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------------------
def _finish_users(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in USER_COLUMNS:
        out[c] = df[c] if c in df.columns else np.nan
    out["account_id"] = out["account_id"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    out["dataset"] = dataset
    out["split"] = out["split"].fillna("").astype(str)
    out["label"] = out["label"].astype(int)
    for c in ("created_at", "crawled_at", "updated_at"):
        if not pd.api.types.is_numeric_dtype(out[c]):
            out[c] = parse_time(out[c])
        out[c] = out[c].astype("float64")
    for c in BOOL_USER_COLUMNS:
        out[c] = to_bool(out[c])
    for c in COUNT_USER_COLUMNS:
        out[c] = to_num(out[c]).astype("float64")
    for c in TEXT_USER_COLUMNS:
        out[c] = out[c].astype("string").str.strip()
        out.loc[out[c].str.lower().isin(["nan", "none", ""]), c] = pd.NA
    # one row per account; drop accounts that appear with contradictory labels
    dup = out.groupby("account_id")["label"].nunique()
    bad = set(dup[dup > 1].index)
    if bad:
        LOG.warning("%s: dropping %d accounts listed as both human and bot", dataset, len(bad))
        out = out[~out["account_id"].isin(bad)]
    out = out.drop_duplicates("account_id", keep="first").reset_index(drop=True)
    return out[USER_COLUMNS]


def _finish_tweets(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in TWEET_COLUMNS:
        out[c] = df[c] if c in df.columns else np.nan
    out["account_id"] = out["account_id"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    out["text"] = out["text"].fillna("").astype(str)
    for c in ("ts", "retweet_count", "reply_count", "favorite_count"):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    for c in ("is_reply", "is_retweet"):
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype("int8")
    out["in_reply_to_user_id"] = out["in_reply_to_user_id"].astype("string")
    return out[TWEET_COLUMNS].reset_index(drop=True)


def _cap(tweets: pd.DataFrame, n: int) -> pd.DataFrame:
    """Keep the n most recent tweets per account (accounts without timestamps keep their first n)."""
    if tweets.empty:
        return tweets
    t = tweets.assign(_o=np.arange(len(tweets)))
    t = t.sort_values(["account_id", "ts", "_o"], ascending=[True, False, True], na_position="last")
    return t.groupby("account_id", sort=False).head(n).drop(columns="_o")


def _norm_cresci_tweets(chunk: pd.DataFrame) -> pd.DataFrame:
    ts = parse_time(chunk["created_at"]) if "created_at" in chunk else pd.Series(np.nan, index=chunk.index)
    if "timestamp" in chunk:
        ts = ts.fillna(parse_time(chunk["timestamp"]))
    rt_status = chunk.get("retweeted_status_id", pd.Series(np.nan, index=chunk.index))
    text = chunk.get("text", pd.Series("", index=chunk.index)).fillna("")
    is_rt = (to_num(rt_status).fillna(0) > 0) | text.str.startswith("RT @")
    reply_to = chunk.get("in_reply_to_status_id", pd.Series(np.nan, index=chunk.index))
    return pd.DataFrame({
        "account_id": chunk["user_id"],
        "text": text,
        "ts": ts,
        "retweet_count": to_num(chunk.get("retweet_count")),
        "reply_count": to_num(chunk.get("reply_count")),
        "favorite_count": to_num(chunk.get("favorite_count")),
        "in_reply_to_user_id": chunk.get("in_reply_to_user_id"),
        "is_reply": (to_num(reply_to).fillna(0) > 0).astype("int8"),
        "is_retweet": is_rt.astype("int8"),
    })


def _stream_tweets(paths: list[str], account_ids: set[str], n: int, chunksize: int) -> pd.DataFrame:
    kept = []
    for p in paths:
        LOG.info("  reading %s", p)
        for chunk in pd.read_csv(p, chunksize=chunksize, low_memory=False, **_CSV_KW):
            if "user_id" not in chunk.columns:
                LOG.warning("  %s has no user_id column, skipped", p)
                break
            chunk["user_id"] = chunk["user_id"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
            chunk = chunk[chunk["user_id"].isin(account_ids)]
            if chunk.empty:
                continue
            kept.append(_cap(_norm_cresci_tweets(chunk), n))
            if len(kept) > 8:  # keep memory bounded
                kept = [_cap(pd.concat(kept, ignore_index=True), n)]
    if not kept:
        return _finish_tweets(pd.DataFrame(columns=TWEET_COLUMNS))
    return _finish_tweets(_cap(pd.concat(kept, ignore_index=True), n))


def _find(folder: Path, pattern: str) -> list[str]:
    return sorted(str(p) for p in folder.glob(pattern))


# ----------------------------------------------------------------------------------------------------------
# Cresci-15
# ----------------------------------------------------------------------------------------------------------
CRESCI15_GROUPS = {"E13.Real": 0, "TFP.Real": 0, "FSF.Fake": 1, "INT.Fake": 1, "TWT.Fake": 1}


def load_cresci15(cfg: dict):
    root = Path(cfg["paths"]["cresci15"])
    users, tweet_files = [], []
    for g, lab in CRESCI15_GROUPS.items():
        folder = root / g
        if not folder.exists():
            LOG.warning("cresci15: %s not found", folder)
            continue
        u = pd.read_csv(folder / "users.csv", **_CSV_KW)
        u = u.rename(columns={"id": "account_id", "updated": "updated_at"})
        u["label"], u["group"] = lab, g.split(".")[0]
        users.append(u)
        tweet_files += _find(folder, "tweets.csv")
    users = _finish_users(pd.concat(users, ignore_index=True), "cresci15")
    tweets = _stream_tweets(tweet_files, set(users["account_id"]), cfg["data"]["max_tweets_per_account"],
                            cfg["data"]["chunksize"])
    return users, tweets


# ----------------------------------------------------------------------------------------------------------
# Cresci-17 (original folder layout, or the flattened Kaggle file names used in the earlier notebooks)
# ----------------------------------------------------------------------------------------------------------
CRESCI17_FOLDERS = {
    "genuine_accounts": 0,
    "social_spambots_1": 1, "social_spambots_2": 1, "social_spambots_3": 1,
    "traditional_spambots_1": 1, "traditional_spambots_2": 1, "traditional_spambots_3": 1,
    "traditional_spambots_4": 1,
    "fake_followers": 1,
}
TWEETLESS = {"traditional_spambots_2", "traditional_spambots_3", "traditional_spambots_4"}
CRESCI17_FLAT = {  # group: (users file, tweets file, label)
    "genuine_accounts": ("GenUsers.csv", "GenTweets.csv", 0),
    "social_spambots_1": ("s1users.csv", "s1tweets.csv", 1),
    "social_spambots_2": ("s2users.csv", "s2tweets.csv", 1),
    "social_spambots_3": ("s3users.csv", "s3tweets.csv", 1),
    "fake_followers": ("FakeUsers.csv", "FakeTweets.csv", 1),
    "traditional_spambots": ("usersAllTS.csv", "TStweets.csv", 1),
}


def load_cresci17(cfg: dict):
    root = Path(cfg["paths"]["cresci17"])
    dcfg = cfg["data"]
    users, tweet_files = [], []
    use_folders = any((root / f"{g}.csv").is_dir() or (root / g).is_dir() for g in CRESCI17_FOLDERS)
    if use_folders:
        for g, lab in CRESCI17_FOLDERS.items():
            if g in TWEETLESS and not dcfg["cresci17_include_tweetless_groups"]:
                continue
            if g == "fake_followers" and not dcfg["cresci17_include_fake_followers"]:
                continue
            folder = root / f"{g}.csv" if (root / f"{g}.csv").is_dir() else root / g
            if not folder.exists():
                LOG.warning("cresci17: %s not found", folder)
                continue
            ufiles = [p for p in _find(folder, "*.csv") if "user" in Path(p).name.lower()]
            tfiles = [p for p in _find(folder, "*.csv") if "tweet" in Path(p).name.lower()]
            u = pd.concat([pd.read_csv(p, **_CSV_KW) for p in ufiles], ignore_index=True)
            u["label"], u["group"] = lab, g
            users.append(u)
            tweet_files += tfiles
    else:
        LOG.info("cresci17: using flattened Kaggle file names")
        for g, (uf, tf, lab) in CRESCI17_FLAT.items():
            if g == "fake_followers" and not dcfg["cresci17_include_fake_followers"]:
                continue
            if not (root / uf).exists():
                LOG.warning("cresci17: %s not found", root / uf)
                continue
            u = pd.read_csv(root / uf, **_CSV_KW)
            u["label"], u["group"] = lab, g
            users.append(u)
            if (root / tf).exists():
                tweet_files.append(str(root / tf))
    u = pd.concat(users, ignore_index=True).rename(columns={"id": "account_id", "updated": "updated_at"})
    users = _finish_users(u, "cresci17")
    tweets = _stream_tweets(tweet_files, set(users["account_id"]), dcfg["max_tweets_per_account"],
                            dcfg["chunksize"])
    if not use_folders and not dcfg["cresci17_include_tweetless_groups"]:
        has = set(tweets["account_id"])
        drop = (users["group"] == "traditional_spambots") & ~users["account_id"].isin(has)
        LOG.info("cresci17: dropping %d traditional spambots that have no tweets", int(drop.sum()))
        users = users[~drop].reset_index(drop=True)
    return users, tweets


# ----------------------------------------------------------------------------------------------------------
# TwiBot-20
# ----------------------------------------------------------------------------------------------------------
_RT = re.compile(r"^\s*RT @")


def _read_json_list(path: Path):
    try:
        import ijson  # streams the 239 MB train.json without loading it whole
        with open(path, "rb") as fh:
            yield from ijson.items(fh, "item")
    except ImportError:
        with open(path, encoding="utf-8") as fh:
            yield from json.load(fh)


def _twibot_records(path: Path, split: str, n: int):
    urows, trows = [], []
    for rec in _read_json_list(path):
        prof = rec.get("profile") or {}
        aid = str(rec.get("ID") or prof.get("id_str") or prof.get("id")).strip()
        urows.append({
            "account_id": aid, "label": int(str(rec.get("label", 0)).strip() or 0), "group": "twibot20",
            "split": split, "created_at": prof.get("created_at"),
            **{c: prof.get(c) for c in BOOL_USER_COLUMNS + COUNT_USER_COLUMNS + TEXT_USER_COLUMNS},
            "domain": ",".join(rec.get("domain") or []),
        })
        for text in (rec.get("tweet") or [])[:n]:
            text = str(text)
            trows.append({"account_id": aid, "text": text, "is_retweet": int(bool(_RT.match(text)))})
    return urows, trows


def load_twibot20(cfg: dict):
    root = Path(cfg["paths"]["twibot20"])
    n = cfg["data"]["max_tweets_per_account"]
    urows, trows = [], []
    for split in ("train", "dev", "test"):
        p = root / f"{split}.json"
        if not p.exists():
            LOG.warning("twibot20: %s not found", p)
            continue
        LOG.info("  reading %s", p)
        u, t = _twibot_records(p, split, n)
        urows += u
        trows += t
    raw = pd.DataFrame(urows)
    users = _finish_users(raw, "twibot20")
    users = users.merge(raw[["account_id", "domain"]].drop_duplicates("account_id"), on="account_id", how="left")
    lsabre_ids = _lsabre_ids(cfg)
    users["in_lsabre"] = users["account_id"].isin(lsabre_ids).astype("int8")
    return users, _finish_tweets(pd.DataFrame(trows))


# ----------------------------------------------------------------------------------------------------------
# fox8-23 (Twitter API v1.1 tweet objects, one JSON account per line)
# ----------------------------------------------------------------------------------------------------------
def load_fox8(cfg: dict):
    path = Path(cfg["paths"]["fox8"])
    n = cfg["data"]["max_tweets_per_account"]
    urows, trows = [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            aid = str(rec.get("user_id"))
            tweets = rec.get("user_tweets") or []
            user = next((t.get("user") for t in tweets if isinstance(t, dict) and t.get("user")), {}) or {}
            urows.append({
                "account_id": aid, "label": 1 if str(rec.get("label")).lower() == "bot" else 0,
                "group": rec.get("dataset", "fox8"), "created_at": user.get("created_at"),
                **{c: user.get(c) for c in BOOL_USER_COLUMNS + COUNT_USER_COLUMNS + TEXT_USER_COLUMNS},
            })
            for t in tweets[:n]:
                text = t.get("full_text") or t.get("text") or ""
                rs = t.get("retweeted_status")
                trows.append({
                    "account_id": aid, "text": text, "ts": t.get("created_at"),
                    "retweet_count": t.get("retweet_count"), "reply_count": t.get("reply_count"),
                    "favorite_count": t.get("favorite_count"),
                    "in_reply_to_user_id": t.get("in_reply_to_user_id_str") or t.get("in_reply_to_user_id"),
                    "is_reply": int(t.get("in_reply_to_status_id") is not None),
                    "is_retweet": int(bool(rs) or bool(_RT.match(text))),
                })
    users = _finish_users(pd.DataFrame(urows), "fox8")
    tw = pd.DataFrame(trows)
    if not tw.empty:
        tw["ts"] = parse_time(tw["ts"])
    return users, _cap(_finish_tweets(tw), n)


# ----------------------------------------------------------------------------------------------------------
# LSABRE: TwiBot-20 bots with LLM-rewritten tweets (github.com/runi-cyber-ai/LSABRE, dataset/ folder)
# ----------------------------------------------------------------------------------------------------------
def _lsabre_frame(cfg: dict) -> pd.DataFrame:
    files = sorted(glob.glob(str(Path(cfg["paths"]["lsabre"]) / "*.csv")))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f, dtype=str) for f in files], ignore_index=True)
    df["ID"] = df["ID"].astype(str).str.strip()
    return df.drop_duplicates("ID")


def _lsabre_ids(cfg: dict) -> set[str]:
    try:
        return set(_lsabre_frame(cfg)["ID"])
    except Exception:
        return set()


def _parse_list(s) -> list[str]:
    if not isinstance(s, str):
        return []
    try:
        v = ast.literal_eval(s)
        return [str(x) for x in v] if isinstance(v, (list, tuple)) else [str(v)]
    except Exception:
        return [s]


def load_lsabre(cfg: dict, twibot_users: pd.DataFrame, twibot_tweets: pd.DataFrame):
    """Build the rewritten test set: for every rewrite variant, the 1,000 LSABRE bots with their profile from
    TwiBot-20 and their tweets replaced by the rewrite, plus the same bots with their original tweets
    (group 'original') and the TwiBot-20 test-split humans. account_id is '<ID>::<variant>'."""
    lf = _lsabre_frame(cfg)
    if lf.empty:
        raise FileNotFoundError(f"no LSABRE csv files in {cfg['paths']['lsabre']}")
    variants = [c for c in lf.columns if c != "ID"]
    if cfg["shift"].get("lsabre_variants", "all") != "all":
        variants = [v for v in variants if v in cfg["shift"]["lsabre_variants"]]
    n = cfg["data"]["max_tweets_per_account"]
    tu = twibot_users.set_index("account_id")
    bots = [i for i in lf["ID"] if i in tu.index]
    LOG.info("lsabre: %d of %d rewritten bots found in TwiBot-20", len(bots), len(lf))
    humans = twibot_users[(twibot_users["split"] == "test") & (twibot_users["label"] == 0)]
    urows, tparts = [], []
    orig = twibot_tweets[twibot_tweets["account_id"].isin(bots)]
    for variant in ["original"] + variants:
        u = tu.loc[bots].reset_index()
        u["group"] = variant
        u["account_id"] = u["account_id"] + "::" + variant
        urows.append(u)
        if variant == "original":
            t = orig.assign(account_id=orig["account_id"] + "::original")
        else:
            rows = []
            for _, r in lf[lf["ID"].isin(bots)].iterrows():
                for text in _parse_list(r[variant])[:n]:
                    rows.append({"account_id": f"{r['ID']}::{variant}", "text": text,
                                 "is_retweet": int(bool(_RT.match(text)))})
            t = _finish_tweets(pd.DataFrame(rows))
        tparts.append(t)
    h = humans.assign(group="human", account_id=humans["account_id"] + "::human")
    urows.append(h)
    ht = twibot_tweets[twibot_tweets["account_id"].isin(set(humans["account_id"]))]
    tparts.append(ht.assign(account_id=ht["account_id"] + "::human"))
    users = pd.concat(urows, ignore_index=True)
    users["dataset"] = "lsabre"
    keep = [c for c in USER_COLUMNS if c in users.columns]
    return users[keep].reset_index(drop=True), _finish_tweets(pd.concat(tparts, ignore_index=True))


LOADERS = {"cresci15": load_cresci15, "cresci17": load_cresci17, "twibot20": load_twibot20, "fox8": load_fox8}
