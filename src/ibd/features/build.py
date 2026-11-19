"""Account-level feature extraction (paper eqs. 1-35), identical for every dataset.

`build_features(users, tweets, cfg)` returns one row per account: meta columns (account_id, label, dataset,
group, split, ...) followed by every feature in `registry.FEATURE_NAMES`. A feature the dataset cannot support
(e.g. tweet timing for TwiBot-20, which has no tweet timestamps) is NaN, never a fake 0.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
import pandas as pd

from ..utils import LOG
from .registry import FEATURE_NAMES

DAY = 86400.0
URL_RE = re.compile(r"https?://\S+|www\.\S+")
HASHTAG_RE = re.compile(r"#\w+")
MENTION_RE = re.compile(r"@\w+")
PUNCT_RE = re.compile(r"[!?,.;:]")
ELONG_RE = re.compile(r"\b\w*(\w)\1{2,}\w*\b")
SENT_SPLIT_RE = re.compile(r"[.!?]+")
WORD_RE = re.compile(r"\b[^\W\d_][\w']*\b", re.UNICODE)
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF\U0001F900-\U0001F9FF]")

META_COLUMNS = ["account_id", "label", "dataset", "group", "split"]


# ----------------------------------------------------------------------------------------------------------
# per-account text features (run in worker processes)
# ----------------------------------------------------------------------------------------------------------
def _readability(text: str) -> tuple[float, float]:
    import textstat
    if len(text.split()) < 3:
        return np.nan, np.nan
    try:
        mr = np.mean([textstat.dale_chall_readability_score(text), textstat.flesch_reading_ease(text),
                      textstat.automated_readability_index(text)])
        fsi = textstat.gunning_fog(text) + textstat.smog_index(text)
        return float(mr), float(fsi)
    except Exception:
        return np.nan, np.nan


def _text_features(texts: list[str], spam_words: frozenset, extreme: float, consistent: float) -> dict:
    from textblob import TextBlob
    n = len(texts)
    if n == 0:
        return {}
    raw = texts
    clean = [URL_RE.sub("", t).strip() for t in raw]
    lengths = np.array([len(t) for t in clean], dtype=float)
    words_per = [WORD_RE.findall(t.lower()) for t in clean]
    all_words = [w for ws in words_per for w in ws]
    total_words = len(all_words)
    joined = " ".join(clean)
    letters = [c for c in joined if c.isalpha()]
    pol, sub = [], []
    for t in clean:
        if t:
            s = TextBlob(t).sentiment
            pol.append(s.polarity)
            sub.append(s.subjectivity)
        else:
            pol.append(0.0)
            sub.append(0.0)
    pol, sub = np.array(pol), np.array(sub)
    hashtags = [h.lower() for t in raw for h in HASHTAG_RE.findall(t)]
    counts = Counter(all_words)
    probs = np.array(list(counts.values()), dtype=float) / max(total_words, 1)
    sentences = [s for s in SENT_SPLIT_RE.split(joined) if s.strip()]
    mr, fsi = _readability(joined)
    sub_drift = float(sub.std())
    return {
        "n_tweets_collected": n,
        "avg_tweet_length": float(lengths.mean()),
        "min_tweet_length": float(lengths.min()),
        "max_tweet_length": float(lengths.max()),
        "tweet_length_irregularity": float(lengths.std()),
        "duplicate_tweet_ratio": 1.0 - len(set(clean)) / n,
        "unique_word_count": float(len(counts)),
        "unique_word_use": len(counts) / n,
        "punctuation_density": sum(len(PUNCT_RE.findall(t)) for t in clean) / max(total_words, 1),
        "avg_elongated_words": sum(len(ELONG_RE.findall(t)) for t in clean) / n,
        "spam_ad": float(sum(1 for w in all_words if w in spam_words)),
        "spam_density": sum(1 for w in all_words if w in spam_words) / max(total_words, 1),
        "avg_sentence_length": total_words / max(len(sentences), 1),
        "capitalization_rate": sum(c.isupper() for c in letters) / max(len(letters), 1),
        "special_char_rate": sum(1 for c in joined if not c.isalnum() and not c.isspace()) / max(len(joined), 1),
        "emoji_rate": sum(len(EMOJI_RE.findall(t)) for t in raw) / n,
        "avg_hashtags": len(hashtags) / n,
        "avg_mentions": sum(len(MENTION_RE.findall(t)) for t in raw) / n,
        "avg_urls": sum(len(URL_RE.findall(t)) for t in raw) / n,
        "hashtag_diversity": len(set(hashtags)) / n,
        "mean_readability": mr,
        "fog_smog_index": fsi,
        "fog_smog_to_tweet": fsi / n if not math.isnan(fsi) else np.nan,
        "content_entropy": float(-(probs * np.log2(probs)).sum()) if total_words else np.nan,
        "avg_polarity": float(pol.mean()),
        "avg_subjectivity": float(sub.mean()),
        "min_polarity": float(pol.min()),
        "max_polarity": float(pol.max()),
        "min_subjectivity": float(sub.min()),
        "max_subjectivity": float(sub.max()),
        "polarity_drift": float(pol.std()),
        "extreme_polarity": float((np.abs(pol) > extreme).mean()),
        "subjectivity_drift": sub_drift,
        "consistent_subjectivity": float(sub_drift < consistent),
    }


def _text_worker(batch, spam_words, extreme, consistent):
    return [(aid, _text_features(texts, spam_words, extreme, consistent)) for aid, texts in batch]


def text_features(tweets: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    from joblib import Parallel, delayed
    fcfg = cfg["features"]
    spam = frozenset(w.lower() for w in fcfg["spam_keywords"])
    groups = list(tweets.groupby("account_id", sort=False)["text"].apply(list).items())
    if not groups:
        return pd.DataFrame(columns=["account_id"])
    n_jobs = fcfg.get("n_jobs", -1)
    size = max(1, min(500, len(groups) // 8 or 1))
    batches = [groups[i:i + size] for i in range(0, len(groups), size)]
    LOG.info("  text features for %d accounts in %d batches", len(groups), len(batches))
    res = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_text_worker)(b, spam, fcfg["extreme_polarity_threshold"], fcfg["consistent_subjectivity_threshold"])
        for b in batches)
    rows = [{"account_id": aid, **feats} for part in res for aid, feats in part]
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------------
# tweet-structure and timing features (vectorised)
# ----------------------------------------------------------------------------------------------------------
def tweet_meta_features(tweets: pd.DataFrame, users: pd.DataFrame, ref_time: pd.Series) -> pd.DataFrame:
    has_ts = tweets["ts"].notna().any()
    has_reply = tweets["in_reply_to_user_id"].notna().any() or tweets["is_reply"].astype(bool).any()
    has_eng = tweets[["retweet_count", "reply_count", "favorite_count"]].notna().any().any()
    g = tweets.groupby("account_id", sort=False)
    out = pd.DataFrame({"account_id": list(g.groups.keys())}).set_index("account_id")
    out["retweet_ratio"] = g["is_retweet"].mean()
    if has_reply:
        out["reply_ratio"] = g["is_reply"].mean()
        replies = tweets[tweets["is_reply"].astype(bool)]
        rd = replies.groupby("account_id")["in_reply_to_user_id"].nunique() / replies.groupby("account_id").size()
        out["reply_diversity"] = rd.reindex(out.index).fillna(0.0)
    if has_eng:
        eng = tweets[["retweet_count", "reply_count", "favorite_count"]].fillna(0).sum(axis=1)
        out["avg_engagement_received"] = eng.groupby(tweets["account_id"]).mean()
    if has_ts:
        t = tweets.dropna(subset=["ts"]).sort_values(["account_id", "ts"])
        dt = t.groupby("account_id")["ts"].diff().dropna() / 3600.0
        dt_g = dt.groupby(t.loc[dt.index, "account_id"])
        out["inter_tweet_time"] = dt_g.mean()
        out["tweet_burstiness"] = dt_g.std(ddof=1)
        span_days = (t.groupby("account_id")["ts"].max() - t.groupby("account_id")["ts"].min()) / DAY
        out["tweet_density_over_time"] = t.groupby("account_id").size() / (span_days + 1.0)
        out["tweet_time_irregularity"] = out["tweet_burstiness"] / out["inter_tweet_time"].replace(0, np.nan)
        u = users.set_index("account_id")
        upd = (u["updated_at"] - u["created_at"]).reindex(t["account_id"]).to_numpy()
        age = (ref_time - u["created_at"]).reindex(t["account_id"]).to_numpy()
        tt = t["ts"].to_numpy()
        # eqs. 21-22 as defined in the paper: they mix an absolute time with a duration, so they mostly encode
        # *when* the tweets were collected; they are tagged artifact-prone for that reason.
        out["temporal_overlap"] = pd.Series(np.abs(tt - upd) / DAY, index=t.index).groupby(t["account_id"]).mean()
        out["temporal_overlap_age"] = pd.Series(np.abs(tt - age) / DAY, index=t.index).groupby(t["account_id"]).mean()
    return out.reset_index()


# ----------------------------------------------------------------------------------------------------------
# profile features
# ----------------------------------------------------------------------------------------------------------
def _name_similarity(a, b) -> float:
    if not isinstance(a, str) or not isinstance(b, str) or not a or not b:
        return np.nan
    sa, sb = set(a.lower()), set(b.lower())
    return len(sa & sb) / max(len(sa), len(sb), 1)


def profile_features(users: pd.DataFrame, ref_time: pd.Series) -> pd.DataFrame:
    u = users
    out = pd.DataFrame({"account_id": u["account_id"]})
    for c in ("verified", "followers_count", "friends_count", "statuses_count", "favourites_count", "listed_count",
              "default_profile", "default_profile_image", "geo_enabled", "profile_background_tile",
              "profile_use_background_image"):
        out[c] = u[c].astype(float)
    name, sname, desc = u["name"].astype("string"), u["screen_name"].astype("string"), u["description"].astype("string")
    out["name_length"] = name.str.len().fillna(0).astype(float)
    out["screen_name_length"] = sname.str.len().fillna(0).astype(float)
    out["name_similarity"] = [_name_similarity(a, b) for a, b in zip(name.fillna(""), sname.fillna(""))]
    out["screen_name_digits"] = sname.fillna("").map(lambda s: sum(ch.isdigit() for ch in s) / max(len(s), 1))
    out["has_description"] = desc.notna().astype(float)
    out["description_length"] = desc.str.len().fillna(0).astype(float)
    out["has_url"] = u["url"].notna().astype(float)
    out["has_location"] = u["location"].notna().astype(float)
    out["profile_completeness"] = (out["has_description"] + out["has_url"] + out["has_location"]
                                   + (1 - out["default_profile_image"])) / 4.0
    fol, fri = out["followers_count"], out["friends_count"]
    out["ff_ratio"] = fri / fol.clip(lower=1)
    out["user_reputation"] = fol / (fol + fri + 1)
    upd_days = ((u["updated_at"] - u["created_at"]) / DAY).clip(lower=0)
    out["profile_update_days"] = upd_days
    den = upd_days.clip(lower=1)
    out["followers_growth"] = fol / den
    out["friends_growth"] = fri / den
    out["status_growth"] = out["statuses_count"] / den
    out["favorites_growth"] = out["favourites_count"] / den
    out["listed_growth"] = out["listed_count"] / den
    out["ffru"] = out["ff_ratio"] / den
    age = ((ref_time.to_numpy() - u["created_at"].to_numpy()) / DAY)
    age = pd.Series(age, index=u.index).clip(lower=0)
    out["account_age_days"] = age
    a = age.clip(lower=1)
    out["followers_over_time"] = fol / a
    out["friends_over_time"] = fri / a
    out["status_over_time"] = out["statuses_count"] / a
    out["favorites_over_time"] = out["favourites_count"] / a
    return out


def reference_time(users: pd.DataFrame, cfg: dict) -> pd.Series:
    """Time_now per account (UNIX seconds): fixed per dataset, or the account's crawl time if configured."""
    ends = cfg["data"]["dataset_end"]
    base = users["dataset"].map(lambda d: pd.Timestamp(ends.get(d, "2020-01-01"), tz="UTC").timestamp())
    if cfg["data"].get("age_reference") == "crawled_at":
        base = users["crawled_at"].fillna(base)
    return pd.Series(base.to_numpy(dtype=float), index=users["account_id"].to_numpy())


def build_features(users: pd.DataFrame, tweets: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    ref = reference_time(users, cfg)
    prof = profile_features(users, ref.reset_index(drop=True))
    meta = tweet_meta_features(tweets, users, ref)
    txt = text_features(tweets, cfg)
    df = users[[c for c in META_COLUMNS if c in users.columns]
               + [c for c in ("domain", "in_lsabre", "lang") if c in users.columns]].copy()
    for part in (prof, meta, txt):
        df = df.merge(part, on="account_id", how="left")
    df["n_tweets_collected"] = df["n_tweets_collected"].fillna(0)
    for f in FEATURE_NAMES:
        if f not in df.columns:
            df[f] = np.nan
    meta_cols = [c for c in df.columns if c not in FEATURE_NAMES]
    df = df[meta_cols + FEATURE_NAMES]
    df[FEATURE_NAMES] = df[FEATURE_NAMES].astype("float64").replace([np.inf, -np.inf], np.nan)
    return df
