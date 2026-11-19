"""Feature registry: the single place that says what each feature is.

Every feature computed in `build.py` must appear here. Scripts read this table to group features by category
(ablation, Table 7), to find the artifact-prone set A (shortcut audit, eq. 38) and to label plots.

artifact_prone=True means the value is driven by *how/when the data was collected* rather than by account
behaviour (crawl time, analysis date, collection depth, platform verification policy). See paper Table 8.
evasion_cost is how hard the feature is for a bot operator to fake: 'low' (edit a profile or text) or 'high'.
source is the prior work the feature comes from ('IBD' = introduced here, 'IBD variant of <ref>' = an IBD
modification of a published feature); new_in_ibd marks both kinds. Sources are set in SOURCES below.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd


@dataclass(frozen=True)
class Feature:
    name: str
    category: str          # user | profile | linguistic | content | engagement | temporal | readability | sentiment
    equation: str          # equation number in an earlier draft (kept for reference; no longer used by the paper)
    artifact_prone: bool
    evasion_cost: str      # low | high
    description: str
    source: str = ""       # prior work, 'IBD' or 'IBD variant of <ref>'
    new_in_ibd: bool = False


F = Feature
FEATURES: list[Feature] = [
    # ---- user (raw account counts) ----
    F("verified", "user", "", True, "high", "Verified badge (platform policy; may reflect how the dataset was sampled)"),
    F("followers_count", "user", "", False, "high", "Number of followers"),
    F("friends_count", "user", "", False, "low", "Number of accounts followed"),
    F("statuses_count", "user", "", False, "low", "Total tweets posted (profile counter)"),
    F("favourites_count", "user", "", False, "low", "Total likes given"),
    F("listed_count", "user", "", False, "high", "Number of public lists containing the account"),
    F("name_length", "user", "", False, "low", "Characters in display name"),
    F("screen_name_length", "user", "", False, "low", "Characters in screen name"),
    F("name_similarity", "user", "1", False, "low", "Character-set overlap of name and screen name"),
    F("screen_name_digits", "user", "", False, "low", "Share of digits in screen name"),
    # ---- profile ----
    F("has_description", "profile", "", False, "low", "Profile has a description"),
    F("description_length", "profile", "", False, "low", "Characters in description"),
    F("has_url", "profile", "", False, "low", "Profile has a URL"),
    F("has_location", "profile", "", False, "low", "Profile has a location"),
    F("profile_completeness", "profile", "", False, "low", "Mean of description, URL, location, custom image"),
    F("default_profile", "profile", "", False, "low", "Default theme"),
    F("default_profile_image", "profile", "", False, "low", "Default avatar"),
    F("geo_enabled", "profile", "", False, "low", "Geotagging enabled"),
    F("profile_background_tile", "profile", "", False, "low", "Background image tiled"),
    F("profile_use_background_image", "profile", "", False, "low", "Uses a background image"),
    # ---- engagement / network growth ----
    F("ff_ratio", "engagement", "", False, "low", "friends / max(followers, 1)"),
    F("user_reputation", "engagement", "", False, "high", "followers / (followers + friends + 1)"),
    F("followers_growth", "engagement", "10", True, "high", "followers / days between creation and P_updated"),
    F("friends_growth", "engagement", "11", True, "low", "friends / days between creation and P_updated"),
    F("status_growth", "engagement", "12", True, "low", "statuses / days between creation and P_updated"),
    F("favorites_growth", "engagement", "13", True, "low", "likes / days between creation and P_updated"),
    F("listed_growth", "engagement", "", True, "high", "listed / days between creation and P_updated"),
    F("ffru", "engagement", "14", True, "low", "ff_ratio / days between creation and P_updated"),
    F("reply_diversity", "engagement", "15", False, "high", "Unique users replied to / replies"),
    F("avg_engagement_received", "engagement", "16", False, "high", "Mean retweets + replies + likes received per tweet"),
    F("retweet_ratio", "engagement", "", False, "low", "Share of collected tweets that are retweets"),
    F("reply_ratio", "engagement", "", False, "low", "Share of collected tweets that are replies"),
    F("avg_hashtags", "engagement", "", False, "low", "Hashtags per tweet"),
    F("avg_mentions", "engagement", "", False, "low", "Mentions per tweet"),
    F("avg_urls", "engagement", "", False, "low", "URLs per tweet"),
    F("hashtag_diversity", "engagement", "", False, "low", "Unique hashtags / tweets"),
    # ---- content ----
    F("n_tweets_collected", "content", "", True, "low", "Tweets available in the dataset (collection depth, capped)"),
    F("avg_tweet_length", "content", "", False, "low", "Mean characters per tweet (URLs removed)"),
    F("min_tweet_length", "content", "7", False, "low", "Shortest tweet"),
    F("max_tweet_length", "content", "8", False, "low", "Longest tweet"),
    F("tweet_length_irregularity", "content", "9", False, "low", "Std of tweet length (TLI)"),
    F("duplicate_tweet_ratio", "content", "", False, "low", "Share of tweets that repeat an earlier tweet"),
    # ---- linguistic ----
    F("unique_word_count", "linguistic", "2", False, "low", "Distinct words over all tweets (UWC)"),
    F("unique_word_use", "linguistic", "3", False, "low", "Distinct words per tweet (UWU)"),
    F("punctuation_density", "linguistic", "4", False, "low", "Punctuation marks per word (PD)"),
    F("avg_elongated_words", "linguistic", "5", False, "low", "Elongated words (e.g. 'sooo') per tweet (AEW)"),
    F("spam_ad", "linguistic", "6", False, "low", "Spam/advertising keyword occurrences (SpamAd)"),
    F("spam_density", "linguistic", "", False, "low", "SpamAd / total words"),
    F("avg_sentence_length", "linguistic", "", False, "low", "Words per sentence"),
    F("capitalization_rate", "linguistic", "", False, "low", "Upper-case share of letters"),
    F("special_char_rate", "linguistic", "", False, "low", "Non-alphanumeric, non-space characters per character"),
    F("emoji_rate", "linguistic", "", False, "low", "Emoji per tweet"),
    # ---- temporal ----
    F("inter_tweet_time", "temporal", "17", False, "high", "Mean hours between consecutive tweets (ITT)"),
    F("tweet_burstiness", "temporal", "18", False, "high", "Std of hours between tweets (TB)"),
    F("tweet_density_over_time", "temporal", "19", False, "high", "Tweets per day over the active span (TDOT)"),
    F("tweet_time_irregularity", "temporal", "20", False, "high", "Coefficient of variation of inter-tweet time (TTI)"),
    F("temporal_overlap", "temporal", "21", True, "high", "Mean |tweet time - (P_updated - P_created)| in days (TO)"),
    F("temporal_overlap_age", "temporal", "22", True, "high", "Mean |tweet time - account age at Time_now| in days (TOA)"),
    F("account_age_days", "temporal", "", True, "high", "Days from creation to Time_now"),
    F("profile_update_days", "temporal", "", True, "high", "Days from creation to P_updated ('profile_update')"),
    F("followers_over_time", "temporal", "23", True, "high", "followers / account age"),
    F("friends_over_time", "temporal", "24", True, "low", "friends / account age"),
    F("status_over_time", "temporal", "25", True, "low", "statuses / account age"),
    F("favorites_over_time", "temporal", "26", True, "low", "likes / account age"),
    # ---- readability ----
    F("mean_readability", "readability", "27", False, "low", "Mean of Dale-Chall, Flesch and ARI scores (MR)"),
    F("fog_smog_index", "readability", "28", False, "low", "Gunning Fog + SMOG (FSI)"),
    F("fog_smog_to_tweet", "readability", "29", False, "low", "FSI / tweets (FST)"),
    F("content_entropy", "readability", "30", False, "low", "Shannon entropy of word distribution (CE, bits)"),
    # ---- sentiment ----
    F("avg_polarity", "sentiment", "31", False, "low", "Mean TextBlob polarity"),
    F("avg_subjectivity", "sentiment", "32", False, "low", "Mean TextBlob subjectivity"),
    F("min_polarity", "sentiment", "", False, "low", "Minimum polarity"),
    F("max_polarity", "sentiment", "", False, "low", "Maximum polarity"),
    F("min_subjectivity", "sentiment", "", False, "low", "Minimum subjectivity"),
    F("max_subjectivity", "sentiment", "", False, "low", "Maximum subjectivity"),
    F("polarity_drift", "sentiment", "", False, "low", "Std of polarity"),
    F("extreme_polarity", "sentiment", "33", False, "low", "Share of tweets with |polarity| > 0.75"),
    F("subjectivity_drift", "sentiment", "34", False, "low", "Std of subjectivity (SD)"),
    F("consistent_subjectivity", "sentiment", "35", False, "low", "1 if SD < 0.1 (CS)"),
]

YANG, VAROL, CHU = "Yang et al. 2020", "Varol et al. 2017", "Chu et al. 2012"
CARDAIOLI, DICKERSON = "Cardaioli et al. 2021", "Dickerson et al. 2014"
NEW_IN_IBD = {  # name -> source, for features introduced or modified in IBD
    **dict.fromkeys(["temporal_overlap", "temporal_overlap_age", "ffru", "extreme_polarity", "consistent_subjectivity",
                     "reply_diversity"], "IBD"),
    **dict.fromkeys(["followers_growth", "friends_growth", "status_growth", "favorites_growth", "listed_growth",
                     "profile_update_days"], f"IBD variant of {YANG}"),
    **dict.fromkeys(["mean_readability", "fog_smog_index", "fog_smog_to_tweet"], f"IBD variant of {CARDAIOLI}"),
    "tweet_time_irregularity": f"IBD variant of {VAROL}",
}
SOURCES = {  # name -> source, for features taken from prior work
    **dict.fromkeys(["followers_count", "friends_count", "statuses_count", "favourites_count", "listed_count",
                     "name_length", "screen_name_length", "description_length", "screen_name_digits",
                     "default_profile", "default_profile_image", "followers_over_time", "friends_over_time",
                     "status_over_time", "favorites_over_time", "ff_ratio", "account_age_days",
                     "tweet_density_over_time"], YANG),
    **dict.fromkeys(["inter_tweet_time", "tweet_burstiness", "content_entropy", "emoji_rate", "avg_polarity",
                     "min_polarity", "max_polarity", "avg_subjectivity", "min_subjectivity", "max_subjectivity",
                     "has_description", "has_url", "has_location", "profile_completeness", "geo_enabled",
                     "profile_background_tile", "profile_use_background_image", "verified", "n_tweets_collected"],
                    VAROL),
    **dict.fromkeys(["user_reputation", "avg_urls", "avg_hashtags", "avg_mentions", "retweet_ratio", "reply_ratio",
                     "spam_ad", "spam_density", "duplicate_tweet_ratio"], CHU),
    **dict.fromkeys(["polarity_drift", "subjectivity_drift"], DICKERSON),
    "name_similarity": "Hayawi et al. 2022",
    "avg_elongated_words": "Ilias & Roussaki 2021",
    **dict.fromkeys(["capitalization_rate", "avg_engagement_received"], "Lopez-Joya et al. 2024"),
    "hashtag_diversity": "Kosmajac & Keselj 2019",
    **dict.fromkeys(["unique_word_count", "unique_word_use", "punctuation_density", "avg_sentence_length",
                     "special_char_rate", "min_tweet_length", "max_tweet_length", "avg_tweet_length",
                     "tweet_length_irregularity"], CARDAIOLI),
}
assert not set(NEW_IN_IBD) & set(SOURCES), "a feature cannot be both new and from prior work"
FEATURES = [replace(f, source=NEW_IN_IBD[f.name], new_in_ibd=True) if f.name in NEW_IN_IBD
            else replace(f, source=SOURCES[f.name]) for f in FEATURES]  # KeyError = feature without a source
assert set(NEW_IN_IBD) | set(SOURCES) == {f.name for f in FEATURES}, "SOURCES / NEW_IN_IBD name unknown features"

BY_NAME = {f.name: f for f in FEATURES}
FEATURE_NAMES = [f.name for f in FEATURES]
CATEGORIES = ["user", "profile", "linguistic", "content", "engagement", "temporal", "readability", "sentiment"]


def table() -> pd.DataFrame:
    return pd.DataFrame([f.__dict__ for f in FEATURES])


def artifact_set(cfg: dict | None = None) -> list[str]:
    extra = (cfg or {}).get("audit", {}).get("extra_artifact_features", []) or []
    return [f.name for f in FEATURES if f.artifact_prone] + [x for x in extra if x in BY_NAME]


def new_in_ibd_set() -> list[str]:
    """Features introduced or modified in IBD (source 'IBD' or 'IBD variant of ...')."""
    return [f.name for f in FEATURES if f.new_in_ibd]


def by_category(names) -> dict[str, list[str]]:
    out = {c: [] for c in CATEGORIES}
    for n in names:
        out[BY_NAME[n].category].append(n)
    return out
