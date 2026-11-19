"""The common account/tweet schema every loader produces.

USERS: one row per account.  TWEETS: one row per tweet (already capped to the most recent N per account).
Columns that a dataset does not provide are present but NaN, so feature code never needs dataset checks.
"""
USER_COLUMNS = [
    "account_id",          # str, unique within a dataset
    "label",               # 0 = human, 1 = bot
    "dataset",             # cresci15 | cresci17 | twibot20 | fox8 | lsabre
    "group",               # sub-dataset, e.g. E13, FSF, social_spambots_1, fox8, zero_shot_rewrite_llama
    "split",               # official split for TwiBot-20 (train/dev/test), '' otherwise
    "created_at",          # account creation, UNIX seconds (UTC)
    "crawled_at",          # when the profile was collected (Cresci-17 only), UNIX seconds
    "updated_at",          # the 'updated' field (Cresci); this is P_updated in eqs. 10-14, 21
    "name", "screen_name", "description", "url", "location", "lang",
    "verified", "default_profile", "default_profile_image", "geo_enabled",
    "profile_background_tile", "profile_use_background_image",
    "followers_count", "friends_count", "statuses_count", "favourites_count", "listed_count",
]

TWEET_COLUMNS = [
    "account_id",
    "text",
    "ts",                  # tweet time, UNIX seconds (NaN for TwiBot-20, which has no tweet timestamps)
    "retweet_count",       # retweets the tweet received (NaN when not provided)
    "reply_count",
    "favorite_count",
    "in_reply_to_user_id", # str or NaN
    "is_reply",            # 1 if the tweet replies to another tweet
    "is_retweet",          # 1 if the account retweeted someone else (RT @... or retweeted_status)
]

BOOL_USER_COLUMNS = ["verified", "default_profile", "default_profile_image", "geo_enabled",
                     "profile_background_tile", "profile_use_background_image"]
COUNT_USER_COLUMNS = ["followers_count", "friends_count", "statuses_count", "favourites_count", "listed_count"]
TEXT_USER_COLUMNS = ["name", "screen_name", "description", "url", "location", "lang"]
