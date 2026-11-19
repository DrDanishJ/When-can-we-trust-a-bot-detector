"""Write a SYNTHETIC fox8-23-style file (same JSON layout as the real release, fake content) for smoke tests.

    python tests/make_fox8_fixture.py data/sample/fox8-23/fox8_23_sample.ndjson

Layout per line: {"user_id", "label": "bot"|"human", "dataset", "user_tweets": [Twitter API v1.1 tweet objects]}.
Numbers produced from this file mean nothing; it only exercises the loader and the pipeline.
"""
import json
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

WORDS = ("crypto market news today great deal thanks follow love new update people time check this "
         "amazing world best project community join free link read more story why what how").split()


def tweet(rng, uid, i, bot, user):
    t = datetime(2023, 5, 1, tzinfo=timezone.utc) - timedelta(hours=i * (2 if bot else rng.uniform(1, 30)))
    words = [rng.choice(WORDS) for _ in range(rng.randint(6, 25))]
    text = " ".join(words).capitalize() + (" #crypto https://t.co/x" if bot and rng.random() < 0.6 else ".")
    return {"created_at": t.strftime("%a %b %d %H:%M:%S +0000 %Y"), "id": uid * 1000 + i, "full_text": text,
            "retweet_count": rng.randint(0, 5), "favorite_count": rng.randint(0, 20),
            "in_reply_to_status_id": None if rng.random() < 0.8 else 1, "in_reply_to_user_id_str": None,
            "user": user}


def main(out: str, n_per_class: int = 20, seed: int = 0):
    rng = random.Random(seed)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for k in range(2 * n_per_class):
            bot = k < n_per_class
            uid = 10_000 + k
            created = datetime(2022 if bot else rng.randint(2009, 2018), rng.randint(1, 12), 1, tzinfo=timezone.utc)
            user = {"id": uid, "name": f"user {k}", "screen_name": f"user{k}{rng.randint(10, 999) if bot else ''}",
                    "description": "" if bot and rng.random() < 0.5 else "I like things",
                    "created_at": created.strftime("%a %b %d %H:%M:%S +0000 %Y"),
                    "followers_count": rng.randint(10, 500) if bot else rng.randint(50, 5000),
                    "friends_count": rng.randint(200, 2000) if bot else rng.randint(50, 800),
                    "statuses_count": rng.randint(500, 9000), "favourites_count": rng.randint(0, 3000),
                    "listed_count": rng.randint(0, 30), "verified": False, "default_profile": bot,
                    "default_profile_image": False, "geo_enabled": rng.random() < 0.3, "lang": None, "url": None,
                    "location": "" if bot else "Somewhere"}
            tweets = [tweet(rng, uid, i, bot, user) for i in range(rng.randint(20, 60))]
            fh.write(json.dumps({"user_id": uid, "label": "bot" if bot else "human",
                                 "dataset": "fox8" if bot else "synthetic-human", "user_tweets": tweets}) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/sample/fox8-23/fox8_23_sample.ndjson")
