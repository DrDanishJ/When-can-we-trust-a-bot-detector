# Data layout

The datasets are not included. Download them from their providers (see the main README) and place them as below,
or change `paths:` in `configs/default.yaml` (or in your own config file passed with `--config`).

```
data/raw/
  cresci-15/                 E13.Real/, TFP.Real/, FSF.Fake/, INT.Fake/, TWT.Fake/
                             each with users.csv and tweets.csv
  cresci-17/                 genuine_accounts.csv/, social_spambots_1.csv/, social_spambots_2.csv/,
                             social_spambots_3.csv/, traditional_spambots_1.csv/ ... fake_followers.csv/
                             each folder with its users and tweets CSV files (the flattened Kaggle file names
                             GenUsers.csv, s1users.csv, ... are also recognised)
  twibot-20/                 train.json, dev.json, test.json
  fox8-23/fox8_23_dataset.ndjson
  lsabre/dataset/            df_rewrites_1000_users_part_*.csv (python scripts/00_get_lsabre.py)
data/cache/                  written by the pipeline: features_<dataset>.parquet and models/
data/sample/                 optional small samples for configs/smoke.yaml (same layout as data/raw)
```

Notes on the raw data that matter for the results:

- Cresci-17 groups were crawled at different times (genuine accounts in 2015, social spambots in 2014, traditional
  spambots in 2010-2011); the fake-follower group records no crawl date, and its snapshot date (6 December 2013) is
  used for the account-age matching of step 10.
- Cresci-17 `traditional_spambots_2/3/4` contain no tweets and are excluded by default.
- The Cresci-15 bots (FSF, INT, TWT) are the Cresci-17 `fake_followers` group.
- TwiBot-20 provides no tweet timestamps, so its temporal features are missing (NaN), not zero.
- Part of the Cresci-15 human accounts tweet in Italian; the sentiment and readability tools are English-based.
