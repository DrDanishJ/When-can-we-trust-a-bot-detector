# iBotDetection (IBD): when can we trust a bot detector?

Code and results for the article

> D. Javed, *When can we trust a bot detector? Shortcut-aware, uncertainty-calibrated and faithfulness-verified
> explanations for social bot detection*, submitted to Knowledge-Based Systems.

IBD is a feature-based social bot detector for X (Twitter) with four verification layers: a shortcut audit of the
explanations (Shortcut Dependence Ratio, joint permutation, retraining, cross-dataset transfer, age matching),
class-conditional split conformal prediction, an evaluation of explanation reliability (deletion AUC, ROAR, bootstrap
stability, SHAP-LIME agreement) and a trust gate that tests whether faithful explanations identify correct
predictions. One pipeline covers Cresci-15, Cresci-17 and TwiBot-20, with fox8-23 and the LSABRE rewritten TwiBot-20
accounts as external test sets.

## 1. Installation

Python 3.10-3.13.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q                            # unit tests, no data needed (6 tests, a few seconds)
```

Step 3 renders the LIME HTML views to PNG/PDF with Playwright and a locally installed Google Chrome
(`channel="chrome"`). Without Chrome, run `playwright install chromium` and remove `channel="chrome"` in
`scripts/03_explain.py`. TabNet is optional (`pip install pytorch-tabnet`) and was not used in the article.

## 2. Data

The datasets are not redistributed here. Obtain them from their sources and place them as described in
[`data/README.md`](data/README.md):

| Dataset | Source | Role |
|---|---|---|
| Cresci-15 | Bot Repository (botometer.osome.iu.edu/bot-repository) | training and testing |
| Cresci-17 | Bot Repository | training and testing |
| TwiBot-20 | on request from its authors | training and testing (official split) |
| fox8-23 | Zenodo, https://doi.org/10.5281/zenodo.8035290 | test only (LLM-powered botnet) |
| LSABRE rewritten TwiBot-20 bots | https://github.com/runi-cyber-ai/LSABRE (`python scripts/00_get_lsabre.py`) | test only |

All paths are set in `configs/default.yaml` (`paths:`) and can be overridden with a second config file
(see `configs/my_pc.yaml`) or on the command line.

## 3. Running the experiments

Every script reads `configs/default.yaml`, plus an optional `--config <file>` and `--set key=value` overrides.

```bash
python scripts/run_all.py                          # steps 1-12 on the full data
python scripts/run_all.py --from 9                 # resume from a step
python scripts/run_all.py --config configs/smoke.yaml --to 8   # steps 1-8 on small samples (pipeline check)
python scripts/05_conformal.py --set models.seeds=[0]   # any step alone, with overrides
```

Step 1 builds a feature cache (`data/cache/features_<dataset>.parquet`); step 2 trains every classifier and caches
the LightGBM explanation models (`data/cache/models/`). All later steps reuse the cache, so they can be rerun
without rebuilding features or retraining.

| Step | Script | Article |
|---|---|---|
| 0 | `00_get_lsabre.py` | downloads the LSABRE files |
| 1 | `01_build_features.py` | Tables 4 and 6, Supplementary Table S1 |
| 2 | `02_train_eval.py` | Table 7, McNemar tests (Section 5.1) |
| 3 | `03_explain.py` | Figs. 3-6, Section 5.2.3 |
| 4 | `04_shortcut_audit.py` | SDR and retraining for A (Table 8, rows "A"), Table 10 |
| 5 | `05_conformal.py` | Table 11, Fig. 7 |
| 6 | `06_reliability.py` | Table 12, Fig. 8 |
| 7 | `07_shift_llm.py` | Table 13 |
| 8 | `08_trust_gate.py` | Fig. 9, Section 5.7 |
| 9 | `09_conformal_checks.py` | Section 5.4: coverage resampling, accuracy of empty sets |
| 10 | `10_shortcut_audit_ext.py` | Table 8 (all rows), Table 9, Section 5.3: age matching, Cresci overlap |
| 11 | `11_shift_ext.py` | Table 13 reference row, Section 5.6: recalibration, tweet-only LSABRE model |
| 12 | `12_trust_gate_ext.py` | Section 5.7: random baseline, AUROC of correctness, DAUC mechanism |

Step 10 also needs the Cresci-17 `users.csv` files (crawl dates) under `paths.cresci17`.

## 4. Outputs

`results/tables/` holds one CSV (and a LaTeX `.tex`) per table and `results/figures/` one PNG and PDF per figure.
File names carry the step's internal table numbers; the mapping to the article is:

| File | Article |
|---|---|
| `dataset_summary`, `feature_registry`, `feature_availability` | Table 6, Table 4, Supplementary Table S1 |
| `table09_classification`, `table09_mcnemar` | Table 7, Section 5.1 |
| `fig_shap_<ds>`, `fig_lime_<ds>_<type>`, `fig_force_*`, `fig_waterfall_*`, `table_fi_vs_shap` | Figs. 3-6, Section 5.2 |
| `table10_sdr`, `table12_ablation` | Table 8 (A rows), Table 10 |
| `table11_cross_dataset` | cross-dataset transfer with the Cresci-15/17 overlap included (superseded by `ext_cross_dataset_dedup`) |
| `table13_conformal`, `table13_calibration`, `fig10_reliability` | Table 11, Fig. 7 |
| `table14_reliability_twibot20`, `table14_roar_twibot20`, `fig12_roar_twibot20` | Table 12, Fig. 8 |
| `table15_shift` | Table 13 |
| `trust_gate_summary`, `trust_gate_auc`, `trust_gate_curve`, `fig13_trust_gate` | Fig. 9, Section 5.7 |
| `*_median`, `*_noA`, `*_m100` | sensitivity checks: median deletion baseline, model without A, M = 100 |
| `ext_coverage_*`, `ext_set_type_accuracy*` | Section 5.4 |
| `ext_audit_sets`, `ext_cross_dataset_dedup`, `ext_overlap`, `ext_age_matched*` | Tables 8-9, Section 5.3 |
| `ext_shift_reference`, `ext_recalibration*`, `ext_lsabre_text_only` | Table 13, Section 5.6 |
| `ext_tg_*` | Section 5.7 |

## 5. Code layout

```
src/ibd/
  config.py                YAML configuration, path resolution, --set overrides
  io/loaders.py            one loader per dataset, common users/tweets schema (io/schema.py)
  features/registry.py     the 78 features: category, source, new in IBD, artifact-prone
  features/build.py        feature formulas, same code for every dataset
  features/store.py        parquet feature cache
  splits.py                train / model-selection / calibration / test splits
  models/zoo.py            classifiers (imputation, scaling and SMOTE fitted on the training split only)
  runs.py                  cached explanation model per dataset and seed
  explain/core.py          TreeSHAP, KernelSHAP, LIME, gain importance
  audit/shortcut.py        Shortcut Dependence Ratio, retraining, cross-dataset transfer
  uq/conformal.py          Mondrian split conformal prediction
  reliability/measures.py  deletion AUC, ROAR, explanation stability, SHAP-LIME agreement
  extended/                coverage checks, extended audit (wider sets, joint permutation, age matching)
  plots.py, metrics.py, utils.py
scripts/                   00-12 and run_all.py
configs/                   default.yaml (full run), smoke.yaml (samples), my_pc.yaml (example local settings)
tests/                     unit tests
```

## 6. Reproducibility notes

- Every model receives a fixed seed; results are mean ± standard deviation over seeds 0-4.
- The test split is used only for scoring: preprocessing and SMOTE are fitted on the training split, early stopping
  and model selection use the model-selection split, conformal thresholds and trust-gate thresholds use the
  calibration split.
- TwiBot-20 accounts that appear in LSABRE are removed from TwiBot-20 training. The Cresci-15 bots are the
  fake-follower group of Cresci-17; the overlap-free cross-dataset results are produced by step 10.
- The Cresci-17 traditional spambot groups without tweets are excluded (`data.cresci17_include_tweetless_groups`).
- Deletion AUC uses a background baseline with M = 50 training accounts (`reliability.deletion_background`).
- The article's experiments were run on an AMD Ryzen 5 3600 (6 cores), 16 GB RAM, with `configs/my_pc.yaml`;
  steps 1-8 take about 100 minutes, most of it in steps 1 and 7.

## 7. Citation and licence

Please cite the article above when using this code (see `CITATION.cff`). The code is released under the MIT
licence (`LICENSE`). The datasets remain under the terms of their original providers.
