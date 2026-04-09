# Thesis Pipeline – Leakage-Free Clustering & Targeting

Automated pipeline that validates fan-level clustering for ticket-purchase
targeting at Aquila Basket Trento.  Produces both evaluation metrics
(Precision@K, Lift@K across 9 methods) and operational marketing packages
(ranked target lists with GDPR consent filtering) for every game.

---

## Quick Start

```bash
cd AnnaPiccoli_capstone_project

# install dependencies (inside your conda/venv)
pip install -r requirements.txt

# run (minimal — evaluation only)
python3 scripts/run_end_to_end.py \
    --raw_csv  "/path/to/export.csv" \
    --opponents_csv "/path/to/trento_opponents_2023_2026.csv" \
    --out_dir runs \
    --test_frac 0.30 \
    --k_list 200 500 1000 \
    --random_seed 42

# run (full — with consent filtering and game packages)
python3 scripts/run_end_to_end.py \
    --raw_csv  "/path/to/export.csv" \
    --opponents_csv "/path/to/trento_opponents_2023_2026.csv" \
    --consent_csv "/path/to/aquila_basket_10103810.csv" \
    --out_dir runs \
    --test_frac 0.30 \
    --k_list 200 500 1000 \
    --random_seed 42 \
    --include_future_games
```

All artefacts land in `runs/run_<timestamp>/`.

---

## CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--raw_csv` | *(required)* | Raw ticket export CSV (semicolon-separated, Italian dates) |
| `--opponents_csv` | None | Opponents schedule CSV (game dates, scores, outcomes) |
| `--consent_csv` | None | Marketing consent CSV (`aquila_basket_10103810.csv`) |
| `--out_dir` | `runs` | Base output directory |
| `--test_frac` | 0.30 | Fraction of games held out as test |
| `--k_list` | 200 500 1000 | K values for Precision@K / Lift@K |
| `--random_seed` | 42 | Random seed for reproducibility |
| `--umap_components` | 10 | UMAP output dimensions |
| `--umap_neighbors` | 15 | UMAP n_neighbors parameter |
| `--umap_min_dist` | 0.0 | UMAP min_dist parameter |
| `--hdbscan_min_cluster` | 50 | HDBSCAN min_cluster_size |
| `--hdbscan_min_samples` | 5 | HDBSCAN min_samples |
| `--use_pca` | False | Use PCA instead of UMAP (faster, deterministic) |
| `--require_consent` | True | Only target fans with `marketing_consent == 1` |
| `--no_require_consent` | — | Override: allow targeting without consent |
| `--require_email` | True | Only target fans with a known email address |
| `--no_require_email` | — | Override: allow targeting without email |
| `--skip_packages` | False | Skip per-game package generation (faster test runs) |
| `--target_top_n` | None | Limit target lists to top N fans per game |
| `--export_bought_already` | False | Export `bought_already.csv` per game |
| `--include_future_games` | False | Also generate packages for future (unplayed) fixtures |
| `--future_n` | None | Max number of future fixtures to include |
| `--future_from_date` | None | Only future fixtures after this date (YYYY-MM-DD) |
| `--future_only_competitions` | None | Filter future fixtures by competition type |
| `--future_only_opponents` | None | Filter future fixtures by opponent name |
| `--config` | None | YAML config file (overrides CLI flags) |

---

## How Leakage Is Prevented

| Risk | Mitigation |
|------|-----------|
| Future-game features in train | **Chronological game split** before any feature building |
| Scaler / UMAP fit on full data | StandardScaler and UMAP are **fit on train fans only**; test fans are projected via `.transform()` |
| Cluster labels from full data | HDBSCAN is **fit on train embeddings**; test fans get labels via `approximate_predict()` |
| Propensity uses test games | Cluster purchase propensity `p_c` is computed **exclusively from train games** |
| Non-match rows leak future info | Season passes / packs are split by purchase date vs. cutoff; rows after cutoff are dropped |
| Opponent affinity uses future games | `_add_opponent_affinity()` only uses opponents whose game_date appears in train tickets |

### Data flow

```
Raw CSV
  │
  ▼
Cleaning & standardisation  (io.py / schema.py)
  │  ↳ parse dates, build person_id (HMAC-SHA256), classify competition,
  │    derive sector/season/age/province, compute price ratios
  ▼
Chronological split  (split.py)
  ├──► TRAIN games (earliest 70%)
  └──► TEST games  (latest 30%)  [held out — not touched until evaluation]
  │
  ▼  (TRAIN only from here)
Build fan features  (feature_building.py)
  │  ↳ features: attendance, recency, price sensitivity, timing,
  │    competition preferences, opponent affinity, seating, churn risk
  ▼
Impute → Scale → UMAP/PCA  (representation.py)
  │  ↳ all .fit() calls on TRAIN only; .transform() applied to TEST
  ▼
HDBSCAN clustering  (clustering.py)
  │  ↳ .fit() on TRAIN embeddings; approximate_predict() for TEST fans
  ▼
Compute cluster propensity p_c  (clustering.py)
  │  ↳ Empirical Bayes Beta-Binomial smoothing on TRAIN games only
  ▼
Advanced scoring  (advanced_scoring.py)
  │  ↳ competition-aware propensity, weighted hybrid (grid-search weights),
  │    frequency-boosted cluster score, supervised logistic regression
  ▼
Evaluate on TEST games  (evaluation.py / baselines.py)
  │  ↳ 9 methods compared: Precision@K, Lift@K
  ▼
Validate clusters  (validation.py)
  │  ↳ cluster profiles, size balance (Gini), propensity spread
  ▼
Generate game packages  (package_generator.py / game_targeting.py)
     ↳ per-game target_list.csv, high_intent.csv, summary.json
       (filtered by GDPR consent + email availability)
```

---

## Evaluation Methods (9 total)

For each **test game** *g*:
1. **Universe** `U_g` = all fans in the train set (those with cluster labels and features)
2. **True buyers** `B_g` = fans from `U_g` who actually purchased game *g*
3. Fans are ranked by each method's score (highest first)
4. Metrics: `Precision@K = |top_K ∩ B_g| / K` and `Lift@K = Precision@K / base_rate(g)`

| Method | Description |
|--------|-------------|
| `random` | Uniform random (averaged over 50 runs) |
| `recency` | Lowest `recency_days` (most recent purchaser first) |
| `frequency` | Highest `games_attended` |
| `cluster` | Cluster-level smoothed propensity `p_c` |
| `cluster_x_freq` | Multiplicative: `p_c_norm × freq_norm` |
| `freq_boost_cluster` | Additive: `0.3 × p_c_norm + 0.7 × freq_norm` (avoids zero-out) |
| `comp_propensity` | Competition-aware propensity `P(buy | cluster, LBA/Eurocup)` |
| `weighted_hybrid` | 5-component weighted score (freq 45%, recency 25%, price 15%, comp 10%, cluster 5%) — weights found by leave-one-out grid search on TRAIN games |
| `supervised_lr` | Logistic regression on per-game binary labels; features: cluster one-hot + games_attended + recency + is_weekend + is_lba |

## GDPR Consent & Marketing Packages

When `--consent_csv` is provided, every game package applies two filters before writing the target list:

1. **Consent filter**: only fans with `marketing_consent == 1` in the consent master file are included (`--require_consent`, on by default)
2. **Email filter**: only fans with a resolvable email address are included (`--require_email`, on by default)

For each game, three files are produced under `packages/<game_date>/`:

| File | Description |
|------|-------------|
| `target_list.csv` | All eligible fans ranked by eligibility score |
| `high_intent.csv` | Top-scoring subset for premium outreach |
| `summary.json` | Campaign statistics (n fans, predicted buyers, expected revenue) |

For future (unplayed) games, use `--include_future_games` — game attributes come from the opponents schedule instead of ticket data.

---

## Assumptions

1. **Person identity** is hashed from PII (email > phone > name+DOB > name+province > operation code) using HMAC-SHA256. Collisions are possible for low-tier identities.
2. **Game date** = normalised `event_dt`. Pre-season friendlies and non-match events are excluded from the game-level split.
3. **Cluster propensity is static** — a single `p_c` per cluster, averaged over all train games. The weighted hybrid partially overcomes this by adding individual-level signals.
4. **UMAP reproducibility** — with `random_state=42` and `n_jobs=1`, UMAP gives identical results across runs on the same machine. Minor numerical differences may occur across UMAP versions.

## Limitations

1. **Cold-start fans** — fans appearing for the first time in test games have no train history and are excluded from the targeting universe.
2. **Small test set** — with ~30% of games as test, there may be as few as 5–10 test games, leading to high variance in Precision@K.
3. **UMAP transform** — `UMAP.transform()` projects new points but does not update the manifold graph. Use `--use_pca` for a fully deterministic fallback.
4. **Noise cluster** — HDBSCAN labels some fans as noise (`cluster = -1`). These get the lowest propensity and are targeted last.
5. **No hyperparameter tuning** — UMAP and HDBSCAN parameters are fixed to the values from the original notebooks. A proper search would strengthen results.

---

## Output Artefacts

| File | Description |
|------|-------------|
| `analytics_clean.csv` | Ticket-level cleaned data (full) |
| `split_info.json` | Train/test game dates, cutoff date, row counts |
| `fan_features_train.csv` | Fan-level features built from train games only |
| `fan_cluster_labels.csv` | `person_id → cluster` mapping |
| `cluster_propensity_per_game.csv` | Per-cluster × per-train-game purchase rate |
| `cluster_propensity_overall.csv` | Overall smoothed `p_c` per cluster |
| `cluster_profiles.csv` | Human-readable cluster profile (median features + auto-label) |
| `size_balance.json` | Cluster size stats: min/max/median, Gini index |
| `propensity_spread.json` | Propensity separation stats: range, IQR, top5/bottom5 ratio |
| `comp_propensity.csv` | Competition-aware `P(buy | cluster, LBA/Eurocup)` |
| `evaluation_detail.csv` | Per-game × per-K results for all 9 methods |
| `evaluation_summary.csv` | Aggregated mean ± std across games per method |
| `packages/<date>/target_list.csv` | Ranked fan target list for each game |
| `packages/<date>/high_intent.csv` | High-intent subset for each game |
| `packages/<date>/summary.json` | Campaign summary for each game |
| `packages/game_index.csv` | Cross-game index of all packages |
| `run_meta.json` | Reproducibility metadata (params, timings) |
| `pipeline.log` | Full execution log |

---

## Project Structure

```
thesis_clean_pipeline/
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml          # default parameter values
├── scripts/
│   └── run_end_to_end.py     # single entry point for the full pipeline
├── src/
│   ├── __init__.py
│   ├── schema.py             # Italian column name discovery & normalisation
│   ├── io.py                 # CSV reading, cleaning, identity (HMAC-SHA256)
│   ├── split.py              # chronological game split (no leakage)
│   ├── feature_building.py   # fan-level feature aggregation
│   ├── representation.py     # impute → scale → UMAP/PCA
│   ├── clustering.py         # HDBSCAN fit/predict + Bayesian propensity
│   ├── evaluation.py         # Precision@K, Lift@K (cluster method)
│   ├── baselines.py          # random, recency, frequency, cluster×freq
│   ├── advanced_scoring.py   # competition propensity, weighted hybrid, LR
│   ├── validation.py         # cluster profiles, size balance, propensity spread
│   ├── consent.py            # GDPR marketing consent loader
│   ├── game_targeting.py     # per-game eligibility scoring (9 components)
│   ├── package_generator.py  # marketing package builder (target list, JSON summary)
│   └── utils.py              # logging setup, timestamped directories, print_summary
└── runs/                     # output directory (git-ignored)
```
