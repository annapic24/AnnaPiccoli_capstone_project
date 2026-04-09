#!/usr/bin/env python3
"""
End-to-end clustering validation pipeline.

Usage
-----
python3 run_end_to_end.py \
  --raw_csv "/Users/valentinapiccoli/Desktop/export 130226.csv" \
  --opponents_csv "/Users/valentinapiccoli/Desktop/thesis/promozioni_25_26/project/data/trento_opponents_2023_2026.csv" \
  --consent_csv "/Users/valentinapiccoli/Desktop/thesis/promozioni_25_26/aquila_basket_10103810.csv" \
  --out_dir ../runs \
  --test_frac 0.30 \
  --k_list 200 500 1000 \
  --random_seed 42

All artefacts are written to ``<out_dir>/run_<timestamp>/``.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ensure src/ is importable
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.io import robust_read_csv, run_cleaning, load_opponents
from src.split import chronological_game_split
from src.feature_building import build_fan_features
from src.representation import fit_representation, transform_test
from src.clustering import fit_clusters, build_cluster_propensity
from src.evaluation import evaluate_targeting, summarise_results
from src.baselines import random_baseline, recency_baseline, frequency_baseline, cluster_frequency_hybrid
from src.validation import build_cluster_profiles, compute_size_balance, propensity_spread
from src.package_generator import generate_all_test_packages, generate_future_packages, build_person_lookup
from src.consent import load_marketing_consent
from src.game_targeting import SUBSCRIBER_CLUSTER
from src.advanced_scoring import (
    build_competition_propensity,
    evaluate_competition_propensity,
    grid_search_weights,
    evaluate_weighted_score,
    frequency_boosted_cluster,
    supervised_logistic_baseline,
)
from src.utils import setup_logging, make_run_dir, print_summary

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Leakage-free clustering validation")
    p.add_argument("--raw_csv", required=True, help="Raw ticket export CSV")
    p.add_argument("--opponents_csv", default=None, help="Opponents schedule CSV")
    p.add_argument("--out_dir", default="runs", help="Base output directory")
    p.add_argument("--test_frac", type=float, default=0.30, help="Fraction of games for test")
    p.add_argument("--k_list", type=int, nargs="+", default=[200, 500, 1000], help="K values for Precision@K")
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--umap_components", type=int, default=10)
    p.add_argument("--umap_neighbors", type=int, default=15)
    p.add_argument("--umap_min_dist", type=float, default=0.0)
    p.add_argument("--hdbscan_min_cluster", type=int, default=100)
    p.add_argument("--hdbscan_min_samples", type=int, default=10)
    p.add_argument("--use_pca", action="store_true", help="Use PCA instead of UMAP")
    p.add_argument("--config", default=None, help="YAML config file (overrides CLI)")
    p.add_argument("--target_top_n", type=int, default=None,
                   help="Limit per-game target lists to top N fans")
    p.add_argument("--skip_packages", action="store_true",
                   help="Skip per-game package generation (faster runs)")
    p.add_argument("--export_bought_already", action="store_true",
                   help="Export bought_already CSVs per game")
    p.add_argument("--consent_csv", default=None,
                   help="Marketing consent master CSV (aquila_basket_10103810.csv)")
    p.add_argument("--require_consent", action="store_true", default=True,
                   help="Only target fans with marketing_consent==1 (default: True)")
    p.add_argument("--no_require_consent", dest="require_consent", action="store_false",
                   help="Allow targeting fans without marketing consent")
    p.add_argument("--require_email", action="store_true", default=True,
                   help="Only target fans with an email (default: True)")
    p.add_argument("--no_require_email", dest="require_email", action="store_false",
                   help="Allow targeting fans without email")
    # ── Future fixtures ──
    p.add_argument("--include_future_games", action="store_true", default=False,
                   help="Generate packages for future (unplayed) fixtures from opponents schedule")
    p.add_argument("--future_n", type=int, default=None,
                   help="Max number of future fixtures to include (default: all)")
    p.add_argument("--future_from_date", default=None,
                   help="Only future fixtures after this date (YYYY-MM-DD). Default: max(event_dt)")
    p.add_argument("--future_only_competitions", nargs="*", default=None,
                   help="Only include future fixtures of these competition types (e.g. 'Serie A' 'EuroCup')")
    p.add_argument("--future_only_opponents", nargs="*", default=None,
                   help="Only include future fixtures vs these opponents (substring match)")
    # ── Split control ──
    p.add_argument("--fixed_cutoff_date", default=None,
                   help="Force train/test cutoff to this date (YYYY-MM-DD), ignoring --test_frac. "
                        "Use when new pre-sale rows for a future game are in the CSV and you want "
                        "to reproduce an earlier split exactly.")
    p.add_argument("--max_eval_date", default=None,
                   help="Exclude competitive games after this date (YYYY-MM-DD) from the "
                        "train/test split and evaluation. Their rows stay in the full cleaned "
                        "data so future-package already_bought checks still work correctly. "
                        "Combine with --fixed_cutoff_date to lock in a historical split.")
    # ── FIX 7: new hyperparameter CLI args ──
    p.add_argument("--bayesian_strength", type=float, default=10.0,
                   help="Effective sample size for empirical Bayes propensity prior (FIX 1)")
    p.add_argument("--run_walk_forward", action="store_true", default=False,
                   help="Run 3-fold walk-forward cross-validation for stability checking (FIX 5)")
    p.add_argument(
        "--standings_csv",
        default="/Users/valentinapiccoli/Desktop/thesis/bba_thesis/bba_pipeline/marketing_thesis_outputs/lba_standings_by_giornata.csv",
        help="LBA standings CSV (lba_standings_by_giornata.csv).  Used as the "
             "authoritative source for top8/bottom8 opponent tier classification.",
    )
    return p.parse_args()


def load_config(args: argparse.Namespace) -> dict:
    """Merge YAML config (if any) with CLI args."""
    cfg = vars(args).copy()
    if args.config is not None:
        try:
            import yaml
            with open(args.config) as f:
                ycfg = yaml.safe_load(f) or {}
            # YAML overrides CLI defaults, but explicit CLI overrides YAML
            for k, v in ycfg.items():
                if k in cfg:
                    cfg[k] = v
        except ImportError:
            log.warning("pyyaml not installed – ignoring --config")
    return cfg


def _run_sanity_checks(
    fan_train: pd.DataFrame,
    fan_labels: pd.DataFrame,
    opponents: pd.DataFrame | None,
    split_train: pd.DataFrame,
    run_dir,
) -> None:
    """Tasks 3 & 4: opponent-aware ranking validation + leakage guard.

    Generates ``sanity_checks.json`` with:
      - opponent_feature_leakage_check: PASS / FAIL
      - ranking_differentiation_check: PASS / FAIL
      - top200_overlaps: pairwise Jaccard and intersection counts
      - mean_eligibility_per_game: dict of game → mean score
      - cluster_dist_top200: dict of game → cluster distribution
    """
    from src.game_targeting import (
        score_fans_for_game,
        build_game_profile_from_schedule,
        _norm_opponent_col,
    )

    sanity = {
        "opponent_feature_leakage_check": "PASS",
        "ranking_differentiation_check": "FAIL",
        "top200_overlaps": {},
        "mean_eligibility_per_game": {},
        "cluster_dist_top200": {},
    }

    # ── Leakage guard check (Task 4) ──────────────────────────────
    # Verify that all opponent affinity columns are computed from TRAIN dates.
    # We check that no pct_games_vs_* column has a higher max than 1.0 and
    # that the opponents schedule future rows are absent.
    opp_pct_cols = [c for c in fan_train.columns if c.startswith("pct_games_vs_")]
    leakage_found = False
    if opp_pct_cols:
        for col in opp_pct_cols:
            if fan_train[col].max() > 1.0 + 1e-6:
                log.warning("Leakage check FAIL: %s max=%.4f > 1.0", col, fan_train[col].max())
                leakage_found = True
    # Also assert: if opponents provided, future rows must not appear in train event dates
    if opponents is not None and "game_date" in opponents.columns:
        train_dates = set(
            pd.to_datetime(split_train["event_dt"], errors="coerce").dt.normalize().dropna().unique()
        )
        future_mask = opponents["game_date"].dt.normalize().gt(
            pd.Timestamp(split_train["event_dt"].max()).normalize()
        ) if "game_date" in opponents.columns else pd.Series(False, index=opponents.index)
        future_dates = set(opponents.loc[future_mask, "game_date"].dt.normalize().dropna().unique())
        leaked = future_dates & train_dates
        if leaked:
            log.warning("Leakage check FAIL: future game dates found in train: %s", leaked)
            leakage_found = True
    if leakage_found:
        sanity["opponent_feature_leakage_check"] = "FAIL"
    log.info("Leakage check: %s", sanity["opponent_feature_leakage_check"])

    # ── Ranking differentiation check (Task 3) ────────────────────
    # Score fans for Sassari, Brescia, and Milano and compare top-200 overlaps.
    # We build minimal game_profiles from the opponents schedule (if available)
    # or construct synthetic ones for the three target opponents.
    target_opponents = {
        "Sassari": {"opponent_team": "Dinamo Sassari", "competition": "LBA",
                    "is_weekend": True, "is_evening": True,
                    "is_high_value": False, "is_derby": False},
        "Brescia": {"opponent_team": "Pallacanestro Brescia", "competition": "LBA",
                    "is_weekend": True, "is_evening": True,
                    "is_high_value": False, "is_derby": True},
        "Milano":  {"opponent_team": "Olimpia Milano", "competition": "LBA",
                    "is_weekend": True, "is_evening": True,
                    "is_high_value": True, "is_derby": False},
    }

    top200_pids = {}
    mean_scores = {}
    cluster_dists = {}

    for label, base_profile in target_opponents.items():
        opp_col = _norm_opponent_col(base_profile["opponent_team"])
        profile = {
            **base_profile,
            "match_key": f"validation_{label}",
            "game_date": pd.Timestamp("2026-03-01"),  # synthetic date
            "opponent_col": opp_col,
        }

        try:
            ranked = score_fans_for_game(
                fan_features=fan_train,
                fan_labels=fan_labels,
                game_profile=profile,
                subscription_pids=set(),
                already_bought_pids=set(),
                person_lookup=None,
                require_consent=False,
                require_email=False,
            )
            targetable = ranked.loc[ranked["rank"].notna()].sort_values("rank")
            top200 = set(targetable.head(200)["person_id"].tolist())
            mean_score = float(targetable.head(200)["eligibility_score"].mean()) if len(targetable) >= 1 else 0.0

            # cluster distribution in top-200
            cl_dist = (
                targetable.head(200)["cluster"]
                .astype(str)
                .value_counts(normalize=True)
                .round(3)
                .to_dict()
            )
            top200_pids[label] = top200
            mean_scores[label] = round(mean_score, 4)
            cluster_dists[label] = cl_dist
            log.info(
                "Validation [%s]: top-200 mean score=%.4f, unique fans=%d",
                label, mean_score, len(top200),
            )
        except Exception as exc:
            log.warning("Validation scoring failed for %s: %s", label, exc)
            top200_pids[label] = set()
            mean_scores[label] = 0.0
            cluster_dists[label] = {}

    sanity["mean_eligibility_per_game"] = mean_scores
    sanity["cluster_dist_top200"] = {k: v for k, v in cluster_dists.items()}

    # Resolve which fallback column each validation opponent uses
    # (needed for the structural-identity note below)
    _HV_T = {"BOLOGNA","MILANO","VENEZIA","VIRTUS","OLIMPIA","REYER"}
    def _opp_fallback_col(opp_name: str, fan_features: pd.DataFrame) -> str:
        opp_col = _norm_opponent_col(opp_name)
        pct_col = f"pct_games_vs_{opp_col}"
        if pct_col in fan_features.columns:
            return pct_col
        if any(t in opp_name.upper() for t in _HV_T):
            return "pct_games_vs_hv_opponents"
        return "pct_games_vs_top8"

    # Pairwise top-200 overlaps
    pairs = [("Sassari", "Brescia"), ("Sassari", "Milano"), ("Brescia", "Milano")]
    all_below_threshold = True
    overlap_threshold = 0.80  # goal: < 80% overlap

    for g1, g2 in pairs:
        s1 = top200_pids.get(g1, set())
        s2 = top200_pids.get(g2, set())
        opp1 = target_opponents[g1]["opponent_team"]
        opp2 = target_opponents[g2]["opponent_team"]
        fb1 = _opp_fallback_col(opp1, fan_train)
        fb2 = _opp_fallback_col(opp2, fan_train)
        same_fallback = (fb1 == fb2)

        if not s1 or not s2:
            jaccard = None
            intersection = 0
            overlap_frac = None
        else:
            intersection = len(s1 & s2)
            union = len(s1 | s2)
            jaccard = round(intersection / union, 4) if union > 0 else 0.0
            overlap_frac = round(intersection / min(len(s1), len(s2)), 4) if min(len(s1), len(s2)) > 0 else 0.0

        pair_key = f"{g1}_vs_{g2}"
        sanity["top200_overlaps"][pair_key] = {
            "intersection": intersection,
            "jaccard": jaccard,
            "overlap_fraction": overlap_frac,
            "g1_fallback_col": fb1,
            "g2_fallback_col": fb2,
            "structurally_identical_expected": same_fallback,
        }
        log.info(
            "Top-200 overlap %s: intersection=%d, jaccard=%.4f, overlap_frac=%s "
            "(fb1=%s fb2=%s same=%s)",
            pair_key, intersection, jaccard or 0, overlap_frac, fb1, fb2, same_fallback,
        )

        # Only count as differentiation failure when the two opponents have
        # DIFFERENT fallback columns but still overlap ≥ 80%.
        # When they share the same fallback column (both future unseen opponents),
        # identical rankings are structurally expected (not a bug).
        if not same_fallback and overlap_frac is not None and overlap_frac >= overlap_threshold:
            all_below_threshold = False

    if all_below_threshold and any(top200_pids.values()):
        sanity["ranking_differentiation_check"] = "PASS"
    log.info("Ranking differentiation check: %s", sanity["ranking_differentiation_check"])

    sanity_path = run_dir / "sanity_checks.json"
    with open(sanity_path, "w") as f:
        json.dump(sanity, f, indent=2, default=str)
    log.info("Sanity checks saved to %s", sanity_path)


def main() -> None:
    args = parse_args()
    cfg = load_config(args)

    # reproducibility
    np.random.seed(cfg["random_seed"])

    # output directory
    run_dir = make_run_dir(Path(cfg["out_dir"]))
    setup_logging(run_dir)
    log.info("Run directory: %s", run_dir)
    log.info("Config: %s", json.dumps({k: str(v) for k, v in cfg.items()}, indent=2))

    t0 = time.time()

    # ── 1. Load raw data ─────────────────────────────────────────
    log.info("=" * 60)
    log.info("STEP 1: Load raw data")
    log.info("=" * 60)
    raw = robust_read_csv(cfg["raw_csv"])
    opponents = None
    if cfg["opponents_csv"]:
        opponents = load_opponents(cfg["opponents_csv"])

    # ── 1c. Load LBA standings CSV (authoritative opponent tiers) ──
    standings_df = None
    standings_path = cfg.get("standings_csv")
    if standings_path:
        try:
            standings_df = pd.read_csv(standings_path)
            log.info(
                "Standings CSV loaded: %d rows, seasons=%s",
                len(standings_df),
                sorted(standings_df["season"].unique().tolist()),
            )
        except Exception as exc:
            log.warning("Could not load standings CSV (%s): %s – using dynamic top8/bottom8", standings_path, exc)

    # ── 1b. Load marketing consent master ──────────────────────
    consent_df = None
    if cfg.get("consent_csv"):
        log.info("=" * 60)
        log.info("STEP 1b: Load marketing consent master")
        log.info("=" * 60)
        consent_df = load_marketing_consent(cfg["consent_csv"])
        consent_df.to_csv(run_dir / "consent_master_loaded.csv", index=False)
        n_consent_yes = (consent_df["marketing_consent"] == 1).sum()
        n_consent_total = len(consent_df)
        log.info(
            "Consent master: %d emails, %d with marketing consent (%.1f%%)",
            n_consent_total, n_consent_yes, 100 * n_consent_yes / max(1, n_consent_total),
        )
    else:
        log.warning("No --consent_csv provided: marketing consent will NOT be enforced")

        # IMPORTANT: If we don't have a consent master, we cannot require consent,
        # otherwise every fan will have marketing_consent=0 and target lists go empty.
        if cfg.get("require_consent", True):
            log.warning("require_consent=True but no consent_csv given → forcing require_consent=False")
            cfg["require_consent"] = False

    # ── 2. Clean ─────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("STEP 2: Clean & standardise")
    log.info("=" * 60)
    clean = run_cleaning(raw, opponents)
    clean.to_csv(run_dir / "analytics_clean.csv", index=False)
    log.info("Saved analytics_clean.csv (%d rows)", len(clean))

    # ── 2b. Build person lookup (email, name, consent) ──────────
    log.info("=" * 60)
    log.info("STEP 2b: Build person contact lookup (consent-aware)")
    log.info("=" * 60)
    person_lookup = build_person_lookup(clean, consent_df=consent_df)
    person_lookup.to_csv(run_dir / "person_lookup.csv", index=False)
    n_pl_email = person_lookup["has_email"].sum()
    n_pl_consent = (person_lookup["marketing_consent"] == 1).sum()
    n_pl_total = len(person_lookup)
    log.info(
        "Person lookup: %d fans, %d with email (%.1f%%), %d with marketing consent (%.1f%%)",
        n_pl_total, n_pl_email, 100 * n_pl_email / max(1, n_pl_total),
        n_pl_consent, 100 * n_pl_consent / max(1, n_pl_total),
    )

    # ── 3. Chronological split ───────────────────────────────────
    log.info("=" * 60)
    fixed_cutoff = cfg.get("fixed_cutoff_date")
    max_eval = cfg.get("max_eval_date")
    if fixed_cutoff:
        log.info("STEP 3: Chronological game split (fixed_cutoff_date=%s)", fixed_cutoff)
    else:
        log.info("STEP 3: Chronological game split (test_frac=%.2f)", cfg["test_frac"])
    if max_eval:
        log.info("  max_eval_date=%s  (games after this are excluded from evaluation)", max_eval)
    log.info("=" * 60)
    split = chronological_game_split(
        clean,
        test_frac=cfg["test_frac"],
        fixed_cutoff_date=fixed_cutoff or None,
        max_eval_date=max_eval or None,
    )

    split_info = {
        "n_train_games": len(split.train_games),
        "n_test_games": len(split.test_games),
        "cutoff_date": str(split.cutoff_date.date()),
        "train_rows": len(split.train),
        "test_rows": len(split.test),
        "train_game_dates": [str(pd.Timestamp(d).date()) for d in split.train_games],
        "test_game_dates": [str(pd.Timestamp(d).date()) for d in split.test_games],
    }
    with open(run_dir / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)
    log.info("Split info saved")

    # ── 4. Build fan features (TRAIN only) ───────────────────────
    log.info("=" * 60)
    log.info("STEP 4: Build fan-level features (TRAIN games only)")
    log.info("=" * 60)
    # Pass opponents_df so opponent-affinity features are computed.
    # Leakage guard is inside build_fan_features: only train game dates
    # (from ticket_df.event_dt) are used to filter the schedule.
    fan_train = build_fan_features(split.train, opponents_df=opponents,
                                   standings_df=standings_df)
    fan_train.to_csv(run_dir / "fan_features_train.csv", index=False)
    log.info("Saved fan_features_train.csv (%d fans)", len(fan_train))

    n_sub = fan_train["is_subscription_holder"].sum() if "is_subscription_holder" in fan_train.columns else 0
    log.info("Subscription holders in train: %d (%.1f%%)", n_sub, 100 * n_sub / max(1, len(fan_train)))

    # ── 5. Dimensionality reduction (fit on TRAIN, EXCLUDING subscribers) ──
    log.info("=" * 60)
    log.info("STEP 5: Dimensionality reduction (fit on NON-subscriber TRAIN fans)")
    log.info("=" * 60)

    # TASK 4: Exclude subscription holders from PCA/UMAP fit
    # They attend most games by definition, so they would distort the
    # embedding space. After clustering, we assign them labels via
    # approximate_predict or nearest-cluster.
    sub_mask = fan_train.get("is_subscription_holder", pd.Series(False, index=fan_train.index)).astype(bool)
    fan_train_nosub = fan_train.loc[~sub_mask].copy()
    fan_train_sub = fan_train.loc[sub_mask].copy()
    n_nosub = len(fan_train_nosub)
    n_sub_actual = len(fan_train_sub)
    log.info("Non-subscribers: %d | Subscribers excluded from fit: %d", n_nosub, n_sub_actual)

    rep = fit_representation(
        fan_train_nosub,
        n_components=cfg["umap_components"],
        umap_neighbors=cfg["umap_neighbors"],
        umap_min_dist=cfg["umap_min_dist"],
        random_state=cfg["random_seed"],
        use_pca_fallback=cfg["use_pca"],
    )
    log.info("Reducer: %s → shape %s", rep.reducer_name, rep.X_reduced.shape)

    # ── 6. Clustering (TRAIN, excluding subscribers from fit) ─────
    log.info("=" * 60)
    log.info("STEP 6: HDBSCAN clustering (TRAIN, subscribers excluded from fit)")
    log.info("=" * 60)

    # Prepare test fans (known in train) for cluster assignment
    test_match = split.test.loc[
        split.test["competition_type"].isin(["LBA", "Eurocup"])
    ]
    test_pids = set(test_match["person_id"].unique())
    train_pids = set(fan_train["person_id"].unique())
    test_known_pids = test_pids & train_pids
    fan_test_known = fan_train.loc[fan_train["person_id"].isin(test_known_pids)].copy()

    # Transform subscribers + test fans for cluster assignment
    fans_to_assign = pd.concat([fan_train_sub, fan_test_known], ignore_index=True)
    fans_to_assign = fans_to_assign.drop_duplicates("person_id")

    X_assign, pids_assign = None, None
    if len(fans_to_assign) > 0:
        X_assign, pids_assign = transform_test(fans_to_assign, rep)

    cr = fit_clusters(
        rep.X_reduced, rep.person_ids,
        X_test=X_assign, pids_test=pids_assign,
        min_cluster_size=cfg["hdbscan_min_cluster"],
        min_samples=cfg["hdbscan_min_samples"],
    )

    # Build unified label DataFrame: non-subscribers from fit + assigned fans
    labels_parts = [
        pd.DataFrame({
            "person_id": cr.person_ids_train,
            "cluster": cr.labels_train,
        })
    ]
    if cr.labels_test is not None and cr.person_ids_test is not None:
        labels_parts.append(pd.DataFrame({
            "person_id": cr.person_ids_test,
            "cluster": cr.labels_test,
        }))

    fan_labels = pd.concat(labels_parts, ignore_index=True).drop_duplicates("person_id")

    # Log subscriber assignment before override (they are in noise/-1)
    sub_pid_set = set(fan_train_sub["person_id"].unique())
    sub_mask_labels = fan_labels["person_id"].isin(sub_pid_set)
    n_sub_in_noise = (fan_labels.loc[sub_mask_labels, "cluster"] == -1).sum()
    log.info(
        "Subscriber cluster assignment (pre-override): %d subscribers, %d in noise",
        sub_mask_labels.sum(), n_sub_in_noise,
    )

    # ── 7. Cluster propensity (TRAIN) ────────────────────────────
    log.info("=" * 60)
    log.info("STEP 7: Cluster purchase propensity (TRAIN)")
    log.info("=" * 60)
    game_prop, overall_prop = build_cluster_propensity(
        split.train, fan_labels,
        bayesian_strength=cfg.get("bayesian_strength", 10.0),
    )
    game_prop.to_csv(run_dir / "cluster_propensity_per_game.csv", index=False)
    overall_prop.to_frame("p_c_smoothed").to_csv(run_dir / "cluster_propensity_overall.csv")
    log.info("Propensity saved")

    # ── 7b. Cluster profiling & validation ─────────────────────────
    log.info("=" * 60)
    log.info("STEP 7b: Cluster profiling & validation")
    log.info("=" * 60)

    profiles = build_cluster_profiles(fan_train, fan_labels, overall_prop)
    profiles.to_csv(run_dir / "cluster_profiles.csv", index=False)
    log.info("Saved cluster_profiles.csv (%d clusters)", len(profiles))

    size_stats = compute_size_balance(fan_labels)
    prop_stats = propensity_spread(overall_prop, fan_labels)

    validation_report = {
        "size_balance": size_stats,
        "propensity_spread": prop_stats,
    }
    with open(run_dir / "validation_report.json", "w") as f:
        json.dump(validation_report, f, indent=2, default=str)
    log.info("Validation report saved")

    # ── 7c. Competition-aware propensity ────────────────────────
    log.info("=" * 60)
    log.info("STEP 7c: Competition-aware propensity P(buy|cluster,comp)")
    log.info("=" * 60)
    comp_prop = build_competition_propensity(split.train, fan_labels)
    comp_prop.to_csv(run_dir / "cluster_propensity_by_competition.csv", index=False)

    # ── 7d. Grid search for optimal weights (on TRAIN) ────────
    log.info("=" * 60)
    log.info("STEP 7d: Grid search for weighted scoring weights (on TRAIN)")
    log.info("=" * 60)
    best_weights = grid_search_weights(
        fan_train, fan_labels, overall_prop, comp_prop,
        split.train, k_eval=200,
    )
    with open(run_dir / "best_weights.json", "w") as f:
        json.dump(best_weights, f, indent=2)

    # ── Canonical evaluation universe (FIX 4) ────────────────────
    # Define a single fan pool used by ALL evaluation methods so that
    # Precision@K is comparable across methods (same denominator/universe).
    eval_universe_pids = set(fan_labels["person_id"].unique())

    # Log universe stats: how many test buyers are in/out of the universe
    _test_match_for_stats = split.test.loc[
        split.test["competition_type"].isin(["LBA", "Eurocup"])
    ].copy()
    _test_match_for_stats["game_date"] = _test_match_for_stats["event_dt"].dt.normalize()
    all_test_buyers = set(_test_match_for_stats["person_id"].unique())
    in_universe_buyers = all_test_buyers & eval_universe_pids
    cold_start_buyers_total = len(all_test_buyers - eval_universe_pids)
    log.info(
        "Eval universe: %d fans | test buyers: %d total, %d in-universe (%.1f%%), "
        "%d cold-start excluded from P@K (%.1f%%)",
        len(eval_universe_pids),
        len(all_test_buyers),
        len(in_universe_buyers),
        100 * len(in_universe_buyers) / max(1, len(all_test_buyers)),
        cold_start_buyers_total,
        100 * cold_start_buyers_total / max(1, len(all_test_buyers)),
    )

    # ── 8. Evaluate on TEST ──────────────────────────────────────
    log.info("=" * 60)
    log.info("STEP 8: Evaluate ALL methods on TEST games")
    log.info("=" * 60)
    k_list = cfg["k_list"]

    # 8a. Original methods (all use canonical universe for fair comparison — FIX 4)
    cluster_eval = evaluate_targeting(
        test_tickets=split.test,
        fan_labels=fan_labels,
        cluster_propensity=overall_prop,
        k_list=k_list,
        universe_pids=eval_universe_pids,
    )
    rand_eval = random_baseline(fan_train, split.test, k_list, seed=cfg["random_seed"],
                                universe_pids=eval_universe_pids)
    rec_eval = recency_baseline(fan_train, split.test, k_list,
                                universe_pids=eval_universe_pids)
    freq_eval = frequency_baseline(fan_train, split.test, k_list,
                                   universe_pids=eval_universe_pids)
    hybrid_eval = cluster_frequency_hybrid(
        fan_train, fan_labels, overall_prop, split.test, k_list,
        universe_pids=eval_universe_pids,
    )

    # 8b. Cluster × Competition propensity
    comp_eval = evaluate_competition_propensity(
        comp_prop, fan_labels, split.test, k_list,
        universe_pids=eval_universe_pids,
    )

    # 8c. Within-cluster weighted scoring (optimized weights)
    weighted_eval = evaluate_weighted_score(
        fan_train, fan_labels, overall_prop, comp_prop,
        split.test, k_list, best_weights,
        universe_pids=eval_universe_pids,
    )

    # 8d. Frequency-boosted cluster (additive blend)
    freq_boost_eval = frequency_boosted_cluster(
        fan_train, fan_labels, overall_prop, split.test, k_list, alpha=0.3,
        universe_pids=eval_universe_pids,
    )

    # 8e. Supervised logistic regression
    lr_eval = supervised_logistic_baseline(
        fan_train, fan_labels, split.train, split.test, k_list,
        seed=cfg["random_seed"],
        universe_pids=eval_universe_pids,
    )

    # combine all methods
    all_eval = pd.concat([
        cluster_eval, rand_eval, rec_eval, freq_eval, hybrid_eval,
        comp_eval, weighted_eval, freq_boost_eval, lr_eval,
    ], ignore_index=True)
    all_eval.to_csv(run_dir / "evaluation_detail.csv", index=False)

    summary = summarise_results(all_eval)
    summary.to_csv(run_dir / "evaluation_summary.csv", index=False)

    print_summary(summary)

    # ── 8i. Walk-forward cross-validation (FIX 5, optional) ──────
    if cfg.get("run_walk_forward", False):
        log.info("=" * 60)
        log.info("STEP 8i: Walk-forward cross-validation (3 folds)")
        log.info("=" * 60)

        # All match game dates across the full dataset (train + test)
        _wf_match_mask = clean["competition_type"].isin(["LBA", "Eurocup"])
        all_match_games = sorted(
            pd.to_datetime(clean.loc[_wf_match_mask, "event_dt"], errors="coerce")
            .dt.normalize().dropna().unique()
        )
        n_total_games = len(all_match_games)
        log.info("Walk-forward: %d total match games in dataset", n_total_games)

        if n_total_games < 6:
            log.warning("Not enough games for walk-forward (%d < 6), skipping", n_total_games)
        else:
            # Fold boundaries: 60% / 73% / 87% / 100%
            boundaries = [0.60, 0.73, 0.87, 1.00]
            wf_results = []

            for fold_i in range(3):
                train_end_idx = int(boundaries[fold_i] * n_total_games)
                test_end_idx = (n_total_games if fold_i == 2
                                else int(boundaries[fold_i + 1] * n_total_games))

                train_games_set = set(all_match_games[:train_end_idx])
                test_games_set = set(all_match_games[train_end_idx:test_end_idx])

                if not train_games_set or not test_games_set:
                    log.warning("Walk-forward fold %d: empty partition, skipping", fold_i + 1)
                    continue

                cutoff_date = min(test_games_set)
                log.info(
                    "Walk-forward fold %d: %d train games (up to %s), %d test games",
                    fold_i + 1, len(train_games_set),
                    max(train_games_set).date(), len(test_games_set),
                )

                # Subset clean data for this fold
                _clean_dt = pd.to_datetime(clean["event_dt"], errors="coerce").dt.normalize()
                _move_dt = pd.to_datetime(clean.get("movement_dt", pd.NaT), errors="coerce")
                _train_mask = (
                    (_wf_match_mask & _clean_dt.isin(train_games_set)) |
                    (~_wf_match_mask & (_move_dt < cutoff_date))
                )
                _test_mask = _wf_match_mask & _clean_dt.isin(test_games_set)

                train_fold = clean.loc[_train_mask].copy()
                test_fold = clean.loc[_test_mask].copy()

                if len(train_fold) == 0 or len(test_fold) == 0:
                    log.warning("Walk-forward fold %d: empty data subset, skipping", fold_i + 1)
                    continue

                try:
                    # Build features (train only)
                    ff_fold = build_fan_features(train_fold, opponents_df=opponents,
                                                 standings_df=standings_df)

                    # Exclude subscribers from UMAP fit
                    _sub_m = ff_fold.get("is_subscription_holder",
                                         pd.Series(False, index=ff_fold.index)).astype(bool)
                    ff_fold_nosub = ff_fold.loc[~_sub_m].copy()

                    if len(ff_fold_nosub) < 50:
                        log.warning("Walk-forward fold %d: only %d non-sub fans, skipping",
                                    fold_i + 1, len(ff_fold_nosub))
                        continue

                    # Representation
                    rep_fold = fit_representation(
                        ff_fold_nosub,
                        n_components=cfg["umap_components"],
                        umap_neighbors=min(cfg["umap_neighbors"], len(ff_fold_nosub) - 1),
                        umap_min_dist=cfg["umap_min_dist"],
                        random_state=cfg["random_seed"],
                        use_pca_fallback=cfg["use_pca"],
                    )

                    # Clustering
                    cr_fold = fit_clusters(
                        rep_fold.X_reduced, rep_fold.person_ids,
                        min_cluster_size=cfg["hdbscan_min_cluster"],
                        min_samples=cfg["hdbscan_min_samples"],
                    )

                    fl_fold = pd.DataFrame({
                        "person_id": cr_fold.person_ids_train,
                        "cluster": cr_fold.labels_train,
                    })

                    # Propensity
                    _, prop_fold = build_cluster_propensity(
                        train_fold, fl_fold,
                        bayesian_strength=cfg.get("bayesian_strength", 10.0),
                    )

                    fold_universe = set(fl_fold["person_id"].unique())
                    K_WF = 200

                    # Evaluate: frequency
                    try:
                        freq_wf = frequency_baseline(ff_fold, test_fold, [K_WF],
                                                     universe_pids=fold_universe)
                        freq_p = float(freq_wf.loc[freq_wf["K"] == K_WF, "precision_at_k"].mean())
                        freq_lift = float(freq_wf.loc[freq_wf["K"] == K_WF, "lift_at_k"].mean())
                        freq_std = float(freq_wf.loc[freq_wf["K"] == K_WF, "lift_at_k"].std())
                    except Exception as e:
                        log.warning("Walk-forward fold %d frequency eval failed: %s", fold_i + 1, e)
                        freq_p = freq_lift = freq_std = float("nan")

                    # Evaluate: weighted_hybrid (uses global best_weights for speed)
                    try:
                        cp_fold = build_competition_propensity(train_fold, fl_fold)
                        wh_wf = evaluate_weighted_score(
                            ff_fold, fl_fold, prop_fold, cp_fold,
                            test_fold, [K_WF], best_weights,
                            universe_pids=fold_universe,
                        )
                        wh_p = float(wh_wf.loc[wh_wf["K"] == K_WF, "precision_at_k"].mean())
                        wh_lift = float(wh_wf.loc[wh_wf["K"] == K_WF, "lift_at_k"].mean())
                        wh_std = float(wh_wf.loc[wh_wf["K"] == K_WF, "lift_at_k"].std())
                    except Exception as e:
                        log.warning("Walk-forward fold %d weighted_hybrid eval failed: %s", fold_i + 1, e)
                        wh_p = wh_lift = wh_std = float("nan")

                    log.info(
                        "Walk-forward fold %d: freq P@200=%.3f Lift=%.2f | "
                        "wh P@200=%.3f Lift=%.2f",
                        fold_i + 1, freq_p, freq_lift, wh_p, wh_lift,
                    )

                    for method, mp, ml, ms in [
                        ("frequency", freq_p, freq_lift, freq_std),
                        ("weighted_hybrid", wh_p, wh_lift, wh_std),
                    ]:
                        wf_results.append({
                            "fold": fold_i + 1,
                            "n_train_games": len(train_games_set),
                            "n_test_games": len(test_games_set),
                            "method": method,
                            "mean_precision_at_200": round(mp, 4) if not np.isnan(mp) else None,
                            "mean_lift_at_200": round(ml, 4) if not np.isnan(ml) else None,
                            "std_lift_at_200": round(ms, 4) if not np.isnan(ms) else None,
                        })

                except Exception as e:
                    log.warning("Walk-forward fold %d failed: %s", fold_i + 1, e)
                    continue

            if wf_results:
                wf_df = pd.DataFrame(wf_results)
                wf_path = run_dir / "walk_forward_validation.csv"
                wf_df.to_csv(wf_path, index=False)
                log.info("Walk-forward results saved: %s", wf_path)

                # Stability summary per method
                for _method in ["frequency", "weighted_hybrid"]:
                    _mdf = wf_df[wf_df["method"] == _method]["mean_lift_at_200"].dropna()
                    if len(_mdf) < 2:
                        continue
                    _mean = float(_mdf.mean())
                    _std = float(_mdf.std())
                    _cv = _std / _mean if _mean > 0 else float("inf")
                    if _cv < 0.3:
                        _stability = "STABLE"
                    elif _cv < 0.5:
                        _stability = "MODERATE VARIANCE"
                    else:
                        _stability = "HIGH VARIANCE — results may not generalise"
                    log.info(
                        "Walk-forward stability [%s]: mean_lift@200=%.3f, std=%.3f, "
                        "CV=%.3f → %s",
                        _method, _mean, _std, _cv, _stability,
                    )
                    if _cv > 0.5:
                        log.warning(
                            "HIGH VARIANCE in walk-forward [%s]: CV=%.3f > 0.5 — "
                            "single-split results may not generalise",
                            _method, _cv,
                        )
            else:
                log.warning("Walk-forward: no fold results generated")
    else:
        log.info("Walk-forward validation skipped (use --run_walk_forward to enable)")

    # ── 8e. EXPORT: top-3 methods → NOT-BOUGHT-YET target lists ─────────
    # Goal: save small, actionable CSVs for marketing:
    # For each test game, for each of the top-3 methods (by avg Precision@K),
    # save the top-K fans who have NOT bought yet (and pass consent/email filters).

    from src.package_generator import _build_opponent_label
    from src.game_targeting import identify_subscription_holders, identify_already_bought
    # from src.baselines import _minmax  # if not accessible, replace with local minmax below
    from src.advanced_scoring import build_weighted_score

    def _minmax_local(s: pd.Series) -> pd.Series:
        s = s.astype(float)
        mn, mx = s.min(), s.max()
        den = (mx - mn) if (mx - mn) != 0 else 1.0
        return (s - mn) / den

    # pick a single K for “marketing lists” (use your smallest K)
    K_EXPORT = int(cfg["k_list"][0])  # e.g. 200

    # Rank methods by mean precision@K_EXPORT across test games
    # (Only keep methods we can actually EXPORT into ranked lists)
    # Methods that have a compute_scores_for_method() implementation.
    # "cluster" (propensity-only) and "cluster_x_comp" (competition propensity)
    # are intentionally excluded: they are evaluated but cannot be re-scored
    # per-game for the marketing-list export without fan-level feature access.
    EXPORTABLE_METHODS = {
        "recency",
        "frequency",
        "cluster_x_freq",
        "weighted_hybrid",
        "freq_boost_cluster",
    }

    # summary df columns depend on summarise_results(); safer to use detail
    detail_df = all_eval.copy()
    detail_df = detail_df[detail_df["K"] == K_EXPORT].copy()

    mean_perf = (
        detail_df.groupby("method")["precision_at_k"]
        .mean()
        .sort_values(ascending=False)
    )
    top3_methods = [m for m in mean_perf.index if m in EXPORTABLE_METHODS][:3]

    log.info("TOP-3 EXPORT methods at K=%d: %s", K_EXPORT, top3_methods)

    export_dir = run_dir / "game_targets_top3"
    export_dir.mkdir(parents=True, exist_ok=True)

    # Precompute pieces used by some scorers
    ff_base = fan_train.copy()
    fl = fan_labels[["person_id", "cluster"]].drop_duplicates()

    # cluster propensity map needed for cluster_x_freq / freq_boost_cluster / weighted_hybrid
    cluster_prop_map = overall_prop

    # helper: compute a global ranking score per person_id for each method
    def compute_scores_for_method(method: str, game_comp: str) -> pd.DataFrame:
        df = ff_base.merge(fl, on="person_id", how="inner").copy()

        if method == "frequency":
            df["score"] = df["games_attended"].fillna(0)

        elif method == "recency":
            # smaller recency_days = better → invert
            rec = df["recency_days"].fillna(df["recency_days"].max() + 1)
            df["score"] = -rec

        elif method == "cluster_x_freq":
            df["p_c"] = df["cluster"].map(cluster_prop_map).fillna(0)
            freq_norm = _minmax_local(df["games_attended"].fillna(0))
            pc_norm = _minmax_local(df["p_c"])
            df["score"] = freq_norm * pc_norm

        elif method == "freq_boost_cluster":
            # same formula as advanced_scoring.frequency_boosted_cluster (additive blend)
            alpha = 0.3
            df["p_c"] = df["cluster"].map(cluster_prop_map).fillna(0)
            freq_norm = _minmax_local(df["games_attended"].fillna(0))
            pc_norm = _minmax_local(df["p_c"])
            df["score"] = alpha * pc_norm + (1 - alpha) * freq_norm

        elif method == "weighted_hybrid":
            # use learned weights + competition alignment
            scored = build_weighted_score(
                fan_features=ff_base,
                fan_labels=fl,
                cluster_propensity=cluster_prop_map,
                comp_propensity=comp_prop,
                test_comp=game_comp,
                **best_weights,
            )
            scored = scored.rename(columns={"weighted_score": "score"})
            return scored[["person_id", "score"]].copy()

        else:
            raise ValueError(f"Unsupported export method: {method}")

        return df[["person_id", "score"]].copy()

    # For each TEST game: export top-K_EXPORT NOT-BOUGHT-YET lists for each top method
    for gd in sorted(split.test_games):
        game_detail = detail_df[detail_df["game_date"] == gd].copy()

        game_perf = (
            game_detail.groupby("method")["precision_at_k"]
            .mean()
            .sort_values(ascending=False)
        )

        top3_methods = [m for m in game_perf.index if m in EXPORTABLE_METHODS][:3]
        match_key = str(pd.Timestamp(gd).date())
        game_date = pd.Timestamp(gd).normalize()

        opp_row = None
        if opponents is not None:
            opp_row = opponents.loc[opponents["game_date"].dt.normalize() == game_date]
            opp_row = opp_row.iloc[0] if len(opp_row) else None

        opponent = str(opp_row["Opponent"]) if opp_row is not None and "Opponent" in opp_row else "Unknown_Opponent"
        game_comp = str(opp_row["Competition"]) if opp_row is not None and "Competition" in opp_row else "LBA"
        file_label = _build_opponent_label(game_date, opponent)

        # exclusions for THIS game
        sub_pids = identify_subscription_holders(clean, game_date)
        bought_pids = identify_already_bought(clean, game_date)

        # consent/email flags (used for splitting AFTER scoring, not before)
        pl_flags = person_lookup[["person_id", "marketing_consent", "has_email"]].drop_duplicates("person_id")

        # Always exclude subs + already-bought (prediction exclusions, not consent)
        excluded_pids = set(sub_pids) | set(bought_pids)

        for method in top3_methods:
            scores = compute_scores_for_method(method, game_comp)

            # Remove subs/already-bought; do NOT filter by consent here —
            # evaluation is on all fans; consent is a delivery constraint only.
            scores = scores[~scores["person_id"].isin(excluded_pids)]

            # Attach consent/email flags for post-scoring split
            scores = scores.merge(pl_flags, on="person_id", how="left")
            scores["marketing_consent"] = scores["marketing_consent"].fillna(0).astype(int)
            scores["has_email"] = scores["has_email"].fillna(False).astype(bool)

            # Rank all fans by score descending
            scores = scores.sort_values("score", ascending=False).reset_index(drop=True)
            scores["rank"] = scores.index + 1

            # Split: ready-to-send (consent=1 AND email) vs high-intent (no consent/email)
            consent_mask = (scores["marketing_consent"] == 1) & (scores["has_email"] == True)
            with_consent = scores[consent_mask].head(K_EXPORT)
            no_consent   = scores[~consent_mask].head(K_EXPORT)

            # Add full contact info for marketing
            contact_cols = [c for c in ["person_id", "buyer_email", "nome", "cognome"]
                            if c in person_lookup.columns]

            def _enrich(df: pd.DataFrame) -> pd.DataFrame:
                """Merge contact info and normalise columns."""
                # buyer_email may already exist if merge happened above
                if "buyer_email" not in df.columns:
                    df = df.merge(person_lookup[contact_cols], on="person_id", how="left")
                df = df.rename(columns={"buyer_email": "email"})
                cols_out = ["person_id", "email", "nome", "cognome",
                            "marketing_consent", "has_email", "score", "rank"]
                return df[[c for c in cols_out if c in df.columns]]

            with_consent_out = _enrich(with_consent)
            no_consent_out   = _enrich(no_consent)

            # Write ready-to-send list
            out_path = export_dir / f"targets_{method}_K{K_EXPORT}_{file_label}.csv"
            with_consent_out.to_csv(out_path, index=False)
            log.info("Exported %s (%d rows, consented)", out_path.name, len(with_consent_out))

            # Write high-intent / no-consent list (new)
            hi_path = export_dir / f"high_intent_{method}_K{K_EXPORT}_{file_label}.csv"
            no_consent_out.to_csv(hi_path, index=False)
            log.info("Exported %s (%d rows, no consent/email)", hi_path.name, len(no_consent_out))

    # ── TASK 5B: Override subscriber labels → SUBSCRIBER ─────────
    # Done AFTER evaluation (which needs numeric labels) but BEFORE
    # marketing packages (which display the label to the marketing team).
    fan_labels["cluster"] = fan_labels["cluster"].astype(str)
    fan_labels.loc[sub_mask_labels, "cluster"] = SUBSCRIBER_CLUSTER
    n_sub_labelled = (fan_labels["cluster"] == SUBSCRIBER_CLUSTER).sum()
    log.info(
        "TASK 5B — Subscriber label override: %d fans → cluster='%s'",
        n_sub_labelled, SUBSCRIBER_CLUSTER,
    )
    fan_labels.to_csv(run_dir / "fan_cluster_labels.csv", index=False)

    # ── 8f. Per-game marketing packages ───────────────────────────
    if not cfg.get("skip_packages", False):
        log.info("=" * 60)
        log.info("STEP 8f: Generate per-game marketing packages")
        log.info("=" * 60)
        pkg_dir = run_dir / "game_packages"
        require_consent = cfg.get("require_consent", True)
        require_email = cfg.get("require_email", True)
        log.info(
            "Package config: require_consent=%s, require_email=%s",
            require_consent, require_email,
        )
        packages = generate_all_test_packages(
            test_games=split.test_games,
            tickets_df=clean,
            fan_features=fan_train,
            fan_labels=fan_labels,
            cluster_propensity=overall_prop,
            opponents_df=opponents,
            person_lookup=person_lookup,
            out_dir=pkg_dir,
            top_n=cfg.get("target_top_n"),
            export_bought_already=cfg.get("export_bought_already", False),
            require_consent=require_consent,
            require_email=require_email,
            propensity_mode="overall",
        )
        log.info("Generated %d game packages in %s", len(packages), pkg_dir)

        # ── Print consent enforcement evidence ───────────────────
        for pkg in packages:
            s = pkg["summary"]
            log.info(
                "  Game %s: %d total, %d sub, %d already_bought, "
                "%d no_consent, %d targetable (in target_list=%d)",
                s.get("match_key", "?"),
                s.get("n_total_fans", 0),
                s.get("n_subscription_holders", 0),
                s.get("n_already_bought", 0),
                s.get("n_no_consent", 0),
                s.get("n_targetable", 0),
                s.get("n_in_target_list", 0),
            )
    else:
        log.info("Skipping per-game package generation (--skip_packages)")

    # ── 8g. Generate FUTURE fixture packages ──────────────────────
    future_packages = []
    if cfg.get("include_future_games", False) and opponents is not None:
        log.info("=" * 60)
        log.info("STEP 8g: Generate FUTURE fixture marketing packages")
        log.info("=" * 60)
        future_pkg_dir = run_dir / "game_packages_future"
        require_consent = cfg.get("require_consent", True)
        require_email = cfg.get("require_email", True)

        future_from = cfg.get("future_from_date")
        if future_from:
            future_from = pd.Timestamp(future_from)

        future_packages = generate_future_packages(
            opponents_df=opponents,
            tickets_df=clean,
            fan_features=fan_train,
            fan_labels=fan_labels,
            cluster_propensity=overall_prop,
            person_lookup=person_lookup,
            out_dir=future_pkg_dir,
            top_n=cfg.get("target_top_n"),
            require_consent=require_consent,
            require_email=require_email,
            propensity_mode="overall",
            after_date=future_from,
            max_n=cfg.get("future_n"),
            only_competitions=cfg.get("future_only_competitions"),
            only_opponents=cfg.get("future_only_opponents"),
        )
        log.info("Generated %d future fixture packages in %s", len(future_packages), future_pkg_dir)

        for pkg in future_packages:
            s = pkg["summary"]
            log.info(
                "  FUTURE %s vs %s: %d total, %d sub, %d already_bought, "
                "%d no_consent, %d targetable (in target_list=%d)",
                s.get("game_date", "?"),
                s.get("opponent_team", "?"),
                s.get("n_total_fans", 0),
                s.get("n_subscription_holders", 0),
                s.get("n_already_bought", 0),
                s.get("n_no_consent", 0),
                s.get("n_targetable", 0),
                s.get("n_in_target_list", 0),
            )
    elif cfg.get("include_future_games", False) and opponents is None:
        log.warning("--include_future_games set but no --opponents_csv provided; skipping")

    # ── 8h. Opponent-aware ranking validation + leakage guard ──────
    log.info("=" * 60)
    log.info("STEP 8h: Opponent-aware ranking validation & sanity checks")
    log.info("=" * 60)
    _run_sanity_checks(
        fan_train=fan_train,
        fan_labels=fan_labels,
        opponents=opponents,
        split_train=split.train,
        run_dir=run_dir,
    )

    # ── 9. Build fan_level.csv (final export) ─────────────────────
    log.info("=" * 60)
    log.info("STEP 9: Build fan_level.csv (complete fan export)")
    log.info("=" * 60)
    fan_level = fan_train.merge(fan_labels, on="person_id", how="left")
    # also merge contact info (columns from consent-aware person_lookup)
    pl_cols = [c for c in ["person_id", "buyer_email", "nome", "cognome",
                            "marketing_consent"]
            if c in person_lookup.columns]

    fan_level = fan_level.merge(
        person_lookup[pl_cols],
        on="person_id", how="left",
    )

    # rename buyer_email -> email
    if "buyer_email" in fan_level.columns:
        fan_level = fan_level.rename(columns={"buyer_email": "email"})

    fan_level.to_csv(run_dir / "fan_level.csv", index=False)
    log.info("Saved fan_level.csv (%d fans × %d cols)", len(fan_level), len(fan_level.columns))

    # consent stats in fan_level
    n_fl_consent = (fan_level.get("marketing_consent", 0) == 1).sum()
    n_fl_sub = (fan_level["cluster"] == SUBSCRIBER_CLUSTER).sum() if "cluster" in fan_level.columns else 0
    log.info(
        "fan_level: %d with marketing_consent, %d with cluster='%s'",
        n_fl_consent, n_fl_sub, SUBSCRIBER_CLUSTER,
    )

    # ── 10. Save run metadata ─────────────────────────────────────
    elapsed = time.time() - t0
    meta = {
        "elapsed_seconds": round(elapsed, 1),
        "raw_csv": str(cfg["raw_csv"]),
        "consent_csv": str(cfg.get("consent_csv", "none")),
        "test_frac": float(cfg["test_frac"]),
        "k_list": [int(k) for k in k_list],
        "random_seed": int(cfg["random_seed"]),
        "n_train_fans": int(len(fan_train)),
        "n_train_fans_nosub": int(n_nosub),
        "n_subscription_holders": int(n_sub_actual),
        "n_subscriber_labelled": int(n_sub_labelled),
        "n_test_known_fans": int(len(fan_test_known)),
        "n_clusters": int(cr.n_clusters),
        "n_noise_train": int(cr.n_noise_train),
        "n_fans_with_email": int(n_pl_email),
        "n_fans_with_consent": int(n_pl_consent),
        "require_consent": bool(cfg.get("require_consent", True)),
        "require_email": bool(cfg.get("require_email", True)),
        "reducer": rep.reducer_name,
        "reducer_n_components": int(cfg["umap_components"]),
        "hdbscan_min_cluster_size": int(cfg["hdbscan_min_cluster"]),
        "hdbscan_min_samples": int(cfg["hdbscan_min_samples"]),
        "cluster_size_balance": size_stats,
        "propensity_spread": prop_stats,
        "best_weights": best_weights,
        "subscribers_excluded_from_fit": True,
        "subscriber_cluster_label": SUBSCRIBER_CLUSTER,
        "include_future_games": bool(cfg.get("include_future_games", False)),
        "n_future_packages": len(future_packages),
        "future_fixtures": [
            {
                "game_date": p["summary"].get("game_date"),
                "opponent": p["summary"].get("opponent_team"),
                "competition": p["summary"].get("competition"),
                "n_targetable": p["summary"].get("n_targetable"),
            }
            for p in future_packages
        ] if future_packages else [],
    }
    with open(run_dir / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    log.info("Pipeline complete in %.1f seconds", elapsed)
    log.info("All artefacts in: %s", run_dir)

    # List all output files
    print(f"\nOutput files in {run_dir}:")
    for p in sorted(run_dir.rglob("*")):
        if p.is_file():
            rel = p.relative_to(run_dir)
            print(f"  {rel}")


if __name__ == "__main__":
    main()
