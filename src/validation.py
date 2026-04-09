"""
Cluster validation: marketing-oriented profiling, stability, and diagnostics.

Focus on actionable metrics (NOT academic silhouette-only):
  - Cluster interpretability profiles
  - Cluster size balance (Gini index)
  - Stability across random seeds or sub-samples
  - Per-cluster propensity spread (are clusters predictive?)
"""
# =============================================================================
# Three validation perspectives:
#
# (1) PROFILES — what does each cluster look like in terms of demographics
#     and behaviour?  Median feature values per cluster, plus an auto-generated
#     human-readable label, help the marketing team understand who they are
#     targeting and craft appropriate messaging.
#
# (2) SIZE BALANCE — are clusters roughly equal in size, or dominated by one
#     giant cluster?  Gini=0 is perfect balance (all clusters identical size);
#     Gini=1 means one cluster contains every fan.  Large imbalance reduces the
#     practical usefulness of cluster-based targeting.
#
# (3) PROPENSITY SPREAD — do clusters have meaningfully different purchase
#     rates?  If all clusters have similar propensity, clustering is not helping
#     targeting — frequency or recency alone would be just as effective.  The
#     separation_ratio (top-5 cluster propensity / bottom-5) quantifies this.
# =============================================================================
from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── profile columns: features that are meaningful for marketing ────
_PROFILE_COLS = [
    "games_attended", "match_games_attended", "total_spend", "match_spend",
    "avg_ticket_price_match_only", "recency_days",
    "pct_early_bird", "pct_last_minute",
    "pct_full_price", "pct_discounted_vs_list", "pct_zero_price",
    "pct_lba_games", "pct_eurocup_games",
    "pct_weekend_games", "pct_evening_games",
    "sector_variety", "sector_consistency",
    "premium_affinity", "discount_dependency",
    "pct_high_value_opponents", "derby_attendance_rate",
    "purchase_occasions", "avg_tickets_per_purchase",
    "active_days", "age_mode",
    "pct_child_tickets", "is_subscription_holder",
]


def build_cluster_profiles(
    fan_features: pd.DataFrame,
    fan_labels: pd.DataFrame,
    cluster_propensity: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Build a human-readable profile for each cluster.

    Parameters
    ----------
    fan_features : DataFrame
        One row per person_id with features.
    fan_labels : DataFrame
        ``[person_id, cluster]``.
    cluster_propensity : Series, optional
        Smoothed purchase propensity per cluster.

    Returns
    -------
    DataFrame with one row per cluster and columns:
        cluster, n_fans, pct_of_total, <median of each profile feature>,
        p_c_smoothed, label (human-readable description).
    """
    fl = fan_labels[["person_id", "cluster"]].drop_duplicates()
    df = fan_features.merge(fl, on="person_id", how="inner")

    # Intersect with what is actually present in fan_features — the
    # pipeline may not always compute every optional feature column.
    avail = [c for c in _PROFILE_COLS if c in df.columns]

    # Summarise each cluster with median values (robust to outliers).
    profiles = df.groupby("cluster")[avail].median()
    profiles["n_fans"] = df.groupby("cluster").size()
    total = profiles["n_fans"].sum()
    profiles["pct_of_total"] = (profiles["n_fans"] / total * 100).round(1)

    if cluster_propensity is not None:
        profiles["p_c_smoothed"] = cluster_propensity

    # _auto_label converts median feature values into a readable string
    # (e.g. "Loyal | LBA-only | High-spend") applied row-by-row.
    profiles["label"] = profiles.apply(_auto_label, axis=1)

    profiles = profiles.reset_index()
    # Reorder columns so the most-used fields appear first in any printed
    # or exported table: cluster identity, size, label, propensity, then
    # all the raw feature medians.
    first = ["cluster", "n_fans", "pct_of_total", "label"]
    if "p_c_smoothed" in profiles.columns:
        first.append("p_c_smoothed")
    rest = [c for c in profiles.columns if c not in first]
    profiles = profiles[first + rest]

    log.info("Cluster profiles: %d clusters, %d features", len(profiles), len(avail))
    return profiles


def _auto_label(row: pd.Series) -> str:
    """Generate a short human-readable label from cluster median features."""
    # Thresholds chosen based on domain knowledge of basketball fan behaviour:
    # e.g. attending 8+ match-day games marks a genuinely loyal season follower,
    # spending 200+ EUR distinguishes high-value customers, and >30% premium-
    # sector purchases signals willingness to pay for better seating.
    parts = []

    # frequency tier
    ga = row.get("match_games_attended", 0)
    if pd.notna(ga):
        if ga >= 8:
            parts.append("Loyal")
        elif ga >= 3:
            parts.append("Regular")
        else:
            parts.append("Casual")

    # competition preference
    lba = row.get("pct_lba_games", 0) or 0
    euro = row.get("pct_eurocup_games", 0) or 0
    if euro > 0.6:
        parts.append("Eurocup-fan")
    elif lba > 0.85:
        parts.append("LBA-only")
    else:
        parts.append("Mixed-comp")

    # spend tier
    spend = row.get("total_spend", 0) or 0
    if spend >= 200:
        parts.append("High-spend")
    elif spend >= 50:
        parts.append("Mid-spend")
    else:
        parts.append("Low-spend")

    # premium flag
    prem = row.get("premium_affinity", 0) or 0
    if prem > 0.3:
        parts.append("Premium")

    # timing
    eb = row.get("pct_early_bird", 0) or 0
    lm = row.get("pct_last_minute", 0) or 0
    if eb > 0.6:
        parts.append("Early-buyer")
    elif lm > 0.6:
        parts.append("Last-minute")

    # subscription
    sub = row.get("is_subscription_holder", 0)
    if pd.notna(sub) and sub > 0.5:
        parts.append("Subscriber")

    return " | ".join(parts) if parts else "Unclassified"


def compute_size_balance(fan_labels: pd.DataFrame) -> dict:
    """Compute cluster size distribution stats.

    Returns dict with:
        n_clusters, n_noise, noise_pct,
        min_size, max_size, median_size, mean_size,
        gini_index (0=perfectly equal, 1=one cluster has all fans).
    """
    sizes = fan_labels.groupby("cluster").size()
    non_noise = sizes.drop(-1, errors="ignore")
    n_noise = int(sizes.get(-1, 0))
    total = int(sizes.sum())

    vals = non_noise.sort_values().values.astype(float)
    n = len(vals)

    if n == 0:
        return {"n_clusters": 0, "n_noise": n_noise, "noise_pct": 100.0}

    # Gini index — standard discrete formula on sorted cluster sizes.
    # Intuition: area between the Lorenz curve and the line of perfect equality.
    # Gini=0.0  → all clusters have identical size (perfect balance).
    # Gini=0.5  → moderate imbalance, typical for real-world clustering.
    # Gini→1.0  → one cluster dominates; cluster-based targeting loses granularity.
    cum = vals.cumsum()
    gini = 1 - 2 * cum.sum() / (n * vals.sum()) + 1 / n if vals.sum() > 0 else 0

    stats = {
        "n_clusters": n,
        "n_noise": n_noise,
        "noise_pct": round(100 * n_noise / max(1, total), 1),
        "min_size": int(vals.min()),
        "max_size": int(vals.max()),
        "median_size": int(np.median(vals)),
        "mean_size": round(vals.mean(), 1),
        "gini_index": round(gini, 4),
    }

    log.info(
        "Size balance: %d clusters, noise=%.1f%%, sizes=[%d, %d], Gini=%.3f",
        n, stats["noise_pct"], stats["min_size"], stats["max_size"], gini,
    )
    return stats


def propensity_spread(
    cluster_propensity: pd.Series,
    fan_labels: pd.DataFrame,
) -> dict:
    """Measure how well clusters separate purchase propensity.

    Returns dict with:
        propensity_range, propensity_std, propensity_iqr,
        top5_mean, bottom5_mean, separation_ratio.
    """
    prop = cluster_propensity.drop(-1, errors="ignore")
    if prop.empty:
        return {}

    sizes = fan_labels[fan_labels["cluster"] != -1].groupby("cluster").size()
    # weight by cluster size
    aligned = prop.reindex(sizes.index).dropna()

    # separation_ratio = mean propensity of the 5 highest-propensity clusters
    # divided by the mean of the 5 lowest.  A ratio of 1.0 means clusters are
    # indistinguishable from a targeting perspective — no cluster signal.
    # A ratio of 2.0+ means the best clusters are at least twice as likely to
    # buy, making cluster-based prioritisation genuinely useful.
    top5 = aligned.nlargest(5).mean()
    bot5 = aligned.nsmallest(5).mean()
    sep = top5 / bot5 if bot5 > 0 else np.inf

    stats = {
        "propensity_range": round(float(aligned.max() - aligned.min()), 4),
        "propensity_std": round(float(aligned.std()), 4),
        "propensity_iqr": round(float(aligned.quantile(0.75) - aligned.quantile(0.25)), 4),
        "top5_mean_propensity": round(float(top5), 4),
        "bottom5_mean_propensity": round(float(bot5), 4),
        "separation_ratio": round(float(sep), 2),
    }

    log.info(
        "Propensity spread: range=%.4f, IQR=%.4f, top5/bot5=%.1fx",
        stats["propensity_range"], stats["propensity_iqr"], sep,
    )
    return stats


def stability_check(
    fan_train: pd.DataFrame,
    fit_fn,
    *,
    n_runs: int = 5,
    subsample_frac: float = 0.8,
    seed: int = 42,
) -> pd.DataFrame:
    """Check cluster stability by sub-sampling train data.

    Parameters
    ----------
    fan_train : DataFrame
        Fan-level features.
    fit_fn : callable
        Function ``fit_fn(fan_subset) -> pd.Series`` that takes a fan
        features subset and returns a Series ``{person_id: cluster_label}``.
    n_runs : int
        Number of sub-sampling runs.
    subsample_frac : float
        Fraction of fans to keep per run.
    seed : int
        Random seed.

    Returns
    -------
    DataFrame with person_id rows and columns ``run_0, run_1, ...``
    showing cluster assignment per run.  Use ``adjusted_rand_score``
    or ``pairwise_agreement`` to compare.
    """
    rng = np.random.RandomState(seed)
    results = {}

    for i in range(n_runs):
        idx = rng.choice(len(fan_train), size=int(len(fan_train) * subsample_frac), replace=False)
        subset = fan_train.iloc[idx].copy()
        labels = fit_fn(subset)
        results[f"run_{i}"] = labels

    out = pd.DataFrame(results)
    log.info("Stability check: %d runs, %.0f%% subsample, %d fans per run",
             n_runs, subsample_frac * 100, int(len(fan_train) * subsample_frac))
    return out
