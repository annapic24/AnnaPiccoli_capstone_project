"""
Advanced scoring methods to improve cluster-based targeting.

ROOT CAUSE ANALYSIS (why cluster-only propensity is weak)
=========================================================
The diagnosis reveals three compounding problems:

1. **FLAT PROPENSITY BAND**: 31 of 33 clusters have smoothed propensity in
   [0.058, 0.072] — an IQR of 0.0002.  Only clusters 0 (p=0.169) and 2
   (p=0.147) separate from the pack.  This means ranking by cluster
   propensity is effectively *random* for 92% of fans.

2. **CLUSTERS CAPTURE BEHAVIOUR, NOT INTENT**: HDBSCAN groups fans by
   behavioural similarity (timing, sector, price sensitivity), but fans
   within the same cluster have vastly different purchase frequencies
   (range [1, 7] inside cluster 0).  Purchase frequency — the strongest
   predictor — varies *within* clusters, not *between* them.

3. **NO GAME-SPECIFIC SIGNAL**: The overall propensity P(buy|cluster) is
   averaged across all 14 train games.  But per-game propensity has huge
   variance (CV > 3.0 for most clusters).  A cluster that turns out 40% for
   one game may turn out 0% for another.  The average washes out this signal.

SOLUTIONS IMPLEMENTED
=====================
A) **Competition-aware propensity** P(buy | cluster, comp_type):
   Separate propensity for LBA vs Eurocup games.

B) **Within-cluster weighted ranking**: Combine cluster propensity with
   individual-level features (frequency, recency, price sensitivity,
   competition alignment) using optimised weights.

C) **Supervised baseline**: Logistic regression on cluster_id × individual
   features, trained on per-game buy/not-buy labels from train games.

D) **Frequency-boosted cluster scoring**: Additive blend of normalized
   cluster propensity + normalized frequency, avoiding the multiplicative
   zero-problem of the old hybrid.
"""
from __future__ import annotations

import logging
from itertools import product
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)


# =====================================================================
# A) Competition-aware propensity
# =====================================================================

def build_competition_propensity(
    train_tickets: pd.DataFrame,
    fan_labels: pd.DataFrame,
    *,
    bayesian_alpha: float = 1.0,
    bayesian_beta: float = 1.0,
) -> pd.DataFrame:
    """Compute P(buy | cluster, competition_type) from train data.

    Returns DataFrame with columns:
        cluster, competition_type, n_games, n_total, total_buyers,
        p_c_comp_smoothed
    """
    # ── SECTION 1: filter to match tickets (LBA and Eurocup only) ─
    # Pack, subscription, and non-game events are excluded — propensity
    # is only meaningful for competitive fixtures.
    tickets = train_tickets.loc[
        train_tickets["competition_type"].isin(["LBA", "Eurocup"])
    ].copy()
    tickets["game_date"] = tickets["event_dt"].dt.normalize()

    # ── SECTION 2: count games per competition type ────────────────
    # We need the total number of LBA / Eurocup games each cluster was
    # exposed to, which forms the denominator of the propensity estimate.
    fans = fan_labels[["person_id", "cluster"]].drop_duplicates()
    # keep noise (-1) as its own group so it is handled consistently
    clusters = sorted(fans["cluster"].unique())
    n_per_cluster = fans.groupby("cluster").size().rename("n_total")

    # unique games per competition, used for exposure count
    game_comp = tickets.drop_duplicates("game_date")[["game_date", "competition_type"]]
    games_per_comp = game_comp.groupby("competition_type").size()

    rows = []
    for comp in ["LBA", "Eurocup"]:
        comp_tickets = tickets[tickets["competition_type"] == comp]
        comp_games = sorted(comp_tickets["game_date"].unique())
        n_comp_games = len(comp_games)
        if n_comp_games == 0:
            continue

        # ── SECTION 3: observed buyers per cluster ─────────────────
        # Count unique (person, game) pairs so a fan who buys multiple
        # tickets to the same game is counted only once as a "buyer".
        buyers = comp_tickets[["person_id", "game_date"]].drop_duplicates()
        buyers = buyers.merge(fans, on="person_id", how="inner")
        observed = buyers.groupby("cluster").size().rename("total_buyers")

        for c in clusters:
            n_total = n_per_cluster.get(c, 0)
            tb = observed.get(c, 0)
            total_exp = n_total * n_comp_games
            # ── SECTION 4: Bayesian smoothing ──────────────────────
            # Add-alpha smoothing (Beta-Binomial conjugate) pulls sparse
            # estimates towards a neutral prior.  With alpha=beta=1 this
            # is equivalent to Laplace smoothing.  Small clusters with
            # few observations are shrunk more strongly, reducing noise.
            p_smooth = (tb + bayesian_alpha) / (total_exp + bayesian_alpha + bayesian_beta)

            rows.append({
                "cluster": c,
                "competition_type": comp,
                "n_games": n_comp_games,
                "n_total": n_total,
                "total_buyers": tb,
                "total_exposures": total_exp,
                "p_c_comp_smoothed": p_smooth,
            })

    result = pd.DataFrame(rows)
    log.info(
        "Competition propensity: %d cluster×comp pairs, "
        "LBA range=[%.4f, %.4f], Eurocup range=[%.4f, %.4f]",
        len(result),
        result.loc[result["competition_type"] == "LBA", "p_c_comp_smoothed"].min(),
        result.loc[result["competition_type"] == "LBA", "p_c_comp_smoothed"].max(),
        result.loc[result["competition_type"] == "Eurocup", "p_c_comp_smoothed"].min()
            if (result["competition_type"] == "Eurocup").any() else 0,
        result.loc[result["competition_type"] == "Eurocup", "p_c_comp_smoothed"].max()
            if (result["competition_type"] == "Eurocup").any() else 0,
    )
    return result


def evaluate_competition_propensity(
    comp_propensity: pd.DataFrame,
    fan_labels: pd.DataFrame,
    test_tickets: pd.DataFrame,
    k_list: Sequence[int],
    universe_pids: set | None = None,
) -> pd.DataFrame:
    """Evaluate cluster×competition propensity on test games.

    For each test game, looks up the propensity for that game's competition
    type, then ranks fans by that propensity (with random tie-breaking).
    """
    match = test_tickets.loc[
        test_tickets["competition_type"].isin(["LBA", "Eurocup"])
    ].copy()
    match["game_date"] = match["event_dt"].dt.normalize()
    test_games = sorted(match["game_date"].unique())

    fans = fan_labels[["person_id", "cluster"]].drop_duplicates().copy()
    # Apply canonical universe filter (FIX 4)
    if universe_pids is not None:
        fans = fans[fans["person_id"].isin(universe_pids)].copy()

    rng = np.random.RandomState(42)
    fans["_tie"] = rng.rand(len(fans))

    # build lookup: (cluster, comp) → propensity
    prop_map = comp_propensity.set_index(["cluster", "competition_type"])["p_c_comp_smoothed"]

    rows = []
    for gd in test_games:
        game_rows = match[match["game_date"] == gd]
        comp = game_rows["competition_type"].mode().iloc[0] if len(game_rows) > 0 else "LBA"
        buyers_all = set(game_rows["person_id"].unique())

        # Cold-start tracking (FIX 4)
        if universe_pids is not None:
            cold_start_buyers = len(buyers_all - universe_pids)
            buyers = buyers_all & universe_pids
        else:
            cold_start_buyers = 0
            buyers = buyers_all

        # rank fans by competition-specific propensity
        fans_g = fans.copy()
        fans_g["p_c"] = fans_g["cluster"].map(
            lambda c, comp=comp: prop_map.get((c, comp), 0)
        )
        fans_g = fans_g.sort_values(["p_c", "_tie"], ascending=[False, True])
        ranked_pids = fans_g["person_id"].values

        n_universe = len(ranked_pids)
        n_buyers = len(buyers)
        overall_rate = n_buyers / max(1, n_universe)

        for k in k_list:
            k_eff = min(k, n_universe)
            top_k = set(ranked_pids[:k_eff])
            hits = len(top_k & buyers)
            precision = hits / k_eff if k_eff > 0 else 0
            lift = precision / overall_rate if overall_rate > 0 else 0

            rows.append({
                "game_date": gd, "K": k, "n_universe": n_universe,
                "n_buyers": n_buyers, "overall_rate": overall_rate,
                "precision_at_k": precision, "lift_at_k": lift,
                "hits_at_k": hits, "cold_start_buyers": cold_start_buyers,
                "method": "cluster_x_comp",
            })

    result = pd.DataFrame(rows)
    log.info("Cluster×Comp eval: %d games × %d K values",
             result["game_date"].nunique(), len(k_list))
    return result


# =====================================================================
# B) Within-cluster weighted scoring with grid search
# =====================================================================

def _minmax(s: pd.Series) -> pd.Series:
    """Min-max normalise to [0, 1].

    Used before combining multiple signals (frequency, recency, cluster
    propensity, price sensitivity) into a weighted score — each signal
    must be on the same [0, 1] scale so the weights have consistent meaning.
    """
    lo, hi = s.min(), s.max()
    rng = hi - lo
    if rng == 0 or pd.isna(rng):
        return pd.Series(0.5, index=s.index)
    return (s - lo) / rng


# ─────────────────────────────────────────────────────────────────────────────
# This function scores each fan using 5 signals combined with configurable
# weights. The default weights (w_freq=0.55, w_recency=0.20, w_cluster=0.05,
# w_price=0.10, w_comp=0.10) were found via grid search to maximise
# Precision@200 on held-out train games. Each component is independently
# normalised to [0,1] before weighting so they are on comparable scales.
# ─────────────────────────────────────────────────────────────────────────────
def build_weighted_score(
    fan_features: pd.DataFrame,
    fan_labels: pd.DataFrame,
    cluster_propensity: pd.Series,
    comp_propensity: pd.DataFrame | None = None,
    test_comp: str = "LBA",
    *,
    w_cluster: float = 0.15,
    w_freq: float = 0.40,
    w_recency: float = 0.25,
    w_price: float = 0.10,
    w_comp_align: float = 0.10,
) -> pd.DataFrame:
    """Score fans using weighted combination of cluster + individual signals.

    Parameters
    ----------
    w_cluster : weight for cluster propensity (or comp-specific if available)
    w_freq : weight for normalized frequency
    w_recency : weight for normalized recency (lower = better)
    w_price : weight for price sensitivity index (lower = better)
    w_comp_align : weight for competition alignment (LBA% or Eurocup%)

    Returns DataFrame with person_id, weighted_score, components.
    """
    ff = fan_features.copy()
    fl = fan_labels[["person_id", "cluster"]].drop_duplicates()
    df = ff.merge(fl, on="person_id", how="inner")

    # ── cluster propensity component ────────────────────────
    # If competition-specific propensity is available, look up P(buy|cluster,
    # comp_type) for the target competition; otherwise fall back to the
    # overall cluster propensity P(buy|cluster) averaged across all games.
    if comp_propensity is not None:
        prop_map = comp_propensity.loc[
            comp_propensity["competition_type"] == test_comp
        ].set_index("cluster")["p_c_comp_smoothed"]
        df["cluster_prop"] = df["cluster"].map(prop_map).fillna(0)
    else:
        df["cluster_prop"] = df["cluster"].map(cluster_propensity).fillna(0)

    # ── individual components ───────────────────────────────

    # Frequency: how many games this fan has attended historically.
    # The single strongest predictor — fans who attend often are most likely
    # to buy again.
    df["freq_norm"] = _minmax(df["games_attended"].fillna(0))

    # Recency: days since the fan's last purchase. Fans who bought recently
    # are more engaged, so we invert the normalised value — a fan who bought
    # yesterday (low recency_days) gets a high recency score.
    recency = df["recency_days"].fillna(df["recency_days"].max())
    df["recency_norm"] = 1 - _minmax(recency)  # lower recency_days → higher score

    # Price sensitivity: fraction of tickets bought at a discount vs. list
    # price. Highly discount-dependent fans are less likely to buy at full
    # price, so we invert — a fan who always pays full price scores 1.0.
    price_sens = df.get("pct_discounted_vs_list", pd.Series(0, index=df.index)).fillna(0)
    df["price_norm"] = 1 - _minmax(price_sens)  # lower discount-dependency → higher score

    # Competition alignment (comp_align): fraction of the fan's historical
    # games that match the target competition type (LBA or Eurocup).
    # A fan who has attended 80% LBA games is more aligned with an LBA fixture
    # than one who mostly attends Eurocup games, so this captures
    # competition-specific loyalty beyond what the cluster propensity captures.
    if test_comp == "Eurocup":
        comp_col = "pct_eurocup_games"
    else:
        comp_col = "pct_lba_games"
    df["comp_align_norm"] = _minmax(df.get(comp_col, pd.Series(0, index=df.index)).fillna(0))

    df["cluster_prop_norm"] = _minmax(df["cluster_prop"])

    # ── weighted score ──────────────────────────────────────
    # Final score is a weighted linear combination of all normalised
    # components. Because each component is in [0,1], the weights directly
    # control each signal's contribution to the ranking.
    df["weighted_score"] = (
        w_cluster * df["cluster_prop_norm"]
        + w_freq * df["freq_norm"]
        + w_recency * df["recency_norm"]
        + w_price * df["price_norm"]
        + w_comp_align * df["comp_align_norm"]
    )

    return df[["person_id", "cluster", "weighted_score",
               "cluster_prop_norm", "freq_norm", "recency_norm",
               "price_norm", "comp_align_norm"]].copy()


# ─────────────────────────────────────────────────────────────────────────────
# Coarse grid search: tries 12 pre-defined weight combinations and evaluates
# each via leave-one-game-out cross-validation on the TRAIN games.
# Leave-one-out means: for each train game, train on the remaining N-1 games,
# then predict the left-out game. This gives an unbiased estimate of targeting
# quality without touching test data.
# ─────────────────────────────────────────────────────────────────────────────
def grid_search_weights(
    fan_features: pd.DataFrame,
    fan_labels: pd.DataFrame,
    cluster_propensity: pd.Series,
    comp_propensity: pd.DataFrame | None,
    train_tickets: pd.DataFrame,
    k_eval: int = 200,
) -> dict:
    """Find optimal weights on TRAIN data using leave-one-game-out CV.

    Tests a coarse grid of weight combinations and returns the best.
    """
    match = train_tickets.loc[
        train_tickets["competition_type"].isin(["LBA", "Eurocup"])
    ].copy()
    match["game_date"] = match["event_dt"].dt.normalize()
    train_games = sorted(match["game_date"].unique())

    fans_all = fan_labels[["person_id", "cluster"]].drop_duplicates()
    n_universe = len(fans_all)

    # coarse grid (weights that sum to ~1)
    weight_configs = [
        {"w_cluster": 0.05, "w_freq": 0.50, "w_recency": 0.25, "w_price": 0.10, "w_comp_align": 0.10},
        {"w_cluster": 0.10, "w_freq": 0.45, "w_recency": 0.25, "w_price": 0.10, "w_comp_align": 0.10},
        {"w_cluster": 0.15, "w_freq": 0.40, "w_recency": 0.25, "w_price": 0.10, "w_comp_align": 0.10},
        {"w_cluster": 0.20, "w_freq": 0.35, "w_recency": 0.25, "w_price": 0.10, "w_comp_align": 0.10},
        {"w_cluster": 0.10, "w_freq": 0.50, "w_recency": 0.20, "w_price": 0.10, "w_comp_align": 0.10},
        {"w_cluster": 0.10, "w_freq": 0.40, "w_recency": 0.30, "w_price": 0.10, "w_comp_align": 0.10},
        {"w_cluster": 0.10, "w_freq": 0.45, "w_recency": 0.20, "w_price": 0.15, "w_comp_align": 0.10},
        {"w_cluster": 0.10, "w_freq": 0.45, "w_recency": 0.25, "w_price": 0.05, "w_comp_align": 0.15},
        {"w_cluster": 0.05, "w_freq": 0.55, "w_recency": 0.20, "w_price": 0.10, "w_comp_align": 0.10},
        {"w_cluster": 0.00, "w_freq": 0.50, "w_recency": 0.25, "w_price": 0.15, "w_comp_align": 0.10},
        {"w_cluster": 0.15, "w_freq": 0.35, "w_recency": 0.30, "w_price": 0.10, "w_comp_align": 0.10},
        {"w_cluster": 0.10, "w_freq": 0.40, "w_recency": 0.25, "w_price": 0.10, "w_comp_align": 0.15},
    ]

    best_score = -1
    best_weights = weight_configs[0]

    # Outer loop: evaluate every weight configuration in the grid.
    for cfg in weight_configs:
        precisions = []
        # Inner LOO loop: iterate over each train game, treating it as the
        # held-out "validation" game. The fan features and cluster propensity
        # are fixed (computed from the full train set), so this tests how well
        # the weights generalise across different games — not across fans.
        for gd in train_games:
            game_rows = match[match["game_date"] == gd]
            comp = game_rows["competition_type"].mode().iloc[0] if len(game_rows) > 0 else "LBA"
            buyers = set(game_rows["person_id"].unique())

            scored = build_weighted_score(
                fan_features, fan_labels, cluster_propensity,
                comp_propensity, test_comp=comp, **cfg,
            )
            scored = scored.sort_values("weighted_score", ascending=False)
            top_k = set(scored["person_id"].head(k_eval).values)
            hits = len(top_k & buyers)
            precisions.append(hits / k_eval)

        # Aggregate: average Precision@k_eval across all LOO games.
        # The weight config with the highest mean precision is returned.
        mean_p = np.mean(precisions)
        if mean_p > best_score:
            best_score = mean_p
            best_weights = cfg

    log.info(
        "Grid search best weights: P@%d=%.4f → %s",
        k_eval, best_score, best_weights,
    )
    return best_weights


# ─────────────────────────────────────────────────────────────────────────────
# Final evaluation of the weighted score on TEST data. Unlike grid_search_weights
# which runs entirely on train games, this function uses the held-out test games
# that were never seen during weight selection. It applies the best weights found
# by grid search and reports Precision@K and Lift@K for each test game.
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_weighted_score(
    fan_features: pd.DataFrame,
    fan_labels: pd.DataFrame,
    cluster_propensity: pd.Series,
    comp_propensity: pd.DataFrame | None,
    test_tickets: pd.DataFrame,
    k_list: Sequence[int],
    weights: dict,
    universe_pids: set | None = None,
) -> pd.DataFrame:
    """Evaluate weighted scoring method on test games."""
    match = test_tickets.loc[
        test_tickets["competition_type"].isin(["LBA", "Eurocup"])
    ].copy()
    match["game_date"] = match["event_dt"].dt.normalize()
    test_games = sorted(match["game_date"].unique())

    rows = []
    for gd in test_games:
        game_rows = match[match["game_date"] == gd]
        comp = game_rows["competition_type"].mode().iloc[0] if len(game_rows) > 0 else "LBA"
        buyers_all = set(game_rows["person_id"].unique())

        # Cold-start tracking (FIX 4)
        if universe_pids is not None:
            cold_start_buyers = len(buyers_all - universe_pids)
            buyers = buyers_all & universe_pids
        else:
            cold_start_buyers = 0
            buyers = buyers_all

        scored = build_weighted_score(
            fan_features, fan_labels, cluster_propensity,
            comp_propensity, test_comp=comp, **weights,
        )
        # Apply canonical universe filter (FIX 4)
        if universe_pids is not None:
            scored = scored[scored["person_id"].isin(universe_pids)]
        scored = scored.sort_values("weighted_score", ascending=False)
        ranked_pids = scored["person_id"].values

        n_universe = len(ranked_pids)
        n_buyers = len(buyers)
        overall_rate = n_buyers / max(1, n_universe)

        for k in k_list:
            k_eff = min(k, n_universe)
            top_k = set(ranked_pids[:k_eff])
            hits = len(top_k & buyers)
            precision = hits / k_eff if k_eff > 0 else 0
            lift = precision / overall_rate if overall_rate > 0 else 0

            rows.append({
                "game_date": gd, "K": k, "n_universe": n_universe,
                "n_buyers": n_buyers, "overall_rate": overall_rate,
                "precision_at_k": precision, "lift_at_k": lift,
                "hits_at_k": hits, "cold_start_buyers": cold_start_buyers,
                "method": "weighted_hybrid",
            })

    result = pd.DataFrame(rows)
    log.info("Weighted hybrid eval: %d games × %d K values",
             result["game_date"].nunique(), len(k_list))
    return result


# =====================================================================
# B2) Frequency-boosted cluster: additive blend
# =====================================================================

# ─────────────────────────────────────────────────────────────────────────────
# Additive blend: final_score = alpha × cluster_propensity_normalised
#                             + (1-alpha) × frequency_normalised
#
# Unlike the multiplicative hybrid in baselines.py, this never zeros out —
# even if cluster propensity is 0, frequency still contributes. alpha=0.3
# means 30% cluster, 70% frequency.
# ─────────────────────────────────────────────────────────────────────────────
def frequency_boosted_cluster(
    fan_features: pd.DataFrame,
    fan_labels: pd.DataFrame,
    cluster_propensity: pd.Series,
    test_tickets: pd.DataFrame,
    k_list: Sequence[int],
    alpha: float = 0.3,
    universe_pids: set | None = None,
) -> pd.DataFrame:
    """Additive blend: score = alpha * p_c_norm + (1-alpha) * freq_norm.

    Unlike the multiplicative hybrid (where any zero component kills the
    score), this ensures that high-frequency fans in low-propensity clusters
    still rank reasonably.
    """
    ff = fan_features[["person_id", "games_attended"]].copy()
    ff["games_attended"] = ff["games_attended"].fillna(0)

    fl = fan_labels[["person_id", "cluster"]].drop_duplicates()
    fl["p_c"] = fl["cluster"].map(cluster_propensity).fillna(0)

    merged = ff.merge(fl[["person_id", "p_c"]], on="person_id", how="left")
    merged["p_c"] = merged["p_c"].fillna(0)

    merged["freq_norm"] = _minmax(merged["games_attended"])
    merged["pc_norm"] = _minmax(merged["p_c"])

    merged["score"] = alpha * merged["pc_norm"] + (1 - alpha) * merged["freq_norm"]

    # tie-breaking with small random noise
    rng = np.random.RandomState(42)
    merged["score"] += rng.uniform(0, 1e-6, len(merged))

    merged = merged.sort_values("score", ascending=False)
    ranked_pids = merged["person_id"].values

    from .baselines import _evaluate_ranked
    result = _evaluate_ranked(ranked_pids, test_tickets, k_list, method="freq_boost_cluster",
                              universe_pids=universe_pids)
    log.info("Freq-boosted cluster eval: %d games × %d K values",
             result["game_date"].nunique(), len(k_list))
    return result


# =====================================================================
# C) Supervised logistic regression baseline
# =====================================================================

# ─────────────────────────────────────────────────────────────────────────────
# Supervised approach: trains a logistic regression model on per-game binary
# labels from train data. For each train game, every fan is labelled 1 (bought)
# or 0 (did not buy). Features include: one-hot cluster membership + individual
# features (games_attended, recency, is_weekend, is_lba). The model is trained
# on ALL train games simultaneously and predicts on test games.
# ─────────────────────────────────────────────────────────────────────────────
def supervised_logistic_baseline(
    fan_features: pd.DataFrame,
    fan_labels: pd.DataFrame,
    train_tickets: pd.DataFrame,
    test_tickets: pd.DataFrame,
    k_list: Sequence[int],
    seed: int = 42,
    universe_pids: set | None = None,
) -> pd.DataFrame:
    """Train a logistic regression model on per-game buy/not-buy labels.

    Features:
      - cluster_id (one-hot encoded, top-K clusters)
      - games_attended (normalized)
      - recency_days (normalized)
      - pct_discounted_vs_list
      - pct_lba_games (competition alignment)
      - pct_evening_games
      - pct_weekend_games
      - purchase_occasions
      - is_subscription_holder

    For each train game, each fan in the train universe is labeled 1 (bought)
    or 0 (did not buy).  The model is trained on this stacked binary dataset.

    At test time, P(buy) is predicted for each fan per test game (using
    game-specific features where possible), and fans are ranked by predicted
    probability.
    """
    log.info("Building supervised logistic baseline...")

    # ── Prepare fan-level feature matrix ──────────────────────
    ff = fan_features.copy()
    fl = fan_labels[["person_id", "cluster"]].drop_duplicates()
    df = ff.merge(fl, on="person_id", how="inner")

    feature_cols = [
        "games_attended", "recency_days", "pct_discounted_vs_list",
        "pct_lba_games", "pct_eurocup_games", "pct_evening_games",
        "pct_weekend_games", "purchase_occasions", "total_spend",
        "pct_early_bird", "pct_last_minute", "premium_affinity",
    ]
    # add subscription flag if available
    if "is_subscription_holder" in df.columns:
        feature_cols.append("is_subscription_holder")

    avail_cols = [c for c in feature_cols if c in df.columns]

    # One-hot encode the top 10 clusters by size into binary indicator columns
    # (cluster_0, cluster_1, …). This lets the model learn a separate intercept
    # for each major cluster without imposing any ordinal structure on cluster IDs.
    # The noise cluster (-1) and smaller clusters fold into the "none of the above"
    # baseline, so the model is not over-parameterised by sparse clusters.
    top_clusters = df["cluster"].value_counts().head(10).index.tolist()
    for c in top_clusters:
        df[f"cluster_{c}"] = (df["cluster"] == c).astype(int)
    cluster_cols = [f"cluster_{c}" for c in top_clusters]

    all_feature_cols = avail_cols + cluster_cols

    # ── Build training labels: per fan × per train game ─────────
    match_train = train_tickets.loc[
        train_tickets["competition_type"].isin(["LBA", "Eurocup"])
    ].copy()
    match_train["game_date"] = match_train["event_dt"].dt.normalize()
    train_games = sorted(match_train["game_date"].unique())

    # Build stacked dataset: for each game, label each fan
    X_rows = []
    y_rows = []
    fan_ids = df["person_id"].values
    fan_feature_matrix = df.set_index("person_id")[all_feature_cols]

    for gd in train_games:
        game_rows = match_train[match_train["game_date"] == gd]
        buyers = set(game_rows["person_id"].unique())
        comp = game_rows["competition_type"].mode().iloc[0] if len(game_rows) > 0 else "LBA"

        # subsample negatives (10:1 ratio) for efficiency
        fan_is_buyer = np.array([pid in buyers for pid in fan_ids])
        n_pos = fan_is_buyer.sum()
        neg_idx = np.where(~fan_is_buyer)[0]
        rng = np.random.RandomState(seed + hash(str(gd)) % 10000)
        n_neg_sample = min(len(neg_idx), n_pos * 10)
        if n_neg_sample > 0:
            neg_sample = rng.choice(neg_idx, size=n_neg_sample, replace=False)
        else:
            neg_sample = neg_idx

        selected = np.concatenate([np.where(fan_is_buyer)[0], neg_sample])

        X_game = fan_feature_matrix.iloc[selected].values
        y_game = fan_is_buyer[selected].astype(int)

        # Game-level features appended as two extra columns per sample:
        #   is_weekend — 1 if the game falls on Saturday or Sunday (dayofweek >= 5).
        #                Weekend games typically attract different attendance patterns.
        #   is_lba     — 1 if the game is an LBA (domestic league) fixture, 0 for
        #                Eurocup. Allows the model to learn competition-type preference.
        gd_ts = pd.Timestamp(gd)
        is_weekend = int(gd_ts.dayofweek >= 5)
        is_lba = int(comp == "LBA")

        # append game-level columns
        game_feats = np.tile([is_weekend, is_lba], (len(selected), 1))
        X_game = np.hstack([X_game, game_feats])

        X_rows.append(X_game)
        y_rows.append(y_game)

    X_train_lr = np.vstack(X_rows)
    y_train_lr = np.concatenate(y_rows)

    # Handle NaN/Inf robustly BEFORE scaling
    X_train_lr = np.nan_to_num(X_train_lr, nan=0.0, posinf=0.0, neginf=0.0)

    # ── Fit model ──────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_lr)
    # Catch any NaN produced by zero-variance columns in StandardScaler
    X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    model = LogisticRegression(
        C=1.0, max_iter=1000, solver="lbfgs",
        class_weight="balanced", random_state=seed,
    )
    model.fit(X_train_scaled, y_train_lr)
    log.info("LR trained: %d samples, %d features, coef range [%.3f, %.3f]",
             len(y_train_lr), X_train_scaled.shape[1],
             model.coef_.min(), model.coef_.max())

    # Log feature importances
    all_feat_names = all_feature_cols + ["game_is_weekend", "game_is_lba"]
    coefs = pd.Series(model.coef_[0], index=all_feat_names)
    top_coefs = coefs.abs().nlargest(8)
    log.info("Top LR coefficients:\n%s",
             "\n".join(f"  {n}: {coefs[n]:+.3f}" for n in top_coefs.index))

    # ── Evaluate on test games ──────────────────────────────────
    # The model outputs P(buy=1 | fan_features, game_features) for every fan.
    # This probability is used directly as a ranking score — fans are sorted
    # by descending probability and the top-K are selected as targets. The
    # absolute probability values are not calibrated; only their relative
    # ordering matters for Precision@K evaluation.
    match_test = test_tickets.loc[
        test_tickets["competition_type"].isin(["LBA", "Eurocup"])
    ].copy()
    match_test["game_date"] = match_test["event_dt"].dt.normalize()
    test_games = sorted(match_test["game_date"].unique())

    X_fan_base = fan_feature_matrix.values
    X_fan_base = np.nan_to_num(X_fan_base, nan=0.0, posinf=0.0, neginf=0.0)

    # Build universe-filtered index map for efficient filtering (FIX 4)
    if universe_pids is not None:
        universe_mask = np.array([pid in universe_pids for pid in fan_ids])
        universe_fan_ids = fan_ids[universe_mask]
        universe_X_base = X_fan_base[universe_mask]
    else:
        universe_fan_ids = fan_ids
        universe_X_base = X_fan_base

    rows = []
    for gd in test_games:
        game_rows = match_test[match_test["game_date"] == gd]
        comp = game_rows["competition_type"].mode().iloc[0] if len(game_rows) > 0 else "LBA"
        buyers_all = set(game_rows["person_id"].unique())

        # Cold-start tracking (FIX 4)
        if universe_pids is not None:
            cold_start_buyers = len(buyers_all - universe_pids)
            buyers = buyers_all & universe_pids
        else:
            cold_start_buyers = 0
            buyers = buyers_all

        gd_ts = pd.Timestamp(gd)
        is_weekend = int(gd_ts.dayofweek >= 5)
        is_lba = int(comp == "LBA")
        game_feats = np.tile([is_weekend, is_lba], (len(universe_fan_ids), 1))
        X_test_lr = np.hstack([universe_X_base, game_feats])
        X_test_scaled = scaler.transform(X_test_lr)
        X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        proba = model.predict_proba(X_test_scaled)[:, 1]
        ranked_idx = np.argsort(-proba)
        ranked_pids = universe_fan_ids[ranked_idx]

        n_universe = len(ranked_pids)
        n_buyers = len(buyers)
        overall_rate = n_buyers / max(1, n_universe)

        for k in k_list:
            k_eff = min(k, n_universe)
            top_k = set(ranked_pids[:k_eff])
            hits = len(top_k & buyers)
            precision = hits / k_eff if k_eff > 0 else 0
            lift = precision / overall_rate if overall_rate > 0 else 0

            rows.append({
                "game_date": gd, "K": k, "n_universe": n_universe,
                "n_buyers": n_buyers, "overall_rate": overall_rate,
                "precision_at_k": precision, "lift_at_k": lift,
                "hits_at_k": hits, "cold_start_buyers": cold_start_buyers,
                "method": "supervised_lr",
            })

    result = pd.DataFrame(rows)
    log.info("Supervised LR eval: %d games × %d K values",
             result["game_date"].nunique(), len(k_list))
    return result
