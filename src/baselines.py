"""
Baseline and hybrid targeting strategies for comparison:
  1. Random
  2. Recency  (rank by most recent purchase)
  3. Frequency (rank by games attended)
  4. Cluster × Frequency hybrid  (combined score)
"""

# =============================================================================
# MODULE OVERVIEW
# =============================================================================
# All four methods rank the SAME pool of fans (the train universe) and evaluate
# against test game buyers.  The comparison is therefore fair: every method
# sees the same candidate fans and the same ground-truth buyers.
#
# The shared _evaluate_ranked() helper handles the evaluation loop so each
# method only needs to produce a single ranked list of person_ids.  Each
# method's entire job is to sort the fan universe by its own criterion and
# pass the result to _evaluate_ranked().
#
# Evaluation metric:
#   Precision@K = (# of top-K recommended fans who actually bought the game)
#                 / K
#
# This measures how useful the top-K targeted list would be in practice:
# if P@K = 0.10 then 10% of the fans we contacted would have bought.
# Lift@K compares that figure against the random (baseline) purchase rate.
# =============================================================================
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _evaluate_ranked(
    ranked_pids: np.ndarray,
    test_tickets: pd.DataFrame,
    k_list: Sequence[int],
    method: str,
    universe_pids: set | None = None,
) -> pd.DataFrame:
    """Shared logic: given a ranked list of person_ids, evaluate on test games.

    Parameters
    ----------
    universe_pids : set or None
        Canonical evaluation universe (FIX 4).  When provided, ``ranked_pids``
        is filtered to this set and buyers outside the universe are tracked as
        ``cold_start_buyers`` but excluded from P@K computation.
    """
    # Apply canonical universe filter: keep only fans that appear in the shared
    # evaluation universe.  This guarantees all methods rank the identical pool
    # of fans so performance differences reflect the ranking criterion, not
    # differences in who was included.
    if universe_pids is not None:
        ranked_pids = np.array([p for p in ranked_pids if p in universe_pids])

    match = test_tickets.loc[
        test_tickets["competition_type"].isin(["LBA", "Eurocup"])
    ].copy()
    match["game_date"] = match["event_dt"].dt.normalize()
    test_games = sorted(match["game_date"].unique())

    rows = []
    n_universe = len(ranked_pids)

    for gd in test_games:
        buyers_all = set(match.loc[match["game_date"] == gd, "person_id"].unique())
        # Cold-start buyers are fans who purchased the test game but were never
        # seen in the train split (no ticket history → not in universe_pids).
        # We count them for reporting purposes but exclude them from P@K because
        # no method could have ranked them – it would be unfair to penalise for
        # failing to predict someone the model has never observed.
        if universe_pids is not None:
            cold_start_buyers = len(buyers_all - universe_pids)
            buyers = buyers_all & universe_pids
        else:
            cold_start_buyers = 0
            buyers = buyers_all

        n_buyers = len(buyers)
        # overall_rate = the purchase rate if we picked fans at random from the
        # universe; this is the denominator for the Lift calculation below.
        overall_rate = n_buyers / max(1, n_universe)

        for k in k_list:
            k_eff = min(k, n_universe)
            top_k = set(ranked_pids[:k_eff])
            hits = len(top_k & buyers)
            precision = hits / k_eff if k_eff > 0 else 0
            # Lift@K = Precision@K / random_purchase_rate
            # A lift of 2.0 means the top-K list is twice as accurate as random.
            lift = precision / overall_rate if overall_rate > 0 else 0

            rows.append({
                "game_date": gd,
                "K": k,
                "n_universe": n_universe,
                "n_buyers": n_buyers,
                "overall_rate": overall_rate,
                "precision_at_k": precision,
                "lift_at_k": lift,
                "hits_at_k": hits,
                "cold_start_buyers": cold_start_buyers,
                "method": method,
            })

    return pd.DataFrame(rows)


# ── Random baseline ──────────────────────────────────────────────────

def random_baseline(
    fan_features: pd.DataFrame,
    test_tickets: pd.DataFrame,
    k_list: Sequence[int],
    n_repeats: int = 50,
    seed: int = 42,
    universe_pids: set | None = None,
) -> pd.DataFrame:
    """Average precision/lift over *n_repeats* random orderings."""
    rng = np.random.RandomState(seed)
    pids = fan_features["person_id"].values

    all_runs = []
    for i in range(n_repeats):
        shuffled = rng.permutation(pids)
        run = _evaluate_ranked(shuffled, test_tickets, k_list, method="random",
                               universe_pids=universe_pids)
        run["run"] = i
        # Averaging over 50 independent shuffles removes the sampling noise
        # inherent in any single random ordering and gives a stable expected
        # value for P@K and Lift@K under random selection.
        all_runs.append(run)

    combined = pd.concat(all_runs, ignore_index=True)

    # average across runs per game/K (cold_start_buyers is constant across runs)
    agg_spec = dict(
        n_universe=("n_universe", "first"),
        n_buyers=("n_buyers", "first"),
        overall_rate=("overall_rate", "first"),
        precision_at_k=("precision_at_k", "mean"),
        lift_at_k=("lift_at_k", "mean"),
        hits_at_k=("hits_at_k", "mean"),
    )
    if "cold_start_buyers" in combined.columns:
        agg_spec["cold_start_buyers"] = ("cold_start_buyers", "first")
    avg = combined.groupby(["game_date", "K"]).agg(**agg_spec).reset_index()
    avg["method"] = "random"

    log.info("Random baseline: %d games × %d K values (averaged over %d runs)",
             avg["game_date"].nunique(), len(k_list), n_repeats)
    return avg


# ── Recency baseline ─────────────────────────────────────────────────

def recency_baseline(
    fan_features: pd.DataFrame,
    test_tickets: pd.DataFrame,
    k_list: Sequence[int],
    universe_pids: set | None = None,
) -> pd.DataFrame:
    """Rank fans by lowest recency_days (most recent purchaser first)."""
    ff = fan_features[["person_id", "recency_days"]].copy()
    ff["recency_days"] = ff["recency_days"].fillna(ff["recency_days"].max() + 1)
    ff = ff.sort_values("recency_days", ascending=True)
    ranked = ff["person_id"].values

    result = _evaluate_ranked(ranked, test_tickets, k_list, method="recency",
                              universe_pids=universe_pids)
    log.info("Recency baseline: %d games × %d K values", result["game_date"].nunique(), len(k_list))
    return result


# ── Frequency baseline ───────────────────────────────────────────────

def frequency_baseline(
    fan_features: pd.DataFrame,
    test_tickets: pd.DataFrame,
    k_list: Sequence[int],
    universe_pids: set | None = None,
) -> pd.DataFrame:
    """Rank fans by highest games_attended."""
    ff = fan_features[["person_id", "games_attended"]].copy()
    ff["games_attended"] = ff["games_attended"].fillna(0)
    ff = ff.sort_values("games_attended", ascending=False)
    ranked = ff["person_id"].values

    result = _evaluate_ranked(ranked, test_tickets, k_list, method="frequency",
                              universe_pids=universe_pids)
    log.info("Frequency baseline: %d games × %d K values", result["game_date"].nunique(), len(k_list))
    return result


# ── Cluster × Frequency hybrid ───────────────────────────────────────

def cluster_frequency_hybrid(
    fan_features: pd.DataFrame,
    fan_labels: pd.DataFrame,
    cluster_propensity: pd.Series,
    test_tickets: pd.DataFrame,
    k_list: Sequence[int],
    universe_pids: set | None = None,
) -> pd.DataFrame:
    """Rank fans by combined cluster propensity × normalized frequency.

    Score = p_c_norm * freq_norm, where both are min-max normalised to [0,1].
    This combines clustering signal with purchase-history signal.
    """
    ff = fan_features[["person_id", "games_attended"]].copy()
    ff["games_attended"] = ff["games_attended"].fillna(0)

    fl = fan_labels[["person_id", "cluster"]].drop_duplicates()
    # The propensity series already contains a smoothed entry for the
    # noise cluster (-1) so no special-casing is needed.
    fl["p_c"] = fl["cluster"].map(cluster_propensity).fillna(0)

    merged = ff.merge(fl[["person_id", "p_c"]], on="person_id", how="left")
    merged["p_c"] = merged["p_c"].fillna(0)

    # Min-max normalisation scales both components to [0, 1] so that the two
    # signals (cluster propensity and games attended) contribute on equal
    # footing regardless of their original magnitudes.  Without normalisation,
    # games_attended (which can reach 30+) would dominate p_c (which sits ~0.04).
    ga_min, ga_max = merged["games_attended"].min(), merged["games_attended"].max()
    pc_min, pc_max = merged["p_c"].min(), merged["p_c"].max()

    merged["freq_norm"] = (merged["games_attended"] - ga_min) / max(1, ga_max - ga_min)
    merged["pc_norm"] = (merged["p_c"] - pc_min) / max(1e-9, pc_max - pc_min)

    # Multiplicative combination: a fan must score well on BOTH dimensions to
    # rank highly.  The downside is that if either component is exactly 0
    # (e.g. a fan who attended no games, or sits in a zero-propensity cluster)
    # the product collapses to 0 and the fan is ranked last regardless of the
    # other signal.  This is intentional for the baseline comparison; the
    # weighted_hybrid in advanced_scoring.py uses additive blending instead
    # to avoid this issue.
    merged["hybrid_score"] = merged["pc_norm"] * merged["freq_norm"]
    merged = merged.sort_values("hybrid_score", ascending=False)
    ranked = merged["person_id"].values

    result = _evaluate_ranked(ranked, test_tickets, k_list, method="cluster_x_freq",
                              universe_pids=universe_pids)
    log.info("Cluster×Freq hybrid: %d games × %d K values",
             result["game_date"].nunique(), len(k_list))
    return result
