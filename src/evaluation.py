"""
Evaluation: Precision@K, Lift@K, and aggregation across test games.
"""

# =============================================================================
# MODULE OVERVIEW
# =============================================================================
# This module evaluates CLUSTER-ONLY targeting.  It ranks all train fans by
# their cluster's smoothed propensity score and measures how many actual game
# buyers appear in the top K.
#
# The two key metrics are:
#
#   Precision@K  – of the top K fans we would target, what fraction actually
#                  bought a ticket to the game?
#                  Precision@K = hits_in_top_K / K
#
#   Lift@K       – how much better is our targeting than picking fans at random?
#                  Lift@K = Precision@K / (n_buyers / n_universe)
#                  A lift of 1.0 means "no better than random"; a lift of 5.0
#                  means we are five times more accurate than random selection.
#
# The cluster-only approach has a structural weakness: all fans in the same
# cluster receive an identical propensity score.  Ties are broken randomly,
# which means within-cluster ordering carries no information.  The advanced
# scoring methods in advanced_scoring.py add per-fan signals (frequency,
# recency, etc.) to break these ties more meaningfully.
# =============================================================================
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def evaluate_targeting(
    test_tickets: pd.DataFrame,
    fan_labels: pd.DataFrame,
    cluster_propensity: pd.Series,
    k_list: Sequence[int] = (200, 500, 1000),
    universe_pids: set | None = None,
) -> pd.DataFrame:
    """Evaluate cluster-based targeting on each TEST game.

    For each test game *g*:
      1. Universe U_g = all fans in ``fan_labels`` (train universe),
         optionally filtered to ``universe_pids`` for a canonical comparison.
      2. True buyers B_g = fans who purchased game *g* (from test_tickets)
         **intersected with U_g** (we can only target known fans).
      3. Rank fans by their cluster's smoothed propensity.
      4. Select top K fans → compute Precision@K and Lift@K.

    Parameters
    ----------
    test_tickets : DataFrame
        Ticket-level test data (must have person_id, event_dt, competition_type).
    fan_labels : DataFrame
        ``[person_id, cluster]`` – cluster assignments from training.
    cluster_propensity : Series
        Indexed by cluster, values = smoothed purchase propensity from train.
    k_list : sequence of int
        Values of K to evaluate.
    universe_pids : set or None
        Canonical evaluation universe.  When provided, ``fan_labels`` is
        filtered to these person_ids before ranking so all methods operate
        on the same fan pool.  Buyers outside the universe are tracked as
        ``cold_start_buyers`` but excluded from P@K computation.

    Returns
    -------
    DataFrame with columns:
        game_date, K, n_universe, n_buyers, overall_rate,
        precision_at_k, lift_at_k, cold_start_buyers, method
    """
    match_tickets = test_tickets.loc[
        test_tickets["competition_type"].isin(["LBA", "Eurocup"])
    ].copy()
    match_tickets["game_date"] = match_tickets["event_dt"].dt.normalize()

    test_games = sorted(match_tickets["game_date"].unique())
    log.info("Evaluating %d test games with K = %s", len(test_games), list(k_list))

    fans = fan_labels[["person_id", "cluster"]].drop_duplicates().copy()

    # Apply canonical evaluation universe if provided (FIX 4)
    if universe_pids is not None:
        fans = fans[fans["person_id"].isin(universe_pids)].copy()

    # Attach propensity – the Series already contains an entry for the
    # noise cluster (-1) if it exists, so a simple .map() suffices.
    fans["p_c"] = fans["cluster"].map(cluster_propensity).fillna(0)

    # Rank fans: highest propensity first, with random tie-breaking.
    # Tie-breaking matters because cluster-only scoring assigns the SAME score
    # to every fan in a given cluster.  Without it, pandas sort_values would
    # break ties by row insertion order, which is an arbitrary artefact of data
    # loading.  A fixed-seed random tiebreak gives a reproducible but unbiased
    # ordering within each cluster, preventing any single cluster from being
    # systematically pushed to the top or bottom of a tie group.
    rng = np.random.RandomState(42)
    fans["_tie"] = rng.rand(len(fans))
    fans = fans.sort_values(["p_c", "_tie"], ascending=[False, True])
    ranked_pids = fans["person_id"].values
    ranked_pid_set = set(ranked_pids)

    rows = []
    for gd in test_games:
        buyers_all = set(
            match_tickets.loc[match_tickets["game_date"] == gd, "person_id"].unique()
        )
        # Cold-start buyers are fans who attended the test game but have no
        # prior purchase history in the train split.  They cannot be ranked by
        # any model and are counted separately so the analyst understands what
        # fraction of real buyers are structurally unrecoverable.
        if universe_pids is not None:
            cold_start_buyers = len(buyers_all - universe_pids)
            buyers = buyers_all & universe_pids
        else:
            cold_start_buyers = 0
            buyers = buyers_all

        n_universe = len(ranked_pids)
        n_buyers = len(buyers)
        # overall_rate is the probability that a randomly chosen fan from the
        # universe bought this particular game – it is the random-baseline rate
        # used as the denominator in the lift formula.
        overall_rate = n_buyers / max(1, n_universe)

        for k in k_list:
            k_eff = min(k, n_universe)
            top_k = set(ranked_pids[:k_eff])
            hits = len(top_k & buyers)
            precision = hits / k_eff if k_eff > 0 else 0
            # Lift@K = Precision@K / overall_rate
            # How many times more accurate is the top-K list compared with
            # selecting K fans uniformly at random from the universe.
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
                "method": "cluster",
            })

    df = pd.DataFrame(rows)
    log.info("Evaluation complete: %d rows", len(df))
    return df


def summarise_results(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate evaluation results across games per (method, K).

    Takes the per-game, per-K rows produced by evaluate_targeting() (or the
    equivalent baseline functions) and collapses them to one summary row per
    (method, K) pair.  The result shows mean and standard deviation of
    Precision@K and Lift@K across all test games, giving both the average
    performance and how consistent that performance is game-to-game.
    """
    agg_spec = {
        "n_games": ("game_date", "nunique"),
        "mean_precision": ("precision_at_k", "mean"),
        "std_precision": ("precision_at_k", "std"),
        "mean_lift": ("lift_at_k", "mean"),
        "std_lift": ("lift_at_k", "std"),
        "mean_hits": ("hits_at_k", "mean"),
        "mean_overall_rate": ("overall_rate", "mean"),
    }
    if "cold_start_buyers" in results.columns:
        agg_spec["mean_cold_start_buyers"] = ("cold_start_buyers", "mean")
    agg = results.groupby(["method", "K"]).agg(**agg_spec).reset_index()
    return agg
