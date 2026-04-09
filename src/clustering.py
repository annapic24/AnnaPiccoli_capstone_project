"""
HDBSCAN clustering: fit on train, predict on test.

Handles noise cluster (-1) explicitly.
"""

# =============================================================================
# MODULE OVERVIEW
# =============================================================================
# This module does two things:
#
#   (1) FITTING CLUSTERS  (fit_clusters)
#       Runs HDBSCAN on the reduced train embeddings (UMAP output) to discover
#       dense groups of fans with similar purchase behaviour.  HDBSCAN is a
#       density-based algorithm: it finds clusters automatically without
#       requiring a fixed number of groups, and explicitly labels low-density
#       "outlier" points as noise (cluster label = -1).  These noise fans are
#       not assigned to any cluster and are handled separately throughout the
#       pipeline.
#
#   (2) COMPUTING PROPENSITY  (build_cluster_propensity)
#       After clustering, we estimate for each cluster how likely its members
#       are to buy a ticket to a given game.  Raw rates are smoothed using
#       Bayesian (Beta-Binomial) shrinkage so that small clusters are pulled
#       toward the global purchase rate rather than being driven by noise.
# =============================================================================
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class ClusterResult:
    labels_train: np.ndarray        # cluster labels for train fans
    person_ids_train: np.ndarray
    labels_test: np.ndarray | None  # cluster labels for test fans (may be None)
    person_ids_test: np.ndarray | None
    n_clusters: int
    n_noise_train: int
    clusterer: object


def fit_clusters(
    X_train: np.ndarray,
    pids_train: np.ndarray,
    X_test: np.ndarray | None = None,
    pids_test: np.ndarray | None = None,
    min_cluster_size: int = 100,
    min_samples: int = 10,
    metric: str = "euclidean",
) -> ClusterResult:
    """Fit HDBSCAN on train embeddings, optionally predict test.

    Parameters
    ----------
    X_train, pids_train
        Train embeddings and corresponding person_ids.
    X_test, pids_test
        Test embeddings (optional).
    min_cluster_size, min_samples, metric
        HDBSCAN hyper-parameters.

    Returns
    -------
    ClusterResult
    """
    import hdbscan

    log.info("Fitting HDBSCAN (min_cluster_size=%d, min_samples=%d) on %d points",
             min_cluster_size, min_samples, len(X_train))

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        prediction_data=True,  # must be True to allow approximate_predict on unseen points later
    )
    labels_train = clusterer.fit_predict(X_train)

    # Labels run from 0 … (n_clusters-1); noise points carry label -1.
    # .max() + 1 gives the cluster count only when at least one real cluster exists.
    n_clusters = labels_train.max() + 1 if labels_train.max() >= 0 else 0
    # Count how many train fans ended up as noise (label == -1)
    n_noise = (labels_train == -1).sum()
    log.info("HDBSCAN: %d clusters, %d noise (%.1f%%)",
             n_clusters, n_noise, 100 * n_noise / max(1, len(labels_train)))

    labels_test = None
    if X_test is not None and len(X_test) > 0:
        try:
            # approximate_predict assigns each test point to the nearest cluster
            # found during training without refitting the model.  The second
            # return value (soft-membership strengths) is discarded here.
            test_labels, _ = hdbscan.approximate_predict(clusterer, X_test)
            labels_test = test_labels
            # Track noise fraction in the test split too – high noise here can
            # indicate test fans who are behaviourally unlike anyone in train.
            n_noise_test = (test_labels == -1).sum()
            log.info("Test prediction: %d fans, %d noise (%.1f%%)",
                     len(test_labels), n_noise_test,
                     100 * n_noise_test / max(1, len(test_labels)))
        except Exception as e:
            log.warning("approximate_predict failed: %s – test fans will be unassigned", e)

    return ClusterResult(
        labels_train=labels_train,
        person_ids_train=pids_train,
        labels_test=labels_test,
        person_ids_test=pids_test,
        n_clusters=n_clusters,
        n_noise_train=n_noise,
        clusterer=clusterer,
    )


def build_cluster_propensity(
    train_tickets: pd.DataFrame,
    fan_labels: pd.DataFrame,
    *,
    exclude_noise: bool = True,
    bayesian_alpha: float | None = None,
    bayesian_beta: float | None = None,
    bayesian_strength: float = 10.0,
    min_support: int = 30,
) -> tuple[pd.DataFrame, pd.Series]:
    """Compute cluster-level purchase propensity from TRAIN data.

    For each cluster *c* and each train game *g*:

        p_c_raw(g) = #fans_in_c_who_bought_game_g / #fans_in_c

    The full Cartesian grid (all clusters × all train games) is built so
    that cluster/game pairs with zero buyers are correctly counted as 0.

    Two smoothed propensity variants are returned:

    * **``p_c_overall``** – plain mean of ``p_c_raw`` across all train
      games (the "MLE" estimate).
    * **``p_c_smoothed``** – Beta-Binomial posterior mean:

          p_smooth = (total_buyers + α) / (total_exposures + α + β)

      where ``total_buyers`` = sum of buyers across *all* train games and
      ``total_exposures`` = n_total × n_train_games.

      The prior is calibrated empirically (Empirical Bayes): if
      ``bayesian_alpha`` / ``bayesian_beta`` are not provided, they are
      computed from the observed global purchase rate so that the prior
      mean equals the data mean and the effective sample size equals
      ``bayesian_strength`` observations.  This correctly shrinks small
      clusters toward the actual purchase rate (~4%) rather than 0.5.

    A ``min_support`` filter is also applied: clusters with fewer than
    *min_support* fans get their smoothed propensity set to the global
    (population-level) rate so they neither dominate nor disappear.

    Parameters
    ----------
    train_tickets : DataFrame
        Ticket-level train data (must have person_id, event_dt,
        competition_type).
    fan_labels : DataFrame
        Must have columns ``[person_id, cluster]``.
    exclude_noise : bool
        If True, noise cluster (label = -1) is excluded from the
        propensity grid and gets a separate entry equal to the global
        purchase rate.
    bayesian_alpha, bayesian_beta : float or None
        Prior pseudo-counts for the Beta-Binomial smoothing.  When None
        (default), both are computed from the observed global purchase
        rate using ``bayesian_strength`` as the effective sample size.
        Pass explicit values to override the empirical calibration.
    bayesian_strength : float
        Effective sample size of the empirical Bayes prior.  A value of
        10 means the prior counts as 10 observations.  Ignored when
        explicit ``bayesian_alpha`` / ``bayesian_beta`` are provided.
    min_support : int
        Clusters with fewer than this many fans are set to the global
        purchase rate in the smoothed column.

    Returns
    -------
    (game_level_propensity, overall_propensity)
        game_level : DataFrame[cluster, game_date, n_buyers, n_total,
                               p_c_raw]
        overall    : Series indexed by cluster with **smoothed**
                     propensity (``p_c_smoothed``).
    """
    # ── match-only tickets ───────────────────────────────────────────────────
    # Only LBA (Serie A) and Eurocup games are used to compute propensity.
    # Pack, Non-partita, and Abbonamento rows are excluded here because they
    # are not individual game-day purchases and would distort the per-game rate.
    tickets = train_tickets.loc[
        train_tickets["competition_type"].isin(["LBA", "Eurocup"])
    ].copy()
    tickets["game_date"] = tickets["event_dt"].dt.normalize()
    train_game_dates = sorted(tickets["game_date"].unique())
    n_train_games = len(train_game_dates)

    # ── fan universe ─────────────────────────────────────────────────────────
    # One row per fan with their assigned cluster label.  The universe here is
    # every fan that appeared in the train split and was assigned a cluster.
    fans = fan_labels[["person_id", "cluster"]].drop_duplicates()

    # Optionally separate noise
    noise_pids = set()
    if exclude_noise and -1 in fans["cluster"].values:
        # Pull noise fans out of the main universe so they don't dilute cluster
        # propensity estimates.  They receive a separate entry at the end.
        noise_pids = set(fans.loc[fans["cluster"] == -1, "person_id"])
        fans = fans.loc[fans["cluster"] != -1].copy()

    clusters = sorted(fans["cluster"].unique())
    # n_total is the denominator for per-cluster purchase rates
    n_per_cluster = fans.groupby("cluster").size().rename("n_total")

    # ── full Cartesian grid: every cluster × every train game ────────────────
    # Building the full cross-product ensures that cluster/game combinations
    # with zero buyers produce a row with n_buyers=0 rather than being silently
    # absent from the data.  Missing rows would overestimate propensity.
    from itertools import product
    grid = pd.DataFrame(
        list(product(train_game_dates, clusters)),
        columns=["game_date", "cluster"],
    )

    # ── observed buyers per cluster per game ─────────────────────────────────
    # De-duplicate (person_id, game_date) so fans who bought multiple tickets
    # to the same game are counted only once (we care about attendance, not
    # ticket quantity).
    buyers = tickets[["person_id", "game_date"]].drop_duplicates()
    buyers = buyers.merge(fans, on="person_id", how="inner")
    observed = (
        buyers
        .groupby(["game_date", "cluster"])
        .size()
        .rename("n_buyers")
        .reset_index()
    )

    # Left-join observed onto the full Cartesian grid so missing pairs become 0
    game_cluster = grid.merge(observed, on=["game_date", "cluster"], how="left")
    game_cluster["n_buyers"] = game_cluster["n_buyers"].fillna(0).astype(int)
    game_cluster = game_cluster.merge(n_per_cluster, on="cluster", how="left")
    # Raw (MLE) propensity per cluster per game: fraction of cluster members who bought
    game_cluster["p_c_raw"] = game_cluster["n_buyers"] / game_cluster["n_total"]

    # ── aggregate: overall MLE propensity ────────────────────────────────────
    # Sum buyers and exposures across all train games to get a stable aggregate
    # estimate for each cluster before applying Bayesian smoothing.
    agg = game_cluster.groupby("cluster").agg(
        total_buyers=("n_buyers", "sum"),
        n_total=("n_total", "first"),          # constant within cluster
    ).reset_index()
    # total_exposures = number of (fan, game) pairs the cluster was "exposed" to
    agg["total_exposures"] = agg["n_total"] * n_train_games
    agg["p_c_overall"] = agg["total_buyers"] / agg["total_exposures"]

    # ── Compute global rate for empirical Bayes calibration ──────────────────
    # The global rate is the population-wide fraction of (fan, game) pairs that
    # resulted in a purchase.  It anchors the Bayesian prior so clusters with
    # little data are shrunk toward the true average (~4%) not toward 0.5.
    global_rate = agg["total_buyers"].sum() / max(1, agg["total_exposures"].sum())

    # Calibrate prior if not explicitly specified.
    # Beta(alpha, beta) prior where mean = global_rate and
    # effective sample size = bayesian_strength.
    if bayesian_alpha is None:
        # alpha = prior "successes" (purchases): encodes how many times we
        # expect a fan to buy, proportional to the global purchase rate
        bayesian_alpha = global_rate * bayesian_strength
    if bayesian_beta is None:
        # beta = prior "failures" (non-purchases): encodes the complementary
        # probability, keeping the prior mean = global_rate
        bayesian_beta = (1.0 - global_rate) * bayesian_strength

    log.info(
        "Empirical Bayes prior: global_rate=%.4f, strength=%.1f "
        "→ alpha=%.4f, beta=%.4f  (prior mean=%.4f)",
        global_rate, bayesian_strength, bayesian_alpha, bayesian_beta,
        bayesian_alpha / max(1e-9, bayesian_alpha + bayesian_beta),
    )

    # ── Bayesian smoothing (Beta-Binomial posterior mean) ────────────────────
    # Empirical Bayes formula: posterior mean of a Beta-Binomial model.
    # Adding alpha/beta pseudo-counts to observed buyers/non-buyers shrinks
    # the estimate toward the global rate, with strength controlled by
    # bayesian_strength.  The effect is strongest for small clusters where
    # the observed data is noisiest.
    #   p_smoothed = (observed_buyers + alpha) / (total_exposures + alpha + beta)
    agg["p_c_smoothed"] = (
        (agg["total_buyers"] + bayesian_alpha)
        / (agg["total_exposures"] + bayesian_alpha + bayesian_beta)
    )

    # ── min-support filter ───────────────────────────────────────────────────
    # Even after Bayesian smoothing, very small clusters (< min_support fans)
    # may have unreliable estimates because they contributed almost nothing to
    # the likelihood.  Override their smoothed score with the global rate so
    # they are treated as "unknown" rather than spuriously high or low.
    small_mask = agg["n_total"] < min_support
    n_small = small_mask.sum()
    if n_small > 0:
        log.info(
            "Min-support filter: %d clusters with < %d fans → "
            "smoothed propensity set to global rate %.4f",
            n_small, min_support, global_rate,
        )
        agg.loc[small_mask, "p_c_smoothed"] = global_rate

    # ── noise cluster entry ──────────────────────────────────────────────────
    # Noise fans (cluster = -1) were excluded from propensity estimation above.
    # We still need a propensity entry for them so the ranking step can handle
    # any fan regardless of cluster label.  We deliberately set their propensity
    # to 0.5× the global rate: noise fans are outliers by definition (they didn't
    # fit any dense cluster), so we penalise them below the population average
    # without discarding them entirely.
    if noise_pids:
        noise_row = pd.DataFrame([{
            "cluster": -1,
            "total_buyers": 0,     # not tracked per-game (noise excluded from grid)
            "n_total": len(noise_pids),
            "total_exposures": len(noise_pids) * n_train_games,
            "p_c_overall": global_rate,
            "p_c_smoothed": global_rate * 0.5,   # penalised: below global rate
        }])
        agg = pd.concat([agg, noise_row], ignore_index=True)

    # ── build output Series ──────────────────────────────────────
    overall = agg.set_index("cluster")["p_c_smoothed"]
    overall.name = "p_c_smoothed"

    log.info(
        "Propensity: %d clusters × %d train games  |  "
        "global rate=%.4f  |  smoothed range=[%.4f, %.4f]",
        len(clusters), n_train_games, global_rate,
        overall.min(), overall.max(),
    )
    log.info(
        "Top-5 clusters by smoothed propensity:\n%s",
        agg.nlargest(5, "p_c_smoothed")[
            ["cluster", "n_total", "total_buyers", "p_c_overall", "p_c_smoothed"]
        ].to_string(index=False),
    )

    return game_cluster, overall
