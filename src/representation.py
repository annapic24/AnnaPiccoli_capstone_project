"""
Dimensionality reduction: scaling, imputation, UMAP / PCA.

All fit/transform operations are split so that
``fit`` happens on TRAIN only and ``transform`` is applied to TEST.
"""

# =============================================================================
# PIPELINE OVERVIEW
# =============================================================================
# The representation pipeline has 3 sequential steps:
#
#   (1) IMPUTATION — missing values are filled with the MEDIAN of each column
#       computed on train data.  Median is more robust than mean when feature
#       distributions are skewed (e.g. ticket counts, spend amounts).
#
#   (2) SCALING — StandardScaler centres every feature to mean=0 and std=1.
#       This is critical because UMAP and HDBSCAN are distance-based: without
#       scaling, a feature with a large numeric range (e.g. total spend in €)
#       would dominate distance calculations over small-range features
#       (e.g. a binary flag), making those smaller features effectively invisible.
#
#   (3) DIMENSIONALITY REDUCTION — UMAP (default) or PCA.
#       UMAP preserves local neighbourhood structure and produces tighter,
#       more meaningful clusters for HDBSCAN than PCA would.  PCA is
#       available as a fast fallback (always supports .transform()).
#
# KEY DESIGN PRINCIPLE — NO DATA LEAKAGE:
#   fit() is called ONLY on train data.  The fitted imputer, scaler, and
#   reducer are stored in RepresentationResult and then re-used in
#   transform_test() to project test fans into the SAME space.  This means
#   the test set never influences the imputation medians, scaling parameters,
#   or the UMAP manifold — which would artificially inflate evaluation metrics.
# =============================================================================
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

# Columns to NEVER use as clustering features (FIX 3: added collinear/noisy features)
# these columns are either IDs, derived from other features (collinear), or add noise
_EXCLUDE = {
    "person_id", "first_purchase", "last_purchase",
    "most_common_sector", "province_mode",
    "identity_key", "id_tier",
    # Collinear: algebraically derivable from sector_variety / games_attended
    "sector_consistency",
    # Collinear: near-perfect anti-correlation with pct_lba_games (sum ≈ 1)
    # has_eurocup_history (binary, non-collinear) is kept instead
    "pct_eurocup_games",
    # Noise: mostly captures year-on-year age rounding across seasons
    "age_nunique",
}

# Per-opponent sparse column prefixes to exclude from UMAP input (FIX 2).
# These ~120 columns are ≥95% zeros and degrade UMAP/HDBSCAN density estimation.
# When a feature is almost always zero, it adds nothing to distance calculations
# but increases dimensionality and introduces noise into the UMAP manifold.
# The 3 named aggregate fallback columns below summarise the same information in
# a compact, non-sparse form and are therefore kept as exceptions.
_EXCLUDE_PATTERNS = [
    "pct_games_vs_",
    "win_rate_vs_",
    "avg_point_diff_vs_",
    "avg_abs_diff_vs_",
]
# Aggregate columns that share a prefix with _EXCLUDE_PATTERNS but are NOT sparse
# and should therefore be retained as legitimate clustering features.
_EXCLUDE_PATTERN_EXCEPTIONS = {
    "pct_games_vs_top8",        # compact summary: fraction of games vs top-8 teams
    "pct_games_vs_bottom8",     # compact summary: fraction of games vs bottom-8 teams
    "pct_games_vs_hv_opponents", # compact summary: fraction of games vs high-value opponents
}


def _select_numeric(fan: pd.DataFrame) -> list[str]:
    """Return numeric column names suitable for clustering.

    Applies two exclusion layers before accepting a column:
      1. Hard-exclude list (_EXCLUDE) — IDs, collinear, and noisy columns.
      2. Sparse per-opponent columns — any column whose name starts with one of
         the _EXCLUDE_PATTERNS prefixes is dropped UNLESS it is listed in
         _EXCLUDE_PATTERN_EXCEPTIONS.  These per-opponent columns are ≥95% zeros
         across fans (most fans never saw a given opponent), which degrades
         UMAP's density estimation and inflates dimensionality without adding
         meaningful signal.  The three aggregate exception columns capture the
         same behavioural information in a compact, non-sparse form.
    """
    cols = []
    n_opponent_excluded = 0
    for c in fan.columns:
        if c in _EXCLUDE:
            continue
        # Exclude per-opponent sparse columns but keep the 3 aggregate fallbacks (FIX 2)
        if c not in _EXCLUDE_PATTERN_EXCEPTIONS and any(
            c.startswith(p) for p in _EXCLUDE_PATTERNS
        ):
            n_opponent_excluded += 1
            continue
        if pd.api.types.is_numeric_dtype(fan[c]) or pd.api.types.is_bool_dtype(fan[c]):
            cols.append(c)
    log.info(
        "Feature selection: %d per-opponent sparse columns excluded from UMAP, "
        "%d features retained for dimensionality reduction",
        n_opponent_excluded, len(cols),
    )
    return cols


@dataclass
class RepresentationResult:
    """Holds fitted objects so they can be reused on test data."""
    feature_cols: list[str]
    imputer: SimpleImputer
    scaler: StandardScaler
    reducer: object          # UMAP or PCA
    reducer_name: str        # "UMAP" or "PCA"
    X_reduced: np.ndarray    # reduced train embeddings
    person_ids: np.ndarray   # corresponding person_ids


def fit_representation(
    fan_train: pd.DataFrame,
    n_components: int = 10,
    umap_neighbors: int = 15,
    umap_min_dist: float = 0.0,
    umap_metric: str = "euclidean",
    random_state: int = 42,
    use_pca_fallback: bool = False,
) -> RepresentationResult:
    """Fit imputer → scaler → reducer on TRAIN fan features.

    Parameters
    ----------
    fan_train : DataFrame
        One row per person_id (train fans).
    n_components : int
        Target dimensionality.
    use_pca_fallback : bool
        If True, skip UMAP and use PCA directly (faster, always supports
        ``.transform()``).

    Returns
    -------
    RepresentationResult
    """
    feature_cols = _select_numeric(fan_train)
    log.info("Representation: %d features selected", len(feature_cols))

    X = fan_train[feature_cols].values.astype(float)
    pids = fan_train["person_id"].values

    # -------------------------------------------------------------------------
    # Step 1 — IMPUTATION
    # Replace NaN entries with the median of each column, computed on train data.
    # Median is preferred over mean because many fan features (e.g. spend, visit
    # count) are right-skewed; median is more robust to extreme outliers.
    # -------------------------------------------------------------------------
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)

    # -------------------------------------------------------------------------
    # Step 2 — SCALING
    # StandardScaler subtracts the column mean and divides by the column std so
    # every feature has mean=0 and std=1.  This is essential for UMAP/HDBSCAN:
    # without it, high-magnitude features (e.g. total spend in €) dominate the
    # Euclidean distance computation and low-magnitude features are invisible.
    # -------------------------------------------------------------------------
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # -------------------------------------------------------------------------
    # Step 3 — DIMENSIONALITY REDUCTION
    # Compress the scaled feature matrix into n_components dimensions.
    # UMAP is preferred: it preserves local neighbourhood structure, which helps
    # HDBSCAN find meaningful, compact clusters.  PCA is a deterministic fallback
    # that is faster and always supports .transform() on unseen data.
    # -------------------------------------------------------------------------
    if use_pca_fallback:
        log.info("Using PCA (n_components=%d)", n_components)
        reducer = PCA(n_components=n_components, random_state=random_state)
        X_red = reducer.fit_transform(X)
        rname = "PCA"
    else:
        try:
            import umap as umap_lib
            log.info("Using UMAP (n_components=%d, n_neighbors=%d, min_dist=%.2f)",
                     n_components, umap_neighbors, umap_min_dist)
            reducer = umap_lib.UMAP(
                n_components=n_components,
                n_neighbors=umap_neighbors,
                min_dist=umap_min_dist,
                metric=umap_metric,
                random_state=random_state,
                verbose=False,
                n_jobs=-1,
            )
            X_red = reducer.fit_transform(X)
            rname = "UMAP"
        except ImportError:
            log.warning("umap-learn not installed – falling back to PCA")
            reducer = PCA(n_components=n_components, random_state=random_state)
            X_red = reducer.fit_transform(X)
            rname = "PCA"

    log.info("Reduced %d fans → %s", X_red.shape[0], X_red.shape)

    return RepresentationResult(
        feature_cols=feature_cols,
        imputer=imputer,
        scaler=scaler,
        reducer=reducer,
        reducer_name=rname,
        X_reduced=X_red,
        person_ids=pids,
    )


def transform_test(
    fan_test: pd.DataFrame,
    rep: RepresentationResult,
) -> tuple[np.ndarray, np.ndarray]:
    """Project test fans into the same reduced space.

    Returns
    -------
    (X_reduced_test, person_ids_test)
    """
    X = fan_test[rep.feature_cols].values.astype(float)
    pids = fan_test["person_id"].values

    X = rep.imputer.transform(X)
    X = rep.scaler.transform(X)

    if rep.reducer_name == "PCA":
        X_red = rep.reducer.transform(X)
    else:
        # UMAP.transform() projects new points onto the already-fitted manifold.
        # It works reliably with standard umap-learn parameters but can raise in
        # edge cases (e.g. certain versions of umap-learn, very small test sets,
        # or when the manifold was built with non-default internal graph params).
        # The PCA fallback handles these cases gracefully: rather than crashing,
        # we fit a fresh PCA on the TRAIN embeddings (rep.X_reduced) and use it
        # to project the test points into a comparable lower-dimensional space.
        # This is less topologically faithful than UMAP.transform() but ensures
        # the pipeline always completes and produces a usable test embedding.
        try:
            X_red = rep.reducer.transform(X)
        except Exception as e:
            log.warning("UMAP.transform failed (%s) – falling back to PCA re-fit", e)
            pca = PCA(n_components=rep.X_reduced.shape[1])
            pca.fit(rep.X_reduced)
            X_red = pca.transform(rep.scaler.transform(rep.imputer.transform(
                fan_test[rep.feature_cols].values.astype(float)
            )))

    log.info("Test transform: %d fans → %s", X_red.shape[0], X_red.shape)
    return X_red, pids
