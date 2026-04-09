"""
Chronological train / test split by game date.

The split is done on UNIQUE GAME DATES, not on individual rows.
All rows belonging to train-set games go to train; likewise for test.
Non-match rows (packs, season passes, …) are split by the same cutoff
date to prevent temporal leakage.
"""

# =============================================================================
# IMPORTANT DESIGN DECISIONS
# =============================================================================
# (1) SPLIT ON GAME DATES, NOT ROWS.
#     The unit of the split is a unique game date, not an individual ticket row.
#     All tickets purchased for a train game stay in train; all tickets for a
#     test game stay in test.  This mirrors the real operational setting where
#     a model is trained on historical seasons and evaluated on future games.
#
# (2) NON-MATCH ROWS (packs, subscriptions, miscellaneous) SPLIT BY PURCHASE DATE.
#     Rows whose competition_type is NOT LBA or Eurocup have no meaningful
#     "event date" for defining train vs. test.  Instead, their movement_dt
#     (purchase/transaction date) is compared against the cutoff.  If it is
#     before the cutoff it goes to train; if at or after, see (3) below.
#
# (3) NON-MATCH ROWS PURCHASED AFTER THE CUTOFF ARE DROPPED.
#     A fan who bought a season pack after the cutoff should not appear to
#     have "historical" data in the train split.  If we kept such rows in
#     train, the fan's features would reflect a future transaction, leaking
#     information from the test period into the training data.  Dropping these
#     rows is therefore the conservative, leak-free choice.
# =============================================================================
from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class SplitResult(NamedTuple):
    train: pd.DataFrame
    test: pd.DataFrame
    train_games: list[pd.Timestamp]
    test_games: list[pd.Timestamp]
    cutoff_date: pd.Timestamp


def chronological_game_split(
    df: pd.DataFrame,
    test_frac: float = 0.30,
    date_col: str = "event_dt",
    fixed_cutoff_date: "str | pd.Timestamp | None" = None,
    max_eval_date: "str | pd.Timestamp | None" = None,
) -> SplitResult:
    """Split ticket-level data by game date chronologically.

    Parameters
    ----------
    df : DataFrame
        Cleaned ticket-level data (must contain *date_col*).
    test_frac : float
        Fraction of **games** (not rows) to hold out as test.
        Ignored when *fixed_cutoff_date* is provided.
    date_col : str
        Column with the event datetime.
    fixed_cutoff_date : str or Timestamp, optional
        If provided, force the train/test cutoff to exactly this date,
        ignoring *test_frac*.  All competitive games strictly before this
        date go to train; games on or after go to test.  Useful for
        reproducing a specific historical split when new data (e.g.
        advance ticket sales for a future game) has been appended to the
        CSV since the original run.
    max_eval_date : str or Timestamp, optional
        If provided, competitive games whose date is strictly after this
        date are excluded from both train and test.  Their rows remain in
        the *full* cleaned DataFrame (so ``generate_future_packages`` can
        still detect pre-sales via the ``already_bought`` check), but they
        are not evaluated.  Combine with *fixed_cutoff_date* to reproduce
        an earlier split exactly while the CSV contains new pre-sale rows.

    Returns
    -------
    SplitResult
        Named tuple with train/test DataFrames and metadata.

    Notes
    -----
    - Only rows with ``competition_type in {LBA, Eurocup}`` are considered
      "games" for the purpose of computing the chronological cutoff.
    - Non-match rows (Abbonamento, Pack, Non partita) are included in
      **train only if their movement_dt (or event_dt) is strictly before
      the cutoff**.  This prevents future non-match transactions (e.g. a
      pack purchased after the cutoff) from leaking into train features.
    - Non-match rows at or after the cutoff are **dropped** (they are
      neither train nor test targets).
    - The cutoff is the *earliest test game date*, so all train games are
      strictly before all test games.
    """
    if date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' not found in DataFrame")

    # Match rows only – these define "games"
    match_mask = df["competition_type"].isin(["LBA", "Eurocup"])
    match_df = df.loc[match_mask].copy()

    # Unique game dates (normalised to day)
    game_dates = match_df[date_col].dt.normalize().dropna().unique()
    game_dates = np.sort(game_dates)

    # ── Apply max_eval_date cap before counting games ─────────────
    # max_eval_date lets you exclude future games (e.g. pre-sales already in the
    # CSV) from both splits while still keeping their rows in the full DataFrame
    # for other purposes (e.g. the already_bought check in package generation).
    # The cap is applied HERE, before n_games is counted, so that test_frac
    # percentages are computed relative to the capped game universe.
    if max_eval_date is not None:
        max_eval_ts = pd.Timestamp(max_eval_date).normalize()
        game_dates = game_dates[game_dates <= max_eval_ts]
        log.info("max_eval_date=%s: capped to %d competitive game dates for split",
                 max_eval_ts.date(), len(game_dates))

    n_games = len(game_dates)

    if n_games < 4:
        raise ValueError(f"Only {n_games} unique game dates – need at least 4 for a split")

    # ── Determine cutoff ──────────────────────────────────────────
    if fixed_cutoff_date is not None:
        # fixed_cutoff_date branch: the caller has supplied an exact date,
        # overriding the test_frac calculation entirely.  This is useful for
        # exactly reproducing an earlier split when new rows have since been
        # appended to the CSV (e.g. pre-sales for upcoming games).
        cutoff = pd.Timestamp(fixed_cutoff_date).normalize()
        train_dates = set(game_dates[game_dates < cutoff])   # strictly before cutoff
        test_dates = set(game_dates[game_dates >= cutoff])   # on or after cutoff
        log.info(
            "Game split (fixed cutoff=%s): %d train games  |  %d test games",
            cutoff.date(), len(train_dates), len(test_dates),
        )
    else:
        # data-driven branch: derive the cutoff from test_frac so the last
        # ceil(n_games * test_frac) games form the test set chronologically.
        n_test = max(1, int(np.ceil(n_games * test_frac)))
        n_train = n_games - n_test
        train_dates = set(game_dates[:n_train])
        test_dates = set(game_dates[n_train:])
        cutoff = pd.Timestamp(min(test_dates))  # earliest test game date
        log.info(
            "Game split: %d train games  |  %d test games  |  cutoff = %s",
            n_train, n_test, cutoff.date(),
        )

    # ── Assign match rows ────────────────────────────────────────
    row_game_date = df[date_col].dt.normalize()
    is_train_game = row_game_date.isin(train_dates) & match_mask
    is_test_game = row_game_date.isin(test_dates) & match_mask

    # ── Assign non-match rows by cutoff ──────────────────────────
    is_non_match = ~match_mask
    # Use movement_dt as the temporal anchor for non-match rows because that
    # reflects when the transaction actually happened (e.g. the date a pack was
    # purchased), which is more meaningful than the event_dt for these row types.
    # Fall back to event_dt only when movement_dt is missing.
    non_match_dt = df["movement_dt"].where(df["movement_dt"].notna(), df[date_col])
    non_match_before_cutoff = is_non_match & (non_match_dt < cutoff)  # goes to train
    non_match_after_cutoff = is_non_match & (non_match_dt >= cutoff)  # dropped (see design note 3)

    n_dropped = non_match_after_cutoff.sum()
    if n_dropped > 0:
        log.info(
            "Non-match rows: %d before cutoff → train, %d at/after cutoff → dropped",
            non_match_before_cutoff.sum(), n_dropped,
        )

    # train_mask includes:
    #   - all ticket rows for games in train_dates (LBA / Eurocup match rows)
    #   - all non-match rows (packs, subscriptions, etc.) purchased before the cutoff
    # Excluded from train_mask (and therefore from both splits):
    #   - non-match rows purchased at or after the cutoff (dropped to prevent leakage)
    train_mask = is_train_game | non_match_before_cutoff
    # test_mask is match rows only — non-match rows have no place in the test set
    # because we evaluate the model on predicting attendance at specific games.
    test_mask = is_test_game
    # non_match_after_cutoff rows are intentionally excluded from both

    train_df = df.loc[train_mask].copy()
    test_df = df.loc[test_mask].copy()

    log.info(
        "Row counts: train=%d  test=%d  dropped=%d  (%.1f%% test rows)",
        len(train_df), len(test_df), n_dropped,
        100 * len(test_df) / max(1, len(df)),
    )

    return SplitResult(
        train=train_df,
        test=test_df,
        train_games=sorted(train_dates),
        test_games=sorted(test_dates),
        cutoff_date=cutoff,
    )
