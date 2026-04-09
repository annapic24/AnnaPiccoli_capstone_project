"""
Fan-level feature aggregation.

This module faithfully reproduces the feature engineering from the original
``pipeline.features.build_fan_level`` + its four enrichment modules, but
is designed to be called on an **arbitrary subset** of ticket-level rows
(e.g. train-only) so that features never leak future information.

Opponent-aware features (Task 1)
---------------------------------
``_add_opponent_affinity(fan, tc, opponents_df)`` computes four groups of
features using ONLY the rows already in ``tc`` (i.e. train split):

  pct_games_vs_<opponent_norm>     – share of that opponent's games attended
  win_rate_vs_<opponent_norm>      – Trento win rate for games that fan attended
  avg_point_diff_vs_<opponent_norm>– mean (Trento_pts − Opp_pts) for fan
  avg_abs_diff_vs_<opponent_norm>  – mean |point diff| (intensity proxy)

Sparse fallbacks (when individual opponent has < 1 home game in train, i.e. unseen):
  pct_games_vs_top8      – fraction attended vs top-8 opponents (win_rate ≥ 0.5)
  pct_games_vs_bottom8   – fraction attended vs bottom-8 opponents
  pct_games_vs_hv_opponents – fraction attended vs high-value/derby opponents

All features are stored as numeric columns on the fan DataFrame; columns for
opponents not seen in training are never created (no data leakage from schedule).
"""

# =============================================================================
# FEATURE ENGINEERING OVERVIEW
# =============================================================================
# This module computes ~67 fan-level features across 9 conceptual groups:
#
#  (1) Attendance
#        games_attended, match_games_attended, tickets_rows
#  (2) Recency & Buying patterns
#        recency_days, purchase_occasions, avg_days_between_purchases
#  (3) Price sensitivity
#        avg_price_ratio_vs_list, pct_full_price, pct_discounted_vs_list
#  (4) Timing
#        pct_early_bird, pct_last_minute, pct_weekend_games
#  (5) Competition
#        pct_lba_games, has_eurocup_history
#  (6) Opponent affinity (per-opponent columns)
#        pct_games_vs_<opponent_norm> columns, plus sparse fallbacks:
#        pct_games_vs_top8, pct_games_vs_bottom8, pct_games_vs_hv_opponents
#  (7) Seating
#        sector_variety, premium_affinity, sector_upgrade_trend
#  (8) Demographics
#        age_mode, pct_child_tickets
#  (9) Churn risk
#        recency_z_score, inactive_risk_flag, engagement_decay
#
# The module is structured as one public entry point (build_fan_features) plus
# four private enrichment functions (_add_temporal, _add_engagement, _add_value,
# _add_churn) and the opponent-affinity subsystem (_add_opponent_affinity,
# _add_sparse_fallbacks, _get_standings_tiers).
# =============================================================================

from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

log = logging.getLogger(__name__)


# ── Score parsing helpers ─────────────────────────────────────────────

def _parse_score(score_str: str) -> tuple[int, int] | None:
    """Parse 'XX-YY' or 'XX - YY' → (trento_pts, opponent_pts).

    Returns None if unparseable.
    The first number is always the home team.  For Trento home games the
    schedule records Trento as the first score, but we use the Outcome column
    ('W'/'L') to determine who won, so we only need the raw integers.
    """
    if not isinstance(score_str, str):
        return None
    m = re.match(r"^\s*(\d+)\s*[-–]\s*(\d+)\s*$", score_str.strip())
    if m is None:
        return None
    try:
        return int(m.group(1)), int(m.group(2))
    except ValueError:
        return None


def _trento_pts_and_diff(score_str: str, outcome: str) -> tuple[float, float] | None:
    """Return (trento_pts, trento_pts − opp_pts) from score + outcome.

    The CSV always puts the Trento score first (home perspective), but we
    verify with the Outcome column to handle any away-game entries:
      - If Outcome=='W' and first > second  → Trento first   (home)
      - If Outcome=='W' and first < second  → Trento second  (away)
      - If Outcome=='L' and first < second  → Trento first   (home)
      - If Outcome=='L' and first > second  → Trento second  (away)
    """
    parsed = _parse_score(score_str)
    if parsed is None:
        return None
    a, b = parsed
    outcome_upper = str(outcome).strip().upper() if pd.notna(outcome) else ""
    if outcome_upper == "W":
        # Trento won → larger score is Trento
        if a >= b:
            return float(a), float(a - b)
        else:
            return float(b), float(b - a)
    elif outcome_upper == "L":
        # Trento lost → smaller score is Trento
        if a <= b:
            return float(a), float(a - b)
        else:
            return float(b), float(b - a)
    else:
        # Unknown outcome – still return raw diff (first assumed Trento)
        return float(a), float(a - b)

# ── helpers ───────────────────────────────────────────────────────────

def _safe_mode(s: pd.Series):
    s = s.dropna()
    if s.empty:
        return np.nan
    return s.mode().iloc[0]


def _ensure(df: pd.DataFrame, col: str, default=np.nan) -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series(default, index=df.index)


# ── main entry point ─────────────────────────────────────────────────

def build_fan_features(
    ticket_df: pd.DataFrame,
    opponents_df: pd.DataFrame | None = None,
    standings_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Aggregate ticket-level rows → one row per person_id.

    Parameters
    ----------
    ticket_df : DataFrame
        Cleaned ticket-level data (TRAIN only).  Must contain at least:
        ``person_id, event_dt, movement_dt, competition_type, settore,
        ticket_price, total_amount, days_before_game, price_ratio_from_list,
        is_child_ticket, is_free_expected_from_age, is_non_match,
        is_bundle_or_pack, sales_channel, season, age, provincia_clean``.
    opponents_df : DataFrame, optional
        Opponents schedule (from ``load_opponents``).  Must have columns
        ``game_date, opponent_team, Score, Outcome``.  When provided,
        opponent-affinity features are computed using ONLY train games
        (those whose ``game_date`` appears in ``ticket_df.event_dt``).
        This is the TRAIN-only guard: no future or test games are used.
    standings_df : DataFrame, optional
        lba_standings_by_giornata.csv.  When provided the top8/bottom8
        fallback tiers are sourced from the external standings CSV
        (authoritative league-wide view) rather than from Trento's
        limited per-season win-rates.

    Returns
    -------
    DataFrame with one row per ``person_id`` and ~67 + opponent features.
    """
    tmp = ticket_df.copy()

    # ── Type coercion ─────────────────────────────────────────────────────
    # Ensure all columns are the expected type even if the CSV has mixed types
    # (e.g. a numeric column that contains a stray header row read as string).
    # _ensure() safely returns a default-filled Series when the column is absent,
    # so downstream code never needs to guard for missing columns.
    tmp["event_dt"] = pd.to_datetime(_ensure(tmp, "event_dt", pd.NaT), errors="coerce")
    tmp["movement_dt"] = pd.to_datetime(_ensure(tmp, "movement_dt", pd.NaT), errors="coerce")
    tmp["competition_type"] = _ensure(tmp, "competition_type", "").astype(str)
    tmp["settore"] = _ensure(tmp, "settore", "").astype(str)
    tmp["age"] = pd.to_numeric(_ensure(tmp, "age", np.nan), errors="coerce")
    tmp["provincia_clean"] = _ensure(tmp, "provincia_clean", np.nan)
    tmp["is_non_match"] = _ensure(tmp, "is_non_match", False).astype(bool)
    tmp["is_child_ticket"] = _ensure(tmp, "is_child_ticket", False).astype(bool)
    tmp["is_free_expected_from_age"] = _ensure(tmp, "is_free_expected_from_age", False).astype(bool)
    tmp["price_ratio_from_list"] = pd.to_numeric(_ensure(tmp, "price_ratio_from_list", np.nan), errors="coerce")
    tmp["days_before_game"] = pd.to_numeric(_ensure(tmp, "days_before_game", np.nan), errors="coerce")
    tmp["total_amount"] = pd.to_numeric(_ensure(tmp, "total_amount", np.nan), errors="coerce")
    tmp["ticket_price"] = pd.to_numeric(_ensure(tmp, "ticket_price", np.nan), errors="coerce")
    tmp["bundle_games_included"] = pd.to_numeric(_ensure(tmp, "bundle_games_included", np.nan), errors="coerce")
    tmp["is_bundle_or_pack"] = _ensure(tmp, "is_bundle_or_pack", False).astype(bool)
    tmp["season"] = _ensure(tmp, "season", np.nan)
    tmp["sales_channel"] = _ensure(tmp, "sales_channel", "").astype(str)

    # ── Bundle effective price ────────────────────────────────────────────
    # For multi-game pack rows the total_amount covers several games, so dividing
    # by bundle_games_included gives a per-game equivalent price.  When the total
    # amount is missing we fall back to ticket_price (which may be the per-game
    # list price for the pack).  Single-ticket rows get NaN (not applicable).
    total_eff = tmp["total_amount"].where(tmp["total_amount"].notna(), tmp["ticket_price"])
    bg = tmp["bundle_games_included"]
    tmp["bundle_effective_price"] = np.where(
        bg.notna() & (bg > 0), total_eff / bg, np.nan
    )

    # ── Primary aggregation — one row per fan ────────────────────────────
    # This is the main fan-level aggregation that collapses all ticket rows for
    # a given person_id into a single summary row.  ALL ticket rows are included
    # here (including non-match, pack, and subscription rows) so that total_spend
    # and tickets_rows reflect the fan's full commercial relationship with the club.
    # Behavioural ratios like pct_non_match_rows are computed as means so they
    # are automatically in [0, 1].
    is_non = tmp["is_non_match"]

    fan = tmp.groupby("person_id").agg(
        games_attended=("event_dt", "nunique"),
        match_games_attended=("event_dt", lambda s: s[~tmp.loc[s.index, "is_non_match"]].nunique()),
        tickets_rows=("person_id", "count"),
        pct_non_match_rows=("is_non_match", "mean"),
        total_spend=("total_amount", "sum"),
        match_spend=("total_amount", lambda s: s[~tmp.loc[s.index, "is_non_match"]].sum()),
        avg_ticket_price_match_only=("ticket_price", lambda s: s[~tmp.loc[s.index, "is_non_match"]].mean()),
        avg_bundle_effective_price=("bundle_effective_price", "mean"),
        total_bundle_games_included=("bundle_games_included", "sum"),
        avg_days_before_game=("days_before_game", lambda s: s[~tmp.loc[s.index, "is_non_match"]].mean()),
        most_common_sector=("settore", _safe_mode),
        sector_variety=("settore", lambda s: s.nunique(dropna=True)),
        province_mode=("provincia_clean", _safe_mode),
        province_nunique=("provincia_clean", lambda s: s.nunique(dropna=True)),
        channel_variety=("sales_channel", "nunique"),
        season_variety=("season", "nunique"),
        competition_variety=("competition_type", "nunique"),
        first_purchase=("movement_dt", "min"),
        last_purchase=("movement_dt", "max"),
        age_mode=("age", _safe_mode),
        age_nunique=("age", lambda s: s.nunique(dropna=True)),
        pct_child_tickets=("is_child_ticket", "mean"),
        pct_discounted_vs_list=("price_ratio_from_list", lambda s: (s < 1.0).mean() if s.notna().any() else np.nan),
        avg_price_ratio_vs_list=("price_ratio_from_list", "mean"),
        has_free_expected_tickets=("is_free_expected_from_age", "max"),
    ).reset_index()

    pid = fan["person_id"]
    def _m(s: pd.Series) -> pd.Series:
        return pid.map(s)

    # ── Clustering subset ─────────────────────────────────────────────────
    # Exclude Pack and Abbonamento (subscription) rows from the behavioural
    # feature calculations below.  These fans attend most games by obligation
    # rather than by active choice, so including their rows would distort
    # behavioural statistics such as recency, purchase frequency, and price
    # sensitivity.  The subscription flag is still kept in the final feature
    # table so that campaign filtering can exclude or target them separately.
    tmp_clust = tmp.loc[
        (~tmp["is_non_match"])
        & (~tmp["competition_type"].isin(["Abbonamento", "Pack"]))
    ].copy()

    if tmp_clust.empty:
        log.warning("tmp_clust is empty – all NaN for secondary features")
        for c in ["recency_days", "purchase_occasions", "avg_tickets_per_purchase",
                   "purchase_spread_ratio", "max_tickets_single_purchase",
                   "tickets_per_game_attended", "pct_early_bird", "pct_last_minute",
                   "pct_full_price", "pct_zero_price", "pct_lba_games",
                   "pct_eurocup_games", "pct_weekend_games", "pct_evening_games",
                   "sector_consistency"]:
            fan[c] = np.nan
    else:
        # purchase_date: prefer movement_dt (the actual transaction timestamp),
        # fall back to event_dt for rows where the movement timestamp is missing.
        purchase_date = tmp_clust["movement_dt"].dt.date
        fallback = tmp_clust["event_dt"].dt.date
        purchase_date = purchase_date.where(purchase_date.notna(), fallback)
        tmp_clust["purchase_date"] = purchase_date

        # ── Recency calculation ───────────────────────────────────────────
        # reference_date = the latest timestamp across both event and movement
        # dates in the entire clustering subset, so that recency is always
        # measured relative to the most recent observation in the training data
        # (never relative to today, which would make the feature time-dependent).
        ref_event = tmp_clust["event_dt"].max()
        ref_move = tmp_clust["movement_dt"].max()
        candidates = [d for d in [ref_event, ref_move] if pd.notna(d)]
        reference_date = max(candidates) if candidates else pd.Timestamp.today().normalize()

        # recency_days = days since the fan's last observed activity (purchase OR
        # event attendance, whichever is more recent).  Taking the max across both
        # timestamps prevents underestimating recency when movement_dt is stale.
        last_p = tmp_clust.groupby("person_id")["movement_dt"].max()
        last_e = tmp_clust.groupby("person_id")["event_dt"].max()
        last_activity = last_p.where(last_p.notna(), last_e)
        # also take the per-fan max across movement/event
        last_activity = pd.concat([last_p, last_e], axis=1).max(axis=1)
        recency = (reference_date - last_activity).dt.days
        recency = recency.clip(lower=0)                   # never negative
        recency = recency.fillna(recency.median())         # impute missing with median
        fan["recency_days"] = _m(recency)

        # purchase occasions
        po = tmp_clust.groupby("person_id")["purchase_date"].nunique()
        total_rows = tmp_clust.groupby("person_id").size()
        fan["purchase_occasions"] = _m(po)
        fan["avg_tickets_per_purchase"] = _m((total_rows / po).replace([np.inf, -np.inf], np.nan))
        fan["purchase_spread_ratio"] = _m((po / total_rows).replace([np.inf, -np.inf], np.nan))

        tpp = tmp_clust.groupby(["person_id", "purchase_date"]).size()
        fan["max_tickets_single_purchase"] = _m(tpp.groupby("person_id").max())

        gaf = tmp_clust.groupby("person_id")["event_dt"].nunique()
        fan["tickets_per_game_attended"] = _m((total_rows / gaf).replace([np.inf, -np.inf], np.nan))

        # ── Behavioural pct_* columns ─────────────────────────────────────
        # Each is_* boolean is computed at the ticket row level, then averaged
        # per fan to produce a fraction in [0, 1].  All computed on the
        # clustering subset (match rows only, no packs / subscriptions).
        #   is_early_bird   : ticket bought more than 7 days before the game
        #   is_last_minute  : ticket bought 2 or fewer days before the game
        #   is_discounted   : paid less than 95% of the official list price
        #   is_full_price   : paid at least 95% of the list price (≈ full price)
        #   is_zero_price   : paid ≤ 1% of list price (free/comp ticket)
        #   is_lba          : the row is a domestic Serie A game
        #   is_eurocup      : the row is a EuroCup European competition game
        #   is_weekend      : the game took place on Saturday or Sunday
        #   is_evening      : the game start time is 18:00 or later
        grp = tmp_clust.groupby("person_id", dropna=False)
        tmp_clust["is_early_bird"] = tmp_clust["days_before_game"] > 7
        tmp_clust["is_last_minute"] = tmp_clust["days_before_game"] <= 2
        tmp_clust["is_discounted"] = tmp_clust["price_ratio_from_list"] < 0.95
        tmp_clust["is_full_price"] = tmp_clust["price_ratio_from_list"] >= 0.95
        tmp_clust["is_zero_price"] = tmp_clust["price_ratio_from_list"] <= 0.01
        tmp_clust["is_lba"] = tmp_clust["competition_type"] == "LBA"
        tmp_clust["is_eurocup"] = tmp_clust["competition_type"] == "Eurocup"
        tmp_clust["is_weekend"] = tmp_clust["event_dt"].dt.dayofweek >= 5
        tmp_clust["is_evening"] = tmp_clust["event_dt"].dt.hour >= 18

        fan["pct_early_bird"] = _m(grp["is_early_bird"].mean())
        fan["pct_last_minute"] = _m(grp["is_last_minute"].mean())
        fan["pct_discounted_vs_list"] = _m(grp["is_discounted"].mean())
        fan["pct_full_price"] = _m(grp["is_full_price"].mean())
        fan["pct_zero_price"] = _m(grp["is_zero_price"].mean())
        fan["avg_price_ratio_vs_list"] = _m(grp["price_ratio_from_list"].mean())
        fan["pct_lba_games"] = _m(grp["is_lba"].mean())
        fan["pct_eurocup_games"] = _m(grp["is_eurocup"].mean())
        fan["competition_variety"] = _m(grp["competition_type"].nunique())
        fan["pct_weekend_games"] = _m(grp["is_weekend"].mean())
        fan["pct_evening_games"] = _m(grp["is_evening"].mean())

        sv = grp["settore"].nunique(dropna=True)
        fan["sector_variety"] = _m(sv)
        sc = 1 - (sv / gaf.clip(lower=1))
        fan["sector_consistency"] = _m(sc.clip(lower=0, upper=1))

    # ── has_eurocup_history binary flag ──────────────────────────────────
    # Non-collinear alternative to pct_eurocup_games for UMAP input.
    # pct_eurocup_games is excluded from the UMAP embedding because it strongly
    # anti-correlates with pct_lba_games (the two sum to ≈ 1 for all match rows),
    # which would introduce redundant variance and distort the embedding geometry.
    # This binary version simply indicates whether a fan has *ever* attended a
    # EuroCup game without the collinearity problem — it adds information about
    # European game engagement while remaining independent of pct_lba_games.
    fan["has_eurocup_history"] = (fan["pct_eurocup_games"].fillna(0) > 0).astype(int)

    # derived
    fan["active_days"] = (fan["last_purchase"] - fan["first_purchase"]).dt.days
    fan["has_bundle_games_info"] = fan["total_bundle_games_included"].fillna(0) > 0
    fan["has_non_match"] = fan["pct_non_match_rows"].fillna(0) > 0

    # ── Subscription holder flag ──────────────────────────────────────────
    # Kept in the features table for campaign filtering (e.g. to exclude
    # existing subscribers from re-acquisition campaigns), but excluded from
    # the UMAP embedding via the _EXCLUDE list in representation.py because
    # subscription holders attend most games by obligation and their behavioural
    # statistics do not reflect discretionary purchase decisions.
    sub_mask = tmp["competition_type"].isin(["Abbonamento", "Pack"])
    sub_pids = set(tmp.loc[sub_mask, "person_id"].unique())
    fan["is_subscription_holder"] = fan["person_id"].isin(sub_pids)

    # ── advanced modules ─────────────────────────────────────────
    fan = _add_temporal(fan, tmp_clust)
    fan = _add_engagement(fan, tmp_clust)
    fan = _add_value(fan, tmp_clust)
    fan = _add_churn(fan, tmp_clust)

    # ── opponent affinity (Task 1) ────────────────────────────────
    # Uses only train games (ticket_df rows); opponents_df provides
    # score/outcome metadata.  No test or future rows used here.
    fan = _add_opponent_affinity(fan, tmp_clust, ticket_df, opponents_df,
                                 standings_df=standings_df)

    log.info("Fan features: %d fans × %d features", len(fan), len(fan.columns) - 1)
    return fan


# ── temporal features ────────────────────────────────────────────────

def _add_temporal(fan: pd.DataFrame, tc: pd.DataFrame) -> pd.DataFrame:
    # Adds time-pattern features: inter-purchase intervals, purchase acceleration
    # (trend in spacing over time), and season-phase purchase distribution.
    out = fan.copy()
    out["days_since_first_purchase"] = out.get("active_days", np.nan)

    if tc.empty:
        for c in ["avg_days_between_purchases", "purchase_regularity_cv",
                   "purchase_acceleration", "last_3_months_activity"]:
            out[c] = np.nan
        for phase in ["early", "mid", "late", "other"]:
            out[f"season_phase_purchases_{phase}"] = 0
        return out

    tc = tc.copy()
    tc["movement_dt"] = pd.to_datetime(tc["movement_dt"], errors="coerce")
    tc["event_dt"] = pd.to_datetime(tc["event_dt"], errors="coerce")

    ref = tc["event_dt"].max()
    if pd.isna(ref):
        ref = tc["movement_dt"].max()
    if pd.isna(ref):
        ref = pd.Timestamp.today().normalize()

    # per-person temporal stats
    records = {}
    for pid, g in tc.groupby("person_id"):
        dates = g["movement_dt"].dropna().sort_values().unique()
        if len(dates) < 2:
            records[pid] = {"avg_days_between_purchases": np.nan,
                            "purchase_regularity_cv": np.nan,
                            "purchase_acceleration": np.nan}
            continue
        ds = pd.Series(dates).sort_values()
        intervals = ds.diff().dt.days.dropna()
        avg_int = intervals.mean()
        std_int = intervals.std()
        cv = std_int / avg_int if avg_int > 0 else np.nan

        if len(dates) >= 3:
            # purchase_acceleration: slope of a linear regression of cumulative
            # days-from-first-purchase against purchase index (0, 1, 2, …).
            # A positive slope means purchases are becoming more spread out over
            # time (slowing down); a negative slope means they are becoming more
            # frequent (accelerating engagement).  Requires at least 3 purchases
            # for a meaningful regression.
            idx = np.arange(len(dates))
            days_from_first = (ds - ds.iloc[0]).dt.days.values
            slope = sp_stats.linregress(idx, days_from_first).slope
        else:
            slope = np.nan

        records[pid] = {"avg_days_between_purchases": avg_int,
                        "purchase_regularity_cv": cv,
                        "purchase_acceleration": slope}

    tdf = pd.DataFrame.from_dict(records, orient="index")
    tdf.index.name = "person_id"
    tdf = tdf.reset_index()
    out = out.merge(tdf, on="person_id", how="left")

    # season_phase splits the basketball calendar into three windows:
    #   early  (Aug–Oct) : pre-season and opening rounds
    #   mid    (Nov–Jan) : Christmas period, typically the densest stretch
    #   late   (Feb–May) : run-in and playoff phase
    #   other  (Jun–Jul) : off-season (rare for ticket purchases)
    # The resulting counts (season_phase_purchases_early/mid/late/other) capture
    # whether a fan concentrates their attendance at the start vs. end of season.
    def _phase(dt):
        if pd.isna(dt):
            return np.nan
        m = dt.month
        if m in (8, 9, 10):
            return "early"
        if m in (11, 12, 1):
            return "mid"
        if m in (2, 3, 4, 5):
            return "late"
        return "other"

    tc["season_phase"] = tc["event_dt"].apply(_phase)
    pc = tc.groupby(["person_id", "season_phase"]).size().unstack(fill_value=0)
    pc = pc.add_prefix("season_phase_purchases_")
    out = out.merge(pc, on="person_id", how="left")
    for ph in ["early", "mid", "late", "other"]:
        c = f"season_phase_purchases_{ph}"
        if c not in out.columns:
            out[c] = 0
        out[c] = out[c].fillna(0)

    # last 3 months
    out["last_3_months_activity"] = (ref - out["last_purchase"]).dt.days <= 90

    return out


# ── engagement features ──────────────────────────────────────────────

def _add_engagement(fan: pd.DataFrame, tc: pd.DataFrame) -> pd.DataFrame:
    # Adds opponent-engagement and group-behaviour features: high-value and derby
    # opponent attendance rates, EuroCup loyalty ratio, companion stability, and
    # sector exploration entropy.
    out = fan.copy()

    if tc.empty:
        for c in ["pct_high_value_opponents", "derby_attendance_rate",
                   "eurocup_loyalty", "companion_group_stability",
                   "sector_exploration_score"]:
            out[c] = 0
        return out

    tc = tc.copy()
    opp = _ensure(tc, "opponent_team", "").astype(str).str.upper()

    # HV (high-value) opponents: the top Italian clubs by national profile and
    # fan-base size.  Games against these teams tend to sell out faster and
    # attract more casual fans, making attendance a signal of high engagement.
    HV = ["BOLOGNA", "MILANO", "VENEZIA", "VIRTUS", "OLIMPIA", "REYER"]

    # DERBY opponents: clubs from geographically close or historically rival
    # cities in the north-east of Italy.  Derby games have local cultural
    # significance beyond pure sporting interest.
    DERBY = ["BRESCIA", "TRIESTE", "TREVISO"]

    tc["is_hv"] = opp.apply(lambda x: any(t in x for t in HV))
    tc["is_derby"] = opp.apply(lambda x: any(t in x for t in DERBY))

    match = tc["competition_type"].isin(["LBA", "Eurocup"])
    hv_pct = tc.loc[match].groupby("person_id")["is_hv"].mean()
    derby_pct = tc.loc[match].groupby("person_id")["is_derby"].mean()

    out["pct_high_value_opponents"] = out["person_id"].map(hv_pct).fillna(0)
    out["derby_attendance_rate"] = out["person_id"].map(derby_pct).fillna(0)

    # eurocup_loyalty: EuroCup games as a fraction of ALL competitive match
    # attendance (LBA + EuroCup).  A value of 1.0 means the fan only ever
    # attended EuroCup games; 0.0 means they only attended LBA games.  This
    # measures how much a fan's interest extends beyond domestic basketball.
    ep = out.get("pct_eurocup_games", 0)
    lp = out.get("pct_lba_games", 0)
    total = ep + lp
    out["eurocup_loyalty"] = np.where(total > 0, ep / total, 0)

    # companion group stability
    cgs = tc.groupby("person_id").apply(
        lambda g: g.groupby("movement_dt").size().std(), include_groups=False
    )
    out["companion_group_stability"] = out["person_id"].map(cgs).fillna(0)

    # sector entropy
    def _entropy(pid):
        s = tc.loc[tc["person_id"] == pid, "settore"]
        if s.empty:
            return 0
        vc = s.value_counts(normalize=True)
        if len(vc) <= 1:
            return 0
        return -(vc * np.log2(vc + 1e-10)).sum()

    out["sector_exploration_score"] = out["person_id"].map(
        {pid: _entropy(pid) for pid in out["person_id"]}
    ).fillna(0)

    return out


# ── value features ───────────────────────────────────────────────────

# SECTOR_SCORE: ordinal ranking of seating sectors by prestige and price level.
# Used to compute sector_upgrade_trend — whether a fan has been choosing
# progressively more premium seats over time.
# Scale: 1 (cheapest/standing) → 5 (courtside, most expensive).
_SECTOR_SCORE = {
    "COURTSIDE": 5, "PARTERRE": 4, "DISTINTI": 3,
    "TRIBUNA": 2, "GRADINATA": 1, "CURVA": 1, "CORNER": 1,
}

def _add_value(fan: pd.DataFrame, tc: pd.DataFrame) -> pd.DataFrame:
    # Adds price-willingness and seating-quality features: sector upgrade trend,
    # price sensitivity index, premium affinity, discount dependency, and
    # willingness-to-pay index.
    out = fan.copy()

    if tc.empty:
        for c in ["sector_upgrade_trend", "price_sensitivity_index",
                   "premium_affinity", "discount_dependency",
                   "willingness_to_pay_index"]:
            out[c] = 0
        return out

    tc = tc.copy()
    tc["settore"] = tc.get("settore", pd.Series(dtype=str)).fillna("").str.upper()
    tc["sector_premium_score"] = tc["settore"].map(
        lambda z: next((v for k, v in _SECTOR_SCORE.items() if k in z), 1)
    )

    # sector_upgrade_trend: slope of a linear regression of sector_premium_score
    # against days-since-first-purchase.  A positive value means the fan has been
    # choosing progressively more expensive seats over time (upgrading behaviour);
    # a negative value means they have been moving to cheaper seats (downgrading).
    # Zero is returned for fans with fewer than 2 purchases, or fans who always
    # sit in the same seat block (no variance in days, so regression is undefined).
    trends = {}
    for pid, g in tc.groupby("person_id"):
        g2 = g[g["movement_dt"].notna()].sort_values("movement_dt")
        if len(g2) < 2:
            trends[pid] = 0
            continue
        days = (g2["movement_dt"] - g2["movement_dt"].iloc[0]).dt.days.values
        scores = g2["sector_premium_score"].values
        if np.std(days) == 0:
            trends[pid] = 0
            continue
        trends[pid] = sp_stats.linregress(days, scores).slope

    out["sector_upgrade_trend"] = out["person_id"].map(trends).fillna(0)
    out["price_sensitivity_index"] = out.get("pct_discounted_vs_list", 0)

    # premium affinity
    prem = ["COURTSIDE", "PARTERRE"]
    tc["is_prem"] = tc["settore"].apply(lambda z: any(p in z for p in prem))
    tc["spend"] = pd.to_numeric(tc.get("total_amount", 0), errors="coerce").fillna(0)
    ps = tc.loc[tc["is_prem"]].groupby("person_id")["spend"].sum()
    ts = tc.groupby("person_id")["spend"].sum()
    pa = ps / ts
    out["premium_affinity"] = out["person_id"].map(pa).fillna(0)

    # discount dependency (>20% off)
    tc["is_hd"] = tc["price_ratio_from_list"] < 0.8
    hd = tc.groupby("person_id")["is_hd"].mean()
    out["discount_dependency"] = out["person_id"].map(hd).fillna(0)

    # willingness to pay
    fp = tc.loc[tc["price_ratio_from_list"] >= 0.95]
    wtp = fp.groupby("person_id")["price_ratio_from_list"].mean()
    out["willingness_to_pay_index"] = out["person_id"].map(wtp).fillna(1.0)

    return out


# ── churn risk features ──────────────────────────────────────────────

def _add_churn(fan: pd.DataFrame, tc: pd.DataFrame) -> pd.DataFrame:
    # Adds churn and disengagement signals: recency z-score, frequency trend
    # over the last 3 months, engagement decay ratio, and an inactive risk flag.
    out = fan.copy()

    recency = out.get("recency_days", pd.Series(dtype=float))
    rmean = recency.mean()
    rstd = recency.std()
    out["recency_z_score"] = (recency - rmean) / rstd if rstd > 0 else 0

    if tc.empty:
        out["frequency_trend_3m"] = 0
        out["engagement_decay"] = 0
        out["inactive_risk_flag"] = False
        return out

    tc = tc.copy()
    tc["movement_dt"] = pd.to_datetime(tc["movement_dt"], errors="coerce")
    ref = tc["movement_dt"].max()
    if pd.isna(ref):
        ref = pd.Timestamp.today().normalize()

    # frequency trend 3m
    cutoff_3m = ref - pd.Timedelta(days=90)
    recent = tc[tc["movement_dt"] >= cutoff_3m]
    ft = {}
    for pid, g in recent.groupby("person_id"):
        g2 = g.sort_values("movement_dt")
        if len(g2) < 2:
            ft[pid] = 0
            continue
        g2["week"] = g2["movement_dt"].dt.to_period("W")
        wc = g2.groupby("week").size()
        if len(wc) < 2:
            ft[pid] = 0
            continue
        ft[pid] = sp_stats.linregress(np.arange(len(wc)), wc.values).slope

    out["frequency_trend_3m"] = out["person_id"].map(ft).fillna(0)

    # engagement_decay: ratio of late-period activity to early-period activity.
    #   late activity  = number of ticket rows in the last 60 days of the dataset
    #   early activity = number of ticket rows in the first 60 days of the dataset
    # A ratio > 1 means the fan is becoming MORE active over time (good signal);
    # a ratio < 1 means they have become LESS active (decay / churn risk).
    # Fans with no early-period activity get NaN, filled with 0 (unknown baseline).
    min_dt = tc["movement_dt"].min()
    if pd.notna(min_dt):
        first_2m = min_dt + pd.Timedelta(days=60)
        last_2m = ref - pd.Timedelta(days=60)
        ff = tc[tc["movement_dt"] <= first_2m].groupby("person_id").size()
        fl = tc[tc["movement_dt"] >= last_2m].groupby("person_id").size()
        ed = fl / ff
        out["engagement_decay"] = out["person_id"].map(ed).fillna(0)
    else:
        out["engagement_decay"] = 0

    # inactive risk flag
    has_abbo = tc.loc[tc["competition_type"] == "Abbonamento"].groupby("person_id").size() > 0
    out["inactive_risk_flag"] = (
        (out["recency_days"] > 90) &
        (~out["person_id"].isin(has_abbo[has_abbo].index))
    )

    return out


# ── opponent affinity features (Task 1) ──────────────────────────────

_OPP_SPARSE_THRESHOLD = 1   # min home games in train to emit per-opponent cols
# Set to 1 so that any opponent seen at least once in the train schedule gets
# its own column. With a single season (14 train games), every opponent is
# encountered at most once. The sparse fallbacks (top8/bottom8/hv) are still
# computed and used when no per-opponent column exists (unseen opponents).
_HV_TOKENS_OPP = {"BOLOGNA", "MILANO", "VENEZIA", "VIRTUS", "OLIMPIA", "REYER"}
_DERBY_TOKENS_OPP = {"BRESCIA", "TRIESTE", "TREVISO"}

# ── Standings → schedule name matching tokens ─────────────────────────
# Maps each LBA team's canonical name (team_name_norm in standings CSV)
# to one or more lowercase tokens that will match as substrings of the
# normalised opponent column name (_norm_opponent output).
# Used by _get_standings_tiers() to map standings tiers to opp_col sets.
_STANDINGS_TEAM_TOKENS: dict[str, list[str]] = {
    "Virtus Bologna":            ["virtus", "bologna"],
    "Olimpia Milano":            ["olimpia", "milano", "armani"],
    "Germani Brescia":           ["germani", "brescia"],
    "Umana Reyer Venezia":       ["reyer", "venezia"],
    "Bertram Derthona Tortona":  ["tortona", "derthona"],
    "UnaHotels Reggio Emilia":   ["reggiana", "reggio"],   # schedule uses "Reggiana"
    "Pallacanestro Trieste":     ["trieste"],
    "NutriBullet Treviso":       ["treviso", "nutribullet"],
    "Banco di Sardegna Sassari": ["sassari", "dinamo"],
    "Estra Pistoia":             ["pistoia"],
    "GeVi Napoli":               ["napoli"],
    "Openjobmetis Varese":       ["varese"],
    "Vanoli Cremona":            ["cremona"],
    "Carpegna Prosciutto Pesaro":["pesaro"],
    "APU Old Wild West Udine":   ["udine"],
    "Acqua San Bernardo Cantù":  ["cantu", "cant"],
    "Givova Scafati":            ["scafati"],
    "Happy Casa Brindisi":       ["brindisi"],
    "Tezenis Verona":            ["verona"],
    "Trapani Shark":             ["trapani"],
    "Fortitudo Kigili Bologna":  ["fortitudo"],
    "Dolomiti Energia Trentino": [],  # home team – never an opponent
}


def _get_standings_tiers(
    standings_df: pd.DataFrame,
    game_info: dict,
    top_n: int = 8,
) -> tuple[set, set]:
    """Derive top-N and bottom-N opponent-column sets from external standings.

    Uses end-of-season final standings (maximum giornata per season) to
    classify LBA opponents by actual league position rather than by Trento's
    training win-rates (which cover at most one game per opponent).

    Parameters
    ----------
    standings_df : DataFrame
        Loaded lba_standings_by_giornata.csv with columns
        season, giornata, position, team_name_norm.
    game_info : dict
        Mapping from game_date → {opp_col, opp_norm, ...} (from train schedule).
    top_n : int
        Number of teams to include in each tier (default 8).

    Returns
    -------
    (top_n_cols, bottom_n_cols) : two sets of opp_col strings.
    """
    # ── 1. Identify seasons present in training games ──────────────
    train_seasons: set[str] = set()
    for info in game_info.values():
        # game_date is a pd.Timestamp; extract season string from year
        pass  # populated below

    # Map train game_dates → approximate season strings
    for gd, info in game_info.items():
        if hasattr(gd, "year"):
            yr = gd.year
            mo = gd.month
            # LBA season runs Sept–June; season label is "YYYY-YY"
            if mo >= 7:
                season_str = f"{yr}-{str(yr + 1)[-2:]}"
            else:
                season_str = f"{yr - 1}-{str(yr)[-2:]}"
            train_seasons.add(season_str)

    # ── 2. Get end-of-season standings for relevant seasons ────────
    sdf = standings_df.copy()
    # Exclude Trento itself from opponent tiers
    sdf = sdf[~sdf["team_name_norm"].str.contains("Trentino|Trento", case=False, na=False)]

    # For each season, find the maximum giornata (end of season)
    max_giornata = sdf.groupby("season")["giornata"].transform("max")
    final_standings = sdf[sdf["giornata"] == max_giornata].copy()

    # Filter to training seasons; fall back to all seasons if none match
    rel = final_standings[final_standings["season"].isin(train_seasons)]
    if rel.empty:
        log.warning(
            "Standings: no rows matched training seasons %s – using all seasons",
            sorted(train_seasons),
        )
        rel = final_standings

    if rel.empty:
        log.warning("Standings DataFrame is empty – falling back to dynamic top8/bottom8")
        return set(), set()

    # For multi-season training, aggregate average position per team
    avg_pos = (
        rel.groupby("team_name_norm")["position"].mean().reset_index()
        .sort_values("position")
    )

    top_n_names = set(avg_pos.head(top_n)["team_name_norm"])
    n_teams = len(avg_pos)
    bottom_n_names = set(avg_pos.tail(top_n)["team_name_norm"])

    log.info(
        "Standings tiers (seasons=%s): top%d=%s | bottom%d=%s",
        sorted(train_seasons), top_n,
        [n.split()[-1] for n in sorted(top_n_names)],
        top_n,
        [n.split()[-1] for n in sorted(bottom_n_names)],
    )

    # ── 3. Map standings team names → opp_col via token matching ──
    all_opp_cols = {info["opp_col"] for info in game_info.values() if info.get("opp_col")}

    def _resolve_to_opp_cols(team_name_set: set[str]) -> set[str]:
        matched: set[str] = set()
        for team_name in team_name_set:
            tokens = _STANDINGS_TEAM_TOKENS.get(team_name, [])
            if not tokens:
                # Auto-extract: words ≥ 5 chars, lowercased
                tokens = [w.lower() for w in team_name.split() if len(w) >= 5]
            for col in all_opp_cols:
                if any(tok in col for tok in tokens):
                    matched.add(col)
        return matched

    top_cols = _resolve_to_opp_cols(top_n_names)
    bottom_cols = _resolve_to_opp_cols(bottom_n_names)

    log.info(
        "Standings → opp_col match: top%d=%s | bottom%d=%s",
        top_n, sorted(top_cols), top_n, sorted(bottom_cols),
    )
    return top_cols, bottom_cols


def _norm_opponent(name: str) -> str:
    """Normalise an opponent name to a safe column suffix."""
    return re.sub(r"[^a-z0-9]", "_", str(name).strip().lower())


def _add_opponent_affinity(
    fan: pd.DataFrame,
    tc: pd.DataFrame,          # tmp_clust (train, match rows only)
    ticket_df: pd.DataFrame,   # full train (for event_dt → game_date join)
    opponents_df: pd.DataFrame | None,
    standings_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add opponent-level attendance, win-rate, and point-diff features.

    Leakage guard
    -------------
    We only look at opponents whose ``game_date`` appears in the **train**
    ticket rows (``ticket_df["event_dt"]``).  The opponents schedule rows
    for test or future games are never touched.

    Parameters
    ----------
    fan : DataFrame
        Current fan-level feature table.
    tc : DataFrame
        Clustering subset: train, match rows (no non-match/abbonamento/pack).
    ticket_df : DataFrame
        Full train tickets (used to get the set of train game dates).
    opponents_df : DataFrame or None
        Opponents schedule with columns:
        ``game_date, opponent_team, Score, Outcome``.
    standings_df : DataFrame or None
        lba_standings_by_giornata.csv.  When provided, top8/bottom8 fallback
        tiers are sourced from the external standings rather than from
        Trento's win-rates (which cover at most one game per opponent).

    Returns
    -------
    fan DataFrame enriched with per-opponent and sparse-fallback columns.
    """
    out = fan.copy()

    if opponents_df is None or opponents_df.empty:
        log.info("No opponents_df provided – skipping opponent affinity features")
        _add_sparse_fallbacks(out, tc, {}, standings_df=standings_df)
        return out

    if "game_date" not in opponents_df.columns:
        log.warning("opponents_df missing 'game_date' column – skipping opponent affinity")
        _add_sparse_fallbacks(out, tc, {}, standings_df=standings_df)
        return out

    # ── 1. Determine TRAIN game dates (leakage guard) ──────────────
    train_event_dates = set(
        pd.to_datetime(ticket_df["event_dt"], errors="coerce")
        .dt.normalize()
        .dropna()
        .unique()
    )

    # Filter opponents schedule to train games only
    opp = opponents_df.copy()
    opp["_gd_norm"] = pd.to_datetime(opp["game_date"], errors="coerce").dt.normalize()
    train_opp = opp.loc[opp["_gd_norm"].isin(train_event_dates)].copy()

    if train_opp.empty:
        log.info("No opponents schedule rows match train game dates – skipping opponent affinity")
        _add_sparse_fallbacks(out, tc, {})
        return out

    log.info(
        "Opponent affinity: %d train games matched in opponents schedule "
        "(leakage guard: %d total schedule rows excluded)",
        len(train_opp), len(opp) - len(train_opp),
    )

    # ── 2. Build per-game lookup: game_date → (opponent, score_pts, outcome) ──
    train_opp["_opp_norm"] = train_opp["opponent_team"].astype(str).str.strip()
    train_opp["_opp_col"] = train_opp["_opp_norm"].apply(_norm_opponent)

    # Parse scores (Outcome used to assign Trento / opponent points)
    score_col = "Score" if "Score" in train_opp.columns else None
    outcome_col = "Outcome" if "Outcome" in train_opp.columns else None

    if score_col:
        parsed = train_opp.apply(
            lambda r: _trento_pts_and_diff(r[score_col], r[outcome_col] if outcome_col else ""),
            axis=1,
        )
        train_opp["_trento_pts"] = parsed.apply(lambda x: x[0] if x else np.nan)
        train_opp["_point_diff"] = parsed.apply(lambda x: x[1] if x else np.nan)
    else:
        train_opp["_trento_pts"] = np.nan
        train_opp["_point_diff"] = np.nan

    train_opp["_trento_won"] = (
        train_opp[outcome_col].astype(str).str.strip().str.upper() == "W"
        if outcome_col else pd.Series(np.nan, index=train_opp.index)
    )

    # Build a mapping: normalised_game_date → opponent info
    game_info = {}
    for _, row in train_opp.iterrows():
        gd = row["_gd_norm"]
        game_info[gd] = {
            "opp_norm": row["_opp_norm"],
            "opp_col": row["_opp_col"],
            "trento_won": row["_trento_won"],
            "point_diff": row["_point_diff"],
            "abs_diff": abs(row["_point_diff"]) if pd.notna(row["_point_diff"]) else np.nan,
        }

    # ── 3. Join game_info onto tc rows ────────────────────────────
    if tc.empty:
        _add_sparse_fallbacks(out, tc, game_info, standings_df=standings_df)
        return out

    tc2 = tc.copy()
    tc2["_gd_norm"] = tc2["event_dt"].dt.normalize()
    tc2["_opp_col"] = tc2["_gd_norm"].map(lambda d: game_info.get(d, {}).get("opp_col"))
    tc2["_opp_norm"] = tc2["_gd_norm"].map(lambda d: game_info.get(d, {}).get("opp_norm"))
    tc2["_trento_won"] = tc2["_gd_norm"].map(lambda d: game_info.get(d, {}).get("trento_won"))
    tc2["_point_diff"] = tc2["_gd_norm"].map(lambda d: game_info.get(d, {}).get("point_diff"))
    tc2["_abs_diff"] = tc2["_gd_norm"].map(lambda d: game_info.get(d, {}).get("abs_diff"))

    # ── 4. Count home games per opponent in train ─────────────────
    opp_game_counts = (
        train_opp.groupby("_opp_col")["_gd_norm"].nunique()
        .to_dict()
    )

    # Total match games per fan (denominator for pct_games_vs_opponent)
    fan_total_games = tc2.groupby("person_id")["event_dt"].nunique()

    # ── 5. Per-opponent features ──────────────────────────────────
    all_opp_cols = sorted(train_opp["_opp_col"].unique())
    dense_cols = [c for c in all_opp_cols if opp_game_counts.get(c, 0) >= _OPP_SPARSE_THRESHOLD]
    sparse_cols = [c for c in all_opp_cols if opp_game_counts.get(c, 0) < _OPP_SPARSE_THRESHOLD]

    log.info(
        "Opponent affinity: %d opponents total – %d dense (≥%d games), %d sparse",
        len(all_opp_cols), len(dense_cols), _OPP_SPARSE_THRESHOLD, len(sparse_cols),
    )

    # For each dense opponent compute fan-level statistics
    for opp_col in dense_cols:
        rows = tc2.loc[tc2["_opp_col"] == opp_col]

        # pct_games_vs_opponent: games attended vs this opp / total games
        games_vs = rows.groupby("person_id")["event_dt"].nunique()
        pct_vs = games_vs / fan_total_games
        pct_vs = pct_vs.clip(upper=1.0)
        out[f"pct_games_vs_{opp_col}"] = out["person_id"].map(pct_vs).fillna(0.0)

        # win_rate_vs_opponent: fraction of attended games that Trento won
        won_rows = rows[rows["_trento_won"].notna()]
        if not won_rows.empty:
            win_rate = won_rows.groupby("person_id")["_trento_won"].mean()
        else:
            win_rate = pd.Series(dtype=float)
        out[f"win_rate_vs_{opp_col}"] = (
            out["person_id"].map(win_rate).fillna(0.5)  # neutral prior
        )

        # avg_point_diff_vs_opponent
        diff_rows = rows[rows["_point_diff"].notna()]
        if not diff_rows.empty:
            avg_diff = diff_rows.groupby("person_id")["_point_diff"].mean()
        else:
            avg_diff = pd.Series(dtype=float)
        out[f"avg_point_diff_vs_{opp_col}"] = out["person_id"].map(avg_diff).fillna(0.0)

        # avg_abs_point_diff_vs_opponent (intensity)
        abs_rows = rows[rows["_abs_diff"].notna()]
        if not abs_rows.empty:
            avg_abs = abs_rows.groupby("person_id")["_abs_diff"].mean()
        else:
            avg_abs = pd.Series(dtype=float)
        out[f"avg_abs_diff_vs_{opp_col}"] = out["person_id"].map(avg_abs).fillna(np.nan)

    # ── 6. Sparse fallbacks (aggregate groups) ────────────────────
    _add_sparse_fallbacks(out, tc2, game_info, fan_total_games, standings_df=standings_df)

    # Store the set of dense opponent columns for use in game_targeting
    out.attrs["_dense_opp_cols"] = dense_cols
    out.attrs["_game_info"] = game_info  # needed by scoring to look up opp stats

    return out


def _add_sparse_fallbacks(
    out: pd.DataFrame,
    tc: pd.DataFrame,
    game_info: dict,
    fan_total_games: pd.Series | None = None,
    standings_df: pd.DataFrame | None = None,
) -> None:
    """Compute aggregate opponent-group fallback features (in-place).

    When ``standings_df`` is provided the top8/bottom8 tiers are derived from
    the external lba_standings_by_giornata.csv (end-of-season final standings)
    rather than being computed dynamically from Trento's training win-rates.
    The hv_opponents set is always derived from the hardcoded ``_HV_TOKENS_OPP``
    / ``_DERBY_TOKENS_OPP`` tokens (brand/geographic value, not performance).
    """
    if tc.empty or not game_info:
        for col in ["pct_games_vs_top8", "pct_games_vs_bottom8", "pct_games_vs_hv_opponents"]:
            out[col] = 0.0
        return

    # ── Determine top8 / bottom8 ──────────────────────────────────
    if standings_df is not None and not standings_df.empty:
        # PREFERRED: use external LBA standings as authoritative tier source
        top8, bottom8 = _get_standings_tiers(standings_df, game_info, top_n=8)
        if not top8 and not bottom8:
            # _get_standings_tiers already logged a warning; fall through to dynamic
            standings_df = None  # trigger dynamic fallback below

    if standings_df is None or standings_df.empty:
        # FALLBACK: derive from Trento's win-rates against each opponent in training
        opp_overall_win: dict[str, list[bool]] = {}
        for gd, info in game_info.items():
            opp_col = info["opp_col"]
            won = info["trento_won"]
            if pd.notna(won):
                if opp_col not in opp_overall_win:
                    opp_overall_win[opp_col] = []
                opp_overall_win[opp_col].append(bool(won))

        opp_wr = {k: np.mean(v) for k, v in opp_overall_win.items() if v}
        if not opp_wr:
            for col in ["pct_games_vs_top8", "pct_games_vs_bottom8", "pct_games_vs_hv_opponents"]:
                out[col] = 0.0
            return

        sorted_opps = sorted(opp_wr.items(), key=lambda x: x[1], reverse=True)
        top8 = {o for o, _ in sorted_opps[:8]}
        bottom8 = {o for o, _ in sorted_opps[-8:]}

    hv_set = set()
    for gd, info in game_info.items():
        opp_upper = info["opp_norm"].upper()
        if any(t in opp_upper for t in _HV_TOKENS_OPP) or any(t in opp_upper for t in _DERBY_TOKENS_OPP):
            hv_set.add(info["opp_col"])

    if fan_total_games is None:
        fan_total_games = tc.groupby("person_id")["event_dt"].nunique()

    def _pct_group(opp_set: set) -> pd.Series:
        mask = tc["_opp_col"].isin(opp_set) if "_opp_col" in tc.columns else pd.Series(False, index=tc.index)
        grp_games = tc.loc[mask].groupby("person_id")["event_dt"].nunique()
        pct = grp_games / fan_total_games
        return pct.clip(upper=1.0)

    out["pct_games_vs_top8"] = out["person_id"].map(_pct_group(top8)).fillna(0.0)
    out["pct_games_vs_bottom8"] = out["person_id"].map(_pct_group(bottom8)).fillna(0.0)
    out["pct_games_vs_hv_opponents"] = out["person_id"].map(_pct_group(hv_set)).fillna(0.0)

    log.info(
        "Sparse fallbacks: top8=%s, bottom8=%s, hv=%s",
        sorted(top8)[:3], sorted(bottom8)[:3], sorted(hv_set)[:3],
    )
