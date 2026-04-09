"""
Game-driven eligibility scoring and per-game targeting.

Ported from try_3.ipynb cells 75–120:  for a given match_key (date+opponent),
score every fan in the train universe by behavioural fit, rank them, and
produce a target list that excludes subscription holders and prior buyers.

The eligibility score combines (each component clipped to [-2, 2] first):
    +0.40  competition fit        (Eurocup / LBA preference)
    +0.20  recency boost          (recent purchasers rank higher)
    -0.15  price sensitivity      (discount-hunters penalised)
    +0.10  premium affinity       (spending in premium sectors)
    +0.10  evening / weekend      (time-slot preference)
    +0.05  legacy opponent affinity (HV / derby aggregate)
    +0.15  opponent affinity      (fan's historical pct vs that specific opponent)
    +0.05  opponent success       (fan win-rate vs that opponent)
    +0.05  opponent intensity     (normalised inverse abs-point-diff)
    ──────────────────────────────────────────────────────────
    +1.10  max positive  /  +0.95  net (after price penalty)

Opponent-aware components (Task 2)
------------------------------------
The three new terms use per-opponent features computed by
``feature_building._add_opponent_affinity``.  Column names follow the
convention ``pct_games_vs_<opp_col>``, ``win_rate_vs_<opp_col>``,
``avg_abs_diff_vs_<opp_col>``, where ``opp_col`` is the normalised opponent
token stored in ``game_profile["opponent_col"]``.

When no per-opponent column is found (sparse opponent, first season encounter,
future fixture with no history) the function falls back to aggregate columns:
    pct_games_vs_hv_opponents, pct_games_vs_top8, pct_games_vs_bottom8.

All components are clipped to [-2, 2] before weighting.
"""

# =============================================================================
# MODULE OVERVIEW
# =============================================================================
# This module produces a ranked list of fans to target for each specific game.
# The eligibility score has 9 components with different weights (total = 100%):
#
#   competition fit (40%)         — does this fan typically attend this
#                                   competition type (LBA vs Eurocup)?
#   recency (20%)                 — how recently did the fan last buy a ticket?
#   price penalty (-15%)          — discount-hunter fans are penalised because
#                                   they are unlikely to buy at full price
#   premium preference (10%)      — does the fan spend on premium seat sectors?
#   timing preference (10%)       — does the fan usually come to evening / weekend
#                                   games that match this fixture's slot?
#   legacy opponent bonus (5%)    — aggregate affinity for high-value or derby
#                                   opponents (e.g. Olimpia Milano, Virtus Bologna)
#   opponent-specific affinity (15%) — share of this fan's games attended that
#                                      were against THIS specific opponent
#   opponent success rate (5%)    — fan's win-rate when attending games vs this
#                                   opponent (winning games feel rewarding)
#   opponent intensity (5%)       — fans who attend close games (small point
#                                   diff) tend to be more emotionally invested
#
# The weights are domain-expert judgements reflecting that competition type fit
# is the strongest predictor of purchase intent for a given game.
# =============================================================================
from __future__ import annotations

import logging
import re
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── High-value and derby opponents (uppercase tokens) ─────────────
_HV_TOKENS = {"BOLOGNA", "MILANO", "VENEZIA", "VIRTUS", "OLIMPIA", "REYER", "BRESCIA"}
_DERBY_TOKENS = {"BRESCIA", "MILANO", "VENEZIA"}


def _norm_opponent_col(name: str) -> str:
    """Normalise opponent name → safe column suffix (mirrors feature_building)."""
    return re.sub(r"[^a-z0-9]", "_", str(name).strip().lower())

# ── Fixed cluster label for subscription holders ──────────────────
SUBSCRIBER_CLUSTER = "SUBSCRIBER"


_COMP_MAP = {
    "Serie A": "LBA",
    "Serie A (Playoff)": "LBA",
    "EuroCup": "Eurocup",
    "Eurocup": "Eurocup",
}


def _lookup_competition(
    game_date: pd.Timestamp,
    opponents_df: pd.DataFrame | None,
) -> str | None:
    """Look up the competition type from the opponents schedule.

    Maps opponents_csv values → internal names:
      "Serie A" / "Serie A (Playoff)" → "LBA"
      "EuroCup" → "Eurocup"

    Returns None if not found (caller should fall back to ticket data).
    """
    if opponents_df is None or opponents_df.empty:
        return None
    if "competition_type" not in opponents_df.columns:
        return None
    gd = pd.Timestamp(game_date).normalize()
    mask = opponents_df["game_date"].dt.normalize() == gd
    rows = opponents_df.loc[mask]
    if rows.empty:
        return None
    raw_comp = str(rows["competition_type"].iloc[0]).strip()
    mapped = _COMP_MAP.get(raw_comp)
    if mapped is None:
        log.warning("Unknown competition_type '%s' in opponents_csv for %s", raw_comp, gd.date())
    return mapped


def lookup_opponent(
    game_date: pd.Timestamp,
    opponents_df: pd.DataFrame | None,
) -> str | None:
    """Look up the opponent team name for a given game date.

    Parameters
    ----------
    game_date : Timestamp
        Normalised game date.
    opponents_df : DataFrame or None
        Must have ``game_date`` (datetime) and ``opponent_team`` columns.

    Returns
    -------
    str or None  –  opponent team name, or None if not found.
    """
    if opponents_df is None or opponents_df.empty:
        return None
    gd = pd.Timestamp(game_date).normalize()
    mask = opponents_df["game_date"].dt.normalize() == gd
    rows = opponents_df.loc[mask]
    if rows.empty:
        return None
    return rows["opponent_team"].iloc[0].strip()


def _clip(s: pd.Series, lo: float = -2.0, hi: float = 2.0) -> pd.Series:
    return s.clip(lower=lo, upper=hi)


def score_fans_for_game(
    fan_features: pd.DataFrame,
    fan_labels: pd.DataFrame,
    game_profile: dict,
    *,
    subscription_pids: set | None = None,
    already_bought_pids: set | None = None,
    person_lookup: pd.DataFrame | None = None,
    require_consent: bool = True,
    require_email: bool = True,
) -> pd.DataFrame:
    """Score every fan's eligibility for a specific game.

    Parameters
    ----------
    fan_features : DataFrame
        Fan-level features (one row per person_id).
    fan_labels : DataFrame
        ``[person_id, cluster]`` from training.
    game_profile : dict
        Must contain competition, is_weekend, is_evening, is_high_value, is_derby.
    subscription_pids : set, optional
        Person-ids that hold a season pass / pack for this game.
    already_bought_pids : set, optional
        Person-ids that already bought a ticket for this game.
    person_lookup : DataFrame, optional
        Must have person_id, marketing_consent, has_email columns.
    require_consent : bool
        If True, fans without marketing_consent==1 are excluded from targeting.
    require_email : bool
        If True, fans without has_email==True are excluded from targeting.

    Returns
    -------
    DataFrame with columns:
        person_id, cluster, eligibility_score,
        is_subscription, already_bought, no_consent, rank
    sorted by rank ascending (1 = best target).
    """
    subscription_pids = subscription_pids or set()
    already_bought_pids = already_bought_pids or set()

    # ── merge fan features with cluster labels ────────────────────
    # Keep only fans who appear in both feature table and label table
    # (inner join drops any fan that was not clustered during training).
    ff = fan_features.copy()
    fl = fan_labels[["person_id", "cluster"]].drop_duplicates()
    df = ff.merge(fl, on="person_id", how="inner")

    # Extract game-level flags from the pre-built profile dict
    comp = game_profile.get("competition", "LBA")
    is_weekend = game_profile.get("is_weekend", False)
    is_evening = game_profile.get("is_evening", True)
    is_hv = game_profile.get("is_high_value", False)
    is_derby = game_profile.get("is_derby", False)

    # =========================================================================
    # COMPUTE SCORE COMPONENTS
    # Each component is clipped to [-2, 2] before being multiplied by its
    # weight, preventing any single outlier fan from dominating the ranking.
    # =========================================================================

    # ── 1. Competition fit (40%) ──────────────────────────────────
    # Measures how well this game's competition type matches what the fan
    # normally attends. A fan who goes to 80% of Eurocup games scores high for
    # a Eurocup fixture; an LBA-only fan scores low. The 0.3 floor for Eurocup
    # prevents LBA-only fans from being scored as zero (they may still attend).
    if comp == "Eurocup":
        # floor at 0.3 so LBA-only fans are not zero'd out
        comp_fit = 0.3 + 0.7 * df.get("pct_eurocup_games", 0).fillna(0)
    else:
        comp_fit = df.get("pct_lba_games", 0).fillna(0)
    comp_fit = _clip(comp_fit)

    # ── 2. Recency boost (20%) ────────────────────────────────────
    # Measures how recently the fan last attended a game. A fan who came last
    # week scores near the top; a fan last seen 3 years ago scores near zero.
    # This captures "active" fans who are currently engaged with the club.
    recency = df.get("recency_days", pd.Series(np.nan, index=df.index))
    recency = recency.fillna(recency.median() if recency.notna().any() else 180)
    # invert: lower recency = higher boost, normalise to [0, 2]
    rec_max = recency.max() if recency.max() > 0 else 1
    recency_adj = _clip(2.0 * (1.0 - recency / rec_max))

    # ── 3. Price sensitivity penalty (−15%) ───────────────────────
    # Measures what fraction of the fan's past tickets were discounted vs full
    # price. High-discount fans are penalised because they are unlikely to
    # purchase spontaneously at face value — they wait for promotions.
    price_sens = df.get("pct_discounted_vs_list", 0).fillna(0)
    price_sens = _clip(price_sens * 2)      # scale [0,1] → [0,2]

    # ── 4. Premium affinity (10%) ─────────────────────────────────
    # Measures whether the fan buys seats in premium/VIP sectors. High-spending
    # fans who sit in premium areas are more likely to respond to targeted offers
    # and represent higher expected revenue per conversion.
    prem = df.get("premium_affinity", 0).fillna(0)
    prem = _clip(prem * 2)

    # ── 5. Time-slot fit (10%) ────────────────────────────────────
    # Measures whether the fan's past attendance pattern matches this game's
    # time slot. A fan who always attends weekday evenings gets a high score for
    # a Wednesday-night game; a weekend-only fan does not.
    evening_fit = _clip(df.get("pct_evening_games", 0).fillna(0)) if is_evening else 0
    weekend_fit = _clip(df.get("pct_weekend_games", 0).fillna(0)) if is_weekend else 0
    time_fit = 0.6 * evening_fit + 0.4 * weekend_fit

    # ── 6. Legacy opponent affinity (5%) — HV / derby aggregate ──
    # Measures the fan's historical attendance at high-value ("HV") games
    # such as Olimpia Milano or Virtus Bologna, or local derby matches.
    # This uses pre-computed aggregate columns, not per-opponent columns.
    if is_hv:
        opp_fit = _clip(df.get("pct_high_value_opponents", 0).fillna(0) * 2)
    elif is_derby:
        opp_fit = _clip(df.get("derby_attendance_rate", 0).fillna(0) * 2)
    else:
        opp_fit = 0

    # ── 7–9. Opponent-aware components (Task 2) ───────────────────
    # Resolve the normalised opponent column key from game_profile.
    # We accept either a pre-computed "opponent_col" key (set by
    # build_game_profile / build_game_profile_from_schedule) or derive
    # it on-the-fly from "opponent_team".
    opponent_team = game_profile.get("opponent_team") or ""
    opp_col = game_profile.get("opponent_col") or _norm_opponent_col(opponent_team)

    # ── 7. Opponent affinity (15%) ────────────────────────────────
    # Measures the share of this fan's all-time attended games that were against
    # THIS specific opponent. A fan who has seen Virtus Bologna 10 times out of
    # 20 total games scores 0.5 (high) for a Virtus Bologna fixture.
    # Use per-opponent column when available, fall back to aggregate groups.
    pct_col = f"pct_games_vs_{opp_col}"
    if pct_col in df.columns:
        opp_affinity = _clip(df[pct_col].fillna(0) * 2)
    elif is_hv:
        # fallback: high-value aggregate
        opp_affinity = _clip(df.get("pct_games_vs_hv_opponents", 0).fillna(0) * 2)
    else:
        # fallback: top8 aggregate
        opp_affinity = _clip(df.get("pct_games_vs_top8", 0).fillna(0) * 2)

    # ── 8. Opponent success (5%) ──────────────────────────────────
    # Measures the fan's win-rate when attending games against this opponent.
    # Fans who have seen their team beat this opponent tend to associate the
    # matchup with positive memories, boosting their likelihood of attending.
    # Fan's win-rate vs this opponent (neutral prior 0.5 → score 0).
    win_col = f"win_rate_vs_{opp_col}"
    if win_col in df.columns:
        # win_rate ∈ [0,1]; centre around 0.5, scale to [-1, +1], clip
        opp_success = _clip((df[win_col].fillna(0.5) - 0.5) * 4)
    else:
        opp_success = pd.Series(0.0, index=df.index)

    # ── 9. Opponent intensity (5%) ────────────────────────────────
    # Measures how close (point-difference-wise) the games were when this fan
    # attended them against this opponent. Fans who attend close, tense games
    # tend to be the most engaged; a large point diff suggests a blowout that
    # may not attract them back for a rematch.
    # Normalise: lower abs_diff → higher score.
    abs_col = f"avg_abs_diff_vs_{opp_col}"
    if abs_col in df.columns:
        abs_diff = df[abs_col].fillna(np.nan)
        # if no data at all default to neutral 0
        if abs_diff.notna().any():
            abs_max = abs_diff.quantile(0.95) if abs_diff.notna().sum() >= 5 else abs_diff.max()
            abs_max = abs_max if (abs_max and abs_max > 0) else 1.0
            # invert: small diff → high score → scale to [0, 2]
            opp_intensity = _clip(2.0 * (1.0 - abs_diff.clip(upper=abs_max) / abs_max))
        else:
            opp_intensity = pd.Series(0.0, index=df.index)
    else:
        opp_intensity = pd.Series(0.0, index=df.index)

    # ── composite score ───────────────────────────────────────────
    score = (
        0.40 * comp_fit
        + 0.20 * recency_adj
        - 0.15 * price_sens
        + 0.10 * prem
        + 0.10 * time_fit
        + 0.05 * opp_fit            # legacy HV/derby aggregate
        + 0.15 * opp_affinity       # per-opponent attendance share
        + 0.05 * opp_success        # per-opponent win rate
        + 0.05 * opp_intensity      # per-opponent game closeness
    )

    # ── deterministic tie-breaker ────────────────────────────────
    # Add a tiny signal from games_attended (scaled to ~1e-4 range)
    # so fans with higher frequency rank higher within score ties.
    # Final fallback: hash of person_id for reproducibility.
    ga = df.get("games_attended", pd.Series(0, index=df.index)).fillna(0)
    ga_max = ga.max() if ga.max() > 0 else 1
    tiebreak = 1e-4 * (ga / ga_max)

    # sub-tiebreak: stable hash of person_id → [0, 1e-8]
    pid_hash = df["person_id"].apply(lambda x: int(x[:8], 16) / 0xFFFFFFFF * 1e-8)
    score = score + tiebreak + pid_hash

    df["eligibility_score"] = score

    # =========================================================================
    # APPLY CONSENT / EMAIL FILTER
    # Consent filtering is done AFTER scoring so that all fans receive a score
    # (useful for analysis). Fans without consent or email are flagged and
    # excluded from the final rank rather than dropped entirely.
    # =========================================================================

    # ── flags ─────────────────────────────────────────────────────
    df["is_subscription"] = df["person_id"].isin(subscription_pids)
    df["already_bought"] = df["person_id"].isin(already_bought_pids)

    # ── consent + email flags ─────────────────────────────────────
    df["marketing_consent"] = 0
    df["has_email"] = False
    if person_lookup is not None:
        pl = person_lookup[["person_id", "marketing_consent", "has_email"]].drop_duplicates("person_id")
        df = df.merge(pl, on="person_id", how="left", suffixes=("_drop", ""))
        # clean up any duplicate columns from merge
        for c in list(df.columns):
            if c.endswith("_drop"):
                df = df.drop(columns=[c])
        df["marketing_consent"] = df["marketing_consent"].fillna(0).astype(int)
        df["has_email"] = df["has_email"].fillna(False).astype(bool)

    # ── build no_consent flag ─────────────────────────────────────
    no_consent = pd.Series(False, index=df.index)
    if require_consent:
        no_consent = no_consent | (df["marketing_consent"] != 1)
    if require_email:
        no_consent = no_consent | (~df["has_email"])
    df["no_consent"] = no_consent

    # =========================================================================
    # RANK FANS
    # Only fans who are not subscription holders, have not already bought a
    # ticket, and pass the consent/email check receive a final rank.
    # All others are assigned NA rank and appear at the bottom of the output.
    # =========================================================================

    # ── rank (exclude subscription + already-bought + no-consent) ──
    targetable = ~df["is_subscription"] & ~df["already_bought"] & ~df["no_consent"]
    df["rank"] = pd.array([pd.NA] * len(df), dtype="Int64")
    if targetable.any():
        df.loc[targetable, "rank"] = (
            df.loc[targetable, "eligibility_score"]
            .rank(ascending=False, method="first")
            .astype("Int64")
        )

    df = df.sort_values("rank", na_position="last")

    n_no_consent = df["no_consent"].sum()
    log.info(
        "Game scoring [%s]: %d fans scored, %d subscription, %d already bought, "
        "%d no consent/email, %d targetable",
        game_profile.get("match_key", "?"),
        len(df),
        df["is_subscription"].sum(),
        df["already_bought"].sum(),
        n_no_consent,
        targetable.sum(),
    )

    return df[
        ["person_id", "cluster", "eligibility_score",
         "is_subscription", "already_bought", "no_consent",
         "marketing_consent", "has_email", "rank"]
    ].reset_index(drop=True)


# ── helpers to build game_profile from ticket data ────────────────

def build_game_profile(
    tickets_df: pd.DataFrame,
    match_key: str,
    opponents_df: pd.DataFrame | None = None,
) -> dict:
    """Build a game_profile dict from the tickets for a specific match_key.

    Assembles all game metadata (competition type, opponent, is_weekend,
    is_evening, is_high_value, is_derby) into a single dict that is consumed
    by score_fans_for_game(). Uses historical ticket rows for the given date
    to infer game attributes; the opponents schedule overrides competition type
    when available because ticket data labels all games as "LBA" regardless.

    Parameters
    ----------
    tickets_df : DataFrame
        Cleaned ticket data (must have event_dt, competition_type, event_title).
    match_key : str
        Format ``"YYYY-MM-DD"`` or ``"YYYY-MM-DD | Opponent"``.
    opponents_df : DataFrame, optional
        Opponents schedule (must have game_date, opponent_team).

    Returns
    -------
    dict suitable for :func:`score_fans_for_game`.
    """
    date_str = match_key.split("|")[0].strip()
    game_date = pd.Timestamp(date_str).normalize()

    # look up opponent from schedule
    opponent = lookup_opponent(game_date, opponents_df)

    # ── competition type: prefer opponents_df (ground truth) over tickets
    # Ticket data labels ALL games as "LBA"; the opponents schedule has
    # the correct split of Serie A vs EuroCup.
    opp_comp = _lookup_competition(game_date, opponents_df)

    mask = tickets_df["event_dt"].dt.normalize() == game_date
    game_rows = tickets_df.loc[mask]

    if game_rows.empty:
        comp = opp_comp if opp_comp else "LBA"
        log.warning("No rows found for match_key='%s', using comp=%s", match_key, comp)
        return {
            "match_key": match_key,
            "game_date": game_date,
            "competition": comp,
            "opponent_team": opponent,
            "opponent_col": _norm_opponent_col(opponent) if opponent else "",
            "is_weekend": game_date.dayofweek >= 5,
            "is_evening": True,
            "is_high_value": False,
            "is_derby": False,
        }

    # Use opponents_df competition as source of truth; fall back to ticket mode
    if opp_comp:
        comp = opp_comp
    else:
        comp_mode = game_rows["competition_type"].mode()
        comp = comp_mode.iloc[0] if len(comp_mode) > 0 else "LBA"
        log.warning(
            "Competition for %s not found in opponents_csv, "
            "falling back to ticket mode: %s", match_key, comp,
        )

    title = game_rows.get("event_title", pd.Series(dtype=str)).iloc[0]
    title_upper = str(title).upper() if pd.notna(title) else ""

    # check opponent tokens in both the event title and the opponent name
    opp_upper = opponent.upper() if opponent else ""
    check_str = title_upper + " " + opp_upper

    is_hv = any(tok in check_str for tok in _HV_TOKENS)
    is_derby = any(tok in check_str for tok in _DERBY_TOKENS)

    hour = game_rows["event_dt"].dt.hour.mode()
    is_evening = (hour.iloc[0] >= 18) if len(hour) > 0 else True

    return {
        "match_key": match_key,
        "game_date": game_date,
        "competition": comp,
        "opponent_team": opponent,
        "opponent_col": _norm_opponent_col(opponent) if opponent else "",
        "is_weekend": game_date.dayofweek >= 5,
        "is_evening": is_evening,
        "is_high_value": is_hv,
        "is_derby": is_derby,
    }


def build_game_profile_from_schedule(
    fixture_row: pd.Series,
    opponents_df: pd.DataFrame | None = None,
) -> dict:
    """Build a game_profile dict from the opponents schedule only (no tickets).

    Used for FUTURE games where no ticket data exists yet — the game has not
    been played, so there are no ticket rows to inspect. All attributes
    (competition type, is_evening, is_high_value, etc.) are derived solely
    from the opponents schedule CSV row. Sets is_future_fixture=True so that
    downstream code can distinguish this path from the historical path.

    Parameters
    ----------
    fixture_row : Series
        A single row from the opponents schedule DataFrame.
        Must have game_date, opponent_team, competition_type.
    opponents_df : DataFrame, optional
        Full opponents schedule (used for opponent lookup).

    Returns
    -------
    dict suitable for :func:`score_fans_for_game`.
    """
    game_date = pd.Timestamp(fixture_row["game_date"]).normalize()
    opponent = str(fixture_row.get("opponent_team", "")).strip() or None

    # competition type from schedule
    raw_comp = str(fixture_row.get("competition_type", "")).strip()
    comp = _COMP_MAP.get(raw_comp, "LBA")

    # time parsing – "TBD" or actual time
    raw_time = str(fixture_row.get("Time", "")).strip()
    is_evening = True  # default for unknown time
    if raw_time and raw_time.upper() != "TBD":
        try:
            hour = int(raw_time.split(":")[0])
            is_evening = hour >= 18
        except (ValueError, IndexError):
            pass

    # opponent tokens for HV / derby detection
    opp_upper = opponent.upper() if opponent else ""
    is_hv = any(tok in opp_upper for tok in _HV_TOKENS)
    is_derby = any(tok in opp_upper for tok in _DERBY_TOKENS)

    match_key = str(game_date.date())
    if opponent:
        match_key = f"{game_date.date()} | {opponent}"

    profile = {
        "match_key": match_key,
        "game_date": game_date,
        "competition": comp,
        "opponent_team": opponent,
        "opponent_col": _norm_opponent_col(opponent) if opponent else "",
        "is_weekend": game_date.dayofweek >= 5,
        "is_evening": is_evening,
        "is_high_value": is_hv,
        "is_derby": is_derby,
        "is_future_fixture": True,
    }

    log.info(
        "Future fixture profile: %s | comp=%s | weekend=%s | evening=%s | HV=%s | derby=%s",
        match_key, comp, profile["is_weekend"], is_evening, is_hv, is_derby,
    )
    return profile


def extract_future_fixtures(
    opponents_df: pd.DataFrame,
    after_date: pd.Timestamp | str | None = None,
    max_n: int | None = None,
    only_competitions: list[str] | None = None,
    only_opponents: list[str] | None = None,
) -> pd.DataFrame:
    """Extract future (unplayed) fixtures from the opponents schedule.

    "Unplayed" means: the game's Status column is not "PLAYED" and its
    Outcome column is empty. A game is also excluded if its date falls on or
    before after_date (defaults to today). These two checks together ensure
    that a game is only returned when it has neither been completed nor
    had its result recorded — i.e. we cannot yet look up attendance data.

    Parameters
    ----------
    opponents_df : DataFrame
        Loaded opponents schedule (from ``load_opponents``).
    after_date : Timestamp or str, optional
        Only include fixtures after this date. If None, uses today.
    max_n : int, optional
        Maximum number of future fixtures to return.
    only_competitions : list of str, optional
        Filter to these competition types (e.g. ["Serie A", "EuroCup"]).
    only_opponents : list of str, optional
        Filter to these opponent names (case-insensitive substring match).

    Returns
    -------
    DataFrame of future fixture rows, sorted by game_date ascending.
    """
    if opponents_df is None or opponents_df.empty:
        log.warning("No opponents schedule provided; cannot extract future fixtures")
        return pd.DataFrame()

    df = opponents_df.copy()

    # Filter: not played (Status is blank, NaN, or not "Played")
    if "Status" in df.columns:
        status = df["Status"].fillna("").astype(str).str.strip()
        df = df.loc[status.str.upper() != "PLAYED"]
    if "Outcome" in df.columns:
        # also exclude rows with an outcome (they were played)
        outcome = df["Outcome"].fillna("").astype(str).str.strip()
        df = df.loc[outcome == ""]

    # Filter: after_date
    if after_date is not None:
        cutoff = pd.Timestamp(after_date).normalize()
    else:
        cutoff = pd.Timestamp("today").normalize()

    if "game_date" in df.columns:
        df = df.loc[df["game_date"].dt.normalize() > cutoff]
    else:
        log.warning("No game_date column in opponents schedule")
        return pd.DataFrame()

    # Filter: competitions
    if only_competitions and "competition_type" in df.columns:
        comp_set = {c.strip().upper() for c in only_competitions}
        df = df.loc[df["competition_type"].str.strip().str.upper().isin(comp_set)]

    # Filter: opponents (substring match, case-insensitive)
    if only_opponents and "opponent_team" in df.columns:
        opp_patterns = [o.strip().upper() for o in only_opponents]
        opp_col = df["opponent_team"].fillna("").str.strip().str.upper()
        mask = pd.Series(False, index=df.index)
        for pat in opp_patterns:
            mask = mask | opp_col.str.contains(pat, na=False)
        df = df.loc[mask]

    df = df.sort_values("game_date").reset_index(drop=True)

    # Limit
    if max_n is not None and max_n > 0:
        df = df.head(max_n)

    log.info(
        "Future fixtures extracted: %d games (after %s)",
        len(df), cutoff.date(),
    )
    for _, row in df.iterrows():
        log.info(
            "  %s | %s | %s",
            pd.Timestamp(row["game_date"]).date(),
            row.get("opponent_team", "?"),
            row.get("competition_type", "?"),
        )
    return df


def identify_subscription_holders(
    tickets_df: pd.DataFrame,
    game_date: pd.Timestamp | None = None,
) -> set:
    """Return set of person_ids who hold a Pack or Abbonamento."""
    # Subscribers attend most games already — targeting them wastes budget.
    # They are excluded entirely from the target list for every game.
    mask = tickets_df["competition_type"].isin(["Abbonamento", "Pack"])
    subs = tickets_df.loc[mask]

    if game_date is not None:
        move = subs["movement_dt"].dt.normalize()
        subs = subs.loc[move <= game_date]

    pids = set(subs["person_id"].unique())
    log.info("Subscription holders: %d (game_date=%s)", len(pids),
             game_date.date() if game_date is not None else "all")
    return pids


def identify_already_bought(
    tickets_df: pd.DataFrame,
    game_date: pd.Timestamp,
) -> set:
    """Return person_ids who already bought a ticket for a specific game date."""
    # Prevents sending promotions to fans who already have a ticket —
    # they would receive a redundant (and potentially confusing) marketing
    # message for a game they are already planning to attend.
    mask = (
        tickets_df["event_dt"].dt.normalize().eq(game_date)
        & tickets_df["competition_type"].isin(["LBA", "Eurocup"])
    )
    pids = set(tickets_df.loc[mask, "person_id"].unique())
    log.info("Already bought for %s: %d fans", game_date.date(), len(pids))
    return pids
