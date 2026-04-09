"""
Package generator: produce per-game marketing outputs.

For a given match_key, generates:
  1. target_list_<match_key>.csv   – ranked fan targets (excluding subs + already bought + no consent)
  2. bought_already_<match_key>.csv – fans who already have tickets (optional)
  3. summary_<match_key>.json      – aggregate stats for the marketing team
  4. game_index.csv                – one row per game with key metadata
"""

# =============================================================================
# MODULE OVERVIEW
# =============================================================================
# This module creates the actual marketing deliverable. For each game it
# produces three output files:
#
#   (1) target_list.csv   — all targetable fans ranked by eligibility score,
#                           with email address and consent status included so
#                           the marketing team can immediately send campaigns
#
#   (2) high_intent.csv   — top-scoring fans who do NOT yet have marketing
#                           consent or a resolved email. These are candidates
#                           for a premium outreach effort (e.g. direct call or
#                           social media re-targeting) to obtain consent first.
#
#   (3) summary.json      — campaign statistics for the marketing team,
#                           including number of targetable fans, breakdown by
#                           consent status, top clusters, and average score.
#
# Two entry points:
#   generate_all_test_packages()  — runs over all held-out TEST games; used
#                                   during model evaluation to measure how well
#                                   the ranking predicts actual purchases.
#   generate_future_packages()    — runs over UPCOMING games (no ticket data);
#                                   this is the operational / production path.
# =============================================================================
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .game_targeting import (
    build_game_profile,
    build_game_profile_from_schedule,
    extract_future_fixtures,
    identify_already_bought,
    identify_subscription_holders,
    lookup_opponent,
    score_fans_for_game,
)

log = logging.getLogger(__name__)


def _sanitise_key(match_key: str) -> str:
    """Convert match_key to a filesystem-safe string."""
    return match_key.replace(" ", "").replace("|", "_").replace("/", "-")


def _dedup_by_email(
    targets_df: pd.DataFrame,
    person_lookup: "pd.DataFrame | None",
) -> pd.DataFrame:
    """Collapse multiple person_ids that share the same email to one row.

    Multiple person_ids can share an email if identity resolution assigned
    different IDs to the same physical person (e.g. they registered with a
    slightly different spelling of their name on a second purchase). Keep only
    the best-scored entry per unique email address so that the same fan does
    not receive duplicate marketing emails and does not inflate the top-N count.

    When identity resolution creates separate hashes for the same physical fan
    (e.g. same email registered under slightly different names), that fan can
    occupy several slots in the top-N target list.  This function keeps only
    the entry with the best rank (lowest rank value) per unique email address,
    and then re-sorts so the output is still rank-ordered.

    Fans with no resolved email are kept as-is (no deduplication possible).
    Must be called *before* the ``head(top_n)`` slice so that the final list
    contains top-N *unique* fans.
    """
    if person_lookup is None or "buyer_email" not in person_lookup.columns:
        return targets_df

    email_map = (
        person_lookup.dropna(subset=["buyer_email"])
        .set_index("person_id")["buyer_email"]
    )
    emails = targets_df["person_id"].map(email_map)
    has_email = emails.notna() & (emails.astype(str).str.strip() != "")

    if not has_email.any():
        return targets_df

    tmp = targets_df.copy()
    tmp["_dedup_email"] = emails

    # Keep the best-ranked row per email (targets are already sorted by rank asc)
    with_email = tmp.loc[has_email].drop_duplicates("_dedup_email", keep="first")
    without_email = tmp.loc[~has_email]

    result = (
        pd.concat([with_email, without_email])
        .sort_values("rank")
        .drop(columns=["_dedup_email"])
        .reset_index(drop=True)
    )

    n_removed = len(targets_df) - len(result)
    if n_removed > 0:
        log.info(
            "Email deduplication: removed %d duplicate person_id rows "
            "(%d unique fans retained in ranked pool)",
            n_removed, len(result),
        )
    return result


def _build_opponent_label(game_date: pd.Timestamp, opponent: str | None) -> str:
    """Build a human-readable game label: 'YYYY-MM-DD_vs_Opponent'."""
    date_str = str(pd.Timestamp(game_date).date())
    if opponent:
        opp_safe = opponent.strip().replace(" ", "_").replace("/", "-")
        return f"{date_str}_vs_{opp_safe}"
    return date_str


def build_person_lookup(
    tickets_df: pd.DataFrame,
    consent_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a person-level lookup table with buyer_email, consent, and names.

    Email resolution hierarchy (buyer-based):
      1. For each person_id, collect all non-null buyer_email values seen across
         all their ticket rows. Pick the most frequently occurring one.
         Tie-break: take the email associated with the most recent movement_dt.
         This ensures that if a fan updated their email address, the current
         one is used rather than an old one.
      2. Join to consent_df on the resolved email (normalised to lowercase) to
         attach marketing_consent, nome, and cognome. The consent file is the
         source of truth for consent status — a fan without a match in consent_df
         is treated as consent=0 (opt-out by default).

    Parameters
    ----------
    tickets_df : DataFrame
        Cleaned ticket-level data with person_id, buyer_email, movement_dt.
    consent_df : DataFrame, optional
        Output of ``load_marketing_consent()``.
        Columns: email, nome, cognome, marketing_consent.

    Returns
    -------
    DataFrame with columns:
        person_id, buyer_email, nome, cognome, marketing_consent, has_email
    """
    cols_needed = ["person_id", "buyer_email", "movement_dt"]
    avail = [c for c in cols_needed if c in tickets_df.columns]
    df = tickets_df[avail].copy()

    # normalise buyer_email
    if "buyer_email" in df.columns:
        df["buyer_email"] = df["buyer_email"].astype(str).str.strip().str.lower()
        df.loc[df["buyer_email"].isin(["", "nan", "none"]), "buyer_email"] = pd.NA
    else:
        df["buyer_email"] = pd.NA

    if "movement_dt" in df.columns:
        df["movement_dt"] = pd.to_datetime(df["movement_dt"], errors="coerce")

    # ── resolve one email per person_id ────────────────────────────
    # Rule: most frequent non-null buyer_email; tie-break by most recent
    valid = df.dropna(subset=["buyer_email"]).copy()

    if len(valid) > 0 and "movement_dt" in valid.columns:
        # count frequency per (person_id, buyer_email) and get latest date
        agg = (
            valid.groupby(["person_id", "buyer_email"])
            .agg(freq=("buyer_email", "size"), latest=("movement_dt", "max"))
            .reset_index()
        )
        # sort: highest freq first, then latest date
        agg = agg.sort_values(["person_id", "freq", "latest"], ascending=[True, False, False])
        best_email = agg.drop_duplicates("person_id", keep="first")[["person_id", "buyer_email"]]
    elif len(valid) > 0:
        best_email = valid.groupby("person_id")["buyer_email"].agg(
            lambda s: s.value_counts().index[0]
        ).reset_index()
    else:
        best_email = pd.DataFrame(columns=["person_id", "buyer_email"])

    # start with all person_ids
    all_pids = pd.DataFrame({"person_id": tickets_df["person_id"].unique()})
    lookup = all_pids.merge(best_email, on="person_id", how="left")

    # has_email flag
    lookup["has_email"] = lookup["buyer_email"].notna()

    # ── merge consent + names from consent master ──────────────────
    if consent_df is not None:
        # consent_df has: email, nome, cognome, marketing_consent
        cons = consent_df.copy()
        cons["email"] = cons["email"].astype(str).str.strip().str.lower()

        lookup = lookup.merge(
            cons.rename(columns={"email": "buyer_email"}),
            on="buyer_email",
            how="left",
        )
        # fans with no match in consent file → consent=0
        lookup["marketing_consent"] = lookup["marketing_consent"].fillna(0).astype(int)
    else:
        lookup["nome"] = pd.NA
        lookup["cognome"] = pd.NA
        lookup["marketing_consent"] = 0

    n_total = len(lookup)
    n_email = lookup["has_email"].sum()
    n_consent = (lookup["marketing_consent"] == 1).sum()
    n_name = lookup["cognome"].notna().sum()

    log.info(
        "Person lookup: %d fans, %d with email (%.1f%%), "
        "%d with marketing consent (%.1f%%), %d with name (%.1f%%)",
        n_total, n_email, 100 * n_email / max(1, n_total),
        n_consent, 100 * n_consent / max(1, n_total),
        n_name, 100 * n_name / max(1, n_total),
    )
    return lookup[["person_id", "buyer_email", "nome", "cognome",
                    "marketing_consent", "has_email"]].reset_index(drop=True)


def generate_game_package(
    match_key: str,
    tickets_df: pd.DataFrame,
    fan_features: pd.DataFrame,
    fan_labels: pd.DataFrame,
    cluster_propensity: pd.Series | None = None,
    opponents_df: pd.DataFrame | None = None,
    person_lookup: pd.DataFrame | None = None,
    out_dir: Path | str | None = None,
    top_n: int | None = None,
    export_bought_already: bool = False,
    require_consent: bool = True,
    require_email: bool = True,
    propensity_mode: str = "overall",
) -> dict:
    """Generate the full marketing package for one game.

    Parameters
    ----------
    match_key : str
        Format ``"YYYY-MM-DD"`` or ``"YYYY-MM-DD | Opponent"``.
    require_consent : bool
        If True, only fans with marketing_consent==1 are targetable.
    require_email : bool
        If True, only fans with has_email==True are targetable.
    propensity_mode : str
        "overall" or "by_competition" — logged in summary for audit.
    """
    # ── STEP 1: build game profile ────────────────────────────────
    # Assemble competition type, opponent, is_weekend, is_evening, etc.
    # from the ticket rows matching this match_key. The opponents schedule
    # overrides competition type when present (ticket data labels everything LBA).
    game_profile = build_game_profile(tickets_df, match_key, opponents_df)
    game_date = game_profile.get("game_date", pd.Timestamp(match_key.split("|")[0].strip()))
    opponent = game_profile.get("opponent_team")

    # build filename label
    file_label = _build_opponent_label(game_date, opponent)

    # ── STEP 2: identify fans to exclude ─────────────────────────
    # Subscription holders and fans who already bought a ticket are removed
    # before scoring so that no marketing budget is wasted on them.
    sub_pids = identify_subscription_holders(tickets_df, game_date)
    bought_pids = identify_already_bought(tickets_df, game_date)

    # ── STEP 3: score ALL fans (consent filtering deferred) ───────
    # Consent is a communication constraint, not a prediction constraint.
    # Score everyone first; split into ready_to_send vs high_intent AFTER.
    # This means we know the propensity of every fan even if we cannot yet
    # legally email them — useful for consent acquisition campaigns.
    scored = score_fans_for_game(
        fan_features, fan_labels, game_profile,
        subscription_pids=sub_pids,
        already_bought_pids=bought_pids,
        person_lookup=person_lookup,
        require_consent=False,
        require_email=False,
    )

    # ── STEP 4: split by consent / email availability ─────────────
    # ranked = all non-sub, non-bought fans (everyone got a rank)
    # targets       = fans with consent AND a known email → ready to send
    # high_intent_df = fans with high score but no consent/email → flag for
    #                  consent-acquisition outreach before the game
    ranked = scored.loc[scored["rank"].notna()].sort_values("rank")
    ready_mask = (ranked["marketing_consent"] == 1) & (ranked["has_email"] == True)
    targets = ranked.loc[ready_mask].copy()
    high_intent_df = ranked.loc[~ready_mask].copy().sort_values(
        "eligibility_score", ascending=False
    )
    # Deduplicate by email before slicing — same physical fan may have multiple
    # person_ids from partial identity resolution
    targets = _dedup_by_email(targets, person_lookup)
    if top_n is not None and top_n > 0:
        targets = targets.head(top_n)
        high_intent_df = high_intent_df.head(top_n)

    # ── STEP 5: merge enrichment columns for output files ─────────
    # Attach behavioural features (spend, recency, sector, etc.) and contact
    # info (email, name) so the exported CSVs are self-contained for the
    # marketing team without needing to cross-reference other tables.

    # merge useful fan features for both lists
    merge_cols = ["person_id"]
    extra_cols = ["games_attended", "total_spend", "recency_days",
                  "pct_lba_games", "pct_eurocup_games", "most_common_sector",
                  "province_mode", "age_mode"]
    for c in extra_cols:
        if c in fan_features.columns:
            merge_cols.append(c)
    feat_lookup = fan_features[merge_cols].drop_duplicates("person_id")
    target_list = targets.merge(feat_lookup, on="person_id", how="left")
    high_intent_list = high_intent_df.merge(feat_lookup, on="person_id", how="left")

    # merge person contact info (buyer_email, nome, cognome) into both lists
    if person_lookup is not None:
        candidate_cols = ["person_id", "buyer_email", "nome", "cognome",
                          "marketing_consent", "has_email"]
        # target_list
        lookup_cols = [c for c in candidate_cols
                       if c in person_lookup.columns
                       and (c == "person_id" or c not in target_list.columns)]
        if len(lookup_cols) > 1:
            target_list = target_list.merge(
                person_lookup[lookup_cols], on="person_id", how="left",
            )
        # high_intent_list
        lookup_cols_hi = [c for c in candidate_cols
                          if c in person_lookup.columns
                          and (c == "person_id" or c not in high_intent_list.columns)]
        if len(lookup_cols_hi) > 1:
            high_intent_list = high_intent_list.merge(
                person_lookup[lookup_cols_hi], on="person_id", how="left",
            )

    # ── already bought list ───────────────────────────────────────
    bought_df = scored.loc[scored["already_bought"]].copy()
    if person_lookup is not None:
        bought_cols = [c for c in ["person_id", "buyer_email", "nome", "cognome"]
                       if c in person_lookup.columns]
        bought_df = bought_df.merge(
            person_lookup[bought_cols],
            on="person_id", how="left",
        )

    # ── summary ───────────────────────────────────────────────────
    n_consent_in_tl = (
        int(target_list["marketing_consent"].sum())
        if "marketing_consent" in target_list.columns else None
    )
    # fans without consent/email who still scored highly
    n_hi_consent = int(len(high_intent_list))
    top10_hi = (
        high_intent_list[["person_id", "eligibility_score"]]
        .head(10)
        .to_dict(orient="records")
        if len(high_intent_list) > 0 else []
    )
    # n_no_consent = fans who have a rank but lack consent / email
    n_no_consent_total = int(len(ranked)) - int(len(targets))

    summary = {
        "match_key": match_key,
        "game_date": str(game_date.date()) if pd.notna(game_date) else None,
        "opponent_team": opponent,
        "competition": game_profile.get("competition", "?"),
        "is_weekend": game_profile.get("is_weekend", False),
        "is_evening": game_profile.get("is_evening", True),
        "is_high_value": game_profile.get("is_high_value", False),
        "is_derby": game_profile.get("is_derby", False),
        "n_total_fans": len(scored),
        "n_subscription_holders": int(scored["is_subscription"].sum()),
        "n_already_bought": int(scored["already_bought"].sum()),
        "n_no_consent": n_no_consent_total,
        "n_targetable": int(len(ranked)),
        "n_in_target_list": len(target_list),
        "n_with_email": int(target_list["has_email"].sum()) if "has_email" in target_list.columns else None,
        "n_with_consent": n_consent_in_tl,
        "n_high_intent_no_consent": n_hi_consent,
        "top10_high_intent": top10_hi,
        "require_consent": require_consent,
        "require_email": require_email,
        "propensity_mode_used": propensity_mode,
        "mean_eligibility_score": round(float(targets["eligibility_score"].mean()), 4)
            if len(targets) > 0 else None,
        "median_eligibility_score": round(float(targets["eligibility_score"].median()), 4)
            if len(targets) > 0 else None,
    }

    # add cluster distribution of targets
    if len(target_list) > 0:
        top_clusters = (
            target_list.head(min(200, len(target_list)))
            .groupby("cluster").size()
            .sort_values(ascending=False)
            .head(5)
        )
        summary["top_clusters_in_top200"] = {
            str(k): int(v) for k, v in top_clusters.items()
        }

    # ── write outputs ─────────────────────────────────────────────
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        target_path = out_dir / f"target_list_{file_label}.csv"
        target_list.to_csv(target_path, index=False)
        log.info("Wrote %s (%d rows)", target_path.name, len(target_list))

        hi_path = out_dir / f"high_intent_{file_label}.csv"
        high_intent_list.to_csv(hi_path, index=False)
        log.info("Wrote %s (%d rows)", hi_path.name, len(high_intent_list))

        if export_bought_already:
            bought_path = out_dir / f"bought_already_{file_label}.csv"
            bought_df.to_csv(bought_path, index=False)
            log.info("Wrote %s (%d rows)", bought_path.name, len(bought_df))

        summary_path = out_dir / f"summary_{file_label}.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        log.info("Wrote %s", summary_path.name)

    return {
        "target_list": target_list,
        "high_intent_no_consent": high_intent_list,
        "bought_already": bought_df,
        "summary": summary,
        "game_profile": game_profile,
    }


def generate_all_test_packages(
    test_games: list[pd.Timestamp],
    tickets_df: pd.DataFrame,
    fan_features: pd.DataFrame,
    fan_labels: pd.DataFrame,
    cluster_propensity: pd.Series | None = None,
    opponents_df: pd.DataFrame | None = None,
    person_lookup: pd.DataFrame | None = None,
    out_dir: Path | str | None = None,
    top_n: int | None = None,
    export_bought_already: bool = False,
    require_consent: bool = True,
    require_email: bool = True,
    propensity_mode: str = "overall",
) -> list[dict]:
    """Generate marketing packages for all test games.

    Iterates over every game date in the held-out test split and calls
    generate_game_package() for each one. Produces one set of output files
    per game (target_list, high_intent, summary JSON) plus a consolidated
    game_index.csv in out_dir that lists all test games with key metrics in
    a single table — used to compare campaign statistics across games.

    Also writes a ``game_index.csv`` summary of all games.
    """
    results = []
    index_rows = []

    for gd in sorted(test_games):
        match_key = str(pd.Timestamp(gd).date())
        pkg = generate_game_package(
            match_key=match_key,
            tickets_df=tickets_df,
            fan_features=fan_features,
            fan_labels=fan_labels,
            cluster_propensity=cluster_propensity,
            opponents_df=opponents_df,
            person_lookup=person_lookup,
            out_dir=out_dir,
            top_n=top_n,
            export_bought_already=export_bought_already,
            require_consent=require_consent,
            require_email=require_email,
            propensity_mode=propensity_mode,
        )
        results.append(pkg)

        # collect index row
        s = pkg["summary"]
        gp = pkg["game_profile"]
        opponent = gp.get("opponent_team")
        file_label = _build_opponent_label(gd, opponent)
        index_rows.append({
            "game_date": s.get("game_date"),
            "opponent_team": opponent or "",
            "competition": s.get("competition"),
            "is_weekend": s.get("is_weekend"),
            "is_evening": s.get("is_evening"),
            "is_high_value": gp.get("is_high_value", False),
            "is_derby": gp.get("is_derby", False),
            "n_subscription_holders": s.get("n_subscription_holders"),
            "n_already_bought": s.get("n_already_bought"),
            "n_no_consent": s.get("n_no_consent"),
            "n_targetable": s.get("n_targetable"),
            "n_in_target_list": s.get("n_in_target_list"),
            "n_with_email": s.get("n_with_email"),
            "mean_eligibility_score": s.get("mean_eligibility_score"),
            "target_list_file": f"target_list_{file_label}.csv",
            "summary_file": f"summary_{file_label}.json",
        })

    # write game_index.csv
    if out_dir is not None:
        idx_df = pd.DataFrame(index_rows)
        idx_path = Path(out_dir) / "game_index.csv"
        idx_df.to_csv(idx_path, index=False)
        log.info("Wrote game_index.csv (%d games)", len(idx_df))

    log.info("Generated packages for %d games", len(results))
    return results


def generate_future_packages(
    opponents_df: pd.DataFrame,
    tickets_df: pd.DataFrame,
    fan_features: pd.DataFrame,
    fan_labels: pd.DataFrame,
    cluster_propensity: pd.Series | None = None,
    person_lookup: pd.DataFrame | None = None,
    out_dir: Path | str | None = None,
    top_n: int | None = None,
    require_consent: bool = True,
    require_email: bool = True,
    propensity_mode: str = "overall",
    after_date: pd.Timestamp | str | None = None,
    max_n: int | None = None,
    only_competitions: list[str] | None = None,
    only_opponents: list[str] | None = None,
) -> list[dict]:
    """Generate marketing packages for FUTURE fixtures (no ticket data yet).

    This is the operational / production entry point. Because these games
    have not yet been played, game attributes are derived from the opponents
    schedule (not ticket rows) via build_game_profile_from_schedule(). The
    already_bought check is still run against the FULL historical ticket
    dataset to catch any pre-sales: a fan who bought an early-bird ticket for
    an upcoming game before this function runs should still be excluded from
    the target list, even though the game is nominally "future".

    Parameters
    ----------
    opponents_df : DataFrame
        Full opponents schedule.
    tickets_df : DataFrame
        Historical ticket data (used only for subscription identification).
    after_date : Timestamp or str, optional
        Only include fixtures after this date. Default: max event_dt in tickets.
    max_n : int, optional
        Maximum number of future fixtures.
    only_competitions, only_opponents : filtering options.

    Returns
    -------
    list of dicts (one per game), same structure as generate_game_package.
    """
    # Default after_date to max event_dt in ticket data
    if after_date is None and tickets_df is not None and "event_dt" in tickets_df.columns:
        max_dt = tickets_df["event_dt"].max()
        after_date = pd.Timestamp(max_dt).normalize()
        log.info("Future fixtures: after_date defaulting to max(event_dt) = %s", after_date.date())

    # Extract future fixtures
    fixtures = extract_future_fixtures(
        opponents_df,
        after_date=after_date,
        max_n=max_n,
        only_competitions=only_competitions,
        only_opponents=only_opponents,
    )

    if fixtures.empty:
        log.warning("No future fixtures found — no packages generated")
        return []

    # Subscription holders (applicable to future games too — season passes)
    sub_pids = identify_subscription_holders(tickets_df) if tickets_df is not None else set()

    results = []
    index_rows = []

    for _, fixture_row in fixtures.iterrows():
        game_profile = build_game_profile_from_schedule(fixture_row, opponents_df)
        game_date = game_profile["game_date"]
        opponent = game_profile.get("opponent_team")
        match_key = game_profile["match_key"]
        file_label = _build_opponent_label(game_date, opponent)

        # For future games, already_bought is empty (no tickets sold yet)
        # unless there are pre-sales in the ticket data
        bought_pids = set()
        if tickets_df is not None and "event_dt" in tickets_df.columns:
            # check for pre-sales: tickets with event_dt matching this future date
            mask = (
                tickets_df["event_dt"].dt.normalize().eq(game_date)
                & tickets_df["competition_type"].isin(["LBA", "Eurocup"])
            )
            if mask.any():
                bought_pids = set(tickets_df.loc[mask, "person_id"].unique())
                log.info("Pre-sales for %s: %d fans already bought", match_key, len(bought_pids))

        # Score ALL fans (consent filtering happens AFTER scoring)
        scored = score_fans_for_game(
            fan_features, fan_labels, game_profile,
            subscription_pids=sub_pids,
            already_bought_pids=bought_pids,
            person_lookup=person_lookup,
            require_consent=False,
            require_email=False,
        )

        # Split ranked fans by consent / email availability
        ranked = scored.loc[scored["rank"].notna()].sort_values("rank")
        ready_mask = (ranked["marketing_consent"] == 1) & (ranked["has_email"] == True)
        targets = ranked.loc[ready_mask].copy()
        high_intent_df = ranked.loc[~ready_mask].copy().sort_values(
            "eligibility_score", ascending=False
        )
        # Deduplicate by email before slicing — same physical fan may have multiple
        # person_ids from partial identity resolution
        targets = _dedup_by_email(targets, person_lookup)
        if top_n is not None and top_n > 0:
            targets = targets.head(top_n)
            high_intent_df = high_intent_df.head(top_n)

        # merge useful fan features for both lists
        merge_cols = ["person_id"]
        extra_cols = ["games_attended", "total_spend", "recency_days",
                      "pct_lba_games", "pct_eurocup_games", "most_common_sector",
                      "province_mode", "age_mode"]
        for c in extra_cols:
            if c in fan_features.columns:
                merge_cols.append(c)
        feat_lookup = fan_features[merge_cols].drop_duplicates("person_id")
        target_list = targets.merge(feat_lookup, on="person_id", how="left")
        high_intent_list = high_intent_df.merge(feat_lookup, on="person_id", how="left")

        # merge person contact info (buyer_email, nome, cognome) into both lists
        if person_lookup is not None:
            candidate_cols = ["person_id", "buyer_email", "nome", "cognome",
                              "marketing_consent", "has_email"]
            # target_list
            lookup_cols = [c for c in candidate_cols
                           if c in person_lookup.columns
                           and (c == "person_id" or c not in target_list.columns)]
            if len(lookup_cols) > 1:
                target_list = target_list.merge(
                    person_lookup[lookup_cols], on="person_id", how="left",
                )
            # high_intent_list
            lookup_cols_hi = [c for c in candidate_cols
                              if c in person_lookup.columns
                              and (c == "person_id" or c not in high_intent_list.columns)]
            if len(lookup_cols_hi) > 1:
                high_intent_list = high_intent_list.merge(
                    person_lookup[lookup_cols_hi], on="person_id", how="left",
                )

        # Already bought list (likely empty for future games)
        bought_df = scored.loc[scored["already_bought"]].copy()
        if person_lookup is not None:
            bought_cols = [c for c in ["person_id", "buyer_email", "nome", "cognome"]
                           if c in person_lookup.columns]
            bought_df = bought_df.merge(
                person_lookup[bought_cols],
                on="person_id", how="left",
            )

        # Summary
        n_consent_in_tl = (
            int(target_list["marketing_consent"].sum())
            if "marketing_consent" in target_list.columns else None
        )
        n_hi_consent = int(len(high_intent_list))
        top10_hi = (
            high_intent_list[["person_id", "eligibility_score"]]
            .head(10)
            .to_dict(orient="records")
            if len(high_intent_list) > 0 else []
        )
        n_no_consent_total = int(len(ranked)) - int(len(targets))

        summary = {
            "match_key": match_key,
            "game_date": str(game_date.date()) if pd.notna(game_date) else None,
            "opponent_team": opponent,
            "competition": game_profile.get("competition", "?"),
            "is_weekend": game_profile.get("is_weekend", False),
            "is_evening": game_profile.get("is_evening", True),
            "is_high_value": game_profile.get("is_high_value", False),
            "is_derby": game_profile.get("is_derby", False),
            "is_future_fixture": True,
            "n_total_fans": len(scored),
            "n_subscription_holders": int(scored["is_subscription"].sum()),
            "n_already_bought": int(scored["already_bought"].sum()),
            "n_no_consent": n_no_consent_total,
            "n_targetable": int(len(ranked)),
            "n_in_target_list": len(target_list),
            "n_with_email": int(target_list["has_email"].sum()) if "has_email" in target_list.columns else None,
            "n_with_consent": n_consent_in_tl,
            "n_high_intent_no_consent": n_hi_consent,
            "top10_high_intent": top10_hi,
            "require_consent": require_consent,
            "require_email": require_email,
            "propensity_mode_used": propensity_mode,
            "mean_eligibility_score": round(float(targets["eligibility_score"].mean()), 4)
                if len(targets) > 0 else None,
            "median_eligibility_score": round(float(targets["eligibility_score"].median()), 4)
                if len(targets) > 0 else None,
        }

        # Cluster distribution of targets
        if len(target_list) > 0:
            top_clusters = (
                target_list.head(min(200, len(target_list)))
                .groupby("cluster").size()
                .sort_values(ascending=False)
                .head(5)
            )
            summary["top_clusters_in_top200"] = {
                str(k): int(v) for k, v in top_clusters.items()
            }

        # Write outputs
        if out_dir is not None:
            od = Path(out_dir)
            od.mkdir(parents=True, exist_ok=True)

            target_path = od / f"target_list_{file_label}.csv"
            target_list.to_csv(target_path, index=False)
            log.info("Wrote %s (%d rows)", target_path.name, len(target_list))

            hi_path = od / f"high_intent_{file_label}.csv"
            high_intent_list.to_csv(hi_path, index=False)
            log.info("Wrote %s (%d rows)", hi_path.name, len(high_intent_list))

            summary_path = od / f"summary_{file_label}.json"
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2, default=str)
            log.info("Wrote %s", summary_path.name)

        results.append({
            "target_list": target_list,
            "high_intent_no_consent": high_intent_list,
            "bought_already": bought_df,
            "summary": summary,
            "game_profile": game_profile,
        })

        # collect index row
        index_rows.append({
            "game_date": summary.get("game_date"),
            "opponent_team": opponent or "",
            "competition": summary.get("competition"),
            "is_weekend": summary.get("is_weekend"),
            "is_evening": summary.get("is_evening"),
            "is_high_value": game_profile.get("is_high_value", False),
            "is_derby": game_profile.get("is_derby", False),
            "is_future_fixture": True,
            "n_subscription_holders": summary.get("n_subscription_holders"),
            "n_already_bought": summary.get("n_already_bought"),
            "n_no_consent": summary.get("n_no_consent"),
            "n_targetable": summary.get("n_targetable"),
            "n_in_target_list": summary.get("n_in_target_list"),
            "n_with_email": summary.get("n_with_email"),
            "mean_eligibility_score": summary.get("mean_eligibility_score"),
            "target_list_file": f"target_list_{file_label}.csv",
            "summary_file": f"summary_{file_label}.json",
        })

    # Write future game_index.csv
    if out_dir is not None and index_rows:
        idx_df = pd.DataFrame(index_rows)
        idx_path = Path(out_dir) / "game_index_future.csv"
        idx_df.to_csv(idx_path, index=False)
        log.info("Wrote game_index_future.csv (%d future games)", len(idx_df))

    log.info("Generated %d future fixture packages", len(results))
    return results
