"""
Marketing consent and contact master table loader.

Source of truth: aquila_basket_10103810.csv
This file has NO header, comma-separated, with the following positional columns:
    [0] Cognome (surname)
    [1] Nome (first name)
    [2] Data di nascita (DOB)
    [3] Città (city)
    [4] Provincia (province)
    [5] Consent type (INFORMATIVA, MARKETING, NEWSLETTER, PROFILAZIONE)
    [6] Consent flag (1=yes, 0=no)
    [7] Full name (concatenated — not used)
    [8] Email

The file is in LONG format: one row per (email × consent_type).
We pivot to WIDE format: one row per email with boolean consent columns,
keeping only the MARKETING consent flag as the authoritative consent.

Name/surname come ONLY from this file (not inferred from tickets).
"""
# =============================================================================
# GDPR compliance: fans can only be targeted for marketing if they gave
# explicit MARKETING consent.  This module loads the consent CSV, extracts
# the marketing consent flag per email, and joins it with name/surname data.
# The result is used later to filter the target list — fans without consent
# are never emailed even if they score highly.
# =============================================================================
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

_COL_NAMES = [
    "cognome", "nome", "dob", "citta", "provincia",
    "consent_type", "consent_flag", "full_name", "email",
]


def load_marketing_consent(path: str | Path) -> pd.DataFrame:
    """Load and clean the marketing consent master CSV.

    Parameters
    ----------
    path : str or Path
        Path to aquila_basket_10103810.csv (no header, comma-separated).

    Returns
    -------
    DataFrame with columns:
        email, nome, cognome, marketing_consent
    One row per unique email.  marketing_consent is int 0/1.
    """
    path = Path(path)
    log.info("Loading marketing consent from %s", path)

    # Read with no header — assign positional names
    raw = pd.read_csv(
        path, header=None, names=_COL_NAMES,
        encoding="utf-8-sig", low_memory=False,
    )
    log.info("  Raw consent file: %d rows × %d cols", raw.shape[0], raw.shape[1])

    # ── SECTION 1: normalise email (lower, strip) ──────────────────
    raw["email"] = raw["email"].astype(str).str.strip().str.lower()
    raw.loc[raw["email"].isin(["", "nan", "none"]), "email"] = pd.NA

    # ── SECTION 2: drop rows without email ────────────────────────
    n_before = len(raw)
    raw = raw.dropna(subset=["email"])
    n_dropped_no_email = n_before - len(raw)
    if n_dropped_no_email > 0:
        log.info("  Dropped %d rows with no email", n_dropped_no_email)

    # ── SECTION 3: validate consent_flag ──────────────────────────
    raw["consent_flag"] = pd.to_numeric(raw["consent_flag"], errors="coerce").fillna(0).astype(int)
    # clamp to 0/1
    raw["consent_flag"] = raw["consent_flag"].clip(0, 1)

    # ── SECTION 4: extract MARKETING consent rows ─────────────────
    mkt_rows = raw.loc[raw["consent_type"].str.strip().str.upper() == "MARKETING"].copy()
    log.info("  MARKETING consent rows: %d", len(mkt_rows))

    # ── SECTION 5: pivot to one row per email ─────────────────────
    # For duplicates (same email, multiple MARKETING rows), keep the
    # row with consent_flag=1 if any (optimistic), else 0.
    # This is deterministic: max() picks 1 over 0.
    # Optimistic strategy: if any row for this email says consent=1,
    # count the fan as consented.  This favours deliverability over
    # over-restriction — a fan who opted in at any point is included.
    consent_by_email = (
        mkt_rows.groupby("email")["consent_flag"]
        .max()
        .rename("marketing_consent")
        .reset_index()
    )

    # ── SECTION 6: build name lookup from ALL rows ────────────────
    # Names come from the full file (not just MARKETING rows) to
    # maximise coverage.  For name/surname, take the first non-null
    # value per email, preferring rows that have both nome and cognome
    # populated.
    name_df = raw[["email", "cognome", "nome"]].copy()
    name_df["cognome"] = name_df["cognome"].astype(str).str.strip()
    name_df["nome"] = name_df["nome"].astype(str).str.strip()
    # replace 'nan' strings with actual NaN
    for c in ["cognome", "nome"]:
        name_df.loc[name_df[c].isin(["nan", "", "None"]), c] = pd.NA

    # Prefer rows where both first name and surname are populated:
    # score: both present=2, one present=1, none=0.  Sorting descending
    # by this score and keeping the first row per email guarantees the
    # richest name record is always selected.
    name_df["_name_score"] = name_df["cognome"].notna().astype(int) + name_df["nome"].notna().astype(int)
    name_df = name_df.sort_values(["email", "_name_score"], ascending=[True, False])
    name_best = name_df.drop_duplicates("email", keep="first")[["email", "cognome", "nome"]]

    # ── SECTION 7: merge consent + names ──────────────────────────
    # Start from ALL unique emails (even those without a MARKETING row
    # — they get marketing_consent=0)
    all_emails = raw[["email"]].drop_duplicates()
    result = all_emails.merge(consent_by_email, on="email", how="left")
    result["marketing_consent"] = result["marketing_consent"].fillna(0).astype(int)
    result = result.merge(name_best, on="email", how="left")

    n_total = len(result)
    n_consent = (result["marketing_consent"] == 1).sum()
    n_with_name = result["cognome"].notna().sum()

    log.info(
        "  Consent master: %d unique emails, %d with marketing consent (%.1f%%), "
        "%d with name (%.1f%%)",
        n_total, n_consent, 100 * n_consent / max(1, n_total),
        n_with_name, 100 * n_with_name / max(1, n_total),
    )

    # ── validation warnings ────────────────────────────────────────
    dup_emails = consent_by_email["email"].duplicated().sum()
    if dup_emails > 0:
        log.warning("  %d duplicate emails in MARKETING rows (resolved by max)", dup_emails)

    invalid_consent = raw.loc[~raw["consent_flag"].isin([0, 1]), "consent_flag"]
    if len(invalid_consent) > 0:
        log.warning("  %d rows had invalid consent values (clamped to 0/1)", len(invalid_consent))

    return result[["email", "nome", "cognome", "marketing_consent"]].reset_index(drop=True)
