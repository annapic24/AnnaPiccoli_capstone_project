"""
I/O helpers: robust CSV reading, date parsing, identity building.

These functions are extracted / simplified from the original
``pipeline.verify_charts`` module so that the clean pipeline
does **not** depend on the old project tree.
"""

# =============================================================================
# MODULE OVERVIEW
# =============================================================================
# This module handles all data loading and cleaning.  It has three main
# responsibilities:
#
#  (1) CSV reading — robust_read_csv() tries multiple encodings automatically
#      because the Italian ticketing system export can use utf-8, cp1252, or
#      latin1 depending on the export tool and Windows locale.
#
#  (2) Identity resolution — build_person_id() creates a pseudonymous
#      person_id for each person using HMAC-SHA256 hashing of PII.  The
#      7-tier hierarchy ensures maximum person linkage: email is most
#      reliable (globally unique and stable), phone is next (can change),
#      name+DOB is a strong fallback (collisions only for common names born
#      on same day), and name+province is a weaker fallback.  The operation
#      code is a last resort that is unique per transaction, not per person.
#
#  (3) Data enrichment — derives competition type, sector, season, age, and
#      province from raw columns so that downstream feature engineering only
#      needs to consume clean, canonical fields.
# =============================================================================
from __future__ import annotations

import hashlib
import hmac
import logging
import os
import re
import unicodedata
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── CSV reading ───────────────────────────────────────────────────────

def robust_read_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    path = str(path)
    log.info("Reading %s", path)

    # The Aquila ticketing system always exports semicolon-separated files
    # (European CSV convention), so we try ";" exclusively rather than
    # auto-detecting the separator, which can misfire on Italian text.
    seps = [";"]
    # Three encodings to cover all known export variants:
    #   utf-8-sig  – modern Windows export with BOM marker
    #   cp1252     – legacy Windows-1252 (default on Italian Windows systems)
    #   latin1     – ISO-8859-1, a safe superset that never raises UnicodeDecodeError
    encodings = ["utf-8-sig", "cp1252", "latin1"]

    # ── Pass 1: fast C-engine ─────────────────────────────────────────
    # The C parser is significantly faster than the Python parser and handles
    # large files well.  low_memory=False forces pandas to scan the whole file
    # before inferring dtypes, which prevents mixed-type warnings on columns
    # such as ticket codes that look numeric but occasionally contain letters.
    for enc in encodings:
        try:
            df = pd.read_csv(
                path,
                sep=";",
                encoding=enc,
                low_memory=False,
                **kwargs,
            )
            if df.shape[1] <= 1:
                # If only one column was parsed the separator was wrong or the
                # encoding garbled the file — treat as failure and try next.
                raise ValueError(f"Parsed {df.shape[1]} column(s) with sep=';'.")
            log.info("Loaded with C-engine sep=';' encoding=%s cols=%d rows=%d", enc, df.shape[1], len(df))
            return df
        except UnicodeDecodeError as e:
            log.warning("C-engine decode failed with encoding=%s: %s", enc, e)
        except Exception as e:
            log.warning("C-engine read failed with encoding=%s: %s", enc, e)

    # ── Pass 2: Python-engine fallback ───────────────────────────────
    # The Python engine is slower but more tolerant of malformed lines,
    # irregular quoting, and unusual line endings.  low_memory is not a
    # valid keyword for the Python engine, so it is intentionally omitted.
    for enc in encodings:
        try:
            df = pd.read_csv(
                path,
                sep=";",
                encoding=enc,
                engine="python",
                **kwargs,
            )
            if df.shape[1] <= 1:
                raise ValueError(f"Parsed {df.shape[1]} column(s) with python engine.")
            log.info("Loaded with python-engine sep=';' encoding=%s cols=%d rows=%d", enc, df.shape[1], len(df))
            return df
        except Exception as e:
            log.warning("python-engine read failed with encoding=%s: %s", enc, e)

    raise RuntimeError(f"Failed to read CSV: {path} (tried encodings {encodings} with sep=';').")


# ── Date parsing ──────────────────────────────────────────────────────

def parse_dates(df: pd.DataFrame, col: str, new_col: str) -> pd.DataFrame:
    """Parse an Italian-format date column (DD/MM/YYYY) robustly."""
    out = df.copy()
    if col not in out.columns:
        out[new_col] = pd.NaT
        return out

    raw = out[col].astype(str).str.strip()
    dt_day_first = pd.to_datetime(raw, errors="coerce", dayfirst=True)
    dt_month_first = pd.to_datetime(raw, errors="coerce", dayfirst=False)

    # pick whichever succeeds more
    n_day = dt_day_first.notna().sum()
    n_month = dt_month_first.notna().sum()
    out[new_col] = dt_day_first if n_day >= n_month else dt_month_first
    log.info("  parse_dates(%s): %d valid  (dayfirst=%s)",
             col, max(n_day, n_month), n_day >= n_month)
    return out


# ── Identity ──────────────────────────────────────────────────────────

def _normalize_text(s: str) -> str:
    """Strip accents, uppercase, remove non-alphanumeric."""
    if pd.isna(s):
        return ""
    s = str(s).strip().upper()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^A-Z0-9@. ]", "", s)
    return s.strip()


def _is_valid_email(v: str) -> bool:
    if pd.isna(v):
        return False
    v = str(v).strip()
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", v))


def _is_valid_phone(v: str) -> bool:
    if pd.isna(v):
        return False
    digits = re.sub(r"\D", "", str(v))
    return len(digits) >= 6 and digits != "0"


def build_person_id(df: pd.DataFrame, salt: str = "thesis_2026") -> pd.DataFrame:
    """Deterministic pseudonymous person_id from PII using HMAC-SHA256.

    Tier hierarchy (first match wins):
      1. holder email
      2. buyer email
      3. holder phone (mobile)
      4. buyer phone (mobile)
      5. name + birth date
      6. name + province
      7. operation code fallback
    """
    out = df.copy()

    def _hmac_id(key: str) -> str:
        # HMAC-SHA256 with a project-specific salt ensures that the same person
        # always maps to the same 24-character hex ID, while making the ID
        # non-invertible without knowing the salt.
        return hmac.new(salt.encode(), key.encode(), hashlib.sha256).hexdigest()[:24]

    keys = []
    tiers = []

    # Pre-fetch every PII column once to avoid repeated .get() calls in the loop.
    holder_email = out.get("holder_email", pd.Series(dtype=str))
    buyer_email = out.get("buyer_email", pd.Series(dtype=str))
    buyer_mobile = out.get("buyer_mobile", pd.Series(dtype=str))
    holder_name = out.get("holder_name", pd.Series(dtype=str)).fillna("")
    holder_surname = out.get("holder_surname", pd.Series(dtype=str)).fillna("")
    buyer_name = out.get("buyer_name", pd.Series(dtype=str)).fillna("")
    buyer_surname = out.get("buyer_surname", pd.Series(dtype=str)).fillna("")
    holder_birthdate = out.get("holder_birthdate", pd.Series(dtype=str)).fillna("")
    buyer_birthdate = out.get("buyer_birthdate", pd.Series(dtype=str)).fillna("")
    holder_province = out.get("holder_province", pd.Series(dtype=str)).fillna("")
    op_code = out.get("operation_code", pd.Series(dtype=str)).fillna("")

    for i in range(len(out)):
        # ── Tier 1: holder email ──────────────────────────────────────
        # Email is the most reliable identifier: globally unique, typically
        # stable across seasons, and provided by the ticket holder (the person
        # who will actually attend).  Preferred over buyer email because the
        # holder is the physical attendee we want to model.
        e = holder_email.iloc[i] if i < len(holder_email) else None
        if _is_valid_email(e):
            keys.append(f"EMAIL:{_normalize_text(e)}")
            tiers.append(1)
            continue

        # ── Tier 2: buyer email ───────────────────────────────────────
        # The buyer may have purchased tickets for someone else (a child, friend).
        # Using buyer email still links multiple purchases by the same purchasing
        # account, which is useful even if the holder varies.
        e = buyer_email.iloc[i] if i < len(buyer_email) else None
        if _is_valid_email(e):
            keys.append(f"EMAIL:{_normalize_text(e)}")
            tiers.append(2)
            continue

        # ── Tier 3: buyer mobile ──────────────────────────────────────
        # Phone number is moderately reliable: more stable than name spelling
        # but can change (number porting, shared family phone).  Digits only
        # to strip country-code prefixes and formatting differences.
        p = buyer_mobile.iloc[i] if i < len(buyer_mobile) else None
        if _is_valid_phone(p):
            keys.append(f"PHONE:{re.sub(r'[^0-9]', '', str(p))}")
            tiers.append(3)
            continue

        # ── Tier 4: holder name + birth date ─────────────────────────
        # Full name + exact date of birth is a strong pseudonymous key:
        # collision probability is very low for uncommon names, and birth
        # date is stable.  We prefer holder over buyer here because the
        # holder DOB is more commonly recorded.
        nm = _normalize_text(f"{holder_surname.iloc[i]} {holder_name.iloc[i]}")
        bd = str(holder_birthdate.iloc[i]).strip()
        if nm and bd and bd != "nan":
            keys.append(f"NAME_DOB:{nm}:{bd}")
            tiers.append(4)
            continue

        # ── Tier 5: buyer name + birth date ──────────────────────────
        # Fallback to the buyer's name + DOB when the holder fields are empty.
        nm2 = _normalize_text(f"{buyer_surname.iloc[i]} {buyer_name.iloc[i]}")
        bd2 = str(buyer_birthdate.iloc[i]).strip()
        if nm2 and bd2 and bd2 != "nan":
            keys.append(f"NAME_DOB:{nm2}:{bd2}")
            tiers.append(5)
            continue

        # ── Tier 6: holder name + province ───────────────────────────
        # Name alone has many collisions (common Italian surnames).  Adding
        # the province narrows this considerably, though it is still weaker
        # than DOB.  Used only when birth date is unavailable.
        prov = str(holder_province.iloc[i]).strip()
        if nm and prov and prov != "nan":
            keys.append(f"NAME_PROV:{nm}:{prov}")
            tiers.append(6)
            continue

        # ── Tier 7: operation code fallback ──────────────────────────
        # The operation code is unique per *transaction*, not per person, so
        # appending the row index makes each key unique.  This tier does NOT
        # link separate purchases — it is purely a "no PII available" fallback
        # that prevents two anonymous rows from being merged incorrectly.
        oc = str(op_code.iloc[i]).strip()
        keys.append(f"OPCODE:{oc}:{i}")
        tiers.append(7)

    out["person_id"] = [_hmac_id(k) for k in keys]
    out["id_tier"] = tiers
    log.info("Identity tiers: %s", pd.Series(tiers).value_counts().sort_index().to_dict())
    return out


# ── Competition type ──────────────────────────────────────────────────

def classify_competition(df: pd.DataFrame, opponents_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Classify each row as LBA / Eurocup / Abbonamento / Pack / Non partita."""
    out = df.copy()
    title = out.get("event_title", pd.Series(dtype=str)).fillna("").astype(str).str.upper()
    event_type = out.get("event_type", pd.Series(dtype=str)).fillna("").astype(str)

    # Default every row to "Non partita"; overwrite below in priority order.
    comp = pd.Series("Non partita", index=out.index)

    # ── Step A: detect subscriptions and multi-game packs from the title ──
    # These are non-individual-match products and must be identified first so
    # that the LBA/Eurocup rules below don't accidentally re-label them.
    is_abbo = title.str.contains("ABBONAM|SEASON", regex=True, na=False)
    is_pack = title.str.contains(r"PACK|\d+\s*INGRESSI|\d+\s*GARE", regex=True, na=False)
    comp = comp.where(~is_abbo, "Abbonamento")
    comp = comp.where(~is_pack, "Pack")

    # ── Step B: use the numeric event_type code for match rows ────────────
    # The ticketing system encodes competition category as a numeric code:
    #   10 → LBA Serie A domestic league game
    #   20 → EuroCup (7DAYS EuroCup European competition)
    # Abbonamento/Pack rows are excluded via the is_abbo | is_pack guard so
    # that a season-ticket row coded as "10" is not relabelled to "LBA".
    lba_mask = event_type.astype(str).str.strip().eq("10")
    euro_mask = event_type.astype(str).str.strip().eq("20")
    comp = comp.where(~lba_mask | is_abbo | is_pack, "LBA")
    comp = comp.where(~euro_mask | is_abbo | is_pack, "Eurocup")

    # ── Step C: title keyword fallback ────────────────────────────────────
    # Some older exports or manually-entered rows do not carry the event_type
    # code at all (field is blank or "0").  For these rows we fall back to
    # substring matching on the event title.  We only apply this to rows that
    # are still "Non partita" after the code-based step, to avoid overwriting
    # correct code-based classifications.
    is_lba_title = title.str.contains("SERIE A|LBA|LEGA BASKET", regex=True, na=False)
    is_euro_title = title.str.contains("EUROCUP|EURO CUP|7DAYS", regex=True, na=False)
    still_non = comp.eq("Non partita")
    comp = comp.where(~(still_non & is_lba_title), "LBA")
    comp = comp.where(~(still_non & is_euro_title), "Eurocup")

    out["competition_type"] = comp

    # ── Override from opponents schedule (source of truth) ──────────
    # Ticket data labels ALL games as LBA (event_type=10); the opponents
    # schedule correctly distinguishes Serie A / EuroCup.
    if opponents_df is not None and "competition_type" in opponents_df.columns:
        opp_map = {
            "Serie A": "LBA",
            "Serie A (Playoff)": "LBA",
            "EuroCup": "Eurocup",
            "Eurocup": "Eurocup",
        }
        opp = opponents_df[["game_date", "competition_type"]].copy()
        opp["comp_override"] = opp["competition_type"].map(opp_map)
        opp = opp.dropna(subset=["comp_override", "game_date"])
        opp["game_date_norm"] = opp["game_date"].dt.normalize()

        if "event_dt" in out.columns:
            out["_gd_norm"] = out["event_dt"].dt.normalize()
            # only override match rows (LBA/Eurocup), not packs/subscriptions
            match_mask = out["competition_type"].isin(["LBA", "Eurocup"])
            override_map = dict(zip(opp["game_date_norm"], opp["comp_override"]))
            corrected = out.loc[match_mask, "_gd_norm"].map(override_map)
            n_corrected = (corrected.notna() & (corrected != out.loc[match_mask, "competition_type"])).sum()
            out.loc[match_mask & corrected.notna(), "competition_type"] = corrected[corrected.notna()]
            out = out.drop(columns=["_gd_norm"])
            if n_corrected > 0:
                log.info("Competition override from opponents_csv: %d rows corrected", n_corrected)

    log.info("Competition types: %s", out["competition_type"].value_counts().to_dict())
    return out


# ── Sector derivation ────────────────────────────────────────────────

def derive_sector(df: pd.DataFrame) -> pd.DataFrame:
    """Derive macro sector from zone column."""
    out = df.copy()
    zone = out.get("zone", pd.Series(dtype=str)).fillna("").astype(str).str.upper()

    def _sector(z: str) -> str:
        z = z.upper()
        if "COURTSIDE" in z:
            return "COURTSIDE"
        if "PARTERRE" in z:
            return "PARTERRE"
        if "DISTINTI" in z:
            return "DISTINTI"
        if "TRIBUNA" in z:
            return "TRIBUNA"
        if "GRADINATA" in z:
            return "GRADINATA"
        if "CURVA" in z:
            return "CURVA"
        if "CORNER" in z:
            return "CORNER"
        return "OTHER"

    out["settore"] = zone.map(_sector)
    return out


# ── Season derivation ────────────────────────────────────────────────

def derive_season(df: pd.DataFrame) -> pd.DataFrame:
    """Derive season string (e.g. '2025-2026') from event_dt."""
    out = df.copy()
    edt = out.get("event_dt", pd.Series(dtype="datetime64[ns]"))

    def _season(dt):
        if pd.isna(dt):
            return np.nan
        y = dt.year
        m = dt.month
        if m >= 7:
            return f"{y}-{y + 1}"
        return f"{y - 1}-{y}"

    out["season"] = edt.apply(_season)
    return out


# ── Age calculation ──────────────────────────────────────────────────

def compute_age(df: pd.DataFrame) -> pd.DataFrame:
    """Compute age from holder_birthdate and event_dt."""
    out = df.copy()
    bd = out.get("holder_birthdate", pd.Series(dtype=str))
    bd_dt = pd.to_datetime(bd, errors="coerce", dayfirst=True)
    edt = out.get("event_dt", pd.Series(dtype="datetime64[ns]"))

    age = (edt - bd_dt).dt.days / 365.25
    # clip to valid range
    age = age.where(age.between(0, 105), np.nan)
    out["age"] = age.round(0)
    return out


# ── Province cleanup ─────────────────────────────────────────────────

def clean_province(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize province column; extract from city if needed."""
    out = df.copy()
    prov = out.get("holder_province", pd.Series(dtype=str)).fillna("").astype(str).str.strip().str.upper()
    city = out.get("holder_city", pd.Series(dtype=str)).fillna("").astype(str).str.strip().str.upper()

    # extract province code from city like "TRENTO (TN)"
    city_prov = city.str.extract(r"\(([A-Z]{2})\)", expand=False)
    prov_clean = prov.where(prov.str.len() == 2, city_prov)

    out["provincia_clean"] = prov_clean
    return out


# ── Opponents loading ────────────────────────────────────────────────

def load_opponents(path: str | Path) -> pd.DataFrame:
    """Load and normalise the opponents schedule CSV."""
    opp = robust_read_csv(path)
    # normalise date
    if "Date" in opp.columns:
        opp["game_date"] = pd.to_datetime(opp["Date"], errors="coerce", dayfirst=True)
    if "opponent_team" in opp.columns:
        opp["opponent_norm"] = opp["opponent_team"].str.strip().str.upper()
    log.info("Loaded %d opponent records", len(opp))
    return opp


# ── Full cleaning pipeline ───────────────────────────────────────────

def run_cleaning(raw_df: pd.DataFrame,
                 opponents_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Run the full cleaning pipeline on raw data.

    Steps:
      1. Parse dates  (event_dt, movement_dt)
      2. Build person_id
      3. Classify competition type
      4. Derive sector, season, age
      5. Clean province

    Returns ticket-level cleaned DataFrame.
    """
    from .schema import rename_to_canonical

    # Rename raw export columns to the canonical internal names defined in schema.py
    df, col_map = rename_to_canonical(raw_df)
    log.info("Cleaning %d rows …", len(df))

    # ── Step 1: Parse dates ───────────────────────────────────────────────
    # event_dt  = when the game was played (DD/MM/YYYY Italian format)
    # movement_dt = when the ticket was purchased / transaction processed
    df = parse_dates(df, "event_dt_raw", "event_dt")
    df = parse_dates(df, "movement_dt_raw", "movement_dt")

    # ── Step 2a: Compute age at time of event ─────────────────────────────
    # Derived from holder_birthdate relative to event_dt so that a fan's age
    # reflects how old they were when they attended, not their current age.
    df = compute_age(df)

    # ── Step 2b: Derive sector and season ─────────────────────────────────
    # sector = macro seating area (COURTSIDE, TRIBUNA, GRADINATA, etc.)
    # season = basketball-year label such as "2025-2026" (July-start convention)
    df = derive_sector(df)
    df = derive_season(df)

    # ── Step 3: Classify competition type ─────────────────────────────────
    # Attaches competition_type: LBA / Eurocup / Abbonamento / Pack / Non partita
    # The opponents_df (if provided) is used to override event_type=10 rows
    # that belong to Eurocup (the ticketing system labels all games as "10").
    df = classify_competition(df, opponents_df)

    # ── Step 4: Clean province ────────────────────────────────────────────
    # Normalises the province field to 2-letter ISO codes; extracts from the
    # city column when the province field itself is missing or malformed.
    df = clean_province(df)

    # ── Step 5: Build pseudonymous person_id ─────────────────────────────
    # Runs the 7-tier HMAC identity resolution (see build_person_id docstring).
    # Done after date/sector/age so that the enriched columns are available if
    # ever needed for future identity logic.
    df = build_person_id(df)

    # ── Pricing helpers ───────────────────────────────────────────────────
    # Italian number format: '.' = thousands separator, ',' = decimal separator
    # e.g. '1.500,00' must become '1500.00', not '1.500.00'
    def _parse_italian_number(s: pd.Series) -> pd.Series:
        cleaned = s.astype(str).str.replace("€", "", regex=False).str.strip()
        has_comma = cleaned.str.contains(",", regex=False)
        # For Italian-format values: strip thousands dots, then swap decimal comma
        italian = (
            cleaned.str.replace(".", "", regex=False)
                   .str.replace(",", ".", regex=False)
        )
        result = cleaned.copy()
        result[has_comma] = italian[has_comma]
        return pd.to_numeric(result, errors="coerce")

    # Only re-parse if the column is still a string (some exports are already numeric)
    ticket_price = df.get("ticket_price", pd.Series(dtype=float))
    if not pd.api.types.is_numeric_dtype(ticket_price):
        df["ticket_price"] = _parse_italian_number(ticket_price)
    total_amt = df.get("total_amount", pd.Series(dtype=float))
    if not pd.api.types.is_numeric_dtype(total_amt):
        df["total_amount"] = _parse_italian_number(total_amt)

    # days_before_game: how far in advance the ticket was bought.
    # Clipped to [-2, 365] to remove clearly erroneous values (e.g. purchases
    # dated years before the game, or returns processed a day after the game).
    if "event_dt" in df.columns and "movement_dt" in df.columns:
        dbg = (df["event_dt"] - df["movement_dt"]).dt.days
        df["days_before_game"] = dbg.where(dbg.between(-2, 365), np.nan)
    else:
        df["days_before_game"] = np.nan

    # Bundle / pack detection flags used by feature_building.py
    title = df.get("event_title", pd.Series(dtype=str)).fillna("").astype(str).str.upper()
    price_type = df.get("price_type", pd.Series(dtype=str)).fillna("").astype(str).str.upper()

    is_abbo = df["competition_type"].eq("Abbonamento")
    is_pack = df["competition_type"].eq("Pack")
    bundle_re = re.compile(r"(?<!\d)(\d{1,2})(?!\d)\s*(INGRESSI|GARE|PARTITE)", re.I)
    df["bundle_games_included"] = title.map(
        lambda t: float(m.group(1)) if (m := bundle_re.search(str(t))) else np.nan
    )
    df["is_non_match"] = df["competition_type"].isin(["Non partita", "Abbonamento", "Pack"])
    bundle_terms = price_type.str.contains("PACK|ABBON", regex=True, na=False)
    df["is_bundle_or_pack"] = is_abbo | is_pack | df["bundle_games_included"].notna() | bundle_terms

    # ── Price ratio (LBA only) ────────────────────────────────────────────
    # Computes price_ratio_from_list = observed_price / official_LBA_list_price
    # for each LBA row.  Used in feature_building to compute price-sensitivity
    # features (pct_full_price, pct_discounted_vs_list, avg_price_ratio_vs_list).
    from .schema import _RAW_CANDIDATES  # noqa: just ensure module imported
    _compute_price_ratio(df)

    log.info("Cleaning complete: %d rows, %d cols", len(df), len(df.columns))
    return df


# ── price ratio helper ───────────────────────────────────────────────

# reference list prices (€) by (sector, age_category)
# These are the official Aquila Basket Trento face-value ticket prices for LBA
# home games.  They serve as the denominator for price_ratio_from_list:
#   ratio < 1.0  → fan paid less than list (discount, promo code, or comp ticket)
#   ratio ≈ 1.0  → fan paid full price
#   ratio > 1.0  → fan paid above list (surcharge, premium resale — rare)
# COURTSIDE and PARTERRE do not have concessionary tiers because those sectors
# are premium and not eligible for youth/student pricing.
_LBA_PRICE_REF = {
    ("COURTSIDE", "INTERO"): 120,
    ("PARTERRE", "INTERO"): 85,
    ("DISTINTI", "INTERO"): 45, ("DISTINTI", "UNDER_20"): 32, ("DISTINTI", "UNDER_12"): 20,
    ("TRIBUNA", "INTERO"): 28, ("TRIBUNA", "UNDER_20"): 20, ("TRIBUNA", "UNDER_12"): 12,
    ("GRADINATA", "INTERO"): 20, ("GRADINATA", "UNDER_20"): 14, ("GRADINATA", "UNDER_12"): 8,
    ("CURVA", "INTERO"): 16, ("CURVA", "UNDER_20"): 8, ("CURVA", "UNDER_12"): 6,
}


def _age_cat(age):
    if pd.isna(age):
        return "INTERO"
    a = float(age)
    if 5 <= a <= 12:
        return "UNDER_12"
    if 13 <= a <= 20:
        return "UNDER_20"
    return "INTERO"


def _compute_price_ratio(df: pd.DataFrame) -> None:
    """Add price_ratio_from_list in-place for LBA rows.

    The reference price table (_LBA_PRICE_REF) contains the official face-value
    ticket prices (in euros) for every combination of seating sector and age
    category.  Dividing the observed ticket_price by this reference gives a
    normalised ratio that is comparable across sectors and age groups, making it
    possible to identify fans who consistently buy at full price vs. those who
    only attend when promotions are active.
    """
    df["expected_age_cat"] = df.get("age", pd.Series(dtype=float)).apply(_age_cat)
    df["is_child_ticket"] = df["expected_age_cat"].isin(["UNDER_12", "UNDER_20"])

    expected = pd.Series(np.nan, index=df.index)
    lba = df["competition_type"].eq("LBA")
    settore = df.get("settore", pd.Series(dtype=str)).fillna("")

    for idx in df.index[lba]:
        key = (settore.at[idx], df.at[idx, "expected_age_cat"])
        if key in _LBA_PRICE_REF:
            expected.at[idx] = _LBA_PRICE_REF[key]

    df["expected_price_from_list"] = expected
    observed = pd.to_numeric(df.get("ticket_price", pd.Series(dtype=float)), errors="coerce")
    df["price_ratio_from_list"] = observed / expected
    df["is_free_expected_from_age"] = df["expected_age_cat"].eq("UNDER_4") | (
        df.get("age", pd.Series(dtype=float)).lt(6) &
        settore.str.contains("CURVA|GRADINATA", na=False)
    )
