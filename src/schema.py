"""
Column name discovery and normalisation.

The raw CSV from the ticketing system has Italian column names that may
vary slightly across exports.  This module maps them to canonical
internal names used throughout the pipeline.
"""

# =============================================================================
# PURPOSE OF THIS MODULE
# =============================================================================
# The CSV exported from the ticketing system uses Italian column names.
# Across different export versions these names can differ in:
#   - accents (e.g. "Citta'" vs "Città" vs "Citta")
#   - capitalisation (e.g. "Data e Ora Spettacolo" vs "data e ora spettacolo")
#   - minor wording differences between software versions
#
# To insulate the rest of the pipeline from these variations, this module
# provides a single mapping layer.  Every column is given a canonical
# internal name (e.g. "event_dt_raw", "buyer_email") that is used
# consistently everywhere downstream.
#
# If a column cannot be found under any of its known candidate names, a
# WARNING is logged but execution continues — the column will simply be
# absent from the renamed DataFrame, and downstream code handles it.
# This avoids hard crashes when a new CSV export drops an optional column.
# =============================================================================
from __future__ import annotations

import logging
import re
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

# ── canonical names → candidate raw names (first match wins) ──────────
# Structure: canonical_name: [possible_raw_name_1, possible_raw_name_2, ...]
# Candidates are tried in order; the first one found in the DataFrame is used.
# Add new candidates to the END of each list to preserve precedence for the
# most common / most recent export format.
_RAW_CANDIDATES: dict[str, list[str]] = {
    "event_dt_raw": [
        "Data e Ora Spettacolo",
        "Data e ora Spettacolo",
        "data e ora spettacolo",
    ],
    "movement_dt_raw": [
        "Data Movimento",
        "data movimento",
    ],
    "event_type": [
        "Tipo Spettacolo",
    ],
    "event_title": [
        "Titolo Spettacolo",
    ],
    "zone": [
        "Zona",
    ],
    "operation_code": [
        "Codice Operazione",
    ],
    "tx_id": [
        "Numero Movimento",
    ],
    "price_type": [
        "Nome Tipologia prezzo",
    ],
    "ticket_price": [
        "Prezzo figurativo Biglietto",
    ],
    "presale_discount": [
        "Prevendita figurativa Biglietto",
    ],
    "total_amount": [
        "Totale Importo figurativo Movimento",
    ],
    "buyer_surname": [
        "Cognome Acquirente",
    ],
    "buyer_name": [
        "Nome Acquirente",
    ],
    "buyer_email": [
        "E-mail Acquirente",
    ],
    "buyer_mobile": [
        "Cellulare Acquirente",
    ],
    "buyer_phone": [
        "Telefono Acquirente",
    ],
    "buyer_birthdate": [
        "Data di Nascita per report Acquirente",
    ],
    "holder_surname": [
        "Cognome Anagrafica biglietto",
    ],
    "holder_name": [
        "Nome Anagrafica biglietto",
    ],
    "holder_email": [
        "E-mail Anagrafica biglietto",
    ],
    "seat_sector": [
        "Settore Posto",
    ],
    "seat_row": [
        "Fila Posto",
    ],
    "seat_number": [
        "Posto",
    ],
    "holder_birthdate": [
        "Data di nascita Anagrafica biglietto",
    ],
    "holder_city": [
        "Citta' Anagrafica biglietto",
        "Citta Anagrafica biglietto",
        "Città Anagrafica biglietto",
    ],
    "holder_province": [
        "Provincia Anagrafica biglietto",
    ],
    "sales_channel": [
        "Canale di vendita Movimento",
    ],
}


def discover_columns(df: pd.DataFrame) -> dict[str, str]:
    """Return mapping *canonical → actual column name* found in *df*.

    Logs warnings for missing columns.
    """
    mapping: dict[str, str] = {}
    # Pre-build a lowercased lookup once to avoid repeated .lower() calls in the loop
    cols_lower = {c.strip().lower(): c for c in df.columns}

    for canonical, candidates in _RAW_CANDIDATES.items():
        found = False
        for cand in candidates:
            # Try exact match first — fastest and safest, preserves original casing
            if cand in df.columns:
                mapping[canonical] = cand
                found = True
                break
            # Fall back to case-insensitive comparison to handle exports where the
            # ticketing system changed capitalisation (e.g. "Data Movimento" vs
            # "data movimento").  Strip leading/trailing whitespace too.
            if cand.strip().lower() in cols_lower:
                mapping[canonical] = cols_lower[cand.strip().lower()]
                found = True
                break
        if not found:
            # Log a warning but do NOT raise — optional columns (e.g. birthdate)
            # may legitimately be absent in some export configurations.
            log.warning("Column '%s' not found (tried %s)", canonical, candidates)
    return mapping


def rename_to_canonical(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Rename raw columns to canonical names.  Returns (df_renamed, mapping).

    Calls discover_columns() to build the canonical→raw mapping, then inverts
    it (raw→canonical) and applies it via DataFrame.rename().  Columns not
    present in the mapping are left with their original names unchanged.
    """
    mapping = discover_columns(df)
    reverse = {v: k for k, v in mapping.items()}  # flip to raw→canonical for rename()
    out = df.rename(columns=reverse)
    log.info("Renamed %d columns to canonical form", len(mapping))
    return out, mapping
