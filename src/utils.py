"""
Utility helpers: logging setup, directory creation, summary printing.
"""
# =============================================================================
# Three simple utilities: setup_logging configures where log messages go,
# make_run_dir creates a timestamped output folder for each pipeline run,
# print_summary prints the evaluation results table in a readable format.
# =============================================================================
from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


def setup_logging(out_dir: Path | None = None, level: int = logging.INFO) -> None:
    """Configure logging to stdout + optional file."""
    fmt = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(out_dir / "pipeline.log", mode="w")
        handlers.append(fh)

    # force=True replaces any existing handlers already attached to the root
    # logger, ensuring this call always takes effect even if basicConfig was
    # called earlier (e.g. by an imported library).
    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)


def make_run_dir(base: Path) -> Path:
    """Create a timestamped run directory."""
    # Timestamped dirs (e.g. run_20260409_143021) prevent overwriting previous
    # runs — every pipeline execution gets its own isolated output folder so
    # earlier results, logs, and artefacts are always preserved for comparison.
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    d = base / f"run_{tag}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def print_summary(
    summary: pd.DataFrame,
    title: str = "EVALUATION SUMMARY",
) -> None:
    """Pretty-print an evaluation summary table."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")
    for _, row in summary.iterrows():
        # Column format breakdown:
        #   {method:<12s}  — left-aligned, 12-char wide method name
        #   K={K:<5d}      — left-aligned K value (e.g. "200  " or "500  ")
        #   Precision=     — 4 decimal places with ± std deviation
        #   Lift=          — 2 decimal places with ± std deviation
        #   [N games]      — how many test games contributed to the averages
        print(
            f"  {row['method']:<12s}  K={int(row['K']):<5d}  "
            f"Precision={row['mean_precision']:.4f} (±{row['std_precision']:.4f})  "
            f"Lift={row['mean_lift']:.2f} (±{row['std_lift']:.2f})  "
            f"[{int(row['n_games'])} games]"
        )
    print(f"{'=' * 70}\n")
