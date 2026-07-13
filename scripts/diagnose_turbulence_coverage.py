#!/usr/bin/env python3
"""
Turbulence/Wind/Attitude Coverage Diagnostic
=============================================
Reports per-campaign non-null coverage for the wind/EDR columns
(Wind_U_ms/Wind_V_ms/Wind_W_ms, EDR_m23s1, EDR_arm), and a source-family
breakdown of the unified EDR_m23s1 histogram -- the concrete check that the
2026-07-13 EDR unification (docs/decisions/2026-07-13-edr-unification.md)
produced a sane combined distribution: the NASA Ames MMS-sourced campaigns
(ATTREX, POSIDON) and UND-pipeline-sourced campaigns (IPHEX, MC3E, MPACE,
OLYMPEX, CRYSTAL-FACE-UND) should occupy overlapping, physically plausible
eps^(1/3) ranges (roughly 0-2 m^(2/3)*s^-1) once converted -- a leftover
scale error in one source family would show up as a disjoint sub-range in
this histogram. EDR_arm is plotted separately since it is deliberately not
part of the unification (units unconfirmed).

Outputs (logs/diagnose_turbulence_coverage/<timestamp>/ and
figs/diagnose_turbulence_coverage/<timestamp>/, with a `latest` symlink in
each kept pointing at the newest run):
  coverage_by_campaign.csv   - per-campaign non-null fraction for every new column
  edr_histograms.png         - EDR_m23s1 by source family (overlaid) + EDR_arm

Usage:
    python scripts/diagnose_turbulence_coverage.py
    python scripts/diagnose_turbulence_coverage.py --env data/out/combined_env_data.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.log_paths import timestamp as _run_timestamp, update_latest

TURBULENCE_COLS = [
    "Wind_U_ms", "Wind_V_ms", "Wind_W_ms", "EDR_m23s1", "EDR_arm",
]

# Source-family split of the unified EDR_m23s1 column, for the overlap
# sanity check -- see docs/decisions/2026-07-13-edr-unification.md.
# EDR_arm is intentionally excluded from EDR_m23s1 (units unconfirmed).
EDR_SOURCE_CAMPAIGNS = {
    "NASA Ames MMS (converted from log10 kW/kg)": ["ATTREX", "POSIDON"],
    "UND pipeline (converted from cm^(2/3)s^-1)": ["IPHEX", "MC3E", "MPACE", "OLYMPEX", "CRYSTAL-FACE-UND"],
}


def _parse_args() -> argparse.Namespace:
    ts = _run_timestamp()
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", type=Path,
                   default=ROOT / "data" / "out" / "combined_env_data.parquet",
                   help="Path to combined L0 parquet file")
    p.add_argument("--out", type=Path, default=None,
                   help="Directory for CSV outputs (default: logs/diagnose_turbulence_coverage/<ts>/)")
    p.add_argument("--figs", type=Path, default=None,
                   help="Directory for plot outputs (default: figs/diagnose_turbulence_coverage/<ts>/)")
    return p.parse_args()


def coverage_by_campaign(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    present = [c for c in cols if c in df.columns]
    counts = df.groupby("Campaign")[present].apply(lambda g: g.notna().mean())
    counts.insert(0, "n_records", df.groupby("Campaign").size())
    return counts


def plot_edr_histograms(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    any_family = False
    for label, campaigns in EDR_SOURCE_CAMPAIGNS.items():
        vals = df.loc[df["Campaign"].isin(campaigns), "EDR_m23s1"].dropna()
        if len(vals):
            ax.hist(vals, bins=60, alpha=0.5, label=label)
            any_family = True
    ax.set_title("EDR_m23s1 by source family (should overlap)")
    ax.set_xlabel("EDR_m23s1 (m^(2/3)*s^-1)")
    if any_family:
        ax.legend(fontsize=7)

    ax = axes[1]
    arm_vals = df["EDR_arm"].dropna() if "EDR_arm" in df.columns else pd.Series(dtype=float)
    if len(arm_vals):
        ax.hist(arm_vals, bins=60, color="gray")
    ax.set_title("EDR_arm (not unified, units unconfirmed)")
    ax.set_xlabel("EDR_arm (raw)")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    ts = _run_timestamp()
    out_dir = args.out or (ROOT / "logs" / "diagnose_turbulence_coverage" / ts)
    figs_dir = args.figs or (ROOT / "figs" / "diagnose_turbulence_coverage" / ts)
    out_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)
    update_latest(out_dir.parent, out_dir)
    update_latest(figs_dir.parent, figs_dir)

    print(f"Loading {args.env} ...")
    df = pd.read_parquet(args.env)
    print(f"  {len(df):,} records, {df['Campaign'].nunique()} campaigns")

    cov = coverage_by_campaign(df, TURBULENCE_COLS)
    cov_path = out_dir / "coverage_by_campaign.csv"
    cov.to_csv(cov_path)
    print(f"Saved {cov_path}")

    hist_path = figs_dir / "edr_histograms.png"
    plot_edr_histograms(df, hist_path)
    if hist_path.exists():
        print(f"Saved {hist_path}")

    print(f"\nLatest run: {out_dir.parent / 'latest'} -> {out_dir.name}")


if __name__ == "__main__":
    main()
