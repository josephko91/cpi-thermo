#!/usr/bin/env python3
"""
Turbulence/Wind/Attitude Coverage Diagnostic
=============================================
Reports per-campaign non-null coverage for the wind/attitude/airspeed/EDR
columns added by docs/todo/2026-07-13-turbulence-measurements-plan.md, and
per-EDR-family value-range histograms -- the concrete check that the
"don't unify EDR" boundary (docs/decisions/2026-07-13-turbulence-schema.md)
was actually respected: Family A (EDR_mms_log10kWkg) should show small
negative-to-single-digit log10 values, Family B (EDR_und_cm23s1) a very
different linear range; a mixup would be obvious in the histogram but
invisible looking at either column in isolation.

Also prints a level-flight Roll_deg/Pitch_deg sanity check per campaign
(should hover near 0, not show a constant offset, which would indicate a
sign/convention mismatch).

Outputs (logs/diagnose_turbulence_coverage/<timestamp>/ and
figs/diagnose_turbulence_coverage/<timestamp>/, with a `latest` symlink in
each kept pointing at the newest run):
  coverage_by_campaign.csv   - per-campaign non-null fraction for every new column
  edr_histograms.png         - per-EDR-family value-range histograms
  attitude_sanity.csv        - per-campaign Roll_deg/Pitch_deg median + IQR

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
    "Wind_U_ms", "Wind_V_ms", "Wind_W_ms", "WindSpeed_ms", "WindDir_deg",
    "Roll_deg", "Pitch_deg", "Heading_deg",
    "AngleOfAttack_deg", "Sideslip_deg",
    "VertVel_ms", "Accel_Vert_ms2",
    "TAS_ms", "IAS_ms", "IAS_ms_nose", "MachNo",
    "DriftAngle_deg", "TrackAngle_deg",
    "EDR_mms_log10kWkg", "EDR_und_cm23s1", "EDR_und_cm23s1_nose", "EDR_arm",
    "REYN_mms",
]

# Per docs/decisions/2026-07-13-turbulence-schema.md: these three (four,
# counting MPACE's nose channel) are deliberately never merged/converted.
EDR_FAMILIES = {
    "EDR_mms_log10kWkg": "Family A (NASA Ames MMS, log10 kW/kg)",
    "EDR_und_cm23s1": "Family B (UND pipeline, linear cm^(2/3)s^-1)",
    "EDR_und_cm23s1_nose": "Family B nose channel (MPACE only)",
    "EDR_arm": "ARM (units unconfirmed)",
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


def attitude_sanity(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for campaign, g in df.groupby("Campaign"):
        row = {"Campaign": campaign}
        for col in ("Roll_deg", "Pitch_deg"):
            if col in g.columns and g[col].notna().any():
                row[f"{col}_median"] = g[col].median()
                row[f"{col}_iqr"] = g[col].quantile(0.75) - g[col].quantile(0.25)
        if len(row) > 1:
            rows.append(row)
    return pd.DataFrame(rows)


def plot_edr_histograms(df: pd.DataFrame, out_path: Path) -> None:
    present = [c for c in EDR_FAMILIES if c in df.columns and df[c].notna().any()]
    if not present:
        return
    fig, axes = plt.subplots(1, len(present), figsize=(5 * len(present), 4))
    if len(present) == 1:
        axes = [axes]
    for ax, col in zip(axes, present):
        vals = df[col].dropna()
        ax.hist(vals, bins=60)
        ax.set_title(EDR_FAMILIES[col])
        ax.set_xlabel(col)
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

    sanity = attitude_sanity(df)
    sanity_path = out_dir / "attitude_sanity.csv"
    sanity.to_csv(sanity_path, index=False)
    print(f"Saved {sanity_path}")
    if not sanity.empty:
        print("\nLevel-flight Roll/Pitch sanity (median should be near 0, not a constant offset):")
        print(sanity.to_string(index=False))

    hist_path = figs_dir / "edr_histograms.png"
    plot_edr_histograms(df, hist_path)
    if hist_path.exists():
        print(f"Saved {hist_path}")

    print(f"\nLatest run: {out_dir.parent / 'latest'} -> {out_dir.name}")


if __name__ == "__main__":
    main()
