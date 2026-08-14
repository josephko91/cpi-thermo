#!/usr/bin/env python3
"""
Derive Particle Size in Microns
=================================
Produces physical-unit (micron) particle-size columns for every campaign in
the COCPIT vgg16 derived-feature CSVs
(/Users/josephko/research/cocpit/final_databases/vgg16/<version>/<CAMPAIGN>.csv),
using only information already present in those CSVs. Not part of the main
cpi-thermo pipeline -- an external, unjoined data source.

Two size fields, from two different sources in the raw data:

  particle_width_microns, particle_height_microns
      Direct passthrough of the CSV's `particle width`/`particle height`
      (v3.1.0: `particle width [microns]`/`particle height [microns]`)
      columns. These are computed by COCPIT's process_sheets.py ::
      particle_dimensions() via cv2.minAreaRect() on the *pre-resize*
      cropped ROI (native CPI sensor pixels), multiplied by the CPI probe's
      fixed 2.3 microns/pixel constant. Unchanged across every COCPIT
      version inspected (v1.2.0/v1.3.0/v1.4.0/3.1.0) -- v3.1.0 just added
      the `[microns]` unit label to the same column. Available for every
      campaign and every version with no exceptions (verified below).

  equiv_d_microns
      Derived here (NOT present in any raw CSV) by rescaling the raw
      `equiv_d` column -- an equivalent circular diameter computed from
      cv2.contourArea() on the *post-resize* 1000x1000 canvas, so it has no
      fixed pixel size of its own -- using that row's own
      frame_width/frame_height (the true pre-resize crop box, in native
      sensor pixels) as the scale reference:

          scale_x = frame_width  * 2.3 / 1000   # microns per resized-x-px
          scale_y = frame_height * 2.3 / 1000   # microns per resized-y-px
          equiv_d_microns = equiv_d * sqrt(scale_x * scale_y)

      Only possible where the raw CSV has `equiv_d` in the first place.
      It's missing in every v1.3.0 file (that release's 17-column schema
      never computed shape descriptors, only classification + width/height)
      and in ATTREX's v1.4.0 file specifically (a 7-column file -- that
      pipeline run evidently stopped before the geometry step for ATTREX;
      every other v1.4.0 campaign has the full 36-column schema). Where
      equiv_d is unavailable, equiv_d_microns is left NaN -- no proxy is
      substituted (an ellipse-based sqrt(width*height) proxy was considered
      and rejected: it measures a different geometric quantity than the
      true contour-area equiv_d and would be misleading to merge into the
      same column without a distinguishing flag).

See docs/decisions/2026-07-24-derived-feature-version-equiv-d-shift.md for
the full investigation into why equiv_d itself (before this conversion)
differs so much between v1.4.0 and v3.1.0, and why this per-particle
frame_width/frame_height rescaling does NOT reconcile that gap (it only
converts pixels to microns within a version, it doesn't fix a cross-version
discrepancy that turned out not to be a units problem).

Outputs (logs/derive_particle_size_microns/<timestamp>/, with `latest`
symlink; one row per campaign per version):
  <CAMPAIGN>_<version>_size_microns.csv   - filename, particle_width_microns,
                                             particle_height_microns,
                                             equiv_d_microns (NaN where n/a)
  coverage_summary.csv                    - per (campaign, version): n rows,
                                             % with particle w/h, % with
                                             equiv_d_microns, and why not
                                             when it's 0%
  summary_report.md                       - condensed human-readable version
                                             of coverage_summary.csv
  run_config.json                         - versions found, paths, package versions

Usage:
    python scripts/derive_particle_size_microns.py
    python scripts/derive_particle_size_microns.py --campaigns ARM ATTREX
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.compare_derived_feature_versions import (
    DERIVED_DB_ROOT, CPI_MICRONS_PER_NATIVE_PX, RESIZED_CANVAS_PX,
    discover_campaign_versions, _normalize_columns,
)
from scripts.log_paths import timestamp as _run_timestamp, update_latest

SIZE_COLS = ["particle_width_microns", "particle_height_microns", "equiv_d_microns"]


def _parse_args() -> argparse.Namespace:
    ts = _run_timestamp()
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--derived-db-root", type=Path, default=DERIVED_DB_ROOT)
    p.add_argument("--campaigns", nargs="+", default=None,
                    help="Restrict to these campaigns (COCPIT naming, e.g. CRYSTAL_FACE_NASA). "
                         "Default: all campaigns with any version available.")
    p.add_argument("--out", type=Path,
                    default=ROOT / "logs" / "derive_particle_size_microns" / ts)
    return p.parse_args()


def derive_size_microns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Adds particle_width_microns/particle_height_microns/equiv_d_microns
    to df (a copy). Returns (df_with_size_cols, gap_notes) where gap_notes
    explains any size field that couldn't be derived."""
    df = df.copy()
    gap_notes = {}

    if "particle_width" in df.columns:
        df["particle_width_microns"] = df["particle_width"]
    else:
        df["particle_width_microns"] = np.nan
        gap_notes["particle_width_microns"] = "raw CSV has no particle_width column"

    if "particle_height" in df.columns:
        df["particle_height_microns"] = df["particle_height"]
    else:
        df["particle_height_microns"] = np.nan
        gap_notes["particle_height_microns"] = "raw CSV has no particle_height column"

    if {"equiv_d", "frame_width", "frame_height"} <= set(df.columns):
        scale_x = df["frame_width"] * CPI_MICRONS_PER_NATIVE_PX / RESIZED_CANVAS_PX
        scale_y = df["frame_height"] * CPI_MICRONS_PER_NATIVE_PX / RESIZED_CANVAS_PX
        df["equiv_d_microns"] = df["equiv_d"] * np.sqrt(scale_x * scale_y)
    else:
        df["equiv_d_microns"] = np.nan
        missing = {"equiv_d", "frame_width", "frame_height"} - set(df.columns)
        gap_notes["equiv_d_microns"] = (
            f"raw CSV missing column(s) needed for conversion: {sorted(missing)}"
        )

    return df, gap_notes


def coverage_row(campaign: str, version: str, df: pd.DataFrame, gap_notes: dict) -> dict:
    n = len(df)
    row = {"Campaign": campaign, "version": version, "n_rows": n}
    for col in SIZE_COLS:
        pct = 100.0 * df[col].notna().sum() / n if n else 0.0
        row[f"{col}_pct_available"] = round(pct, 1)
        row[f"{col}_note"] = gap_notes.get(col, "")
    return row


def write_summary_report(coverage_df: pd.DataFrame, out_dir: Path) -> None:
    lines = ["# Particle size in microns -- coverage by campaign/version\n"]
    lines.append(
        "particle_width_microns/particle_height_microns are a direct passthrough "
        "of the raw CSV's particle width/height columns (always microns, computed "
        "pre-resize with the CPI probe's fixed 2.3 microns/pixel constant). "
        "equiv_d_microns is derived here from the raw equiv_d column, rescaled "
        "per-particle via that row's frame_width/frame_height. See the script "
        "docstring and docs/decisions/2026-07-24-derived-feature-version-equiv-d-shift.md "
        "for why equiv_d_microns is not simply a fixed-constant conversion.\n"
    )

    always_100 = coverage_df[
        (coverage_df["particle_width_microns_pct_available"] == 100.0)
        & (coverage_df["particle_height_microns_pct_available"] == 100.0)
    ]
    lines.append(
        f"**particle_width_microns/particle_height_microns: {len(always_100)}/"
        f"{len(coverage_df)} (campaign, version) combinations have 100% coverage.**"
    )
    incomplete = coverage_df[coverage_df["particle_width_microns_pct_available"] < 100.0]
    if not incomplete.empty:
        lines.append("Incomplete width/height coverage:")
        for _, r in incomplete.iterrows():
            lines.append(
                f"- {r['Campaign']} {r['version']}: "
                f"{r['particle_width_microns_pct_available']}% "
                f"({r['particle_width_microns_note']})"
            )
    lines.append("")

    eq_missing = coverage_df[coverage_df["equiv_d_microns_pct_available"] == 0.0]
    eq_present = coverage_df[coverage_df["equiv_d_microns_pct_available"] > 0.0]
    lines.append(
        f"\n**equiv_d_microns: {len(eq_present)}/{len(coverage_df)} (campaign, version) "
        f"combinations have it available; {len(eq_missing)} do not.**\n"
    )
    if not eq_missing.empty:
        lines.append("Not available (and why):")
        for _, r in eq_missing.iterrows():
            lines.append(f"- {r['Campaign']} {r['version']}: {r['equiv_d_microns_note']}")
    lines.append("")

    report_path = out_dir / "summary_report.md"
    report_path.write_text("\n".join(lines))
    print(f"  Saved {report_path}")


def main() -> None:
    args = _parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Discovering campaign versions under {args.derived_db_root} ...")
    camp_versions = discover_campaign_versions(args.derived_db_root)
    if args.campaigns:
        camp_versions = {c: vs for c, vs in camp_versions.items() if c in args.campaigns}
    print(f"  {len(camp_versions)} campaigns, "
          f"{sum(len(v) for v in camp_versions.values())} (campaign, version) files")

    coverage_rows = []
    for camp, versions in sorted(camp_versions.items()):
        for v in versions:
            path = args.derived_db_root / v / f"{camp}.csv"
            df = pd.read_csv(path, low_memory=False)
            df = _normalize_columns(df)
            df, gap_notes = derive_size_microns(df)

            out_cols = [c for c in ["filename"] + SIZE_COLS if c in df.columns]
            out_path = args.out / f"{camp}_{v}_size_microns.csv"
            df[out_cols].to_csv(out_path, index=False)

            row = coverage_row(camp, v, df, gap_notes)
            coverage_rows.append(row)
            print(f"  {camp} {v}: n={row['n_rows']:,}  "
                  f"width/height={row['particle_width_microns_pct_available']}%  "
                  f"equiv_d_microns={row['equiv_d_microns_pct_available']}%")

    coverage_df = pd.DataFrame(coverage_rows)
    coverage_path = args.out / "coverage_summary.csv"
    coverage_df.to_csv(coverage_path, index=False)
    print(f"\nSaved {coverage_path}")

    write_summary_report(coverage_df, args.out)

    run_config = {
        "derived_db_root": str(args.derived_db_root),
        "campaigns": sorted(camp_versions),
        "cpi_microns_per_native_px": CPI_MICRONS_PER_NATIVE_PX,
        "resized_canvas_px": RESIZED_CANVAS_PX,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }
    config_path = args.out / "run_config.json"
    with open(config_path, "w") as f:
        json.dump(run_config, f, indent=2, default=str)
    print(f"Saved {config_path}")

    update_latest(args.out.parent, args.out)
    print(f"\nLatest run: {args.out.parent / 'latest'} -> {args.out.name}")


if __name__ == "__main__":
    main()
