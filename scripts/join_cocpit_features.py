#!/usr/bin/env python3
"""
Join COCPIT Particle Size/Geometric Features into L1/L2
=========================================================
Joins per-particle size (microns), geometric-shape, and habit-classification
features from the external COCPIT vgg16 derived-feature database into this
pipeline's L1/L2 tiers, keyed on the exact filename match already
established between `cpi_filename` (this pipeline) and COCPIT's own
`filename` column -- both are the same underlying CPI-archive naming
convention (see `parsers/cpi_timestamps.py`).

This is NOT part of the main cpi-thermo pipeline: like the other COCPIT-
reading scripts (`derive_particle_size_microns.py`,
`compare_derived_feature_versions.py`, ...), it depends on an external,
non-portable path (`/Users/josephko/research/cocpit/final_databases/vgg16`)
that only exists on this machine. It writes NEW output files rather than
overwriting `combined_env_data_L1.parquet`/`_L2.parquet`, so the core
pipeline's reproducibility never depends on this external data being
present.

Uses COCPIT version v1.4.0 by default -- the only version with all 15
campaigns present, and the version whose `filename` column was confirmed to
exact-match this pipeline's `cpi_filename` (v3.1.0's filenames are missing
the millisecond field entirely and would silently fail a direct join; see
`docs/decisions/2026-07-24-derived-feature-version-equiv-d-shift.md`).

The join is a LEFT join (L1/L2 rows kept even with no COCPIT match) so that
match-rate coverage is measurable rather than silently dropping unmatched
rows -- directly answers "does every available record have corresponding
size/geometric features."

Outputs:
  data/out/combined_env_data_L1_cocpit.parquet
  data/out/combined_env_data_L2_cocpit.parquet
  logs/join_cocpit_features/<timestamp>/match_coverage_by_campaign.csv
  logs/join_cocpit_features/<timestamp>/feature_completeness.csv
  logs/join_cocpit_features/<timestamp>/summary_stats.csv

Usage:
    python scripts/join_cocpit_features.py
    python scripts/join_cocpit_features.py --version v1.4.0 --campaigns ARM ISDAC
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from parsers.cpi_timestamps import CPI_TO_ENV_CAMPAIGN
from scripts.compare_derived_feature_versions import (
    DERIVED_DB_ROOT,
    discover_campaign_versions,
    load_campaign_version,
)
from scripts.log_paths import timestamp as _run_timestamp, update_latest

# Columns pulled from the normalized COCPIT frame (post _normalize_columns +
# _add_equiv_d_microns -- see compare_derived_feature_versions.py). Only
# columns actually present in the loaded frame are kept; a version/campaign
# missing a given column simply omits it rather than erroring.
SIZE_COLS = ["particle_width_microns", "particle_height_microns", "equiv_d_microns"]
GEOMETRIC_COLS = [
    "circularity", "solidity", "complexity", "phi", "perim_area_ratio",
    "roundness", "filled_circular_area_ratio", "convex_perim", "hull_area",
    "perim", "cnt_area", "extreme_points",
]
HABIT_COLS = [
    "classification", "agg", "budding", "bullet", "column", "compact_irreg",
    "fragment", "planar_polycrystal", "rimed", "sphere",
]
QUALITY_COLS = ["cutoff", "blur", "contours", "edges", "std", "contrast"]
COCPIT_FEATURE_COLS = SIZE_COLS + GEOMETRIC_COLS + HABIT_COLS + QUALITY_COLS


def _parse_args() -> argparse.Namespace:
    ts = _run_timestamp()
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cocpit-root", type=Path, default=DERIVED_DB_ROOT,
                   help="Root of the COCPIT vgg16 derived-feature database "
                        "(external, non-portable path -- default only valid "
                        "on this machine)")
    p.add_argument("--version", type=str, default="v1.4.0",
                   help="COCPIT database version to join (default: v1.4.0, "
                        "the only version with all 15 campaigns and a "
                        "confirmed filename match against cpi_filename)")
    p.add_argument("--l1", type=Path,
                   default=ROOT / "data" / "out" / "combined_env_data_L1.parquet")
    p.add_argument("--l2", type=Path,
                   default=ROOT / "data" / "out" / "combined_env_data_L2.parquet")
    p.add_argument("--l1-out", type=Path,
                   default=ROOT / "data" / "out" / "combined_env_data_L1_cocpit.parquet")
    p.add_argument("--l2-out", type=Path,
                   default=ROOT / "data" / "out" / "combined_env_data_L2_cocpit.parquet")
    p.add_argument("--out", type=Path,
                   default=ROOT / "logs" / "join_cocpit_features" / ts)
    p.add_argument("--campaigns", nargs="+", default=None,
                   help="COCPIT-style campaign keys to restrict to (e.g. ARM "
                        "ISDAC); default: every campaign found in --version")
    return p.parse_args()


def load_cocpit_features(cocpit_root: Path, version: str,
                          cocpit_campaigns: list[str]) -> pd.DataFrame:
    """Load and stack every requested campaign's COCPIT frame for `version`,
    tagged with the mapped env-style Campaign name."""
    frames = []
    for cocpit_campaign in cocpit_campaigns:
        env_campaign = CPI_TO_ENV_CAMPAIGN.get(cocpit_campaign)
        if env_campaign is None:
            print(f"  SKIP {cocpit_campaign}: no entry in CPI_TO_ENV_CAMPAIGN")
            continue
        df = load_campaign_version(cocpit_root, cocpit_campaign, version, quick_test=False)
        df["Campaign"] = env_campaign
        cols = ["filename", "Campaign"] + [c for c in COCPIT_FEATURE_COLS if c in df.columns]
        frames.append(df[cols])
        print(f"  {cocpit_campaign:20s} -> {env_campaign:20s} {len(df):>9,} COCPIT rows")
    if not frames:
        return pd.DataFrame(columns=["filename", "Campaign"] + COCPIT_FEATURE_COLS)
    return pd.concat(frames, ignore_index=True)


def join_tier(tier: pd.DataFrame, cocpit: pd.DataFrame) -> pd.DataFrame:
    """Per-campaign left join of a tier (L1 or L2) against COCPIT features
    on cpi_filename == filename. Left join keeps every tier row, NaN COCPIT
    columns where unmatched."""
    if "cpi_filename" not in tier.columns:
        raise ValueError("Tier has no cpi_filename column -- expected L1/L2, not L0")

    merged = pd.merge(
        tier,
        cocpit.rename(columns={"filename": "cpi_filename"}),
        on=["cpi_filename", "Campaign"],
        how="left",
    )
    return merged


def match_coverage(tier_name: str, tier: pd.DataFrame, joined: pd.DataFrame,
                    cocpit: pd.DataFrame) -> pd.DataFrame:
    """Per-campaign: tier row count, # matched to a COCPIT row, % matched,
    and (reverse direction) # COCPIT rows for that campaign with no tier
    match at all."""
    match_col = "particle_width_microns" if "particle_width_microns" in joined.columns else None
    joined = joined.copy()
    joined["_matched"] = joined[match_col].notna() if match_col else False

    matched_by_campaign = joined.groupby("Campaign")["_matched"].sum()
    tier_counts = tier.groupby("Campaign").size()
    cocpit_counts = cocpit.groupby("Campaign").size()
    matched_filenames_by_campaign = (
        joined.loc[joined["_matched"], ["Campaign", "cpi_filename"]]
        .groupby("Campaign")["cpi_filename"].apply(set)
    )

    all_campaigns = sorted(set(tier["Campaign"].unique()) | set(cocpit["Campaign"].unique()))
    rows = []
    for camp in all_campaigns:
        n_tier = int(tier_counts.get(camp, 0))
        n_matched = int(matched_by_campaign.get(camp, 0))
        n_cocpit = int(cocpit_counts.get(camp, 0))
        matched_fns = matched_filenames_by_campaign.get(camp, set())
        cocpit_fns = set(cocpit.loc[cocpit["Campaign"] == camp, "filename"])
        n_cocpit_unmatched = len(cocpit_fns - matched_fns)
        rows.append({
            "tier": tier_name,
            "Campaign": camp,
            "n_tier_rows": n_tier,
            "n_matched": n_matched,
            "pct_matched": round(n_matched / n_tier * 100, 2) if n_tier else 0.0,
            "n_cocpit_rows": n_cocpit,
            "n_cocpit_unmatched": n_cocpit_unmatched,
        })
    return pd.DataFrame(rows)


def main() -> None:
    args = _parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Discovering COCPIT campaigns in {args.cocpit_root / args.version} ...")
    camp_versions = discover_campaign_versions(args.cocpit_root)
    available = sorted(c for c, vs in camp_versions.items() if args.version in vs)
    cocpit_campaigns = args.campaigns if args.campaigns else available
    print(f"  {len(cocpit_campaigns)} campaigns: {cocpit_campaigns}")

    print(f"\nLoading COCPIT {args.version} features ...")
    cocpit = load_cocpit_features(args.cocpit_root, args.version, cocpit_campaigns)
    print(f"  Total: {len(cocpit):,} COCPIT particle rows")

    coverage_frames = []
    completeness_rows = []
    stats_rows = []

    for tier_name, in_path, out_path in [
        ("L1", args.l1, args.l1_out),
        ("L2", args.l2, args.l2_out),
    ]:
        print(f"\nLoading {tier_name} from {in_path} ...")
        tier = pd.read_parquet(in_path)
        print(f"  {len(tier):,} rows, {tier['Campaign'].nunique()} campaigns")

        print(f"Joining COCPIT features onto {tier_name} ...")
        joined = join_tier(tier, cocpit)
        joined.to_parquet(out_path, index=False)
        print(f"  Saved {out_path} ({len(joined):,} rows, {len(joined.columns)} columns)")

        cov = match_coverage(tier_name, tier, joined, cocpit)
        coverage_frames.append(cov)

        for col in COCPIT_FEATURE_COLS:
            if col in joined.columns:
                pct = joined[col].notna().mean() * 100
                completeness_rows.append({"tier": tier_name, "column": col, "pct_non_null": round(pct, 2)})

        matched = joined[joined["particle_width_microns"].notna()] if "particle_width_microns" in joined.columns else joined.iloc[0:0]
        for col in SIZE_COLS + [c for c in GEOMETRIC_COLS if c in ("circularity", "solidity", "complexity", "roundness")]:
            if col in matched.columns and pd.api.types.is_numeric_dtype(matched[col]):
                s = matched[col].dropna()
                if len(s):
                    stats_rows.append({
                        "tier": tier_name, "column": col, "n": len(s),
                        "mean": round(s.mean(), 3), "std": round(s.std(), 3),
                        "min": round(s.min(), 3), "max": round(s.max(), 3),
                    })

    coverage_df = pd.concat(coverage_frames, ignore_index=True)
    coverage_path = args.out / "match_coverage_by_campaign.csv"
    coverage_df.to_csv(coverage_path, index=False)
    print(f"\nSaved {coverage_path}")
    print(coverage_df.to_string(index=False))

    completeness_df = pd.DataFrame(completeness_rows)
    completeness_path = args.out / "feature_completeness.csv"
    completeness_df.to_csv(completeness_path, index=False)
    print(f"\nSaved {completeness_path}")

    stats_df = pd.DataFrame(stats_rows)
    stats_path = args.out / "summary_stats.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"\nSaved {stats_path}")
    print(stats_df.to_string(index=False))

    update_latest(args.out.parent, args.out)
    print(f"\nLatest run: {args.out.parent / 'latest'} -> {args.out.name}")


if __name__ == "__main__":
    main()
