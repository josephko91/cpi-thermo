#!/usr/bin/env python3
"""
Build L0/L1/L2 data tiers
==========================
Derives the L1 and L2 data tiers from the combined env parquet (L0), per
GitHub issue #12: no merge tolerance anywhere in this pipeline, so a second
without a real measurement is NaN rather than a fabricated nearest-match
value.

  L0 - data/out/combined_env_data.parquet itself: every whole second where
       *any* instrument in a campaign reported *anything* (the union of all
       instrument timestamps within that campaign, built by the parsers).
  L1 - one row per CPI particle image, joined to its exact-second L0 env
       record (parsers/cpi_timestamps.py's canonical CPI timestamp
       loader). Multiple images sharing a second each get their own row,
       with L0's env columns duplicated across them; a `cpi_filename`
       column keeps every row traceable to its source image. Images with
       no matching L0 second (no instrument reported anything then) are
       dropped.
  L2 - L1 filtered further to rows with a complete record: every column in
       CORE_COLS present (CPI presence is already guaranteed by L1).

The CPI join is done per campaign (not globally) so a CPI image from one
campaign can never spuriously match an L0 row from a different campaign
that happens to share the same wall-clock second.

Outputs:
  data/out/combined_env_data_L1.parquet
  data/out/combined_env_data_L2.parquet
  logs/build_data_tiers/<timestamp>/tier_summary.csv  (rows per tier per
    campaign, with a `latest` symlink)

Usage:
    python scripts/build_data_tiers.py
    python scripts/build_data_tiers.py --env data/out/combined_env_data.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from parsers.cpi_timestamps import (
    CPI_TO_ENV_CAMPAIGN,
    load_cpi_embeddings_timestamps,
)
from parsers.utils import round_timestamp_to_second
from scripts.log_paths import timestamp as _run_timestamp, update_latest

CPI_CSV = ROOT / "data" / "raw" / "cpi_embeddings_timestamps.csv"

# Core "complete record" columns for L2 -- the resolved/"best" variables,
# not per-instrument variants (Si_HW, qv_dlh, ...). Sw is excluded: it's
# derived from Si + Tair_C, so it adds no independent completeness signal.
# Campaigns with no Si/qv source at all (e.g. MPACE) will have zero L2 rows
# under this definition -- expected, not a bug.
CORE_COLS = ["Tair_C", "P_hPa", "Si", "qv", "Lat", "Lon", "Alt_m"]


def _parse_args() -> argparse.Namespace:
    ts = _run_timestamp()
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", type=Path,
                   default=ROOT / "data" / "out" / "combined_env_data.parquet",
                   help="Path to the L0 combined parquet")
    p.add_argument("--cpi", type=Path, default=CPI_CSV,
                   help="Path to the CPI embeddings timestamp CSV")
    p.add_argument("--out", type=Path,
                   default=ROOT / "logs" / "build_data_tiers" / ts,
                   help="Directory for the tier-summary CSV")
    p.add_argument("--l1-out", type=Path,
                   default=ROOT / "data" / "out" / "combined_env_data_L1.parquet")
    p.add_argument("--l2-out", type=Path,
                   default=ROOT / "data" / "out" / "combined_env_data_L2.parquet")
    return p.parse_args()


def build_l1(l0: pd.DataFrame, cpi: pd.DataFrame) -> pd.DataFrame:
    """One row per CPI image, joined to its exact-second L0 env record.

    This is a one-to-many inner merge, not a filter: every CPI image with
    a matching (Campaign, Timestamp) second in L0 gets its own L1 row,
    with L0's env columns duplicated across every image sharing that
    second. An image with no matching L0 second is dropped -- no
    instrument reported anything at that time, so there's nothing to
    join it to.

    CPI timestamps are whole-second (no sub-second precision), but L0
    itself is not always 1Hz-unique per (Campaign, Timestamp) -- e.g. ARM
    is a genuine native 4Hz stream (0.25s intervals), so flooring its
    Timestamp to the second produces up to 4 L0 rows per second. Since a
    CPI image can't be attributed to one specific sub-second reading over
    another, L0 is deduped to one row per (Campaign, floored second) --
    keeping the first sample in that second -- *for this merge only* (L0
    itself, and its sub-second resolution, is untouched). Without this,
    each image would fan out across every sub-second L0 row sharing its
    second instead of getting exactly one matching env record.
    """
    l0 = l0.copy()
    l0["Timestamp"] = round_timestamp_to_second(l0["Timestamp"])
    l0 = l0.drop_duplicates(subset=["Campaign", "Timestamp"], keep="first")
    cpi = cpi.copy()
    cpi["datetime"] = round_timestamp_to_second(cpi["datetime"])

    l1_frames = []
    for env_campaign, cpi_sub in cpi.groupby("campaign_env"):
        l0_sub = l0[l0["Campaign"] == env_campaign]
        if l0_sub.empty:
            continue
        matched = pd.merge(
            cpi_sub[["filename", "datetime"]].rename(columns={"filename": "cpi_filename"}),
            l0_sub,
            left_on="datetime",
            right_on="Timestamp",
            how="inner",
        ).drop(columns=["datetime"])
        if not matched.empty:
            l1_frames.append(matched)

    if not l1_frames:
        empty = l0.iloc[0:0].copy()
        empty.insert(0, "cpi_filename", pd.Series(dtype=object))
        return empty
    return pd.concat(l1_frames, ignore_index=True)


def build_l2(l1: pd.DataFrame) -> pd.DataFrame:
    """L1 rows with every CORE_COLS variable present."""
    present = [c for c in CORE_COLS if c in l1.columns]
    return l1[l1[present].notna().all(axis=1)].copy()


def main() -> None:
    args = _parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Loading L0 from {args.env} ...")
    l0 = pd.read_parquet(args.env)
    print(f"  L0: {len(l0):,} records, {l0['Campaign'].nunique()} campaigns")

    print(f"Loading CPI timestamps from {args.cpi} ...")
    cpi = load_cpi_embeddings_timestamps(args.cpi)
    print(f"  CPI: {len(cpi):,} images across {cpi['campaign_env'].nunique()} campaigns")

    print("Building L1 (one row per CPI image) ...")
    l1 = build_l1(l0, cpi)
    print(f"  L1: {len(l1):,} records")

    print("Building L2 (complete records) ...")
    l2 = build_l2(l1)
    print(f"  L2: {len(l2):,} records")

    args.l1_out.parent.mkdir(parents=True, exist_ok=True)
    l1.to_parquet(args.l1_out, index=False)
    l2.to_parquet(args.l2_out, index=False)
    print(f"Saved {args.l1_out}")
    print(f"Saved {args.l2_out}")

    # --- per-campaign summary ---
    images_per_campaign = cpi.groupby("campaign_env").size()
    campaigns = sorted(set(l0["Campaign"].unique()) | set(CPI_TO_ENV_CAMPAIGN.values()))
    rows = []
    for camp in campaigns:
        n0 = int((l0["Campaign"] == camp).sum())
        n1 = int((l1["Campaign"] == camp).sum()) if not l1.empty else 0
        n2 = int((l2["Campaign"] == camp).sum()) if not l2.empty else 0
        n_images = int(images_per_campaign.get(camp, 0))
        rows.append({
            "Campaign": camp,
            "n_cpi_images": n_images,
            "n_L0": n0,
            "n_L1": n1,
            "n_L2": n2,
            "pct_images_matched": round(n1 / n_images * 100, 2) if n_images else 0.0,
            "pct_L2_of_L1": round(n2 / n1 * 100, 2) if n1 else 0.0,
        })
    summary = pd.DataFrame(rows)
    summary_path = args.out / "tier_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved {summary_path}")
    print("\n" + summary.to_string(index=False))

    update_latest(args.out.parent, args.out)
    print(f"\nLatest run: {args.out.parent / 'latest'} -> {args.out.name}")


if __name__ == "__main__":
    main()
