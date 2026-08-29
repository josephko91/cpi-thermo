#!/usr/bin/env python3
"""
Verify overlap between raw CPI image files and COCPIT derived-feature rows
=============================================================================
Standalone script for xcite (HPC cluster) -- no dependency on the
cpi-thermo repo, only pandas (stdlib otherwise). For each campaign,
compares the set of raw single-particle image filenames on disk against
the `filename` column of that campaign's COCPIT v1.4.0 derived-feature
CSV, and reports:
  - how many raw images have a corresponding COCPIT feature row
  - how many COCPIT feature rows have no corresponding raw image on disk
  - overlap counts/percentages in both directions

This answers, independent of any env-data join: does COCPIT's own
processing coverage gap (found in cpi-thermo's
docs/reports/2026-08-29-cocpit-particle-feature-join.md -- e.g. MACPEX at
2.8% coverage, ARM at 8.6%) come from COCPIT simply never having processed
most of a campaign's raw images, or from some other mismatch (e.g.
filename-format drift). Run this on xcite where the raw image directories
actually live; the derived-feature CSVs are the same ones already used on
the other machine.

Usage:
    python xcite_verify_cpi_cocpit_overlap.py
    python xcite_verify_cpi_cocpit_overlap.py --campaigns ARM MACPEX
    python xcite_verify_cpi_cocpit_overlap.py --out overlap_report.csv
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

# campaign -> (raw image directory, COCPIT CSV stem)
CAMPAIGNS: dict[str, str] = {
    "MC3E":              "/home/vanessa/hulk/cocpit/cpi_data/campaigns/MC3E/single_imgs_v1.4.0",
    "ARM":               "/home/vanessa/hulk/cocpit/cpi_data/campaigns/ARM/single_imgs_v1.4.0",
    "MPACE":             "/home/vanessa/hulk/cocpit/cpi_data/campaigns/MPACE/single_imgs_v1.4.0",
    "IPHEX":             "/home/vanessa/hulk/cocpit/cpi_data/campaigns/IPHEX/single_imgs_v1.4.0",
    "AIRS_II":           "/home/vanessa/hulk/cocpit/cpi_data/campaigns/AIRS_II/single_imgs_v1.4.0",
    "ICE_L":             "/home/vanessa/hulk/cocpit/cpi_data/campaigns/ICE_L/single_imgs_v1.4.0",
    "CRYSTAL_FACE_NASA": "/home/vanessa/hulk/cocpit/cpi_data/campaigns/CRYSTAL_FACE_NASA/single_imgs_v1.4.0",
    "MACPEX":            "/home/vanessa/hulk/cocpit/cpi_data/campaigns/MACPEX/single_imgs_v1.4.0",
    "MIDCIX":            "/home/vanessa/hulk/cocpit/cpi_data/campaigns/MIDCIX/single_imgs_v1.4.0",
    "ISDAC":             "/home/vanessa/hulk/cocpit/cpi_data/campaigns/ISDAC/single_imgs_v1.4.0",
    "ATTREX":            "/home/vanessa/hulk/cocpit/cpi_data/campaigns/ATTREX/single_imgs_v1.4.0",
    "CRYSTAL_FACE_UND":  "/home/vanessa/hulk/cocpit/cpi_data/campaigns/CRYSTAL_FACE_UND/single_imgs_v1.4.0",
    "ESCAPE":  "/home/vanessa/hulk/cocpit/cpi_data/campaigns/ESCAPE/single_imgs_v1.4.0",
    "OLYMPEX":  "/home/vanessa/hulk/cocpit/cpi_data/campaigns/OLYMPEX/single_imgs_v1.4.0",
    "POSIDON":  "/home/vanessa/hulk/cocpit/cpi_data/campaigns/POSIDON/single_imgs_v1.4.0",
}

COCPIT_CSV_ROOT = Path("/home/vanessa/hulk/cocpit/final_databases/vgg16/v1.4.0")

# Image extensions to count as "raw CPI images" -- adjust if xcite's
# single_imgs_v1.4.0 dirs use something other than .png.
IMAGE_EXTS = {".png"}

# Print a heartbeat every this many scanned directory entries, so a slow
# scan on a network filesystem is visibly progressing, not silently stuck.
PROGRESS_EVERY = 50_000


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--campaigns", nargs="+", default=None,
                   help="Restrict to these campaign keys (default: all 12 above)")
    p.add_argument("--out", type=Path, default=Path("cpi_cocpit_overlap.csv"),
                   help="Where to write the per-campaign overlap CSV")
    p.add_argument("--dump-unmatched", type=Path, default=None,
                   help="Optional directory to dump, per campaign, the list "
                        "of raw-image filenames with no COCPIT row and "
                        "COCPIT filenames with no raw image on disk")
    return p.parse_args()


def list_raw_images(img_dir: str) -> set[str]:
    """Filenames (not full paths) of every raw image file in img_dir.

    Filters by extension only (no entry.is_file()/is_dir() call) -- on a
    network filesystem (NFS/Lustre), is_file() forces a stat() syscall per
    entry, which over hundreds of thousands of files is the dominant cost
    and can make this look "stuck" for many minutes with zero output. A
    directory literally named e.g. "x.png" is not a realistic case here.
    Prints a heartbeat every PROGRESS_EVERY entries so a long-running scan
    is visibly alive rather than silent.
    """
    p = Path(img_dir)
    if not p.is_dir():
        print(f"  WARNING: directory not found: {img_dir}")
        return set()
    names = set()
    n_seen = 0
    with os.scandir(p) as it:
        for entry in it:
            n_seen += 1
            if n_seen % PROGRESS_EVERY == 0:
                print(f"    ... scanned {n_seen:,} entries so far ({len(names):,} images)", flush=True)
            if Path(entry.name).suffix.lower() in IMAGE_EXTS:
                names.add(entry.name)
    return names


def load_cocpit_filenames(campaign: str) -> set[str]:
    csv_path = COCPIT_CSV_ROOT / f"{campaign}.csv"
    if not csv_path.is_file():
        print(f"  WARNING: COCPIT CSV not found: {csv_path}")
        return set()
    df = pd.read_csv(csv_path, usecols=["filename"], low_memory=False)
    return set(df["filename"].dropna().astype(str))


def main() -> None:
    args = _parse_args()
    campaigns = args.campaigns if args.campaigns else list(CAMPAIGNS)

    if args.dump_unmatched:
        args.dump_unmatched.mkdir(parents=True, exist_ok=True)

    rows = []
    for campaign in campaigns:
        img_dir = CAMPAIGNS.get(campaign)
        if img_dir is None:
            print(f"SKIP {campaign}: not in CAMPAIGNS map")
            continue

        print(f"{campaign} ...")
        raw_images = list_raw_images(img_dir)
        cocpit_filenames = load_cocpit_filenames(campaign)

        n_raw = len(raw_images)
        n_cocpit = len(cocpit_filenames)
        overlap = raw_images & cocpit_filenames
        n_overlap = len(overlap)
        raw_only = raw_images - cocpit_filenames
        cocpit_only = cocpit_filenames - raw_images

        pct_raw_matched = round(n_overlap / n_raw * 100, 2) if n_raw else 0.0
        pct_cocpit_matched = round(n_overlap / n_cocpit * 100, 2) if n_cocpit else 0.0

        print(f"  raw images on disk:      {n_raw:>10,}")
        print(f"  COCPIT feature rows:     {n_cocpit:>10,}")
        print(f"  overlap (both):          {n_overlap:>10,}")
        print(f"  raw images w/o COCPIT:   {len(raw_only):>10,}  ({pct_raw_matched}% of raw matched)")
        print(f"  COCPIT rows w/o raw img: {len(cocpit_only):>10,}  ({pct_cocpit_matched}% of COCPIT matched)")

        rows.append({
            "campaign": campaign,
            "n_raw_images": n_raw,
            "n_cocpit_rows": n_cocpit,
            "n_overlap": n_overlap,
            "n_raw_only": len(raw_only),
            "n_cocpit_only": len(cocpit_only),
            "pct_raw_matched": pct_raw_matched,
            "pct_cocpit_matched": pct_cocpit_matched,
        })

        if args.dump_unmatched:
            with open(args.dump_unmatched / f"{campaign}_raw_only.txt", "w") as f:
                f.write("\n".join(sorted(raw_only)))
            with open(args.dump_unmatched / f"{campaign}_cocpit_only.txt", "w") as f:
                f.write("\n".join(sorted(cocpit_only)))

    result = pd.DataFrame(rows)
    result.to_csv(args.out, index=False)
    print(f"\nSaved {args.out}")
    print(result.to_string(index=False))

    if len(result):
        tot_raw = result["n_raw_images"].sum()
        tot_overlap = result["n_overlap"].sum()
        tot_cocpit = result["n_cocpit_rows"].sum()
        print(f"\nOverall: {tot_overlap:,} / {tot_raw:,} raw images have a COCPIT row "
              f"({round(tot_overlap / tot_raw * 100, 2) if tot_raw else 0.0}%)")
        print(f"Overall: {tot_overlap:,} / {tot_cocpit:,} COCPIT rows have a raw image "
              f"({round(tot_overlap / tot_cocpit * 100, 2) if tot_cocpit else 0.0}%)")


if __name__ == "__main__":
    main()
