#!/usr/bin/env python3
"""
Verify overlap between raw CPI image files (single_imgs_v1.4.0) and COCPIT
v3.1.0 derived-feature rows
=============================================================================
Standalone script for xcite (HPC cluster) -- no dependency on the
cpi-thermo repo, only pandas (stdlib otherwise). Variant of
xcite_verify_cpi_cocpit_overlap.py that checks the SAME raw
single_imgs_v1.4.0 image directories against the v3.1.0 derived-feature
CSVs instead of v1.4.0's.

*** IMPORTANT KNOWN CAVEAT, read before interpreting results ***
v3.1.0's filename column was found (separately, on a different machine) to
use a 4-field format missing the millisecond segment entirely, e.g.
"2011_0523_183354_0.png", versus single_imgs_v1.4.0's 5-field format,
e.g. "2011_0523_163505_942_29.png" (YYYY_MMDD_HHMMSS_ms_particleID).
A direct filename-set join across that format difference will show near-0%
overlap REGARDLESS of how much of the raw archive v3.1.0 actually
processed -- that would be a format mismatch, not evidence v3.1.0 covers
little of the data. This script still runs the same direct-match check
(useful to *confirm* the format break is real and see its magnitude), but
prints this warning again in the output so a near-0% result isn't
misread. If you need an apples-to-apples coverage number for v3.1.0, the
millisecond field would need to be stripped from single_imgs_v1.4.0's
names (or reconstructed on v3.1.0's side) before matching -- not done
here, since that requires deciding how to disambiguate multiple particles
from the same sheet whose only difference was the stripped ms field.

v3.1.0 also only has 11 of the 15 campaigns (confirmed via `ls` on xcite):
AIRS_II, ARM, ATTREX, CRYSTAL_FACE_NASA, CRYSTAL_FACE_UND, ESCAPE, ICE_L,
IPHEX, ISDAC, MACPEX, MC3E -- MIDCIX, MPACE, OLYMPEX, POSIDON are absent
and excluded from the default campaign list below.

Usage:
    python xcite_verify_cpi_cocpit_overlap_v3.py
    python xcite_verify_cpi_cocpit_overlap_v3.py --campaigns ARM MACPEX
    python xcite_verify_cpi_cocpit_overlap_v3.py --out overlap_report_v3.1.0.csv
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

# campaign -> raw single-particle image directory (still single_imgs_v1.4.0
# -- v3.1.0 has no separate raw-image directory of its own on xcite, per
# the user's request to check v3.1.0's CSV rows against these same dirs)
CAMPAIGNS: dict[str, str] = {
    "MC3E":              "/home/vanessa/hulk/cocpit/cpi_data/campaigns/MC3E/single_imgs_v1.4.0",
    "ARM":               "/home/vanessa/hulk/cocpit/cpi_data/campaigns/ARM/single_imgs_v1.4.0",
    "IPHEX":             "/home/vanessa/hulk/cocpit/cpi_data/campaigns/IPHEX/single_imgs_v1.4.0",
    "AIRS_II":           "/home/vanessa/hulk/cocpit/cpi_data/campaigns/AIRS_II/single_imgs_v1.4.0",
    "ICE_L":             "/home/vanessa/hulk/cocpit/cpi_data/campaigns/ICE_L/single_imgs_v1.4.0",
    "CRYSTAL_FACE_NASA": "/home/vanessa/hulk/cocpit/cpi_data/campaigns/CRYSTAL_FACE_NASA/single_imgs_v1.4.0",
    "MACPEX":            "/home/vanessa/hulk/cocpit/cpi_data/campaigns/MACPEX/single_imgs_v1.4.0",
    "ISDAC":             "/home/vanessa/hulk/cocpit/cpi_data/campaigns/ISDAC/single_imgs_v1.4.0",
    "ATTREX":            "/home/vanessa/hulk/cocpit/cpi_data/campaigns/ATTREX/single_imgs_v1.4.0",
    "CRYSTAL_FACE_UND":  "/home/vanessa/hulk/cocpit/cpi_data/campaigns/CRYSTAL_FACE_UND/single_imgs_v1.4.0",
    "ESCAPE":            "/home/vanessa/hulk/cocpit/cpi_data/campaigns/ESCAPE/single_imgs_v1.4.0",
    # MIDCIX, MPACE, OLYMPEX, POSIDON have no v3.1.0 CSV -- omitted, not
    # just left to warn-and-skip, so a default run doesn't waste time
    # scandir-ing large raw directories with nothing to compare against.
}

COCPIT_CSV_ROOT = Path("/home/vanessa/hulk/cocpit/final_databases/vgg16/v3.1.0")

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
                   help="Restrict to these campaign keys (default: all 11 above)")
    p.add_argument("--out", type=Path, default=Path("overlap_report_v3.1.0.csv"),
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

    print("=" * 78)
    print("CAVEAT: v3.1.0 filenames are known to be missing the millisecond")
    print("field vs. single_imgs_v1.4.0's naming -- a low/near-0% match rate")
    print("below may indicate that format break, not a real coverage gap.")
    print("See this script's docstring for detail.")
    print("=" * 78)

    if args.dump_unmatched:
        args.dump_unmatched.mkdir(parents=True, exist_ok=True)

    rows = []
    for campaign in campaigns:
        img_dir = CAMPAIGNS.get(campaign)
        if img_dir is None:
            print(f"SKIP {campaign}: not in CAMPAIGNS map")
            continue

        print(f"\n{campaign} ...")
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

        print(f"  raw images (single_imgs_v1.4.0): {n_raw:>10,}")
        print(f"  COCPIT v3.1.0 feature rows:       {n_cocpit:>10,}")
        print(f"  overlap (both):                   {n_overlap:>10,}")
        print(f"  raw images w/o COCPIT:            {len(raw_only):>10,}  ({pct_raw_matched}% of raw matched)")
        print(f"  COCPIT rows w/o raw img:           {len(cocpit_only):>10,}  ({pct_cocpit_matched}% of COCPIT matched)")

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
        print(f"\nOverall: {tot_overlap:,} / {tot_raw:,} raw images have a COCPIT v3.1.0 row "
              f"({round(tot_overlap / tot_raw * 100, 2) if tot_raw else 0.0}%)")
        print(f"Overall: {tot_overlap:,} / {tot_cocpit:,} COCPIT v3.1.0 rows have a raw image "
              f"({round(tot_overlap / tot_cocpit * 100, 2) if tot_cocpit else 0.0}%)")
        print("\nIf both percentages above are near 0%, re-read the CAVEAT at the "
              "top of this script's output before concluding v3.1.0 has little data.")


if __name__ == "__main__":
    main()
