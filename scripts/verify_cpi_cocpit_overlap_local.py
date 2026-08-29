#!/usr/bin/env python3
"""
Verify overlap between CPI image filenames and COCPIT derived-feature rows (local)
=====================================================================================
Local-machine counterpart to `xcite_verify_cpi_cocpit_overlap.py` (run on
the xcite HPC cluster against actual raw image directories). This machine
doesn't have the raw single-particle image files, but
`data/raw/cpi_embeddings_timestamps.csv` is this pipeline's own canonical
CPI image manifest -- confirmed to carry the exact same per-campaign
filename counts as xcite's raw `single_imgs_v1.4.0` directories (e.g.
CRYSTAL_FACE_UND 1,617,826, ARM 295,703, MC3E 187,558 -- identical on both
machines), so it's a reliable stand-in for "does a raw image exist" without
needing the image files themselves.

For each campaign, compares the embeddings CSV's `filename` set against
that campaign's COCPIT v1.4.0 derived-feature CSV's `filename` set, and
reports overlap both directions -- same method, same output schema as the
xcite script, for direct comparison against xcite's `overlap_report.csv`.

Outputs:
  logs/verify_cpi_cocpit_overlap_local/<timestamp>/overlap_report.csv
  logs/verify_cpi_cocpit_overlap_local/<timestamp>/<campaign>_raw_only.txt (optional, --dump-unmatched)
  logs/verify_cpi_cocpit_overlap_local/<timestamp>/<campaign>_cocpit_only.txt (optional, --dump-unmatched)

Usage:
    python scripts/verify_cpi_cocpit_overlap_local.py
    python scripts/verify_cpi_cocpit_overlap_local.py --campaigns ARM MACPEX
    python scripts/verify_cpi_cocpit_overlap_local.py --dump-unmatched
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.log_paths import timestamp as _run_timestamp, update_latest

EMBEDDINGS_CSV = ROOT / "data" / "raw" / "cpi_embeddings_timestamps.csv"
COCPIT_CSV_ROOT = Path("/Users/josephko/research/cocpit/final_databases/vgg16/v1.4.0")


def _parse_args() -> argparse.Namespace:
    ts = _run_timestamp()
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--embeddings-csv", type=Path, default=EMBEDDINGS_CSV)
    p.add_argument("--cocpit-root", type=Path, default=COCPIT_CSV_ROOT)
    p.add_argument("--campaigns", nargs="+", default=None,
                   help="Restrict to these COCPIT-style campaign keys "
                        "(default: every campaign in the embeddings CSV)")
    p.add_argument("--out", type=Path,
                   default=ROOT / "logs" / "verify_cpi_cocpit_overlap_local" / ts)
    p.add_argument("--dump-unmatched", action="store_true",
                   help="Also write, per campaign, the list of embeddings-"
                        "only filenames and COCPIT-only filenames")
    return p.parse_args()


def load_embeddings_filenames(embeddings_csv: Path) -> dict[str, set[str]]:
    """COCPIT-style campaign -> set of filenames, straight from the
    embeddings CSV's own `campaign` column (already COCPIT-style naming,
    e.g. AIRS_II, CRYSTAL_FACE_NASA -- no mapping needed)."""
    df = pd.read_csv(embeddings_csv, usecols=["campaign", "filename"])
    return {camp: set(sub["filename"]) for camp, sub in df.groupby("campaign")}


def load_cocpit_filenames(cocpit_root: Path, campaign: str) -> set[str]:
    csv_path = cocpit_root / f"{campaign}.csv"
    if not csv_path.is_file():
        print(f"  WARNING: COCPIT CSV not found: {csv_path}")
        return set()
    df = pd.read_csv(csv_path, usecols=["filename"], low_memory=False)
    return set(df["filename"].dropna().astype(str))


def main() -> None:
    args = _parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Loading embeddings manifest from {args.embeddings_csv} ...")
    embeddings_by_campaign = load_embeddings_filenames(args.embeddings_csv)
    campaigns = args.campaigns if args.campaigns else sorted(embeddings_by_campaign)
    print(f"  {len(campaigns)} campaigns: {campaigns}")

    rows = []
    for campaign in campaigns:
        print(f"\n{campaign} ...")
        raw_images = embeddings_by_campaign.get(campaign, set())
        cocpit_filenames = load_cocpit_filenames(args.cocpit_root, campaign)

        n_raw = len(raw_images)
        n_cocpit = len(cocpit_filenames)
        overlap = raw_images & cocpit_filenames
        n_overlap = len(overlap)
        raw_only = raw_images - cocpit_filenames
        cocpit_only = cocpit_filenames - raw_images

        pct_raw_matched = round(n_overlap / n_raw * 100, 2) if n_raw else 0.0
        pct_cocpit_matched = round(n_overlap / n_cocpit * 100, 2) if n_cocpit else 0.0

        print(f"  embeddings filenames:    {n_raw:>10,}")
        print(f"  COCPIT feature rows:     {n_cocpit:>10,}")
        print(f"  overlap (both):          {n_overlap:>10,}")
        print(f"  embeddings w/o COCPIT:   {len(raw_only):>10,}  ({pct_raw_matched}% of embeddings matched)")
        print(f"  COCPIT rows w/o embed:   {len(cocpit_only):>10,}  ({pct_cocpit_matched}% of COCPIT matched)")

        rows.append({
            "campaign": campaign,
            "n_embeddings_filenames": n_raw,
            "n_cocpit_rows": n_cocpit,
            "n_overlap": n_overlap,
            "n_embeddings_only": len(raw_only),
            "n_cocpit_only": len(cocpit_only),
            "pct_embeddings_matched": pct_raw_matched,
            "pct_cocpit_matched": pct_cocpit_matched,
        })

        if args.dump_unmatched:
            (args.out / f"{campaign}_embeddings_only.txt").write_text("\n".join(sorted(raw_only)))
            (args.out / f"{campaign}_cocpit_only.txt").write_text("\n".join(sorted(cocpit_only)))

    result = pd.DataFrame(rows)
    out_csv = args.out / "overlap_report.csv"
    result.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")
    print(result.to_string(index=False))

    if len(result):
        tot_raw = result["n_embeddings_filenames"].sum()
        tot_overlap = result["n_overlap"].sum()
        tot_cocpit = result["n_cocpit_rows"].sum()
        print(f"\nOverall: {tot_overlap:,} / {tot_raw:,} embeddings filenames have a COCPIT row "
              f"({round(tot_overlap / tot_raw * 100, 2) if tot_raw else 0.0}%)")
        print(f"Overall: {tot_overlap:,} / {tot_cocpit:,} COCPIT rows have an embeddings filename "
              f"({round(tot_overlap / tot_cocpit * 100, 2) if tot_cocpit else 0.0}%)")

    update_latest(args.out.parent, args.out)
    print(f"\nLatest run: {args.out.parent / 'latest'} -> {args.out.name}")


if __name__ == "__main__":
    main()
