#!/usr/bin/env python3
"""
Data Tiers Schematic (L0 -> L1 -> L2)
========================================
A conceptual, paper-ready schematic of this pipeline's three data tiers:
what each tier is, how you get from one to the next, and current
row/column/campaign counts for each. Unlike
`scripts/diagnose_data_tiers.py`'s per-campaign funnel bar chart (a
diagnostic figure), this is a small, static, three-box diagram intended
for a manuscript's methods section, meant to sit alongside a table with
the same numbers (e.g. `latex/section-dataset-construction-and-
harmonization-2026-08-31.tex`'s `tab:data-tiers`).

Row/column/campaign counts are computed live from the current parquets
(not hardcoded), so this figure stays in sync if the dataset is rebuilt.

Output:
  figs/plot_data_tiers_schematic/<timestamp>/data_tiers_schematic.png

Usage:
    python scripts/plot_data_tiers_schematic.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrow, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.log_paths import timestamp as _run_timestamp, update_latest  # noqa: E402

CORE_COLS = ["Tair_C", "P_hPa", "Si", "qv", "Lat", "Lon", "Alt_m"]

# Same 3-color subset of the Okabe-Ito colorblind-safe palette used in
# scripts/plot_campaign_geography.py, kept consistent across this
# dataset's paper figures.
TIER_COLOR = {"L0": "#56B4E9", "L1": "#E69F00", "L2": "#009E73"}


def _parse_args() -> argparse.Namespace:
    ts = _run_timestamp()
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--l0", type=Path, default=ROOT / "data" / "out" / "combined_env_data.parquet")
    p.add_argument("--l1", type=Path, default=ROOT / "data" / "out" / "combined_env_data_L1.parquet")
    p.add_argument("--l2", type=Path, default=ROOT / "data" / "out" / "combined_env_data_L2.parquet")
    p.add_argument("--out", type=Path,
                   default=ROOT / "figs" / "plot_data_tiers_schematic" / ts)
    return p.parse_args()


def _tier_stats(path: Path) -> dict:
    df = pd.read_parquet(path)
    return {
        "rows": len(df),
        "cols": df.shape[1],
        "campaigns": df["Campaign"].nunique(),
    }


def _fmt(n: int) -> str:
    return f"{n:,}"


def main() -> None:
    args = _parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("Loading L0/L1/L2 for live row/column/campaign counts ...")
    stats = {
        "L0": _tier_stats(args.l0),
        "L1": _tier_stats(args.l1),
        "L2": _tier_stats(args.l2),
    }
    for tier, s in stats.items():
        print(f"  {tier}: {s['rows']:,} rows, {s['cols']} columns, {s['campaigns']} campaigns")

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    box_w, box_h = 2.6, 3.2
    box_y = 0.9
    box_x = {"L0": 0.3, "L1": 3.7, "L2": 7.1}

    definitions = {
        "L0": "Every whole second where\nany instrument in a campaign\nreported a measurement",
        "L1": "One row per CPI particle\nimage, joined to its exact-\nsecond L0 record",
        "L2": "L1 filtered to rows with all\nseven core variables present\n(Tair_C, P_hPa, Si, qv,\nLat, Lon, Alt_m)",
    }

    for tier, x in box_x.items():
        s = stats[tier]
        box = FancyBboxPatch(
            (x, box_y), box_w, box_h,
            boxstyle="round,pad=0.08,rounding_size=0.12",
            facecolor=TIER_COLOR[tier], edgecolor="black", linewidth=1.2,
            alpha=0.85, zorder=2,
        )
        ax.add_patch(box)
        ax.text(x + box_w / 2, box_y + box_h - 0.35, tier,
                 ha="center", va="top", fontsize=20, fontweight="bold", zorder=3)
        ax.text(x + box_w / 2, box_y + box_h - 0.95, definitions[tier],
                 ha="center", va="top", fontsize=9.5, zorder=3, linespacing=1.4)
        stat_text = (f"{_fmt(s['rows'])} rows\n"
                     f"{s['cols']} columns\n"
                     f"{s['campaigns']} campaigns")
        ax.text(x + box_w / 2, box_y + 0.62, stat_text,
                 ha="center", va="center", fontsize=10, fontweight="bold",
                 zorder=3, linespacing=1.6)

    arrow_specs = [
        (box_x["L0"] + box_w, box_x["L1"], "exact-second join\nwith CPI image\ntimestamps"),
        (box_x["L1"] + box_w, box_x["L2"], "core-variable\ncompleteness\nfilter"),
    ]
    for x_start, x_end, label in arrow_specs:
        arrow = FancyArrow(
            x_start + 0.08, box_y + box_h / 2, x_end - x_start - 0.16, 0,
            width=0.03, head_width=0.28, head_length=0.18,
            length_includes_head=True, facecolor="dimgray", edgecolor="dimgray",
            zorder=2,
        )
        ax.add_patch(arrow)
        ax.text((x_start + x_end) / 2 + 0.08, box_y + box_h / 2 + 0.55, label,
                 ha="center", va="bottom", fontsize=8.5, color="dimgray",
                 style="italic", linespacing=1.3)

    ax.text(
        5.0, 0.35,
        "Three campaigns (ESCAPE, OLYMPEX, POSIDON) have environmental data only and contribute 0 rows to L1/L2.\n"
        "MPACE reaches L1 but flew no water-vapor instrument, so it fails L2's completeness filter and drops out there.",
        ha="center", va="center", fontsize=8, color="dimgray", style="italic", linespacing=1.5,
    )

    fig.tight_layout()
    out_path = args.out / "data_tiers_schematic.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Saved {out_path}")

    update_latest(args.out.parent, args.out)
    print(f"\nLatest run: {args.out.parent / 'latest'} -> {args.out.name}")


if __name__ == "__main__":
    main()
