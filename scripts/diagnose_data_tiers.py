#!/usr/bin/env python3
"""
L0/L1/L2 Data Tier Diagnostic
==============================
Reports on the three data tiers produced by scripts/build_data_tiers.py:

  L0 - data/out/combined_env_data.parquet: every whole second where *any*
       instrument in a campaign reported *anything* (union of all
       instrument timestamps, exact-second joins only -- no merge_asof
       tolerance; see docs/decisions/2026-07-07-exact-second-merge-rewrite.md).
  L1 - data/out/combined_env_data_L1.parquet: one row per CPI particle
       image, joined to its exact-second L0 env record (`cpi_filename`
       column identifies the source image).
  L2 - data/out/combined_env_data_L2.parquet: L1 filtered to rows with a
       complete record (every core variable present).

For each tier this measures: row counts, date coverage, and per-campaign
coverage % for the core variables (Tair_C, P_hPa, Si, qv, Lat, Lon, Alt_m).

Outputs (logs/diagnose_data_tiers/<timestamp>/ and
figs/diagnose_data_tiers/<timestamp>/, with `latest` symlinks):
  tier_row_counts.csv          - n_L0/L1/L2 and date range per campaign
  tier_variable_coverage.csv   - per-campaign, per-tier, per-variable % valid
  tier_funnel.png              - L0->L1->L2 row-count funnel per campaign
  tier_coverage_heatmap.png    - per-tier coverage heatmap (campaign x variable)

Usage:
    python scripts/diagnose_data_tiers.py
    python scripts/diagnose_data_tiers.py --l0 data/out/combined_env_data.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from parsers.cpi_timestamps import load_cpi_embeddings_timestamps
from scripts.log_paths import timestamp as _run_timestamp, update_latest

# Core "complete record" columns -- mirrors build_data_tiers.py's CORE_COLS.
CORE_COLS = ["Tair_C", "P_hPa", "Si", "qv", "Lat", "Lon", "Alt_m"]

CAMPAIGN_ORDER = [
    "ARM", "CRYSTAL-FACE-NASA", "CRYSTAL-FACE-UND", "MIDCIX", "MPACE",
    "AIRS-II", "ICE-L", "ISDAC", "MACPEX", "MC3E",
    "ATTREX", "IPHEX", "OLYMPEX", "POSIDON", "ESCAPE",
]

STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor":   "#f8f8f8",
    "axes.grid":        True,
    "grid.color":       "white",
    "grid.linewidth":   0.8,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "font.size":        9,
}

TIER_COLORS = {"L0": "#4c72b0", "L1": "#dd8452", "L2": "#55a868"}


def _parse_args() -> argparse.Namespace:
    ts = _run_timestamp()
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--l0", type=Path,
                   default=ROOT / "data" / "out" / "combined_env_data.parquet")
    p.add_argument("--l1", type=Path,
                   default=ROOT / "data" / "out" / "combined_env_data_L1.parquet")
    p.add_argument("--l2", type=Path,
                   default=ROOT / "data" / "out" / "combined_env_data_L2.parquet")
    p.add_argument("--cpi", type=Path,
                   default=ROOT / "data" / "raw" / "cpi_embeddings_timestamps.csv")
    p.add_argument("--out", type=Path,
                   default=ROOT / "logs" / "diagnose_data_tiers" / ts)
    p.add_argument("--figs", type=Path,
                   default=ROOT / "figs" / "diagnose_data_tiers" / ts)
    return p.parse_args()


def row_counts_table(
    tiers: dict[str, pd.DataFrame],
    campaigns: list[str],
    images_per_campaign: pd.Series,
) -> pd.DataFrame:
    """n_L1 is an image count (one row per CPI image), not a second count,
    so it's compared against the true CPI image total (pct_images_matched)
    rather than against n_L0 -- an L1/L0 ratio would be meaningless (and
    can exceed 100%) whenever a campaign has multiple images per second.
    """
    rows = []
    for camp in campaigns:
        row = {"Campaign": camp, "n_cpi_images": int(images_per_campaign.get(camp, 0))}
        for tier_name, df in tiers.items():
            sub = df[df["Campaign"] == camp] if not df.empty else df.iloc[0:0]
            row[f"n_{tier_name}"] = len(sub)
            if len(sub):
                row[f"{tier_name}_date_start"] = str(sub["Timestamp"].min().date())
                row[f"{tier_name}_date_end"] = str(sub["Timestamp"].max().date())
            else:
                row[f"{tier_name}_date_start"] = ""
                row[f"{tier_name}_date_end"] = ""
        n_images, n1 = row["n_cpi_images"], row["n_L1"]
        row["pct_images_matched"] = round(n1 / n_images * 100, 2) if n_images else 0.0
        row["pct_L2_of_L1"] = round(row["n_L2"] / n1 * 100, 2) if n1 else 0.0
        rows.append(row)

    totals = {"Campaign": "TOTAL", "n_cpi_images": int(images_per_campaign.sum())}
    for tier_name, df in tiers.items():
        totals[f"n_{tier_name}"] = len(df)
        totals[f"{tier_name}_date_start"] = str(df["Timestamp"].min().date()) if len(df) else ""
        totals[f"{tier_name}_date_end"] = str(df["Timestamp"].max().date()) if len(df) else ""
    n_images, n1 = totals["n_cpi_images"], totals["n_L1"]
    totals["pct_images_matched"] = round(n1 / n_images * 100, 2) if n_images else 0.0
    totals["pct_L2_of_L1"] = round(totals["n_L2"] / n1 * 100, 2) if n1 else 0.0
    rows.append(totals)

    return pd.DataFrame(rows)


def variable_coverage_table(tiers: dict[str, pd.DataFrame], campaigns: list[str]) -> pd.DataFrame:
    rows = []
    for camp in campaigns:
        for tier_name, df in tiers.items():
            sub = df[df["Campaign"] == camp] if not df.empty else df.iloc[0:0]
            row = {"Campaign": camp, "tier": tier_name, "n_rows": len(sub)}
            for col in CORE_COLS:
                if len(sub) and col in sub.columns:
                    row[f"pct_{col}"] = round(sub[col].notna().mean() * 100, 2)
                else:
                    row[f"pct_{col}"] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def plot_funnel(counts: pd.DataFrame, campaigns: list[str], figs_dir: Path) -> None:
    plot_df = counts[counts["Campaign"] != "TOTAL"].set_index("Campaign").loc[
        [c for c in campaigns if c in counts["Campaign"].values]
    ]
    x = np.arange(len(plot_df))
    width = 0.26

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(13, 5.5), constrained_layout=True)
        for i, tier in enumerate(["L0", "L1", "L2"]):
            vals = plot_df[f"n_{tier}"].values
            ax.bar(x + (i - 1) * width, vals, width, label=tier, color=TIER_COLORS[tier])
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df.index, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("rows (log scale)")
        ax.set_title("L0 -> L1 -> L2 row-count funnel by campaign", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        out_p = figs_dir / "tier_funnel.png"
        fig.savefig(out_p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_p}")


def plot_coverage_heatmap(var_cov: pd.DataFrame, campaigns: list[str], figs_dir: Path) -> None:
    cmap = LinearSegmentedColormap.from_list("cov", ["#f0f0f0", "#4c72b0"])
    tiers = ["L0", "L1", "L2"]

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(16, 6), constrained_layout=True)
        for ax, tier in zip(axes, tiers):
            sub = var_cov[var_cov["tier"] == tier].set_index("Campaign")
            sub = sub.loc[[c for c in campaigns if c in sub.index]]
            mat = sub[[f"pct_{c}" for c in CORE_COLS]].values.astype(float)
            im = ax.imshow(mat, aspect="auto", vmin=0, vmax=100, cmap=cmap)
            ax.set_xticks(range(len(CORE_COLS)))
            ax.set_xticklabels(CORE_COLS, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(len(sub.index)))
            ax.set_yticklabels(sub.index, fontsize=7)
            ax.set_title(tier, fontsize=10, fontweight="bold")
            for r in range(mat.shape[0]):
                for c in range(mat.shape[1]):
                    v = mat[r, c]
                    if not np.isnan(v):
                        ax.text(c, r, f"{v:.0f}", ha="center", va="center", fontsize=6,
                                 color="white" if v > 55 else "black")
        fig.colorbar(im, ax=axes, shrink=0.6, label="% valid")
        fig.suptitle("Core-variable coverage by tier", fontsize=11, fontweight="bold")
        out_p = figs_dir / "tier_coverage_heatmap.png"
        fig.savefig(out_p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_p}")


def main() -> None:
    args = _parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.figs.mkdir(parents=True, exist_ok=True)

    print(f"Loading L0 from {args.l0} ...")
    l0 = pd.read_parquet(args.l0)
    print(f"Loading L1 from {args.l1} ...")
    l1 = pd.read_parquet(args.l1)
    print(f"Loading L2 from {args.l2} ...")
    l2 = pd.read_parquet(args.l2)
    tiers = {"L0": l0, "L1": l1, "L2": l2}
    print(f"  L0: {len(l0):,}  L1: {len(l1):,}  L2: {len(l2):,}\n")

    print(f"Loading CPI timestamps from {args.cpi} ...")
    cpi = load_cpi_embeddings_timestamps(args.cpi)
    images_per_campaign = cpi.groupby("campaign_env").size()

    campaigns = [c for c in CAMPAIGN_ORDER if c in l0["Campaign"].unique()]

    counts = row_counts_table(tiers, campaigns, images_per_campaign)
    counts_path = args.out / "tier_row_counts.csv"
    counts.to_csv(counts_path, index=False)
    print(f"Saved {counts_path}")
    print(counts[["Campaign", "n_cpi_images", "n_L0", "n_L1", "n_L2", "pct_images_matched", "pct_L2_of_L1"]]
          .to_string(index=False))

    var_cov = variable_coverage_table(tiers, campaigns)
    var_cov_path = args.out / "tier_variable_coverage.csv"
    var_cov.to_csv(var_cov_path, index=False)
    print(f"\nSaved {var_cov_path}")

    plot_funnel(counts, campaigns, args.figs)
    plot_coverage_heatmap(var_cov, campaigns, args.figs)

    update_latest(args.out.parent, args.out)
    update_latest(args.figs.parent, args.figs)
    print(f"\nLatest run: {args.out.parent / 'latest'} -> {args.out.name}")
    print(f"Latest figs: {args.figs.parent / 'latest'} -> {args.figs.name}")


if __name__ == "__main__":
    main()
