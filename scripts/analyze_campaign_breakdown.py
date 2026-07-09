#!/usr/bin/env python3
"""
Per-campaign descriptive breakdown across L0/L1/L2
====================================================
Answers the basic "what's actually in this dataset, campaign by campaign?"
questions that `analyze_data_tiers.py`'s whole-tier descriptive stats don't
surface on their own: how many rows does each campaign contribute at each
tier, what fraction of the tier's total that is, how complete is each core
variable *within* that campaign (not just pooled across all 15), what date
range and how many distinct flight-dates each campaign covers, and how much
of a campaign's L0 rows survive the L0->L1->L2 funnel (CPI-image join, then
core-variable-complete filter).

Reuses (does not reimplement) `descriptive_by_campaign` and its CORE_COLS /
CAMPAIGN_ORDER / COLORS / STYLE constants from analyze_data_tiers.py, so the
per-campaign completeness numbers here are identical to (not a re-derivation
of) `<tier>_descriptive_by_campaign.csv` -- this script adds row-share,
date-range, and funnel-retention on top, plus visualizations of all of it.

Outputs (logs/analyze_campaign_breakdown/<ts>/ and
figs/analyze_campaign_breakdown/<ts>/, with `latest` symlinks):
  campaign_row_counts_and_share.csv   - n_rows + % of tier total, per campaign, L0/L1/L2
  campaign_date_ranges.csv            - first/last Timestamp + n unique flight-dates, per campaign, per tier
  campaign_funnel_retention.csv       - per campaign: L0->L1 and L1->L2 retention %
  campaign_completeness_l0.csv        - alias of analyze_data_tiers.py's l0_descriptive_by_campaign.csv
  campaign_completeness_l1.csv        - same, L1
  campaign_completeness_l2.csv        - same, L2
  run_config.json

  01_row_counts_by_campaign_and_tier.png   - grouped bar, log scale
  02_funnel_retention_by_campaign.png      - L0->L1->L2 retention %, per campaign
  03_completeness_heatmap_l0.png           - campaign x core-variable % valid
  04_completeness_heatmap_l1.png           - same, L1
  05_completeness_heatmap_l2.png           - same, L2 (should be all 100% by construction)
  06_date_range_by_campaign.png            - horizontal timeline, L0 first/last Timestamp per campaign

Usage:
    python scripts/analyze_campaign_breakdown.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_data_tiers import descriptive_by_campaign, CORE_COLS, CAMPAIGN_ORDER, COLORS, STYLE
from scripts.log_paths import timestamp as _run_timestamp, update_latest

TIER_PATHS = {
    "L0": ROOT / "data" / "out" / "combined_env_data.parquet",
    "L1": ROOT / "data" / "out" / "combined_env_data_L1.parquet",
    "L2": ROOT / "data" / "out" / "combined_env_data_L2.parquet",
}
TIER_ORDER = ["L0", "L1", "L2"]
LOAD_COLS = ["Campaign", "Timestamp"] + CORE_COLS


def _parse_args() -> argparse.Namespace:
    ts = _run_timestamp()
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    for tier, default in TIER_PATHS.items():
        p.add_argument(f"--{tier.lower()}", type=Path, default=default)
    p.add_argument("--out", type=Path, default=ROOT / "logs" / "analyze_campaign_breakdown" / ts)
    p.add_argument("--figs", type=Path, default=ROOT / "figs" / "analyze_campaign_breakdown" / ts)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Per-tier computations
# ---------------------------------------------------------------------------

def row_counts_and_share(df: pd.DataFrame, tier: str) -> pd.DataFrame:
    total = len(df)
    counts = df["Campaign"].value_counts()
    rows = [{"tier": tier, "Campaign": camp, "n_rows": int(counts.get(camp, 0)),
             "pct_of_tier_total": round(100 * counts.get(camp, 0) / total, 2)}
            for camp in CAMPAIGN_ORDER]
    return pd.DataFrame(rows)


def date_ranges(df: pd.DataFrame, tier: str) -> pd.DataFrame:
    rows = []
    for camp, sub in df.groupby("Campaign"):
        dates = sub["Timestamp"].dt.date
        rows.append({
            "tier": tier, "Campaign": camp,
            "first_timestamp": sub["Timestamp"].min(), "last_timestamp": sub["Timestamp"].max(),
            "n_unique_flight_dates": dates.nunique(),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_row_counts(counts_by_tier: dict[str, pd.DataFrame], figs_dir: Path) -> None:
    campaigns = [c for c in CAMPAIGN_ORDER]
    x = np.arange(len(campaigns))
    width = 0.25
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(13, 5.5), constrained_layout=True)
        for i, tier in enumerate(TIER_ORDER):
            df = counts_by_tier[tier].set_index("Campaign").reindex(campaigns)
            ax.bar(x + (i - 1) * width, df["n_rows"].fillna(0), width=width, label=tier)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(campaigns, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("n rows (log scale)")
        ax.set_title("Row count by campaign and data tier", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        out_p = figs_dir / "01_row_counts_by_campaign_and_tier.png"
        fig.savefig(out_p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_p}")


def plot_funnel_retention(funnel_df: pd.DataFrame, figs_dir: Path) -> None:
    campaigns = [c for c in CAMPAIGN_ORDER if c in funnel_df["Campaign"].values]
    df = funnel_df.set_index("Campaign").reindex(campaigns)
    x = np.arange(len(campaigns))
    width = 0.35
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
        ax.bar(x - width / 2, df["l0_to_l1_pct"], width=width, label="L0 -> L1 (CPI image join)", color="#4c72b0")
        ax.bar(x + width / 2, df["l1_to_l2_pct"], width=width, label="L1 -> L2 (core-vars complete)", color="#dd8452")
        ax.set_xticks(x)
        ax.set_xticklabels(campaigns, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("retention %")
        ax.set_title("Funnel retention by campaign", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        out_p = figs_dir / "02_funnel_retention_by_campaign.png"
        fig.savefig(out_p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_p}")


def plot_completeness_heatmap(by_campaign: pd.DataFrame, tier: str, index_num: int, figs_dir: Path) -> None:
    pivot = by_campaign.pivot(index="Campaign", columns="variable", values="pct_valid")
    pivot = pivot.reindex([c for c in CAMPAIGN_ORDER if c in pivot.index])
    pivot = pivot[CORE_COLS]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
        ax.set_xticks(range(len(CORE_COLS)))
        ax.set_xticklabels(CORE_COLS, fontsize=8, rotation=30, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=8)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6.5,
                            color="black" if v > 40 else "white")
        ax.set_title(f"{tier}: % valid by campaign x core variable", fontsize=10, fontweight="bold")
        fig.colorbar(im, ax=ax, shrink=0.8, label="% valid")
        out_p = figs_dir / f"0{index_num}_completeness_heatmap_{tier.lower()}.png"
        fig.savefig(out_p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_p}")


def plot_date_ranges(l0_dates: pd.DataFrame, figs_dir: Path) -> None:
    df = l0_dates.set_index("Campaign").reindex(CAMPAIGN_ORDER).dropna(subset=["first_timestamp"])
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
        y = np.arange(len(df))
        starts = mdates.date2num(df["first_timestamp"].dt.tz_localize(None))
        ends = mdates.date2num(df["last_timestamp"].dt.tz_localize(None))
        ax.barh(y, ends - starts, left=starts, height=0.5,
                color=[COLORS.get(c, "gray") for c in df.index])
        ax.set_yticks(y)
        ax.set_yticklabels(df.index, fontsize=8)
        ax.xaxis_date()
        ax.set_xlabel("date (UTC)")
        ax.set_title("Campaign date coverage (L0, first-to-last Timestamp)", fontsize=10, fontweight="bold")
        ax.invert_yaxis()
        out_p = figs_dir / "06_date_range_by_campaign.png"
        fig.savefig(out_p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_p}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.figs.mkdir(parents=True, exist_ok=True)

    paths = {"L0": args.l0, "L1": args.l1, "L2": args.l2}
    counts_by_tier: dict[str, pd.DataFrame] = {}
    completeness_by_tier: dict[str, pd.DataFrame] = {}
    dates_by_tier: dict[str, pd.DataFrame] = {}
    total_rows_by_tier: dict[str, int] = {}

    for tier in TIER_ORDER:
        print(f"Loading {tier} from {paths[tier]} ...")
        df = pd.read_parquet(paths[tier], columns=LOAD_COLS)
        total_rows_by_tier[tier] = len(df)
        print(f"  {tier}: {len(df):,} rows")

        counts_by_tier[tier] = row_counts_and_share(df, tier)
        completeness_by_tier[tier] = descriptive_by_campaign(df)
        completeness_by_tier[tier].to_csv(args.out / f"campaign_completeness_{tier.lower()}.csv", index=False)
        dates_by_tier[tier] = date_ranges(df, tier)
        del df

    counts_all = pd.concat(counts_by_tier.values(), ignore_index=True)
    counts_all.to_csv(args.out / "campaign_row_counts_and_share.csv", index=False)

    dates_all = pd.concat(dates_by_tier.values(), ignore_index=True)
    dates_all.to_csv(args.out / "campaign_date_ranges.csv", index=False)

    print("\nComputing L0->L1->L2 funnel retention per campaign ...")
    l0 = counts_by_tier["L0"].set_index("Campaign")["n_rows"]
    l1 = counts_by_tier["L1"].set_index("Campaign")["n_rows"]
    l2 = counts_by_tier["L2"].set_index("Campaign")["n_rows"]
    funnel_df = pd.DataFrame({
        "Campaign": CAMPAIGN_ORDER,
        "l0_n_rows": [int(l0.get(c, 0)) for c in CAMPAIGN_ORDER],
        "l1_n_rows": [int(l1.get(c, 0)) for c in CAMPAIGN_ORDER],
        "l2_n_rows": [int(l2.get(c, 0)) for c in CAMPAIGN_ORDER],
    })
    funnel_df["l0_to_l1_pct"] = (100 * funnel_df["l1_n_rows"] / funnel_df["l0_n_rows"].replace(0, np.nan)).round(2)
    funnel_df["l1_to_l2_pct"] = (100 * funnel_df["l2_n_rows"] / funnel_df["l1_n_rows"].replace(0, np.nan)).round(2)
    funnel_df["l0_to_l2_pct"] = (100 * funnel_df["l2_n_rows"] / funnel_df["l0_n_rows"].replace(0, np.nan)).round(2)
    funnel_df.to_csv(args.out / "campaign_funnel_retention.csv", index=False)

    print("\n=== Plots ===")
    plot_row_counts(counts_by_tier, args.figs)
    plot_funnel_retention(funnel_df, args.figs)
    for i, tier in enumerate(TIER_ORDER, start=3):
        plot_completeness_heatmap(completeness_by_tier[tier], tier, i, args.figs)
    plot_date_ranges(dates_by_tier["L0"], args.figs)

    run_config = {
        "total_rows_by_tier": total_rows_by_tier,
        "core_cols": CORE_COLS,
        "campaign_order": CAMPAIGN_ORDER,
    }
    with open(args.out / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2, default=str)
    print(f"\nSaved {args.out / 'run_config.json'}")

    update_latest(args.out.parent, args.out)
    update_latest(args.figs.parent, args.figs)
    print(f"Latest run: {args.out.parent / 'latest'} -> {args.out.name}")
    print(f"Latest figs: {args.figs.parent / 'latest'} -> {args.figs.name}")


if __name__ == "__main__":
    main()
