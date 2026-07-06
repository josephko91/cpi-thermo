#!/usr/bin/env python3
"""
CPI Data Fusion Diagnostic
===========================
Tests the combined environmental data from main.py against CPI imagery timestamps.
Measures how much of the CPI image archive has concurrent Tair_C and Si measurements.

Outputs:
  logs/diagnostics/cpi_fusion_summary.csv   - per-campaign coverage table
  logs/diagnostics/cpi_fusion_report.txt    - text summary + bug notes
  figs/cpi_fusion_coverage.png              - bar chart of coverage rates
  figs/cpi_fusion_si_tair_scatter.png       - Si vs Tair_C colored by campaign

Usage:
    python scripts/diagnose_cpi_fusion.py
    python scripts/diagnose_cpi_fusion.py --env-data data/out/combined_env_data.parquet
    python scripts/diagnose_cpi_fusion.py --rebuild   # re-run main.py first
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Ensure project root is on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main import process_all_campaigns, DEFAULT_CAMPAIGN_CONFIG
from parsers.cpi_timestamps import (
    CPI_TO_ENV_CAMPAIGN as CPI_TO_ENV,
    load_cpi_embeddings_timestamps,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CPI_CSV       = ROOT / "data" / "raw" / "cpi_embeddings_timestamps.csv"
DEFAULT_ENV   = ROOT / "data" / "out" / "combined_env_data.parquet"
DIAG_DIR      = ROOT / "logs" / "diagnostics"
FIGS_DIR      = ROOT / "figs"

MATCH_TOLERANCE_S = 1   # seconds; CPI timestamps are whole-second


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_cpi(path: Path) -> pd.DataFrame:
    # Campaign-name normalisation and known UTC-offset corrections (e.g. MC3E
    # recorded in CDT) live in parsers/cpi_timestamps.py, the single source of
    # truth for anyone loading this archive.
    return load_cpi_embeddings_timestamps(path)


def load_or_build_env(env_path: Path, rebuild: bool) -> pd.DataFrame:
    if rebuild or not env_path.exists():
        print("Building combined env data via main.py …")
        df = process_all_campaigns(verbose=True)
        env_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(env_path, index=False)
        print(f"Saved to {env_path}")
    else:
        print(f"Loading env data from {env_path}")
        df = pd.read_parquet(env_path)
    return df


def merge_campaign(
    cpi_sub: pd.DataFrame,
    env_sub: pd.DataFrame,
    tol_s: int = MATCH_TOLERANCE_S,
) -> pd.DataFrame:
    """Nearest-timestamp merge (merge_asof) within ±tol_s seconds."""
    cpi_s = cpi_sub.dropna(subset=["datetime"]).sort_values("datetime")
    env_s = env_sub.dropna(subset=["Timestamp"]).sort_values("Timestamp")
    # Ensure both keys share the same datetime resolution (ns, UTC)
    cpi_s = cpi_s.copy()
    cpi_s["datetime"] = cpi_s["datetime"].dt.as_unit("ns")
    env_s = env_s.copy()
    env_s["Timestamp"] = env_s["Timestamp"].dt.as_unit("ns")

    merged = pd.merge_asof(
        cpi_s,
        env_s[["Timestamp", "Tair_C", "Si"]],
        left_on="datetime",
        right_on="Timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=tol_s),
    )
    return merged


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def compute_coverage(cpi: pd.DataFrame, env: pd.DataFrame) -> pd.DataFrame:
    """
    For each campaign present in the CPI data, compute:
      - n_cpi_images   : total CPI images
      - n_env_records  : env records available
      - n_matched      : CPI images with a env timestamp within tolerance
      - n_tair_valid   : matched images with non-null Tair_C
      - n_si_valid     : matched images with non-null Si
      - n_both_valid   : matched images with both valid
      - pct_matched    : n_matched / n_cpi_images
      - pct_both       : n_both_valid / n_cpi_images
    """
    env_campaigns = set(env["Campaign"].unique())
    rows = []

    for cpi_name, env_name in sorted(CPI_TO_ENV.items()):
        cpi_sub = cpi[cpi["campaign"] == cpi_name]
        if cpi_sub.empty:
            continue

        n_cpi = len(cpi_sub)
        in_env = env_name in env_campaigns
        n_env = int((env["Campaign"] == env_name).sum()) if in_env else 0

        if not in_env or n_env == 0:
            rows.append({
                "campaign_cpi": cpi_name,
                "campaign_env": env_name,
                "n_cpi_images": n_cpi,
                "n_env_records": 0,
                "n_matched": 0,
                "n_tair_valid": 0,
                "n_si_valid": 0,
                "n_both_valid": 0,
                "pct_matched": 0.0,
                "pct_tair": 0.0,
                "pct_si": 0.0,
                "pct_both": 0.0,
                "status": "env_missing",
            })
            continue

        env_sub = env[env["Campaign"] == env_name]
        merged = merge_campaign(cpi_sub, env_sub)

        n_matched    = int(merged["Timestamp"].notna().sum())
        n_tair_valid = int(merged["Tair_C"].notna().sum())
        n_si_valid   = int(merged["Si"].notna().sum())
        n_both_valid = int(merged[["Tair_C", "Si"]].notna().all(axis=1).sum())

        rows.append({
            "campaign_cpi": cpi_name,
            "campaign_env": env_name,
            "n_cpi_images": n_cpi,
            "n_env_records": n_env,
            "n_matched": n_matched,
            "n_tair_valid": n_tair_valid,
            "n_si_valid": n_si_valid,
            "n_both_valid": n_both_valid,
            "pct_matched": round(n_matched / n_cpi * 100, 1),
            "pct_tair":    round(n_tair_valid / n_cpi * 100, 1),
            "pct_si":      round(n_si_valid   / n_cpi * 100, 1),
            "pct_both":    round(n_both_valid  / n_cpi * 100, 1),
            "status": "ok",
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

CAMPAIGN_COLORS = {
    "ARM": "#1f77b4", "AIRS_II": "#ff7f0e", "ATTREX": "#2ca02c",
    "CRYSTAL_FACE_NASA": "#d62728", "CRYSTAL_FACE_UND": "#9467bd",
    "ICE_L": "#8c564b", "IPHEX": "#e377c2", "ISDAC": "#7f7f7f",
    "MACPEX": "#bcbd22", "MC3E": "#17becf", "MIDCIX": "#aec7e8",
    "MPACE": "#ffbb78", "OLYMPEX": "#98df8a",
}


def plot_coverage_bars(summary: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    campaigns = summary["campaign_cpi"].tolist()
    x = np.arange(len(campaigns))
    w = 0.28

    bars_matched = ax.bar(x - w, summary["pct_matched"], w, label="Timestamp matched", color="#4c72b0")
    bars_tair    = ax.bar(x,     summary["pct_tair"],    w, label="Tair_C valid",       color="#dd8452")
    bars_si      = ax.bar(x + w, summary["pct_si"],      w, label="Si valid",            color="#55a868")

    ax.set_xticks(x)
    ax.set_xticklabels(campaigns, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("CPI images covered (%)")
    ax.set_title("CPI image coverage by concurrent env measurements (±1 s tolerance)")
    ax.set_ylim(0, 110)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend(loc="upper right", fontsize=8)
    ax.axhline(100, color="gray", lw=0.8, ls="--")

    # Annotate missing campaigns
    for i, row in summary.iterrows():
        idx = campaigns.index(row["campaign_cpi"])
        if row["status"] == "env_missing":
            ax.text(idx, 3, "no env\ndata", ha="center", va="bottom",
                    fontsize=6, color="red", style="italic")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_si_tair_scatter(env: pd.DataFrame, out_path: Path) -> None:
    """Si vs Tair_C scatter, colored by campaign, for matched env data only."""
    cols = ["Tair_C", "Si", "Campaign"]
    valid = env[cols].dropna()
    # Clip to physical range for plotting
    valid = valid[(valid["Tair_C"].between(-80, 30)) & (valid["Si"].between(-1, 1))]
    if valid.empty:
        return

    campaigns = sorted(valid["Campaign"].unique())
    cmap = plt.get_cmap("tab20", len(campaigns))

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, camp in enumerate(campaigns):
        sub = valid[valid["Campaign"] == camp]
        ax.scatter(sub["Tair_C"], sub["Si"], s=0.5, alpha=0.3,
                   color=cmap(i), label=camp, rasterized=True)

    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Air temperature (°C)")
    ax.set_ylabel("Ice supersaturation (Si)")
    ax.set_title("Si vs Tair_C in combined env dataset (all campaigns)")
    leg = ax.legend(markerscale=8, fontsize=6, loc="upper right",
                    ncol=2, framealpha=0.7)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def write_report(summary: pd.DataFrame, env: pd.DataFrame, out_path: Path) -> str:
    ok      = summary[summary["status"] == "ok"]
    missing = summary[summary["status"] == "env_missing"]

    total_cpi     = int(summary["n_cpi_images"].sum())
    total_matched = int(ok["n_matched"].sum())
    total_both    = int(ok["n_both_valid"].sum())
    pct_matched   = total_matched / total_cpi * 100 if total_cpi else 0
    pct_both      = total_both    / total_cpi * 100 if total_cpi else 0

    env_campaigns = sorted(env["Campaign"].unique().tolist())
    n_tair_env    = int(env["Tair_C"].notna().sum())
    n_si_env      = int(env["Si"].notna().sum())

    report_lines = [
        "=" * 70,
        "CPI DATA FUSION DIAGNOSTIC REPORT",
        "=" * 70,
        "",
        "ENVIRONMENTAL DATA SUMMARY",
        "-" * 40,
        f"  Total env records    : {len(env):,}",
        f"  Campaigns with data  : {len(env_campaigns)}  {env_campaigns}",
        f"  Records w/ Tair_C    : {n_tair_env:,}  ({n_tair_env/len(env)*100:.1f}%)",
        f"  Records w/ Si        : {n_si_env:,}  ({n_si_env/len(env)*100:.1f}%)",
        "",
        "CPI IMAGE ARCHIVE SUMMARY",
        "-" * 40,
        f"  Total CPI images       : {total_cpi:,}",
        f"  CPI campaigns          : {sorted(summary['campaign_cpi'].unique().tolist())}",
        f"  Campaigns w/o env data : {missing['campaign_cpi'].tolist()}",
        "",
        "FUSION COVERAGE (±1 s timestamp tolerance)",
        "-" * 40,
        f"  CPI images with matched env timestamp : {total_matched:,} / {total_cpi:,}  ({pct_matched:.1f}%)",
        f"  CPI images with both Tair_C & Si      : {total_both:,} / {total_cpi:,}  ({pct_both:.1f}%)",
        "",
        "PER-CAMPAIGN BREAKDOWN",
        "-" * 40,
    ]

    # Table header
    hdr = f"  {'Campaign':<22} {'CPI imgs':>9} {'matched%':>9} {'Tair%':>7} {'Si%':>7} {'both%':>7}"
    report_lines.append(hdr)
    report_lines.append("  " + "-" * (len(hdr) - 2))
    for _, row in summary.sort_values("campaign_cpi").iterrows():
        flag = " [NO ENV]" if row["status"] == "env_missing" else ""
        report_lines.append(
            f"  {row['campaign_cpi']:<22} {int(row['n_cpi_images']):>9,}"
            f" {row['pct_matched']:>8.1f}% {row['pct_tair']:>6.1f}% {row['pct_si']:>6.1f}%"
            f" {row['pct_both']:>6.1f}%{flag}"
        )

    report_lines += [
        "",
        "=" * 70,
        "REMAINING KNOWN ISSUES",
        "=" * 70,
        "",
        "1. CAMPAIGN NAME MISMATCH (CPI vs env data) — HANDLED CENTRALLY",
        "   CPI CSV uses underscores (CRYSTAL_FACE_UND, AIRS_II, ICE_L).",
        "   main.py / env data uses hyphens (CRYSTAL-FACE-UND, AIRS-II, ICE-L).",
        "   Impact: any direct join on campaign name will produce zero matches.",
        "   Mitigation: parsers/cpi_timestamps.py CPI_TO_ENV_CAMPAIGN normalises",
        "   names; load_cpi_embeddings_timestamps() is the canonical loader.",
        "",
        "2. MPACE HAS NO WATER-VAPOR INSTRUMENT",
        "   parsers/mpace.py loads Tair_C/P_hPa/Lat/Lon/Alt_m from the UND Citation",
        "   NASA Ames files (99.9% timestamp match, ~76% Tair coverage vs CPI",
        "   images), but no dew/frost-point or laser hygrometer channel exists in",
        "   any of the 15 raw files, so Si/qv are NaN for every record.",
        "",
        "3. OLYMPEX / ESCAPE IN ENV BUT ABSENT FROM CPI",
        "   OLYMPEX and ESCAPE have env records but zero CPI images in the archive.",
        "   Likely intentional: CPI was not deployed on those platforms.",
        "",
        "4. STALE PARQUET — REMOVED 2026-07-06",
        "   data/raw/combined_env_si_airtemp_01.parquet used the old 'Tair'",
        "   column name (not 'Tair_C') and covered only 6 campaigns — a footgun",
        "   for anything that referenced it by accident. Deleted; fully",
        "   superseded by data/out/combined_env_data.parquet.",
        "",
        "5. CRYSTAL_FACE_UND LOW Si COVERAGE (~53%)",
        "   Jul 7, 9, 11: MIS.CIT RH column is entirely fill values — the humidity",
        "   instrument was not operating those days. No alternative water vapor",
        "   source exists in the UND Citation dataset for those dates.",
        "",
        "6. MC3E CPI TIMEZONE CORRECTION APPLIED",
        "   MC3E CPI timestamps were recorded in CDT (UTC-5), not UTC — a bug in",
        "   the raw CPI archive's datetime column, not in the env pipeline.",
        "   Corrected centrally in parsers/cpi_timestamps.py",
        "   (CPI_UTC_OFFSET_HOURS), so every consumer of this archive gets the",
        "   fix automatically rather than re-discovering it.",
        "   Evidence: CPI images on 2011-05-23 fall 16:35-19:32, while env data",
        "   starts at 21:20 UTC — consistent with a 5-hour CDT offset.",
        "",
        "7. MIDCIX Si/Tair STILL ~26% — JLH GENUINELY ABSENT ON 2 DATES",
        "   JLH (water vapor) data files exist only for Apr 17, 19, 30, May 2,",
        "   3, 5, 6 — no Tair_C/Si/qv is possible for Apr 22 (17k) or Apr 27",
        "   (18k) CPI images. This part is a genuine data gap, not fixable",
        "   without new raw data. Timestamp MATCH itself was recovered from",
        "   ~26% to ~65%, though: FP/ (navigation) files exist for those same",
        "   2 dates and load_midcix() now adds position-only fallback rows",
        "   (Lat/Lon/Alt_m, no Tair/Si/qv) from them, mirroring the HW/ALIAS",
        "   fallback pattern in crystal_face_nasa.py.",
        "",
        "8. CRYSTAL_FACE_NASA LOW COVERAGE (~44%) — GENUINE DATA LIMITATION",
        "   JLH (1 Hz, primary Si source) has large gaps during cloud penetrations",
        "   — exactly when CPI is most active (anti-correlation). Harvard WV (HW)",
        "   provides a fallback but at 10-second resolution. With ±1 s merge tolerance,",
        "   only ~20–30% of CPI-active seconds inside HW-covered periods can match.",
        "   Bug fixes applied: JLH+MM primary merge, HW+MM/NM gap-fill added for",
        "   JLH-absent dates and in-flight JLH gaps. Coverage improved from 24% to 44%.",
        "   Further improvement would require relaxing merge tolerance to ±5 s or",
        "   interpolating HW to 1 Hz — both involve data quality trade-offs.",
        "",
    ]

    report = "\n".join(report_lines)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    print(f"Saved: {out_path}")
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose CPI–env data fusion coverage")
    p.add_argument("--env-data", type=Path, default=DEFAULT_ENV,
                   help="Path to combined env parquet (default: data/out/combined_env_data.parquet)")
    p.add_argument("--cpi-csv", type=Path, default=CPI_CSV,
                   help="Path to CPI embeddings timestamps CSV")
    p.add_argument("--rebuild", action="store_true",
                   help="Re-run main.py to rebuild combined_env_data.parquet")
    p.add_argument("--tol", type=int, default=MATCH_TOLERANCE_S,
                   help="Timestamp match tolerance in seconds (default: 1)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    global MATCH_TOLERANCE_S
    MATCH_TOLERANCE_S = args.tol

    # Load data
    cpi = load_cpi(args.cpi_csv)
    env = load_or_build_env(args.env_data, args.rebuild)

    print(f"\nCPI images : {len(cpi):,} across {cpi['campaign'].nunique()} campaigns")
    print(f"Env records: {len(env):,} across {env['Campaign'].nunique()} campaigns\n")

    # Coverage analysis
    print("Computing per-campaign coverage …")
    summary = compute_coverage(cpi, env)

    # Save summary CSV
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = DIAG_DIR / "cpi_fusion_summary.csv"
    summary.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Plots
    plot_coverage_bars(summary, FIGS_DIR / "cpi_fusion_coverage.png")
    plot_si_tair_scatter(env, FIGS_DIR / "cpi_fusion_si_tair_scatter.png")

    # Text report
    report = write_report(summary, env, DIAG_DIR / "cpi_fusion_report.txt")
    print()
    print(report)


if __name__ == "__main__":
    main()
