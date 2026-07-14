#!/usr/bin/env python3
"""
QA/QC diagnostic checks for the combined campaign dataset.

Each check is a self-contained function that:
  - Reads the combined parquet
  - Writes CSV reports to --out  (logs/qaqc/<timestamp>/, logs/qaqc/latest symlink)
  - Writes plots to   --figs (figs/qaqc/<timestamp>/, figs/qaqc/latest symlink)
  - Returns a summary dict (check_id, n_flags, pct_flagged, n_campaigns_affected)

Usage:
  python scripts/qa_checks.py                          # run all checks
  python scripts/qa_checks.py --checks 1 2             # run QC1 and QC2 only
  python scripts/qa_checks.py --env data/out/combined_env_data_20260701.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import BoundaryNorm
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Make parsers/ importable when running as a script
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from parsers.utils import es_liq_hPa, es_ice_hPa  # noqa: E402
from scripts.log_paths import timestamp as _run_timestamp, update_latest  # noqa: E402

# ---------------------------------------------------------------------------
# Shared constants (mirrors plot_all_campaigns.py)
# ---------------------------------------------------------------------------
CAMPAIGN_ORDER = [
    "ARM", "CRYSTAL-FACE-NASA", "CRYSTAL-FACE-UND", "MIDCIX", "MPACE",
    "AIRS-II", "ICE-L", "ISDAC", "MACPEX", "MC3E",
    "ATTREX", "IPHEX", "OLYMPEX", "POSIDON", "ESCAPE",
]

COLORS = {
    "ARM":                "#1f77b4",
    "CRYSTAL-FACE-NASA":  "#ff7f0e",
    "CRYSTAL-FACE-UND":   "#2ca02c",
    "MIDCIX":             "#d62728",
    "MPACE":              "#9edae5",
    "AIRS-II":            "#9467bd",
    "ICE-L":              "#8c564b",
    "ISDAC":              "#e377c2",
    "MACPEX":             "#7f7f7f",
    "MC3E":               "#bcbd22",
    "ATTREX":             "#17becf",
    "IPHEX":              "#aec7e8",
    "OLYMPEX":            "#ffbb78",
    "POSIDON":            "#98df8a",
    "ESCAPE":             "#ff9896",
}

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

# Hard physical bounds for QC1
HARD_BOUNDS: dict[str, tuple[float, float]] = {
    "Tair_C": (-95.0,  60.0),
    "P_hPa":  ( 50.0, 1050.0),
    "Si":     ( -1.0,   2.0),
    "Alt_m":  (-500.0, 25000.0),
    "Lat":    (-90.0,  90.0),
    "Lon":    (-180.0, 180.0),
    # EDR and Wind_* are deliberately NOT bounded here: turbulence/wind speed
    # are legitimately unbounded by storm severity, and a hard bound would
    # false-flag real severe-weather events.
}
# qv columns share the same bounds
QV_BOUNDS: tuple[float, float] = (0.0, 100.0)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    ts = _run_timestamp()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", type=Path,
                   default=ROOT / "data" / "out" / "combined_env_data.parquet",
                   help="Path to combined parquet file")
    p.add_argument("--out", type=Path,
                   default=ROOT / "logs" / "qaqc" / ts,
                   help="Directory for CSV diagnostic outputs "
                        "(default: logs/qaqc/<timestamp>/, with logs/qaqc/latest "
                        "kept pointing at the newest run)")
    p.add_argument("--figs", type=Path,
                   default=ROOT / "figs" / "qaqc" / ts,
                   help="Directory for plot outputs")
    p.add_argument("--checks", nargs="+", type=int,
                   default=list(range(1, 10)),
                   help="Which QC checks to run (1–9, default: all)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# QC1 — Physical Range Checks
# ---------------------------------------------------------------------------

def check_01_physical_range(
    df: pd.DataFrame,
    out_dir: Path,
    figs_dir: Path,
) -> dict:
    """Flag rows where any variable falls outside absolute physical hard bounds."""
    print("QC1: Physical range checks ...")

    campaigns = [c for c in CAMPAIGN_ORDER if c in df["Campaign"].unique()]
    n_total = len(df)

    # Build bounds list: include all qv_* columns
    bounds = dict(HARD_BOUNDS)
    qv_cols = [c for c in df.columns if c == "qv" or c.startswith("qv_")]
    for col in qv_cols:
        bounds[col] = QV_BOUNDS

    # -- Collect flag records --------------------------------------------------
    flag_rows = []
    for var, (lo, hi) in bounds.items():
        if var not in df.columns:
            continue
        s = pd.to_numeric(df[var], errors="coerce")
        mask_lo = s < lo
        mask_hi = s > hi
        for mask, bound_type in [(mask_lo, "lo"), (mask_hi, "hi")]:
            bad = df[mask][["Campaign", "Timestamp", "source_file"]].copy()
            bad["variable"] = var
            bad["value"] = s[mask].values
            bad["bound"] = lo if bound_type == "lo" else hi
            bad["bound_violated"] = bound_type
            flag_rows.append(bad)

    flags = pd.concat(flag_rows, ignore_index=True) if flag_rows else pd.DataFrame()

    # -- Per-campaign × per-variable summary ----------------------------------
    summary_rows = []
    for camp in campaigns:
        camp_n = len(df[df["Campaign"] == camp])
        row = {"Campaign": camp, "n_records": camp_n}
        for var, (lo, hi) in bounds.items():
            if var not in df.columns:
                row[f"{var}_n_lo"] = 0
                row[f"{var}_n_hi"] = 0
                row[f"{var}_pct_flagged"] = 0.0
                continue
            sub = df.loc[df["Campaign"] == camp, var]
            s = pd.to_numeric(sub, errors="coerce")
            n_lo = int((s < lo).sum())
            n_hi = int((s > hi).sum())
            row[f"{var}_n_lo"] = n_lo
            row[f"{var}_n_hi"] = n_hi
            row[f"{var}_pct_flagged"] = round((n_lo + n_hi) / camp_n * 100, 4) if camp_n else 0.0
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)

    # Save CSVs
    out_dir.mkdir(parents=True, exist_ok=True)
    flags_path = out_dir / "01_physical_range_flags.csv"
    summary_path = out_dir / "01_physical_range_summary.csv"
    flags.to_csv(flags_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    print(f"  Saved {flags_path}  ({len(flags):,} flagged rows)")
    print(f"  Saved {summary_path}")

    # -- Plot A: heatmap of flag% per campaign × variable ----------------------
    figs_dir.mkdir(parents=True, exist_ok=True)

    # Only plot variables that have ≥1 flag somewhere (keeps heatmap clean)
    pct_cols = [f"{v}_pct_flagged" for v in bounds if v in df.columns]
    mat_vars = [v for v in bounds if v in df.columns]
    mat = summary_df[pct_cols].values  # shape (n_camp, n_var)

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(mat_vars) + 2), max(4, 0.4 * len(campaigns) + 1.5)))
        cmap = plt.cm.RdYlGn_r
        norm = BoundaryNorm([0, 0.01, 0.1, 1, 5, 100.01], cmap.N)
        im = ax.imshow(mat, aspect="auto", interpolation="nearest",
                       cmap=cmap, norm=norm, origin="upper")

        for ci in range(len(campaigns)):
            for vi in range(len(mat_vars)):
                v = mat[ci, vi]
                txt = f"{v:.2f}%" if v > 0 else "0"
                clr = "white" if v >= 5 else "0.2"
                ax.text(vi, ci, txt, ha="center", va="center", fontsize=6.5, color=clr)

        ax.set_xticks(range(len(mat_vars)))
        ax.set_xticklabels(mat_vars, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(len(campaigns)))
        ax.set_yticklabels(campaigns, fontsize=8)
        ax.tick_params(axis="both", length=0)
        ax.set_title("QC1 — % records outside physical hard bounds\n(per campaign × variable)",
                     fontsize=10, fontweight="bold", pad=6)

        cb = fig.colorbar(im, ax=ax, orientation="vertical", pad=0.02, shrink=0.8)
        cb.set_label("% records flagged", fontsize=8)
        cb.ax.tick_params(labelsize=7)

        plt.tight_layout()
        out_a = figs_dir / "01a_range_summary_heatmap.png"
        fig.savefig(out_a, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_a}")

    # -- Plot B: tail histograms per variable with hard-limit lines ------------
    core_vars = [v for v in ["Tair_C", "P_hPa", "Si", "Alt_m", "qv"] if v in df.columns]
    n_vars = len(core_vars)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, int(np.ceil(n_vars / 2)),
                                 figsize=(14, 6), constrained_layout=True)
        axes_flat = list(axes.flat)

        for i, var in enumerate(core_vars):
            ax = axes_flat[i]
            lo, hi = bounds.get(var, (None, None))
            for camp in campaigns:
                s = pd.to_numeric(df.loc[df["Campaign"] == camp, var], errors="coerce").dropna()
                if len(s) < 10:
                    continue
                p001, p999 = s.quantile(0.001), s.quantile(0.999)
                ax.scatter(camp, p001, marker="v", s=18, color=COLORS.get(camp, "gray"), alpha=0.8)
                ax.scatter(camp, p999, marker="^", s=18, color=COLORS.get(camp, "gray"), alpha=0.8)

            if lo is not None:
                ax.axhline(lo, color="red", lw=1.2, ls="--", label=f"lo={lo}")
            if hi is not None:
                ax.axhline(hi, color="red", lw=1.2, ls="-", label=f"hi={hi}")
            ax.set_title(var, fontsize=9, fontweight="bold")
            ax.set_xticks(range(len(campaigns)))
            ax.set_xticklabels(campaigns, rotation=45, ha="right", fontsize=6)
            ax.tick_params(labelsize=7)
            ax.set_ylabel(var, fontsize=7)
            ax.legend(fontsize=6, loc="best")

        for ax in axes_flat[n_vars:]:
            ax.set_visible(False)

        fig.suptitle("QC1 — p0.1 (▼) and p99.9 (▲) per campaign with hard limits (red)\n"
                     "Points outside red lines are flagged", fontsize=10, fontweight="bold")
        out_b = figs_dir / "01b_variable_tails.png"
        fig.savefig(out_b, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_b}")

    n_flags = len(flags)
    n_camps = flags["Campaign"].nunique() if n_flags else 0
    return {
        "check_id": "QC1",
        "check_name": "Physical range checks",
        "n_flags_total": n_flags,
        "pct_dataset_flagged": round(n_flags / n_total * 100, 4) if n_total else 0,
        "n_campaigns_affected": n_camps,
        "notes": f"Hard bounds: Tair[−95,60], P[50,1050], Si[−1,2], qv[0,100], Alt[−500,25000], Lat/Lon",
    }


# ---------------------------------------------------------------------------
# QC2 — Internal Consistency Checks
# ---------------------------------------------------------------------------

def check_02_internal_consistency(
    df: pd.DataFrame,
    out_dir: Path,
    figs_dir: Path,
) -> dict:
    """Cross-validate physically coupled variables: qv vs saturation, T vs altitude."""
    print("QC2: Internal consistency checks ...")

    campaigns = [c for c in CAMPAIGN_ORDER if c in df["Campaign"].unique()]
    n_total = len(df)
    out_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    work = df[["Campaign", "Timestamp", "source_file", "Tair_C", "P_hPa", "qv", "Alt_m"]].copy()
    work["Tair_C"] = pd.to_numeric(work["Tair_C"], errors="coerce")
    work["P_hPa"]  = pd.to_numeric(work["P_hPa"],  errors="coerce")
    work["qv"]     = pd.to_numeric(work["qv"],      errors="coerce")
    work["Alt_m"]  = pd.to_numeric(work["Alt_m"],   errors="coerce")

    flag_rows = []

    # ── Sub-check A: qv > qv_sat_liq ─────────────────────────────────────────
    # Two severity tiers are used rather than a single threshold:
    #
    #   MILD   (1.05 < ratio ≤ 1.20) — chilled-mirror instruments measure at
    #     liquid saturation (not the ice frost point) while in cloud, causing
    #     qv to approach qv_sat_liq.  Instrument thermal lag and droplet
    #     flooding can push the reported value 5–20% above saturation for
    #     seconds to minutes.  This is a known artifact of chilled-mirror
    #     sensors in precipitation campaigns (IPHEX, MC3E, OLYMPEX, ICE-L,
    #     ARM) and should be annotated rather than discarded.
    #
    #   SEVERE (ratio > 1.20) — exceedances this large cannot be explained by
    #     instrument lag alone.  Root causes include sensor malfunction, unit
    #     conversion errors, or pressure/temperature inconsistencies.  These
    #     should be treated as suspect data.
    #
    # Both tiers are written to the flag CSV so that downstream users can
    # apply their own threshold.  The summary reports them separately.
    MILD_THRESHOLD   = 1.05
    SEVERE_THRESHOLD = 1.20

    # Campaigns that fly primarily through cloud / precipitation — mild
    # exceedances are physically expected for these.
    # AIRS-II (Alliance Icing Research Study II) uses the same chilled-mirror
    # dew-point instrument as ARM/IPHEX/etc. and by design samples in-cloud
    # icing conditions -- it was missing from this set (2026-07-06 audit:
    # 399 rows misclassified as "severe"/unexplained that are the same
    # expected in-cloud chilled-mirror behavior documented for the other
    # campaigns here).
    IN_CLOUD_CAMPAIGNS = {
        "ARM", "IPHEX", "MC3E", "OLYMPEX", "ICE-L",
        "CRYSTAL-FACE-UND", "CRYSTAL-FACE-NASA", "AIRS-II",
    }

    has_vars = work[["Tair_C", "P_hPa", "qv"]].notna().all(axis=1)
    sub_a = work[has_vars].copy()
    es_liq = es_liq_hPa(sub_a["Tair_C"].values)
    denom = sub_a["P_hPa"].values - es_liq
    qv_sat = np.where(denom > 0, 0.622 * es_liq / denom * 1000, np.nan)
    sub_a["qv_sat_liq"] = qv_sat
    sub_a["qv_ratio"]   = sub_a["qv"] / sub_a["qv_sat_liq"]

    mask_a = sub_a["qv_ratio"] > MILD_THRESHOLD
    flagged_a = sub_a[mask_a].copy()
    flagged_a["check_type"] = "qv_exceeds_saturation"
    # Severity tier label
    flagged_a["severity"] = np.where(
        flagged_a["qv_ratio"] > SEVERE_THRESHOLD, "severe", "mild"
    )
    # Expected-in-cloud annotation
    flagged_a["in_cloud_campaign"] = flagged_a["Campaign"].isin(IN_CLOUD_CAMPAIGNS)
    flag_rows.append(flagged_a[["Campaign", "Timestamp", "source_file",
                                 "check_type", "severity", "in_cloud_campaign",
                                 "Tair_C", "P_hPa", "qv",
                                 "qv_sat_liq", "qv_ratio"]])

    # ── Sub-check B: T vs altitude (ICAO lapse rate) ─────────────────────────
    # ISA: T(K) = 288.15 - 6.5e-3 * h_m  for h_m ≤ 11000 m
    #            = 216.65 (isothermal) for 11000 < h_m ≤ 20000 m
    has_ta = work[["Tair_C", "Alt_m"]].notna().all(axis=1)
    sub_b = work[has_ta].copy()
    h = sub_b["Alt_m"].values
    T_isa_K = np.where(
        h <= 11000,
        288.15 - 6.5e-3 * h,
        np.where(h <= 20000, 216.65, 216.65 - 1e-3 * (h - 20000)),
    )
    T_isa_C = T_isa_K - 273.15
    sub_b["T_isa_C"]   = T_isa_C
    sub_b["T_dev_C"]   = sub_b["Tair_C"].values - T_isa_C  # positive = warmer than ISA

    TOLERANCE_C = 40.0  # flag only gross failures
    mask_b = sub_b["T_dev_C"].abs() > TOLERANCE_C
    flagged_b = sub_b[mask_b].copy()
    flagged_b["check_type"] = "T_altitude_inconsistent"
    flag_rows.append(flagged_b[["Campaign", "Timestamp", "source_file",
                                 "check_type", "Tair_C", "Alt_m",
                                 "T_isa_C", "T_dev_C"]])

    # Combine flags (align columns loosely)
    flags = pd.concat(flag_rows, ignore_index=True, sort=False) if flag_rows else pd.DataFrame()

    # Summary per check_type × campaign
    summary_rows = []
    for ct in ["qv_exceeds_saturation", "T_altitude_inconsistent"]:
        sub_ct = flags[flags["check_type"] == ct] if len(flags) else pd.DataFrame()
        for camp in campaigns:
            n_camp = len(df[df["Campaign"] == camp])
            sub_camp = sub_ct[sub_ct["Campaign"] == camp] if len(sub_ct) else pd.DataFrame()
            n_flag = len(sub_camp)
            n_mild   = int((sub_camp.get("severity", pd.Series()) == "mild").sum())  if len(sub_camp) else 0
            n_severe = int((sub_camp.get("severity", pd.Series()) == "severe").sum()) if len(sub_camp) else 0
            in_cloud = camp in IN_CLOUD_CAMPAIGNS if ct == "qv_exceeds_saturation" else False
            summary_rows.append({
                "check_type":       ct,
                "Campaign":         camp,
                "in_cloud_campaign": in_cloud,
                "n_records":        n_camp,
                "n_flagged":        n_flag,
                "n_mild":           n_mild,
                "n_severe":         n_severe,
                "pct_flagged":      round(n_flag / n_camp * 100, 4) if n_camp else 0.0,
                "pct_severe":       round(n_severe / n_camp * 100, 4) if n_camp else 0.0,
            })
    summary_df = pd.DataFrame(summary_rows)

    flags_path   = out_dir / "02_consistency_flags.csv"
    summary_path = out_dir / "02_consistency_summary.csv"
    flags.to_csv(flags_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    n_severe_total = int((flags.get("severity", pd.Series()) == "severe").sum()) if len(flags) else 0
    n_mild_total   = int((flags.get("severity", pd.Series()) == "mild").sum())   if len(flags) else 0
    print(f"  Saved {flags_path}  ({len(flags):,} flagged rows: {n_mild_total:,} mild, {n_severe_total:,} severe)")
    print(f"  Saved {summary_path}")

    # ── Plot A: qv / qv_sat_liq vs pressure, coloured by campaign ─────────────
    NCOLS, NROWS = 4, int(np.ceil(len(campaigns) / 4))
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(NROWS, NCOLS, figsize=(14, NROWS * 2.8),
                                 constrained_layout=True)
        axes_flat = list(axes.flat)
        for i, camp in enumerate(campaigns):
            ax = axes_flat[i]
            sub = sub_a[sub_a["Campaign"] == camp].dropna(subset=["qv_ratio", "P_hPa"])
            color = COLORS.get(camp, "gray")
            if len(sub) > 5000:
                sub = sub.sample(5000, random_state=42)
            ax.scatter(sub["qv_ratio"], sub["P_hPa"], s=1.5, alpha=0.3,
                       color=color, rasterized=True)
            ax.axvline(MILD_THRESHOLD,   color="orange", lw=1.0, ls="--")
            ax.axvline(SEVERE_THRESHOLD, color="red",    lw=1.0, ls="-")
            ax.set_xlim(0, 3)
            ax.set_ylim(1050, 50)
            ax.set_xlabel("qv / qv_sat_liq", fontsize=7)
            ax.set_ylabel("P (hPa)", fontsize=7)
            in_cloud_label = " [cloud]" if camp in IN_CLOUD_CAMPAIGNS else ""
            ax.set_title(camp + in_cloud_label, fontsize=8, fontweight="bold", color=color, pad=3)
            ax.tick_params(labelsize=7)
            n_mild_c   = int(((flagged_a["Campaign"] == camp) & (flagged_a["severity"] == "mild")).sum())
            n_severe_c = int(((flagged_a["Campaign"] == camp) & (flagged_a["severity"] == "severe")).sum())
            ax.text(0.97, 0.05,
                    f"mild={n_mild_c:,}\nsevere={n_severe_c:,}",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=6, color="red")
        for ax in axes_flat[len(campaigns):]:
            ax.set_visible(False)
        fig.suptitle("QC2a — qv / qv_sat_liq vs pressure\n"
                     "orange dashed = 1.05 (mild); red solid = 1.20 (severe); [cloud] = in-cloud campaign",
                     fontsize=10, fontweight="bold", y=1.01)
        out_a = figs_dir / "02a_qv_vs_qvsat.png"
        fig.savefig(out_a, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_a}")

    # ── Plot B: Tair_C vs Alt_m per campaign, ISA overlaid, flagged highlighted
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(NROWS, NCOLS, figsize=(14, NROWS * 2.8),
                                 constrained_layout=True)
        axes_flat = list(axes.flat)
        h_ref = np.linspace(0, 25000, 300)
        T_ref = np.where(h_ref <= 11000, 288.15 - 6.5e-3 * h_ref,
                         np.where(h_ref <= 20000, 216.65, 216.65 - 1e-3 * (h_ref - 20000))) - 273.15

        for i, camp in enumerate(campaigns):
            ax = axes_flat[i]
            color = COLORS.get(camp, "gray")
            sub_ok = sub_b[(sub_b["Campaign"] == camp) & ~mask_b[sub_b.index]]
            sub_fl = sub_b[(sub_b["Campaign"] == camp) & mask_b[sub_b.index]]
            if len(sub_ok) > 5000:
                sub_ok = sub_ok.sample(5000, random_state=42)
            ax.scatter(sub_ok["Tair_C"], sub_ok["Alt_m"] / 1000, s=1.5,
                       alpha=0.25, color=color, rasterized=True, label="ok")
            if len(sub_fl):
                ax.scatter(sub_fl["Tair_C"], sub_fl["Alt_m"] / 1000, s=6,
                           alpha=0.8, color="red", rasterized=True, label="flagged")
            ax.plot(T_ref, h_ref / 1000, color="k", lw=1.0, ls="--", alpha=0.7)
            ax.fill_betweenx(h_ref / 1000, T_ref - TOLERANCE_C, T_ref + TOLERANCE_C,
                             alpha=0.08, color="gray")
            ax.set_xlim(-100, 50)
            ax.set_ylim(0, 25)
            ax.set_xlabel("Tair_C (°C)", fontsize=7)
            ax.set_ylabel("Alt (km)", fontsize=7)
            ax.set_title(camp, fontsize=8, fontweight="bold", color=color, pad=3)
            ax.tick_params(labelsize=7)
            if len(sub_fl):
                ax.text(0.97, 0.95, f"n_flag={len(sub_fl):,}", transform=ax.transAxes,
                        ha="right", va="top", fontsize=6, color="red")
        for ax in axes_flat[len(campaigns):]:
            ax.set_visible(False)
        fig.suptitle("QC2b — Tair_C vs Altitude\n(dashed = ISA; grey band = ±40°C tolerance; red = flagged)",
                     fontsize=10, fontweight="bold", y=1.01)
        out_b = figs_dir / "02b_T_vs_altitude.png"
        fig.savefig(out_b, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_b}")

    n_flags  = len(flags)
    n_severe = int((flags.get("severity", pd.Series()) == "severe").sum()) if n_flags else 0
    n_mild   = int((flags.get("severity", pd.Series()) == "mild").sum())   if n_flags else 0
    n_camps  = flags["Campaign"].nunique() if n_flags else 0
    return {
        "check_id": "QC2",
        "check_name": "Internal consistency",
        "n_flags_total": n_flags,
        "n_flags_mild": n_mild,
        "n_flags_severe": n_severe,
        "pct_dataset_flagged": round(n_flags / n_total * 100, 4) if n_total else 0,
        "pct_severe": round(n_severe / n_total * 100, 4) if n_total else 0,
        "n_campaigns_affected": n_camps,
        "notes": (
            f"qv: mild={MILD_THRESHOLD}×qv_sat (in-cloud instrument physics expected); "
            f"severe={SEVERE_THRESHOLD}×qv_sat (suspect data); "
            "T: |Tair_C − T_ISA| > 40°C"
        ),
    }


# ---------------------------------------------------------------------------
# QC3 — Stuck-Sensor / Temporal Continuity
# ---------------------------------------------------------------------------

def check_03_stuck_sensor(
    df: pd.DataFrame,
    out_dir: Path,
    figs_dir: Path,
) -> dict:
    """Detect variables frozen at a constant value for ≥30 consecutive readings."""
    print("QC3: Stuck-sensor and temporal continuity checks ...")

    campaigns = [c for c in CAMPAIGN_ORDER if c in df["Campaign"].unique()]
    n_total = len(df)
    out_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    CHECK_VARS   = [v for v in ["Tair_C", "P_hPa", "Alt_m", "qv", "Si"] if v in df.columns]
    MIN_RUN      = 30   # flag runs of ≥30 identical values
    GAP_HOURS    = 2.0  # flag intra-file gaps > 2 h

    stuck_rows = []
    gap_rows   = []

    for camp in campaigns:
        sub = df[df["Campaign"] == camp].copy()
        sub["Timestamp"] = pd.to_datetime(sub["Timestamp"], utc=True, errors="coerce")

        # ── Stuck-sensor: per source_file, per variable ───────────────────────
        for sf, sf_grp in sub.groupby("source_file"):
            sf_grp = sf_grp.sort_values("Timestamp").reset_index(drop=True)

            # Timestamp gap check within file
            dt = sf_grp["Timestamp"].diff().dt.total_seconds() / 3600
            big_gaps = dt[dt > GAP_HOURS]
            for idx, gap_h in big_gaps.items():
                gap_rows.append({
                    "Campaign": camp,
                    "source_file": sf,
                    "gap_start": sf_grp.loc[idx - 1, "Timestamp"],
                    "gap_end":   sf_grp.loc[idx,     "Timestamp"],
                    "gap_hours": round(gap_h, 2),
                })

            for var in CHECK_VARS:
                vals = pd.to_numeric(sf_grp[var], errors="coerce").values
                if np.isnan(vals).all():
                    continue

                # Run-length encoding
                run_val, run_start, run_len = None, 0, 1
                for j in range(1, len(vals)):
                    v_prev, v_curr = vals[j - 1], vals[j]
                    same = (np.isnan(v_prev) and np.isnan(v_curr)) or (
                        not np.isnan(v_prev) and not np.isnan(v_curr)
                        and abs(v_curr - v_prev) < 1e-6
                    )
                    if same:
                        run_len += 1
                    else:
                        if run_len >= MIN_RUN and not np.isnan(vals[run_start]):
                            ts_start = sf_grp["Timestamp"].iloc[run_start]
                            ts_end   = sf_grp["Timestamp"].iloc[j - 1]
                            stuck_rows.append({
                                "Campaign": camp,
                                "source_file": sf,
                                "variable": var,
                                "run_start": ts_start,
                                "run_end":   ts_end,
                                "run_length": run_len,
                                "stuck_value": vals[run_start],
                            })
                        run_val, run_start, run_len = vals[j], j, 1
                # Handle run at end of file
                if run_len >= MIN_RUN and not np.isnan(vals[run_start]):
                    stuck_rows.append({
                        "Campaign": camp,
                        "source_file": sf,
                        "variable": var,
                        "run_start": sf_grp["Timestamp"].iloc[run_start],
                        "run_end":   sf_grp["Timestamp"].iloc[-1],
                        "run_length": run_len,
                        "stuck_value": vals[run_start],
                    })

    stuck_df = pd.DataFrame(stuck_rows)
    gap_df   = pd.DataFrame(gap_rows)

    stuck_path = out_dir / "03_stuck_sensor_flags.csv"
    gap_path   = out_dir / "03_timestamp_gaps.csv"
    stuck_df.to_csv(stuck_path, index=False)
    gap_df.to_csv(gap_path, index=False)
    print(f"  Saved {stuck_path}  ({len(stuck_df):,} stuck-value runs)")
    print(f"  Saved {gap_path}  ({len(gap_df):,} timestamp gaps > {GAP_HOURS}h)")

    # ── Plot: stuck-sensor timeline ──────────────────────────────────────────
    camps_with_stuck = stuck_df["Campaign"].unique() if len(stuck_df) else []
    if len(camps_with_stuck):
        n_panels = len(camps_with_stuck)
        with plt.rc_context(STYLE):
            fig, axes = plt.subplots(n_panels, 1,
                                     figsize=(14, max(3, 1.5 * n_panels)),
                                     constrained_layout=True)
            if n_panels == 1:
                axes = [axes]
            var_colors = {"Tair_C": "#1f77b4", "P_hPa": "#ff7f0e",
                          "Alt_m": "#2ca02c", "qv": "#d62728", "Si": "#9467bd"}
            for ax, camp in zip(axes, camps_with_stuck):
                sub_c = df[df["Campaign"] == camp]
                ts_min = pd.to_datetime(sub_c["Timestamp"].min())
                ts_max = pd.to_datetime(sub_c["Timestamp"].max())
                ax.set_xlim(ts_min, ts_max)
                ax.set_ylim(-0.5, len(CHECK_VARS) - 0.5)
                ax.set_yticks(range(len(CHECK_VARS)))
                ax.set_yticklabels(CHECK_VARS, fontsize=8)

                sub_stuck = stuck_df[stuck_df["Campaign"] == camp]
                for _, row in sub_stuck.iterrows():
                    vi = CHECK_VARS.index(row["variable"]) if row["variable"] in CHECK_VARS else -1
                    if vi < 0:
                        continue
                    ax.barh(vi, left=row["run_start"],
                            width=(pd.Timestamp(row["run_end"]) - pd.Timestamp(row["run_start"])),
                            height=0.4, color=var_colors.get(row["variable"], "gray"),
                            alpha=0.7,
                            label=f"{row['variable']} (n={row['run_length']})")

                ax.set_title(f"{camp} — stuck-sensor runs (≥{MIN_RUN} identical readings)",
                             fontsize=9, fontweight="bold", color=COLORS.get(camp, "k"))
                ax.xaxis.set_major_formatter(
                    matplotlib.dates.DateFormatter("%m-%d") if hasattr(matplotlib, "dates")
                    else mticker.AutoFormatter()
                )
                ax.tick_params(labelsize=7)

            fig.suptitle("QC3 — Stuck sensor runs by campaign and variable",
                         fontsize=10, fontweight="bold")
            out_p = figs_dir / "03_stuck_sensor_timeline.png"
            fig.savefig(out_p, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved {out_p}")
    else:
        print("  No stuck-sensor runs found — skipping plot")

    n_flags = len(stuck_df)
    n_camps = stuck_df["Campaign"].nunique() if n_flags else 0
    return {
        "check_id": "QC3",
        "check_name": "Stuck-sensor / temporal continuity",
        "n_flags_total": n_flags,
        "pct_dataset_flagged": round(n_flags / n_total * 100, 4) if n_total else 0,
        "n_campaigns_affected": n_camps,
        "notes": f"Stuck: ≥{MIN_RUN} consecutive identical readings; Gaps: >{GAP_HOURS}h intra-file",
    }


# ---------------------------------------------------------------------------
# QC4 — Fill / Sentinel Value Detection
# ---------------------------------------------------------------------------

def check_04_sentinel_values(
    df: pd.DataFrame,
    out_dir: Path,
    figs_dir: Path,
) -> dict:
    """Check for un-converted fill/sentinel values surviving in the combined dataset."""
    print("QC4: Fill/sentinel value detection ...")

    campaigns = [c for c in CAMPAIGN_ORDER if c in df["Campaign"].unique()]
    n_total = len(df)
    out_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    SENTINELS = [-9999.0, -9999.99, -999.0, -999.99, -8888.0, -7777.0,
                 9999.0, 99999.0, -1000.0, 1000.0]
    # 1000.0/-1000.0/9999.0 are ambiguous for Alt_m (meters), P_hPa, and
    # *_ppmv concentration columns -- all three routinely pass through these
    # magnitudes as real physical values (Alt_m crossing 1000 m or 9999 m
    # during a climb/descent; P_hPa near 1000 hPa at low altitude; water
    # vapor mixing ratios of 1000-10000 ppmv in the lower troposphere).
    # 2026-07-06 audit: every one of 34 flagged (campaign, variable,
    # sentinel) combinations using these two magnitudes was confirmed to be
    # real, continuously-varying data (verified against the raw parquet: the
    # matched values are either many distinct floats spanning the tolerance
    # band, or -- for whole-meter-quantized GPS altitude -- short, separate
    # multi-second bursts of a real repeated integer reading at different
    # times in the flight), not an un-converted fill code. No true sentinel
    # anywhere in the dataset uses either magnitude. Excluding these
    # variable/sentinel combinations avoids the false positives while
    # keeping both values active for every other column (e.g. Tair_C, Si,
    # qv), where they remain unambiguous and useful.
    AMBIGUOUS_SENTINELS = {1000.0, -1000.0, 9999.0}
    def _sentinel_applies(col: str, sent: float) -> bool:
        if sent not in AMBIGUOUS_SENTINELS:
            return True
        return not (col in ("Alt_m", "P_hPa") or col.endswith("_ppmv"))

    NUMERIC_COLS = [c for c in df.columns
                    if pd.api.types.is_numeric_dtype(df[c])
                    and c not in ("Lat", "Lon")]  # lat/lon have no sentinels in this dataset

    sentinel_rows = []
    tail_rows = []

    for camp in campaigns:
        sub = df[df["Campaign"] == camp]
        n_camp = len(sub)
        for col in NUMERIC_COLS:
            if col not in sub.columns:
                continue
            s = pd.to_numeric(sub[col], errors="coerce").dropna()
            if len(s) == 0:
                continue

            # Tail stats
            tail_rows.append({
                "Campaign": camp,
                "variable": col,
                "n_valid":  len(s),
                "p0_01":    round(float(s.quantile(0.0001)), 6),
                "p0_1":     round(float(s.quantile(0.001)), 6),
                "p99_9":    round(float(s.quantile(0.999)), 6),
                "p99_99":   round(float(s.quantile(0.9999)), 6),
                "min":      round(float(s.min()), 6),
                "max":      round(float(s.max()), 6),
            })

            # Sentinel check: exact matches or within 0.01% of sentinel
            for sent in SENTINELS:
                if sent == 0:
                    continue  # skip 0 — too many legitimate zeros
                if not _sentinel_applies(col, sent):
                    continue
                tol = abs(sent) * 0.0001
                mask_sent = (pd.to_numeric(sub[col], errors="coerce") - sent).abs() <= tol
                n_sent = int(mask_sent.sum())
                if n_sent > 0:
                    sentinel_rows.append({
                        "Campaign": camp,
                        "variable": col,
                        "sentinel_value": sent,
                        "n_matching": n_sent,
                        "pct_of_campaign": round(n_sent / n_camp * 100, 4),
                    })

    sentinel_df = pd.DataFrame(sentinel_rows, columns=[
        "Campaign", "variable", "sentinel_value", "n_matching", "pct_of_campaign",
    ])
    tail_df     = pd.DataFrame(tail_rows)

    sent_path = out_dir / "04_sentinel_flags.csv"
    tail_path = out_dir / "04_tail_stats.csv"
    sentinel_df.to_csv(sent_path, index=False)
    tail_df.to_csv(tail_path, index=False)
    print(f"  Saved {sent_path}  ({len(sentinel_df):,} sentinel matches)")
    print(f"  Saved {tail_path}")

    # ── Plot: tail distributions for core variables ─────────────────────────
    core_vars = [v for v in ["Tair_C", "P_hPa", "Si", "qv", "Alt_m"] if v in df.columns]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(core_vars),
                                 figsize=(3.5 * len(core_vars), 5),
                                 constrained_layout=True)
        if len(core_vars) == 1:
            axes = [axes]

        for ax, var in zip(axes, core_vars):
            sub_tail = tail_df[tail_df["variable"] == var]
            camps_v  = sub_tail["Campaign"].tolist()
            p001     = sub_tail["p0_01"].tolist()
            p999     = sub_tail["p99_99"].tolist()
            colors_v = [COLORS.get(c, "gray") for c in camps_v]

            for ci, (camp, lo_v, hi_v, col) in enumerate(zip(camps_v, p001, p999, colors_v)):
                ax.plot([ci, ci], [lo_v, hi_v], color=col, lw=2, alpha=0.7)
                ax.scatter([ci], [lo_v], marker="v", s=20, color=col, zorder=3)
                ax.scatter([ci], [hi_v], marker="^", s=20, color=col, zorder=3)

            # Sentinel reference lines
            sent_sub = sentinel_df[sentinel_df["variable"] == var]["sentinel_value"].unique()
            for sv in sent_sub:
                ax.axhline(sv, color="red", lw=0.8, ls=":", alpha=0.8)
                ax.text(len(camps_v) - 0.5, sv, f" {sv:.0f}", fontsize=5, color="red", va="bottom")

            ax.set_xticks(range(len(camps_v)))
            ax.set_xticklabels(camps_v, rotation=60, ha="right", fontsize=6)
            ax.set_title(var, fontsize=9, fontweight="bold")
            ax.tick_params(labelsize=7)

        fig.suptitle("QC4 — Variable tails (p0.01▼ to p99.99▲) per campaign\n"
                     "Red dotted lines = known sentinel values",
                     fontsize=10, fontweight="bold")
        out_p = figs_dir / "04_tail_distributions.png"
        fig.savefig(out_p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_p}")

    n_flags = int(sentinel_df["n_matching"].sum()) if len(sentinel_df) else 0
    n_camps = sentinel_df["Campaign"].nunique() if len(sentinel_df) else 0
    return {
        "check_id": "QC4",
        "check_name": "Fill/sentinel value detection",
        "n_flags_total": n_flags,
        "pct_dataset_flagged": round(n_flags / n_total * 100, 4) if n_total else 0,
        "n_campaigns_affected": n_camps,
        "notes": f"Checked sentinels: {SENTINELS}",
    }


# ---------------------------------------------------------------------------
# QC5 — Inter-Instrument Cross-Validation
# ---------------------------------------------------------------------------

def check_05_instrument_crossval(
    df: pd.DataFrame,
    out_dir: Path,
    figs_dir: Path,
) -> dict:
    """Compare simultaneous measurements from multiple instruments."""
    print("QC5: Inter-instrument cross-validation ...")

    n_total = len(df)
    out_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    # Instrument pairs to compare: (campaign, col_A, label_A, col_B, label_B, base_var)
    PAIRS = [
        ("MACPEX",           "Si_HWV",            "HWV",  "Si_DLH",            "DLH",  "Si"),
        ("MACPEX",           "Si_HWV",            "HWV",  "Si_JLH",            "JLH",  "Si"),
        ("MACPEX",           "Si_DLH",            "DLH",  "Si_JLH",            "JLH",  "Si"),
        ("ATTREX",           "Si_DLH",            "DLH",  "Si_NOAA",           "NOAA", "Si"),
        ("ATTREX",           "Si_DLH",            "DLH",  "Si_UCATS",          "UCATS","Si"),
        ("ATTREX",           "Si_NOAA",           "NOAA", "Si_UCATS",          "UCATS","Si"),
        ("IPHEX",            "Si_chilled_mirror", "CM",   "Si_ophir_tdl",      "TDL",  "Si"),
        ("ICE-L",            "Si_chilled_mirror", "CM",   "Si_MRTDL",          "MRTDL","Si"),
        ("CRYSTAL-FACE-NASA","Si_JLH",            "JLH",  "Si_HW",             "HW",   "Si"),
        # qv pairs mirror Si pairs
        ("MACPEX",           "qv_hwv",            "HWV",  "qv_dlh",            "DLH",  "qv"),
        ("MACPEX",           "qv_hwv",            "HWV",  "qv_jlh",            "JLH",  "qv"),
        ("ATTREX",           "qv_dlh",            "DLH",  "qv_noaa",           "NOAA", "qv"),
        ("ATTREX",           "qv_dlh",            "DLH",  "qv_ucats",          "UCATS","qv"),
        ("IPHEX",            "qv_chilled_mirror", "CM",   "qv_ophir_tdl",      "TDL",  "qv"),
        ("ICE-L",            "qv_chilled_mirror", "CM",   "qv_mrtdl",          "MRTDL","qv"),
    ]

    stat_rows  = []
    sample_dfs = []

    for camp, col_a, lbl_a, col_b, lbl_b, base_var in PAIRS:
        if camp not in df["Campaign"].unique():
            continue
        if col_a not in df.columns or col_b not in df.columns:
            continue
        sub = df[df["Campaign"] == camp][[col_a, col_b, "Timestamp", "P_hPa"]].copy()
        sub[col_a] = pd.to_numeric(sub[col_a], errors="coerce")
        sub[col_b] = pd.to_numeric(sub[col_b], errors="coerce")
        co_valid = sub.dropna(subset=[col_a, col_b])
        n = len(co_valid)
        if n < 10:
            continue

        a, b = co_valid[col_a].values, co_valid[col_b].values
        bias    = float(np.mean(a - b))
        rmse    = float(np.sqrt(np.mean((a - b) ** 2)))
        r       = float(np.corrcoef(a, b)[0, 1]) if n > 1 else np.nan
        # linear fit slope
        slope = float(np.polyfit(b, a, 1)[0]) if n > 1 else np.nan

        stat_rows.append({
            "Campaign": camp, "base_var": base_var,
            "instrument_A": lbl_a, "col_A": col_a,
            "instrument_B": lbl_b, "col_B": col_b,
            "n_co_valid": n,
            "bias_A_minus_B": round(bias, 6),
            "rmse": round(rmse, 6),
            "pearson_r": round(r, 6),
            "slope_A_on_B": round(slope, 4),
        })

        # Save up to 2000 co-valid rows for inspection
        samp = co_valid.head(2000).copy()
        samp["Campaign"]     = camp
        samp["instrument_A"] = lbl_a
        samp["instrument_B"] = lbl_b
        samp["base_var"]     = base_var
        sample_dfs.append(samp)

    stat_df   = pd.DataFrame(stat_rows)
    sample_df = pd.concat(sample_dfs, ignore_index=True) if sample_dfs else pd.DataFrame()

    stat_path   = out_dir / "05_instrument_crossval_stats.csv"
    sample_path = out_dir / "05_instrument_crossval_samples.csv"
    stat_df.to_csv(stat_path, index=False)
    sample_df.to_csv(sample_path, index=False)
    print(f"  Saved {stat_path}  ({len(stat_df)} pairs)")
    print(f"  Saved {sample_path}")

    # ── Plot A: scatter per instrument pair ───────────────────────────────────
    si_pairs = stat_df[stat_df["base_var"] == "Si"]
    n_si = len(si_pairs)
    if n_si:
        NCOLS = min(3, n_si)
        NROWS = int(np.ceil(n_si / NCOLS))
        with plt.rc_context(STYLE):
            fig, axes = plt.subplots(NROWS, NCOLS, figsize=(5 * NCOLS, 4 * NROWS),
                                     constrained_layout=True)
            axes_flat = list(np.array(axes).flat) if n_si > 1 else [axes]
            for i, (_, row) in enumerate(si_pairs.iterrows()):
                ax = axes_flat[i]
                samp = sample_df[
                    (sample_df["Campaign"] == row["Campaign"]) &
                    (sample_df["instrument_A"] == row["instrument_A"]) &
                    (sample_df["instrument_B"] == row["instrument_B"]) &
                    (sample_df["base_var"] == "Si")
                ]
                if len(samp) == 0:
                    ax.set_visible(False)
                    continue
                color = COLORS.get(row["Campaign"], "gray")
                ax.scatter(samp[row["col_B"]], samp[row["col_A"]], s=4, alpha=0.4,
                           color=color, rasterized=True)
                lims = (-1.2, 2.2)
                ax.plot(lims, lims, "k--", lw=0.8, alpha=0.6)
                ax.set_xlim(lims); ax.set_ylim(lims)
                ax.set_xlabel(f"Si_{row['instrument_B']}", fontsize=8)
                ax.set_ylabel(f"Si_{row['instrument_A']}", fontsize=8)
                ax.set_title(f"{row['Campaign']}: {row['instrument_A']} vs {row['instrument_B']}\n"
                             f"bias={row['bias_A_minus_B']:+.3f}  r={row['pearson_r']:.3f}  n={row['n_co_valid']:,}",
                             fontsize=8, fontweight="bold")
                ax.tick_params(labelsize=7)
            for ax in axes_flat[n_si:]:
                ax.set_visible(False)
            fig.suptitle("QC5a — Si inter-instrument scatter (1:1 dashed)",
                         fontsize=11, fontweight="bold")
            out_a = figs_dir / "05a_instrument_scatter.png"
            fig.savefig(out_a, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved {out_a}")

    # ── Plot B: bias bar chart ────────────────────────────────────────────────
    if len(stat_df):
        with plt.rc_context(STYLE):
            fig, axes = plt.subplots(1, 2, figsize=(14, 4), constrained_layout=True)
            for ax, bvar in zip(axes, ["Si", "qv"]):
                sub_v = stat_df[stat_df["base_var"] == bvar]
                if len(sub_v) == 0:
                    ax.set_visible(False)
                    continue
                labels = [f"{r['Campaign']}\n{r['instrument_A']}−{r['instrument_B']}"
                          for _, r in sub_v.iterrows()]
                biases = sub_v["bias_A_minus_B"].values
                bar_colors = [COLORS.get(r["Campaign"], "gray") for _, r in sub_v.iterrows()]
                bars = ax.bar(range(len(biases)), biases, color=bar_colors, alpha=0.75)
                ax.axhline(0, color="k", lw=0.8)
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
                ax.set_ylabel(f"Mean bias ({bvar}_A − {bvar}_B)", fontsize=8)
                ax.set_title(f"{bvar} — instrument bias summary", fontsize=9, fontweight="bold")
                ax.tick_params(labelsize=7)
            fig.suptitle("QC5b — Inter-instrument bias (A − B)", fontsize=10, fontweight="bold")
            out_b = figs_dir / "05b_instrument_bias_summary.png"
            fig.savefig(out_b, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved {out_b}")

    n_flags = len(stat_df[stat_df["pearson_r"].abs() < 0.5]) if len(stat_df) else 0
    n_camps = stat_df["Campaign"].nunique() if len(stat_df) else 0
    return {
        "check_id": "QC5",
        "check_name": "Inter-instrument cross-validation",
        "n_flags_total": n_flags,
        "pct_dataset_flagged": 0.0,
        "n_campaigns_affected": n_camps,
        "notes": f"{len(stat_df)} instrument pairs compared; n_flags = pairs with Pearson r < 0.5",
    }


# ---------------------------------------------------------------------------
# QC6 — Data Coverage Audit (per flight date)
# ---------------------------------------------------------------------------

def check_06_coverage_audit(
    df: pd.DataFrame,
    out_dir: Path,
    figs_dir: Path,
) -> dict:
    """Characterise data coverage at the individual-flight level."""
    print("QC6: Per-flight coverage audit ...")

    campaigns = [c for c in CAMPAIGN_ORDER if c in df["Campaign"].unique()]
    n_total = len(df)
    out_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    KEY_VARS = [v for v in ["Si", "Tair_C", "P_hPa", "qv", "Alt_m"] if v in df.columns]

    df2 = df.copy()
    df2["flight_date"] = pd.to_datetime(df2["Timestamp"], utc=True, errors="coerce").dt.date

    flight_rows = []
    summary_rows = []

    for camp in campaigns:
        sub = df2[df2["Campaign"] == camp]
        for fd, grp in sub.groupby("flight_date"):
            n = len(grp)
            row = {"Campaign": camp, "flight_date": fd, "n_records": n}
            for var in KEY_VARS:
                row[f"{var}_pct_valid"] = round(grp[var].notna().mean() * 100, 2)
            flight_rows.append(row)

        # Per-campaign summary
        flight_sub = pd.DataFrame([r for r in flight_rows if r["Campaign"] == camp])
        if len(flight_sub) == 0:
            continue
        n_flights   = len(flight_sub)
        n_zero_si   = int((flight_sub.get("Si_pct_valid", pd.Series()) == 0).sum()) if "Si_pct_valid" in flight_sub else 0
        all_key_avg = flight_sub[[f"{v}_pct_valid" for v in KEY_VARS if f"{v}_pct_valid" in flight_sub]].min(axis=1)
        n_lt50      = int((all_key_avg < 50).sum())
        n_complete  = int((all_key_avg >= 90).sum())
        summary_rows.append({
            "Campaign": camp,
            "n_flights": n_flights,
            "n_flights_zero_Si": n_zero_si,
            "n_flights_lt50pct_all_vars": n_lt50,
            "n_flights_complete_90pct": n_complete,
            "pct_flights_complete": round(n_complete / n_flights * 100, 1) if n_flights else 0,
        })

    flight_df  = pd.DataFrame(flight_rows)
    summary_df = pd.DataFrame(summary_rows)

    flight_path  = out_dir / "06_per_flight_coverage.csv"
    summary_path = out_dir / "06_coverage_summary.csv"
    flight_df.to_csv(flight_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    print(f"  Saved {flight_path}  ({len(flight_df):,} flight-days)")
    print(f"  Saved {summary_path}")

    # ── Plot: per-flight heatmap (campaigns on y, flight index on x, 5 vars) ──
    with plt.rc_context(STYLE):
        n_camps = len(campaigns)
        fig, axes = plt.subplots(len(KEY_VARS), 1,
                                 figsize=(18, 2.0 * len(KEY_VARS)),
                                 constrained_layout=True)
        if len(KEY_VARS) == 1:
            axes = [axes]

        max_flights = flight_df.groupby("Campaign").size().max()

        cmap_cov = plt.cm.RdYlGn
        norm_cov = BoundaryNorm([0, 25, 50, 75, 90, 100.01], cmap_cov.N)

        for ax, var in zip(axes, KEY_VARS):
            pct_col = f"{var}_pct_valid"
            mat = np.full((n_camps, max_flights), np.nan)
            for ci, camp in enumerate(campaigns):
                camp_flights = flight_df[flight_df["Campaign"] == camp].sort_values("flight_date")
                for fi, (_, frow) in enumerate(camp_flights.iterrows()):
                    if pct_col in frow:
                        mat[ci, fi] = frow[pct_col]

            masked = np.ma.masked_invalid(mat)
            ax.imshow(masked, aspect="auto", interpolation="nearest",
                      cmap=cmap_cov, norm=norm_cov, origin="upper")
            ax.set_yticks(range(n_camps))
            ax.set_yticklabels(campaigns, fontsize=7)
            ax.set_xlabel("Flight number within campaign", fontsize=7)
            ax.set_title(f"{var} valid % per flight", fontsize=8, fontweight="bold", pad=3)
            ax.set_xticks(np.arange(0, max_flights, 5))
            ax.set_xticklabels(np.arange(1, max_flights + 1, 5), fontsize=6)
            ax.tick_params(axis="y", length=0)

        # Shared colourbar on right
        sm = plt.cm.ScalarMappable(cmap=cmap_cov, norm=norm_cov)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=axes, orientation="vertical", pad=0.01, shrink=0.6)
        cb.set_label("% valid", fontsize=8)
        cb.set_ticks([0, 25, 50, 75, 90, 100])
        cb.ax.tick_params(labelsize=7)

        fig.suptitle("QC6 — Per-flight data coverage for all key variables",
                     fontsize=10, fontweight="bold")
        out_p = figs_dir / "06_per_flight_heatmap.png"
        fig.savefig(out_p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_p}")

    n_flags = int(summary_df["n_flights_zero_Si"].sum()) if len(summary_df) else 0
    n_camps_aff = int((summary_df["n_flights_zero_Si"] > 0).sum()) if len(summary_df) else 0
    return {
        "check_id": "QC6",
        "check_name": "Per-flight coverage audit",
        "n_flags_total": n_flags,
        "pct_dataset_flagged": 0.0,
        "n_campaigns_affected": n_camps_aff,
        "notes": "n_flags = flight-days with zero valid Si",
    }


# ---------------------------------------------------------------------------
# QC7 — Duplicate and Out-of-Order Timestamps
# ---------------------------------------------------------------------------

def check_07_timestamps(
    df: pd.DataFrame,
    out_dir: Path,
    figs_dir: Path,
) -> dict:
    """Detect duplicate, near-duplicate, and out-of-order timestamps."""
    print("QC7: Timestamp quality checks ...")

    campaigns = [c for c in CAMPAIGN_ORDER if c in df["Campaign"].unique()]
    n_total = len(df)
    out_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    dup_rows     = []
    ooo_rows     = []
    summary_rows = []

    df2 = df.copy()
    df2["Timestamp"] = pd.to_datetime(df2["Timestamp"], utc=True, errors="coerce")
    df2["ts_ns"] = df2["Timestamp"].astype("int64")  # nanoseconds since epoch

    for camp in campaigns:
        sub = df2[df2["Campaign"] == camp].sort_values("Timestamp").reset_index()
        n_camp = len(sub)

        # Exact duplicates
        dup_mask  = sub.duplicated(subset=["Timestamp"], keep=False)
        exact_dup = sub[dup_mask]
        n_exact   = int(dup_mask.sum())

        # Near-duplicates (within 0.5 s = 5e8 ns)
        dt_ns    = sub["ts_ns"].diff().abs()
        near_dup_mask = (dt_ns > 0) & (dt_ns <= 5e8)
        n_near   = int(near_dup_mask.sum())

        # Out-of-order (strictly decreasing)
        dt_signed = sub["ts_ns"].diff()
        ooo_mask  = dt_signed < 0
        n_ooo     = int(ooo_mask.sum())

        # Out-of-bounds (outside known campaign window ± 1 day)
        ts_min = sub["Timestamp"].min() - pd.Timedelta(days=1)
        ts_max = sub["Timestamp"].max() + pd.Timedelta(days=1)
        oob_mask = (sub["Timestamp"] < ts_min) | (sub["Timestamp"] > ts_max)
        n_oob    = int(oob_mask.sum())

        # Collect exact dup rows
        if n_exact:
            d = exact_dup[["Campaign", "Timestamp", "source_file"]].copy()
            d["issue"] = "exact_duplicate"
            dup_rows.append(d)

        # Collect OOO rows
        if n_ooo:
            o = sub[ooo_mask][["Campaign", "Timestamp", "source_file"]].copy()
            o["issue"] = "out_of_order"
            ooo_rows.append(o)

        summary_rows.append({
            "Campaign":       camp,
            "n_records":      n_camp,
            "n_exact_dupes":  n_exact,
            "n_near_dupes":   n_near,
            "n_out_of_order": n_ooo,
            "n_out_of_bounds": n_oob,
            "pct_exact_dupes": round(n_exact / n_camp * 100, 4) if n_camp else 0,
        })

    dup_df  = pd.concat(dup_rows, ignore_index=True)  if dup_rows  else pd.DataFrame()
    ooo_df  = pd.concat(ooo_rows, ignore_index=True)  if ooo_rows  else pd.DataFrame()
    sum_df  = pd.DataFrame(summary_rows)

    dup_path  = out_dir / "07_timestamp_duplicates.csv"
    ooo_path  = out_dir / "07_timestamp_ooo.csv"
    sum_path  = out_dir / "07_timestamp_summary.csv"
    dup_df.to_csv(dup_path, index=False)
    ooo_df.to_csv(ooo_path, index=False)
    sum_df.to_csv(sum_path, index=False)
    print(f"  Saved {dup_path}  ({len(dup_df):,} exact-duplicate rows)")
    print(f"  Saved {ooo_path}  ({len(ooo_df):,} out-of-order rows)")
    print(f"  Saved {sum_path}")

    # ── Plot: bar chart of timestamp issue counts per campaign ─────────────
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(14, 4), constrained_layout=True)
        issue_cols = ["n_exact_dupes", "n_near_dupes", "n_out_of_order"]
        issue_lbls = ["Exact dupes", "Near-dupes (≤0.5 s)", "Out-of-order"]
        x = np.arange(len(campaigns))
        width = 0.25
        for ax, (col, lbl), offset in zip(
            [axes[0]] * 3, zip(issue_cols, issue_lbls), [-width, 0, width]
        ):
            vals = [sum_df.loc[sum_df["Campaign"] == c, col].values[0]
                    if c in sum_df["Campaign"].values else 0
                    for c in campaigns]
            ax.bar(x + offset, vals, width=width * 0.9, label=lbl, alpha=0.8)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(campaigns, rotation=40, ha="right", fontsize=8)
        axes[0].set_ylabel("Count", fontsize=9)
        axes[0].set_title("Timestamp issues — counts per campaign", fontsize=9, fontweight="bold")
        axes[0].legend(fontsize=8)
        axes[0].tick_params(labelsize=7)

        # Pct exact dupes
        pcts = [sum_df.loc[sum_df["Campaign"] == c, "pct_exact_dupes"].values[0]
                if c in sum_df["Campaign"].values else 0
                for c in campaigns]
        bar_cols = [COLORS.get(c, "gray") for c in campaigns]
        axes[1].bar(x, pcts, color=bar_cols, alpha=0.8)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(campaigns, rotation=40, ha="right", fontsize=8)
        axes[1].set_ylabel("% exact duplicates", fontsize=9)
        axes[1].set_title("Exact duplicate timestamps (% of campaign)", fontsize=9, fontweight="bold")
        axes[1].tick_params(labelsize=7)

        fig.suptitle("QC7 — Timestamp quality by campaign", fontsize=10, fontweight="bold")
        out_p = figs_dir / "07_timestamp_issues.png"
        fig.savefig(out_p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_p}")

    n_flags = int(sum_df["n_exact_dupes"].sum() + sum_df["n_out_of_order"].sum()) if len(sum_df) else 0
    n_camps = int(((sum_df["n_exact_dupes"] + sum_df["n_out_of_order"]) > 0).sum()) if len(sum_df) else 0
    return {
        "check_id": "QC7",
        "check_name": "Duplicate/out-of-order timestamps",
        "n_flags_total": n_flags,
        "pct_dataset_flagged": round(n_flags / n_total * 100, 4) if n_total else 0,
        "n_campaigns_affected": n_camps,
        "notes": "n_flags = exact duplicates + out-of-order rows",
    }


# ---------------------------------------------------------------------------
# QC8 — Vertical Profile Plausibility
# ---------------------------------------------------------------------------

def check_08_vertical_profiles(
    df: pd.DataFrame,
    out_dir: Path,
    figs_dir: Path,
) -> dict:
    """Bin T and qv by pressure level and compare against ISA and saturation."""
    print("QC8: Vertical profile plausibility ...")

    campaigns = [c for c in CAMPAIGN_ORDER if c in df["Campaign"].unique()]
    n_total = len(df)
    out_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    # Pressure bin edges (hPa) — standard levels
    P_EDGES   = [1050, 925, 850, 700, 500, 400, 300, 200, 150, 100, 70, 0]
    P_CENTERS = [(P_EDGES[i] + P_EDGES[i + 1]) / 2 for i in range(len(P_EDGES) - 1)]

    # ICAO standard atmosphere temperature at each pressure center (°C)
    # Computed via h = (1 - (P/1013.25)^(1/5.25588)) / 2.25577e-5 for troposphere
    def _icao_T_from_P(P_hPa: float) -> float:
        if P_hPa <= 0:
            return np.nan
        if P_hPa >= 226.32:  # troposphere
            h = (1.0 - (P_hPa / 1013.25) ** (1.0 / 5.25588)) / 2.25577e-5
            T_K = 288.15 - 6.5e-3 * h
        else:  # lower stratosphere (isothermal)
            T_K = 216.65
        return T_K - 273.15

    df2 = df.copy()
    df2["P_hPa"]  = pd.to_numeric(df2["P_hPa"],  errors="coerce")
    df2["Tair_C"] = pd.to_numeric(df2["Tair_C"], errors="coerce")
    df2["qv"]     = pd.to_numeric(df2["qv"],     errors="coerce")
    df2["P_bin"]  = pd.cut(df2["P_hPa"], bins=P_EDGES[::-1], labels=False)

    profile_rows = []
    flag_rows    = []

    for camp in campaigns:
        sub = df2[df2["Campaign"] == camp]
        for bi, (p_lo, p_hi) in enumerate(zip(P_EDGES[1:], P_EDGES[:-1])):
            bin_sub = sub[(sub["P_hPa"] > p_lo) & (sub["P_hPa"] <= p_hi)]
            n_bin   = len(bin_sub)
            pc      = P_CENTERS[-(bi + 1)]  # reversed: bins go high-P to low-P
            # Reindex properly
            pc      = (p_lo + p_hi) / 2

            T_isa = _icao_T_from_P(pc)

            T_mean = float(bin_sub["Tair_C"].mean()) if n_bin else np.nan
            T_std  = float(bin_sub["Tair_C"].std())  if n_bin else np.nan
            qv_mean= float(bin_sub["qv"].mean())     if n_bin else np.nan
            qv_std = float(bin_sub["qv"].std())      if n_bin else np.nan

            # Saturation reference for the qv_exceeds_saturation check: use
            # the bin's own OBSERVED mean temperature and the LIQUID
            # saturation formula (matching QC2's row-level check), not the
            # ISA theoretical temperature with the ICE formula. Real
            # near-surface/lower-troposphere air is routinely warmer than
            # the ISA standard atmosphere, and ice saturation vapor pressure
            # is always <= liquid at the same temperature -- combining both
            # systematically underestimated the true reference and produced
            # widespread false positives (verified: 41 of 47 originally
            # flagged bins no longer exceed saturation once corrected).
            # T_isa is still used for the separate T_deviation_from_ISA
            # check below and for the T_isa_C/T_dev_C profile columns, where
            # comparing against the ISA reference is the intended check.
            if not np.isnan(T_mean):
                es_pc = float(es_liq_hPa(np.array([T_mean]))[0])
                denom = pc - es_pc
                qv_sat_pc = 0.622 * es_pc / denom * 1000 if denom > 0 else np.nan
            else:
                qv_sat_pc = np.nan

            row = {
                "Campaign": camp,
                "P_lo_hPa": p_lo, "P_hi_hPa": p_hi, "P_center_hPa": pc,
                "n_records": n_bin,
                "T_mean_C": round(T_mean, 3) if not np.isnan(T_mean) else np.nan,
                "T_std_C":  round(T_std,  3) if not np.isnan(T_std)  else np.nan,
                "T_isa_C":  round(T_isa,  3),
                "T_dev_C":  round(T_mean - T_isa, 3) if (not np.isnan(T_mean)) else np.nan,
                "qv_mean_gkg": round(qv_mean, 4) if not np.isnan(qv_mean) else np.nan,
                "qv_std_gkg":  round(qv_std,  4) if not np.isnan(qv_std)  else np.nan,
                "qv_sat_gkg":  round(qv_sat_pc, 4) if not np.isnan(qv_sat_pc) else np.nan,
            }
            profile_rows.append(row)

            if n_bin >= 10:
                if not np.isnan(T_mean) and abs(T_mean - T_isa) > 30:
                    flag_rows.append({**row,
                                      "flag_type": "T_deviation_from_ISA",
                                      "deviation": round(T_mean - T_isa, 2)})
                if not np.isnan(qv_mean) and not np.isnan(qv_sat_pc) and qv_mean > qv_sat_pc:
                    flag_rows.append({**row,
                                      "flag_type": "qv_exceeds_saturation",
                                      "deviation": round(qv_mean - qv_sat_pc, 4)})

    profile_df = pd.DataFrame(profile_rows)
    flag_df    = pd.DataFrame(flag_rows)

    profile_path = out_dir / "08_vertical_profiles.csv"
    flag_path    = out_dir / "08_vertical_profile_flags.csv"
    profile_df.to_csv(profile_path, index=False)
    flag_df.to_csv(flag_path, index=False)
    print(f"  Saved {profile_path}")
    print(f"  Saved {flag_path}  ({len(flag_df)} flagged pressure bins)")

    # ── Plot A: T profiles ─────────────────────────────────────────────────
    p_vals_isa = np.linspace(70, 1050, 200)
    T_isa_line = [_icao_T_from_P(p) for p in p_vals_isa]

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6, 8))
        for camp in campaigns:
            sub_p = profile_df[(profile_df["Campaign"] == camp) & (profile_df["n_records"] >= 10)]
            if len(sub_p) == 0:
                continue
            ax.plot(sub_p["T_mean_C"], sub_p["P_center_hPa"],
                    color=COLORS.get(camp, "gray"), lw=1.5, marker="o",
                    markersize=4, label=camp)
        ax.plot(T_isa_line, p_vals_isa, "k--", lw=2, label="ISA", zorder=10)
        ax.set_ylim(1050, 50)
        ax.set_ylabel("Pressure (hPa)", fontsize=9)
        ax.set_xlabel("Mean Tair_C (°C)", fontsize=9)
        ax.set_title("QC8a — Campaign-mean T profiles vs ISA\n(dashed = ICAO standard)",
                     fontsize=10, fontweight="bold")
        ax.legend(fontsize=7, loc="upper left", ncol=2)
        ax.tick_params(labelsize=8)
        plt.tight_layout()
        out_a = figs_dir / "08a_T_profiles.png"
        fig.savefig(out_a, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_a}")

    # ── Plot B: qv profiles with saturation envelope ───────────────────────
    qv_sat_line = []
    for p in p_vals_isa:
        T_p = _icao_T_from_P(p)
        es  = float(es_ice_hPa(np.array([T_p]))[0])
        dn  = p - es
        qv_sat_line.append(0.622 * es / dn * 1000 if dn > 0 else np.nan)

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6, 8))
        for camp in campaigns:
            sub_p = profile_df[(profile_df["Campaign"] == camp) & (profile_df["n_records"] >= 10)
                               & profile_df["qv_mean_gkg"].notna()]
            if len(sub_p) == 0:
                continue
            ax.plot(sub_p["qv_mean_gkg"], sub_p["P_center_hPa"],
                    color=COLORS.get(camp, "gray"), lw=1.5, marker="o",
                    markersize=4, label=camp)
        ax.plot(qv_sat_line, p_vals_isa, "k--", lw=2, label="qv_sat (ISA T)", zorder=10)
        ax.set_ylim(1050, 50)
        ax.set_xlim(left=0)
        ax.set_xscale("log")
        ax.set_ylabel("Pressure (hPa)", fontsize=9)
        ax.set_xlabel("Mean qv (g/kg, log scale)", fontsize=9)
        ax.set_title("QC8b — Campaign-mean qv profiles vs saturation\n(dashed = qv_sat at ISA temperature)",
                     fontsize=10, fontweight="bold")
        ax.legend(fontsize=7, loc="upper left", ncol=2)
        ax.tick_params(labelsize=8)
        plt.tight_layout()
        out_b = figs_dir / "08b_qv_profiles.png"
        fig.savefig(out_b, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_b}")

    n_flags = len(flag_df)
    n_camps = flag_df["Campaign"].nunique() if n_flags else 0
    return {
        "check_id": "QC8",
        "check_name": "Vertical profile plausibility",
        "n_flags_total": n_flags,
        "pct_dataset_flagged": 0.0,
        "n_campaigns_affected": n_camps,
        "notes": "n_flags = pressure bins where |T_mean − T_ISA| > 30°C or qv_mean > qv_sat",
    }


# ---------------------------------------------------------------------------
# QC9 — LWC cross-check for severe Si supersaturation flags (IPHEX/OLYMPEX)
# ---------------------------------------------------------------------------

# Campaign-specific raw loader + candidate LWC columns (in priority order).
# King hot-wire is preferred (robust, widely used reference instrument);
# Nevzorov/CDP are fallbacks when King is unavailable for a given flight.
_LWC_CROSSVAL_CAMPAIGNS = {
    "IPHEX": {
        "raw_dir": "IPHEX",
        "pattern": "*.iphex",
        "lwc_cols": ["King_LWC_ad", "Nev_LWC", "CDP_LWC", "CSI_CWC"],
    },
    "OLYMPEX": {
        "raw_dir": "OLYMPEX",
        "pattern": "*",
        "lwc_cols": ["King_LWC_ad", "Nev_LWCcor", "CDP_LWC", "CSI_CWC"],
    },
}

# Si above which a row is considered a "severe" supersaturation flag
# (matches the ad hoc Si > 1.05 threshold used in scripts/full_diagnostic.py).
SI_SEVERE_THRESHOLD = 1.05

# Below this LWC (g/m^3), a severe Si flag is treated as a sensor-error
# candidate rather than plausible in-cloud/rain contamination. Chosen as a
# conservative floor above typical hot-wire probe noise floor (~0.01-0.02 g/m^3).
LWC_INCLOUD_THRESHOLD = 0.05

# Below this Tair_C, a sensor-error candidate is treated as a cold-regime
# chilled-mirror amplification candidate rather than an unambiguous fault --
# complementary to the mirror-fault mask threshold in parsers/olympex.py
# (Air_Temp > -10°C there is masked outright as physically implausible).
COLD_REGIME_THRESHOLD = -10.0


def check_09_lwc_crossval(
    df: pd.DataFrame,
    out_dir: Path,
    figs_dir: Path,
) -> dict:
    """Cross-check severe Si>1.05 flags (IPHEX/OLYMPEX) against raw LWC data.

    Si and qv computed from chilled-mirror/frost-point sensors can spike when
    rain or cloud water contaminates the sensor. LWC isn't part of the
    standard schema (QC-only per project decision), so this re-reads the raw
    files for just the already-flagged rows and joins on (source_file,
    Timestamp) to split the flags into plausible in-cloud/precip vs. likely
    sensor error.
    """
    print("QC9: LWC cross-check for severe Si flags (IPHEX/OLYMPEX) ...")

    from parsers.iphex import load_iphex_file
    from parsers.olympex import load_olympex_file

    loaders = {"IPHEX": load_iphex_file, "OLYMPEX": load_olympex_file}

    out_dir.mkdir(parents=True, exist_ok=True)

    result_rows = []
    for camp, spec in _LWC_CROSSVAL_CAMPAIGNS.items():
        if camp not in df["Campaign"].unique():
            continue
        sub = df[(df["Campaign"] == camp) & (df["Si"] > SI_SEVERE_THRESHOLD)]
        if sub.empty:
            continue

        raw_dir = ROOT / "data" / "raw" / spec["raw_dir"]
        for source_file, flagged in sub.groupby("source_file"):
            fpath = raw_dir / source_file
            if not fpath.exists():
                print(f"  Warning: raw file not found for cross-check: {fpath}")
                continue
            try:
                raw = loaders[camp](fpath)
            except Exception as e:
                print(f"  Warning: could not reload {source_file}: {e}")
                continue
            lwc_col = next((c for c in spec["lwc_cols"] if c in raw.columns), None)
            if lwc_col is None or "Timestamp" not in raw.columns:
                continue

            raw_small = pd.DataFrame({
                "Timestamp": raw["Timestamp"],
                "LWC_gm3": pd.to_numeric(raw[lwc_col], errors="coerce"),
                "lwc_source": lwc_col,
            })
            merged = flagged[["Timestamp", "Si", "Tair_C"]].merge(raw_small, on="Timestamp", how="left")
            merged["Campaign"] = camp
            merged["source_file"] = source_file
            result_rows.append(merged)

    if not result_rows:
        print("  No severe Si>1.05 flags found for IPHEX/OLYMPEX — nothing to cross-check.")
        return {
            "check_id": "QC9",
            "check_name": "LWC cross-check (severe Si flags)",
            "n_flags_total": 0,
            "pct_dataset_flagged": 0.0,
            "n_campaigns_affected": 0,
            "notes": "No Si > 1.05 rows found for IPHEX/OLYMPEX in this dataset.",
        }

    result_df = pd.concat(result_rows, ignore_index=True)
    result_df["likely_cause"] = np.where(
        result_df["LWC_gm3"].notna() & (result_df["LWC_gm3"] > LWC_INCLOUD_THRESHOLD),
        "in_cloud_or_precip",
        "sensor_error_candidate",
    )
    # Among sensor-error candidates, separate cold-regime cases (Tair_C <
    # COLD_REGIME_THRESHOLD) where chilled-mirror hysteresis is a documented,
    # physically plausible explanation for an inflated Si -- these are flagged
    # for visibility only, not masked, since e.g. IPHEX's 2014-06-13 flight
    # (Tair_C ~ -55°C) may partly reflect real cirrus-regime supersaturation
    # near the homogeneous-freezing threshold rather than a pure instrument
    # fault. Unambiguous mirror-fault rows (FrostPoint far above a near-0°C
    # Air_Temp) are masked directly in parsers/olympex.py instead and won't
    # appear here as Si>1.05 after rebuild. See
    # docs/decisions/2026-07-05-qc9-iphex-olympex.md.
    result_df["cold_regime_amplification_candidate"] = (
        (result_df["likely_cause"] == "sensor_error_candidate")
        & (result_df["Tair_C"] < COLD_REGIME_THRESHOLD)
    )

    out_path = out_dir / "09_lwc_crossval.csv"
    result_df.to_csv(out_path, index=False)

    n_total = len(result_df)
    n_incloud = int((result_df["likely_cause"] == "in_cloud_or_precip").sum())
    n_sensor = int((result_df["likely_cause"] == "sensor_error_candidate").sum())
    n_cold = int(result_df["cold_regime_amplification_candidate"].sum())
    n_no_lwc = int(result_df["LWC_gm3"].isna().sum())
    n_camps = result_df["Campaign"].nunique()

    print(f"  Saved {out_path}  ({n_total:,} severe Si>1.05 rows cross-checked)")
    print(f"    in-cloud/precip (LWC > {LWC_INCLOUD_THRESHOLD} g/m³): {n_incloud:,}")
    print(f"    sensor-error candidates (LWC ≤ {LWC_INCLOUD_THRESHOLD} g/m³ or missing): {n_sensor:,}")
    print(f"      of which cold-regime amplification candidates (Tair_C < {COLD_REGIME_THRESHOLD}°C, flagged not masked): {n_cold:,}")

    return {
        "check_id": "QC9",
        "check_name": "LWC cross-check (severe Si flags)",
        "n_flags_total": n_sensor,
        "pct_dataset_flagged": round(n_sensor / len(df) * 100, 4) if len(df) else 0.0,
        "n_campaigns_affected": n_camps,
        "notes": (
            f"Of {n_total:,} Si>{SI_SEVERE_THRESHOLD} rows (IPHEX/OLYMPEX), {n_incloud:,} "
            f"have LWC>{LWC_INCLOUD_THRESHOLD} g/m³ (plausible in-cloud/precip contamination), "
            f"{n_sensor:,} have LWC at/near zero or missing (sensor-error candidates), of which "
            f"{n_cold:,} are cold-regime (Tair_C<{COLD_REGIME_THRESHOLD}°C) amplification "
            f"candidates flagged but not masked; {n_no_lwc:,} rows had no LWC value available."
        ),
    }


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------

def write_summary(results: list[dict], out_dir: Path) -> None:
    summary_df = pd.DataFrame(results)
    path = out_dir / "00_qaqc_summary.csv"
    summary_df.to_csv(path, index=False)
    print(f"\nSaved overall summary → {path}")
    print("\n" + "=" * 72)
    print(f"  {'Check':<6}  {'Name':<38}  {'N flags':>9}  {'% flagged':>10}  {'Campaigns':>9}")
    print("  " + "-" * 70)
    for r in results:
        print(f"  {r['check_id']:<6}  {r['check_name']:<38}  "
              f"{r['n_flags_total']:>9,}  {r['pct_dataset_flagged']:>9.3f}%  "
              f"{r['n_campaigns_affected']:>9}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    print(f"Loading {args.env} ...")
    df = pd.read_parquet(args.env)
    print(f"  {len(df):,} records, {len(df.columns)} columns, "
          f"{df['Campaign'].nunique()} campaigns\n")

    check_fns = {
        1: check_01_physical_range,
        2: check_02_internal_consistency,
        3: check_03_stuck_sensor,
        4: check_04_sentinel_values,
        5: check_05_instrument_crossval,
        6: check_06_coverage_audit,
        7: check_07_timestamps,
        8: check_08_vertical_profiles,
        9: check_09_lwc_crossval,
    }

    results = []
    for n in sorted(args.checks):
        if n not in check_fns:
            print(f"  Warning: no check #{n} defined — skipping")
            continue
        result = check_fns[n](df, args.out, args.figs)
        results.append(result)
        print()

    if results:
        write_summary(results, args.out)

    update_latest(args.out.parent, args.out)
    update_latest(args.figs.parent, args.figs)
    print(f"\nLatest run: {args.out.parent / 'latest'} -> {args.out.name}")


if __name__ == "__main__":
    main()
