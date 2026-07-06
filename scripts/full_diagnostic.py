"""
Full diagnostic of combined_env_data.parquet.
Outputs:
  - Console: per-variable stats + per-campaign availability table
  - figs/full_diagnostic/<timestamp>/  — one PNG per variable with per-campaign
    distributions (figs/full_diagnostic/latest symlink kept pointing at the
    newest run)
"""

import argparse
import sys
import textwrap
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.log_paths import timestamp as _run_timestamp, update_latest

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PARQUET = ROOT / "data" / "out" / "combined_env_data.parquet"
FIGDIR  = ROOT / "figs" / "full_diagnostic" / _run_timestamp()

VARIABLES = {
    "Tair_C":  dict(label="Air Temperature (°C)",        xlim=(-90, 40),   bins=65),
    "P_hPa":   dict(label="Pressure (hPa)",               xlim=(50, 1050),  bins=60),
    "Alt_m":   dict(label="Altitude (m)",                  xlim=(-200, 16000), bins=60),
    "Si":      dict(label="Ice supersaturation (Si)",      xlim=(-0.5, 1.5), bins=60),
    "Sw":      dict(label="Liquid supersaturation (Sw)",   xlim=(-0.5, 1.0), bins=60),
    "qv":      dict(label="Water vapor mixing ratio (g/kg)", xlim=(0, 20),  bins=60),
}

CAMPAIGN_ORDER = [
    "ARM", "AIRS-II", "ATTREX", "CRYSTAL-FACE-NASA", "CRYSTAL-FACE-UND",
    "ESCAPE", "ICE-L", "IPHEX", "ISDAC", "MACPEX", "MC3E",
    "MIDCIX", "OLYMPEX", "POSIDON",
]

PALETTE = [
    "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
    "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
    "#aec7e8","#ffbb78","#98df8a","#ff9896",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pct(n, total):
    return f"{100*n/total:.1f}%" if total else "N/A"


def availability_table(df):
    campaigns = sorted(df["Campaign"].unique(), key=lambda c: CAMPAIGN_ORDER.index(c) if c in CAMPAIGN_ORDER else 99)
    rows = []
    for c in campaigns:
        sub = df[df["Campaign"] == c]
        n   = len(sub)
        row = {"Campaign": c, "N_rows": n}
        for v in VARIABLES:
            if v in sub.columns:
                nn  = sub[v].notna().sum()
                row[v] = f"{nn:,} ({pct(nn,n)})"
            else:
                row[v] = "—"
        rows.append(row)
    return pd.DataFrame(rows)


def variable_stats(df):
    rows = []
    for v in VARIABLES:
        if v not in df.columns:
            continue
        s   = df[v].dropna()
        n   = len(s)
        tot = len(df)
        row = {
            "Variable":  v,
            "Non-null":  n,
            "Pct_avail": pct(n, tot),
            "Min":       f"{s.min():.3g}" if n else "—",
            "P5":        f"{s.quantile(.05):.3g}" if n else "—",
            "Median":    f"{s.median():.3g}" if n else "—",
            "Mean":      f"{s.mean():.3g}" if n else "—",
            "P95":       f"{s.quantile(.95):.3g}" if n else "—",
            "Max":       f"{s.max():.3g}" if n else "—",
            "Std":       f"{s.std():.3g}" if n else "—",
        }
        rows.append(row)
    return pd.DataFrame(rows)


def plot_distributions(df, figdir):
    figdir.mkdir(parents=True, exist_ok=True)
    campaigns = [c for c in CAMPAIGN_ORDER if c in df["Campaign"].unique()]
    colors    = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(campaigns)}
    n_camp    = len(campaigns)

    for var, cfg in VARIABLES.items():
        if var not in df.columns:
            print(f"  [skip] {var} not in parquet")
            continue

        ncols  = 4
        nrows  = int(np.ceil(n_camp / ncols))
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(4.5 * ncols, 3.2 * nrows),
                                 sharey=False, sharex=True)
        axes = axes.flatten()

        xmin, xmax = cfg["xlim"]
        bins = np.linspace(xmin, xmax, cfg["bins"] + 1)

        for i, camp in enumerate(campaigns):
            ax  = axes[i]
            sub = df.loc[df["Campaign"] == camp, var].dropna()
            n   = len(sub)
            col = colors[camp]

            if n > 0:
                sub_clipped = sub.clip(xmin, xmax)
                ax.hist(sub_clipped, bins=bins, color=col, alpha=0.80,
                        edgecolor="white", linewidth=0.3)
                med = sub.median()
                ax.axvline(med, color="k", lw=1.2, ls="--", alpha=0.7)
                stats_txt = (
                    f"n={n:,}\n"
                    f"med={med:.2g}\n"
                    f"NaN={100*(1-n/len(df[df['Campaign']==camp])):.0f}%"
                )
                ax.text(0.97, 0.95, stats_txt, transform=ax.transAxes,
                        ha="right", va="top", fontsize=7,
                        bbox=dict(fc="white", ec="none", alpha=0.7))
            else:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                        ha="center", va="center", color="gray", fontsize=9)

            ax.set_title(camp, fontsize=9, fontweight="bold", color=col)
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3g"))
            ax.tick_params(labelsize=7)
            ax.set_xlim(xmin, xmax)

        # hide unused panels
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(cfg["label"], fontsize=13, fontweight="bold", y=1.01)
        fig.supxlabel(cfg["label"], fontsize=9)
        fig.supylabel("Count", fontsize=9)
        plt.tight_layout()
        out = figdir / f"{var}_per_campaign.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out}")


def plot_availability_heatmap(avail_df, figdir):
    """Availability heatmap: campaigns × variables."""
    vars_list  = list(VARIABLES.keys())
    camps      = avail_df["Campaign"].tolist()

    # extract raw counts for the heatmap value
    pct_mat = np.full((len(camps), len(vars_list)), np.nan)
    for i, row in avail_df.iterrows():
        for j, v in enumerate(vars_list):
            cell = row.get(v, "—")
            if cell == "—":
                continue
            # parse "12,345 (67.8%)"
            try:
                p = float(cell.split("(")[1].rstrip("%)"))
                pct_mat[i, j] = p
            except Exception:
                pass

    fig, ax = plt.subplots(figsize=(len(vars_list) * 1.2 + 1, len(camps) * 0.45 + 1))
    im = ax.imshow(pct_mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, label="% non-null")

    ax.set_xticks(range(len(vars_list)))
    ax.set_xticklabels(vars_list, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(camps)))
    ax.set_yticklabels(camps, fontsize=8)

    for i in range(len(camps)):
        for j in range(len(vars_list)):
            v = pct_mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}%", ha="center", va="center",
                        fontsize=7, color="black" if 20 < v < 85 else "white")

    ax.set_title("Data availability (% non-null) per campaign × variable",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    out = figdir / "availability_heatmap.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def plot_temporal_coverage(df, figdir):
    """Strip chart: each campaign's time span."""
    camps = [c for c in CAMPAIGN_ORDER if c in df["Campaign"].unique()]
    colors = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(camps)}

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, camp in enumerate(camps):
        sub = df[df["Campaign"] == camp]["Timestamp"]
        t0, t1 = sub.min(), sub.max()
        ax.barh(i, (t1 - t0).total_seconds() / 86400,
                left=t0.timestamp() / 86400,
                height=0.6, color=colors[camp], alpha=0.8)
        ax.text((t1.timestamp() + t0.timestamp()) / 2 / 86400,
                i, camp, ha="center", va="center", fontsize=7, color="white",
                fontweight="bold")

    # x-axis: convert day-numbers back to years (rough)
    from matplotlib.dates import date2num, DateFormatter, YearLocator
    ax.set_yticks([])
    ax.set_xlabel("Year")
    ax.set_title("Campaign temporal coverage", fontsize=11, fontweight="bold")

    # re-draw with real dates
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_visible(False)

    # simpler: just put date labels on x
    import matplotlib.dates as mdates
    fig2, ax3 = plt.subplots(figsize=(10, 5))
    for i, camp in enumerate(camps):
        sub = df[df["Campaign"] == camp]["Timestamp"]
        t0, t1 = sub.min(), sub.max()
        ax3.barh(i, t1 - t0, left=t0, height=0.6,
                 color=colors[camp], alpha=0.85)
        ax3.text(t0 + (t1 - t0) / 2, i, camp,
                 ha="center", va="center", fontsize=7.5,
                 color="white", fontweight="bold")
    ax3.set_yticks([])
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax3.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)
    ax3.set_title("Campaign temporal coverage", fontsize=11, fontweight="bold")
    ax3.invert_yaxis()
    plt.tight_layout()
    out = figdir / "temporal_coverage.png"
    fig2.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig2)
    plt.close(fig)
    print(f"  saved {out}")


def plot_altitude_vs_tair(df, figdir):
    """Scatter: Alt_m vs Tair_C, coloured by campaign."""
    camps  = [c for c in CAMPAIGN_ORDER if c in df["Campaign"].unique()]
    colors = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(camps)}

    fig, ax = plt.subplots(figsize=(9, 6))
    for camp in reversed(camps):  # bottom layers first
        sub = df[df["Campaign"] == camp].dropna(subset=["Alt_m", "Tair_C"])
        if sub.empty:
            continue
        ax.scatter(sub["Tair_C"], sub["Alt_m"], s=0.3, alpha=0.15,
                   color=colors[camp], label=camp, rasterized=True)

    ax.set_xlabel("Air Temperature (°C)", fontsize=10)
    ax.set_ylabel("Altitude (m)", fontsize=10)
    ax.set_title("Altitude vs Temperature — all campaigns", fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, ncol=2, loc="upper right",
              framealpha=0.9, markerscale=8,
              handler_map={},
              scatterpoints=1)
    plt.tight_layout()
    out = figdir / "alt_vs_tair_all.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def plot_si_vs_tair(df, figdir):
    """Si vs Tair scatter, faceted by campaign."""
    camps  = [c for c in CAMPAIGN_ORDER if c in df["Campaign"].unique()]
    colors = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(camps)}

    ncols = 4
    nrows = int(np.ceil(len(camps) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows),
                              sharex=True, sharey=True)
    axes = axes.flatten()

    for i, camp in enumerate(camps):
        ax  = axes[i]
        sub = df[df["Campaign"] == camp].dropna(subset=["Tair_C", "Si"])
        col = colors[camp]
        if not sub.empty:
            ax.scatter(sub["Tair_C"], sub["Si"], s=0.3, alpha=0.12,
                       color=col, rasterized=True)
        ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.5)
        ax.axhline(1, color="r", lw=0.8, ls="--", alpha=0.5)
        ax.set_xlim(-85, 35)
        ax.set_ylim(-0.5, 1.6)
        ax.set_title(camp, fontsize=9, fontweight="bold", color=col)
        ax.tick_params(labelsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Si vs Tair per campaign  (dashed: Si=0, Si=1)", fontsize=12, fontweight="bold")
    fig.supxlabel("Air Temperature (°C)", fontsize=9)
    fig.supylabel("Si (ice supersaturation)", fontsize=9)
    plt.tight_layout()
    out = figdir / "Si_vs_Tair_per_campaign.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"\n{'='*70}")
    print(f"  CPI-THERMO Full Diagnostic  —  {date.today()}")
    print(f"{'='*70}\n")

    print(f"Loading {PARQUET} …")
    df = pd.read_parquet(PARQUET)
    print(f"  {len(df):,} rows × {len(df.columns)} columns\n")

    # ---- 1. overall shape & dtypes ----------------------------------------
    print("── Columns in parquet ──────────────────────────────────────────────")
    for col in df.columns:
        nn  = df[col].notna().sum()
        pct_v = pct(nn, len(df))
        print(f"  {col:<30} dtype={str(df[col].dtype):<12} non-null={nn:>9,}  ({pct_v})")

    # ---- 2. per-variable stats ---------------------------------------------
    print("\n── Per-variable summary stats ──────────────────────────────────────")
    stats = variable_stats(df)
    print(stats.to_string(index=False))

    # ---- 3. per-campaign availability --------------------------------------
    print("\n── Per-campaign data availability ──────────────────────────────────")
    avail = availability_table(df)
    print(avail.to_string(index=False))

    # ---- 4. row counts by campaign -----------------------------------------
    print("\n── Row counts by campaign ──────────────────────────────────────────")
    counts = df["Campaign"].value_counts().reindex(CAMPAIGN_ORDER).dropna()
    total  = len(df)
    for camp, n in counts.items():
        bar = "█" * int(40 * n / total)
        print(f"  {camp:<22} {n:>8,} ({pct(n,total):>5})  {bar}")
    print(f"  {'TOTAL':<22} {total:>8,}")

    # ---- 5. known-issue quick check ----------------------------------------
    print("\n── Known-issue quick checks ────────────────────────────────────────")
    esc = df[df["Campaign"] == "ESCAPE"]
    esc_low = (esc["P_hPa"] < 50).sum()
    print(f"  ESCAPE P_hPa < 50 hPa  : {esc_low:,} rows")

    iph = df[df["Campaign"] == "IPHEX"]
    if "Si" in df.columns:
        iph_sat = (iph["Si"] > 1.05).sum()
        print(f"  IPHEX Si > 1.05        : {iph_sat:,} rows")

    olym = df[df["Campaign"] == "OLYMPEX"]
    olym_sat = (olym["Si"] > 1.05).sum() if "Si" in df.columns else 0
    print(f"  OLYMPEX Si > 1.05      : {olym_sat:,} rows")

    arm_nan = df[df["Campaign"] == "ARM"]["qv"].isna().sum()
    arm_tot = len(df[df["Campaign"] == "ARM"])
    print(f"  ARM qv NaN             : {arm_nan:,} / {arm_tot:,} ({pct(arm_nan,arm_tot)})")

    # ---- 6. figures --------------------------------------------------------
    print(f"\n── Generating figures → {FIGDIR} ────────────────────────────────")
    FIGDIR.mkdir(parents=True, exist_ok=True)

    print("\n  [1/5] Variable distributions per campaign …")
    plot_distributions(df, FIGDIR)

    print("\n  [2/5] Availability heatmap …")
    plot_availability_heatmap(avail, FIGDIR)

    print("\n  [3/5] Temporal coverage …")
    plot_temporal_coverage(df, FIGDIR)

    print("\n  [4/5] Alt vs Tair scatter …")
    plot_altitude_vs_tair(df, FIGDIR)

    print("\n  [5/5] Si vs Tair per campaign …")
    plot_si_vs_tair(df, FIGDIR)

    update_latest(FIGDIR.parent, FIGDIR)
    print(f"\n{'='*70}")
    print(f"  Done. Figures in {FIGDIR}/")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
