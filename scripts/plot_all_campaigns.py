#!/usr/bin/env python3
"""
Diagnostic plots for all campaigns.

Outputs (in figs/all-campaigns/):
  01_si_distributions.png        — Si KDE per campaign, faceted grid
  02_tair_distributions.png      — Tair_C KDE per campaign, faceted grid
  03_si_vs_tair_hexbin.png       — 2-D hexbin Si vs Tair_C per campaign, faceted grid
  04_data_availability.png       — flight-day availability matrix
  05_summary_boxplots.png        — Si + Tair box plots across campaigns
  06_si_instrument_coverage.png  — heatmap of per-instrument Si fill rates per campaign

Usage (standalone):
  python scripts/plot_all_campaigns.py
  python scripts/plot_all_campaigns.py --env data/out/combined_env_data.parquet --out figs/all-campaigns

Called automatically by main.py after processing.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.colors import BoundaryNorm


def _kde(data, x):
    """Simple KDE via Gaussian kernel — no scipy required."""
    h = 1.06 * data.std() * len(data) ** (-0.2)
    h = max(h, 1e-6)
    diff = (x[:, None] - data[None, :]) / h
    return np.exp(-0.5 * diff ** 2).mean(axis=1) / (h * np.sqrt(2 * np.pi))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args():
    ROOT = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=Path,
                   default=ROOT / "data" / "out" / "combined_env_data.parquet")
    p.add_argument("--out", type=Path,
                   default=ROOT / "figs" / "all-campaigns")
    return p.parse_args()

_args = _parse_args()
ROOT     = Path(__file__).resolve().parents[1]
ENV_PATH = _args.env
OUT_DIR  = _args.out
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Campaign order (roughly chronological)
CAMPAIGN_ORDER = [
    "ARM", "CRYSTAL-FACE-NASA", "CRYSTAL-FACE-UND", "MIDCIX",
    "AIRS-II", "ICE-L", "ISDAC", "MACPEX", "MC3E",
    "ATTREX", "IPHEX", "OLYMPEX", "POSIDON", "ESCAPE",
]

# Colour palette — one colour per campaign
COLORS = {
    "ARM":                "#1f77b4",
    "CRYSTAL-FACE-NASA":  "#ff7f0e",
    "CRYSTAL-FACE-UND":   "#2ca02c",
    "MIDCIX":             "#d62728",
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

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print(f"Loading {ENV_PATH} ...")
env = pd.read_parquet(ENV_PATH)
env = env[env["Campaign"].isin(CAMPAIGN_ORDER)].copy()
env["date"] = env["Timestamp"].dt.date

campaigns = [c for c in CAMPAIGN_ORDER if c in env["Campaign"].unique()]
n = len(campaigns)
NCOLS = 4
NROWS = int(np.ceil(n / NCOLS))

# ---------------------------------------------------------------------------
# Helper: blank unused axes
# ---------------------------------------------------------------------------
def hide_unused(axes_flat, n_used):
    for ax in axes_flat[n_used:]:
        ax.set_visible(False)


# ---------------------------------------------------------------------------
# 1. Si distributions — KDE per campaign
# ---------------------------------------------------------------------------
print("Plot 1: Si distributions ...")
with plt.rc_context(STYLE):
    fig, axes = plt.subplots(NROWS, NCOLS, figsize=(14, NROWS * 2.6),
                             constrained_layout=True)
    axes_flat = axes.flat

    for i, camp in enumerate(campaigns):
        ax = axes_flat[i]
        data = env.loc[env["Campaign"] == camp, "Si"].dropna()
        color = COLORS[camp]

        # KDE
        x = np.linspace(-1.0, 2.0, 400)
        if len(data) > 10:
            try:
                d = data.clip(-1.0, 2.0).values
                y = _kde(d, x)
                ax.fill_between(x, y, alpha=0.35, color=color)
                ax.plot(x, y, color=color, lw=1.5)
            except Exception:
                ax.hist(data.clip(-1.0, 2.0), bins=60, density=True,
                        color=color, alpha=0.6)

        ax.axvline(0, color="k", lw=0.8, ls="--", alpha=0.5)
        ax.set_xlim(-1.0, 2.0)
        ax.set_title(camp, fontsize=8, fontweight="bold", color=color, pad=3)
        ax.set_xlabel("Si", fontsize=7)
        ax.set_ylabel("density", fontsize=7)
        ax.tick_params(labelsize=7)

        # Annotation: median and n
        med = data.median()
        ax.text(0.97, 0.92, f"n={len(data):,}\nmed={med:.3f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=6.5, color="0.35")

    hide_unused(axes_flat, len(campaigns))
    fig.suptitle("Si distributions — all campaigns", fontsize=11, fontweight="bold", y=1.01)
    out = OUT_DIR / "01_si_distributions.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# 2. Tair_C distributions
# ---------------------------------------------------------------------------
print("Plot 2: Tair_C distributions ...")
with plt.rc_context(STYLE):
    fig, axes = plt.subplots(NROWS, NCOLS, figsize=(14, NROWS * 2.6),
                             constrained_layout=True)
    axes_flat = axes.flat

    for i, camp in enumerate(campaigns):
        ax = axes_flat[i]
        data = env.loc[env["Campaign"] == camp, "Tair_C"].dropna()
        color = COLORS[camp]

        t_lo, t_hi = -90.0, 30.0
        x = np.linspace(t_lo, t_hi, 500)
        if len(data) > 10:
            try:
                d = data.clip(t_lo, t_hi).values
                y = _kde(d, x)
                ax.fill_between(x, y, alpha=0.35, color=color)
                ax.plot(x, y, color=color, lw=1.5)
            except Exception:
                ax.hist(data.clip(t_lo, t_hi), bins=60, density=True,
                        color=color, alpha=0.6)

        ax.axvline(0, color="k", lw=0.8, ls="--", alpha=0.5)
        ax.set_xlim(t_lo, t_hi)
        ax.set_title(camp, fontsize=8, fontweight="bold", color=color, pad=3)
        ax.set_xlabel("Tair_C (°C)", fontsize=7)
        ax.set_ylabel("density", fontsize=7)
        ax.tick_params(labelsize=7)

        med = data.median()
        ax.text(0.97, 0.92, f"n={len(data):,}\nmed={med:.1f}°C",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=6.5, color="0.35")

    hide_unused(axes_flat, len(campaigns))
    fig.suptitle("Air temperature distributions — all campaigns", fontsize=11,
                 fontweight="bold", y=1.01)
    out = OUT_DIR / "02_tair_distributions.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# 3. Si vs Tair_C — 2-D hexbin, one panel per campaign
# ---------------------------------------------------------------------------
print("Plot 3: Si vs Tair_C hexbin ...")
with plt.rc_context(STYLE):
    fig, axes = plt.subplots(NROWS, NCOLS, figsize=(14, NROWS * 3.0),
                             constrained_layout=True)
    axes_flat = axes.flat

    for i, camp in enumerate(campaigns):
        ax = axes_flat[i]
        sub = env.loc[env["Campaign"] == camp, ["Tair_C", "Si"]].dropna()

        if len(sub) > 20:
            hb = ax.hexbin(
                sub["Tair_C"].clip(-90, 30),
                sub["Si"].clip(-1.0, 2.0),
                gridsize=40,
                cmap="YlOrRd",
                mincnt=1,
                linewidths=0.2,
            )
            fig.colorbar(hb, ax=ax, pad=0.02, shrink=0.8, label="count")
        else:
            ax.text(0.5, 0.5, "insufficient data",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=8, color="0.5")

        ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.5)
        ax.set_xlim(-90, 30)
        ax.set_ylim(-1.0, 2.0)
        ax.set_title(camp, fontsize=8, fontweight="bold",
                     color=COLORS[camp], pad=3)
        ax.set_xlabel("Tair_C (°C)", fontsize=7)
        ax.set_ylabel("Si", fontsize=7)
        ax.tick_params(labelsize=7)

    hide_unused(axes_flat, len(campaigns))
    fig.suptitle("Si vs air temperature — all campaigns", fontsize=11,
                 fontweight="bold", y=1.01)
    out = OUT_DIR / "03_si_vs_tair_hexbin.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# 4. Data availability timeline matrix
#    Rows = campaigns, columns = calendar dates (each campaign on its own
#    relative-day axis within its deployment window).
#    Each cell is coloured by fraction of that day's records that have valid Si.
# ---------------------------------------------------------------------------
print("Plot 4: Data availability timeline ...")

# Build per-campaign × flight-date stats
records = []
for camp, grp in env.groupby("Campaign"):
    for d, dg in grp.groupby("date"):
        n_total = len(dg)
        n_si    = dg["Si"].notna().sum()
        n_tair  = dg["Tair_C"].notna().sum()
        records.append(dict(
            campaign=camp,
            date=pd.Timestamp(d),
            n_total=n_total,
            n_si=n_si,
            n_tair=n_tair,
            si_frac=n_si / n_total if n_total else 0.0,
        ))

avail = pd.DataFrame(records)

# ---- Panel A: campaign-relative flight-day index -------------------------
# For each campaign, rank its flight dates 1, 2, 3 …
avail["flight_day"] = avail.groupby("campaign")["date"].rank(method="dense").astype(int)
max_flights = int(avail["flight_day"].max())

# Build matrix: rows = campaigns (top to bottom), cols = flight day index
mat_si    = np.full((n, max_flights), np.nan)
mat_tair  = np.full((n, max_flights), np.nan)
mat_n     = np.zeros((n, max_flights))

camp_idx = {c: i for i, c in enumerate(campaigns)}
for _, row in avail.iterrows():
    ci = camp_idx.get(row["campaign"])
    if ci is None:
        continue
    di = int(row["flight_day"]) - 1
    mat_si[ci, di]   = row["si_frac"]
    mat_tair[ci, di] = row["n_tair"] / row["n_total"] if row["n_total"] else 0.0
    mat_n[ci, di]    = row["n_total"]

# ---- Figure layout -------------------------------------------------------
fig = plt.figure(figsize=(18, 8), facecolor="white")

# Two sub-panels stacked: top = Si coverage, bottom = Tair coverage
gs = fig.add_gridspec(2, 1, hspace=0.45, left=0.17, right=0.95,
                      top=0.90, bottom=0.08)

cmap_avail = plt.cm.RdYlGn
norm = BoundaryNorm([0, 0.25, 0.5, 0.75, 0.9, 1.001], cmap_avail.N)

def draw_matrix(ax, mat, title):
    masked = np.ma.masked_where(mat_n == 0, mat)
    im = ax.imshow(masked, aspect="auto", interpolation="nearest",
                   cmap=cmap_avail, norm=norm, origin="upper")
    ax.set_yticks(range(n))
    ax.set_yticklabels(campaigns, fontsize=8)
    ax.set_xlabel("Flight number within campaign", fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=4)

    # Draw grey for no-data cells
    no_data = np.ma.masked_where(mat_n > 0, mat_n)
    ax.imshow(no_data, aspect="auto", interpolation="nearest",
              cmap=plt.cm.Greys, vmin=0, vmax=1, origin="upper", alpha=0.15)

    # Flight number ticks: every 5 starting at 1
    ax.set_xticks(np.arange(0, max_flights, 5))
    ax.set_xticklabels(np.arange(1, max_flights + 1, 5), fontsize=7)
    ax.tick_params(axis="y", length=0)
    return im

ax0 = fig.add_subplot(gs[0])
ax1 = fig.add_subplot(gs[1])

im0 = draw_matrix(ax0, mat_si,   "Si valid fraction per flight day")
im1 = draw_matrix(ax1, mat_tair, "Tair_C valid fraction per flight day")

# Shared colorbar
cbar_ax = fig.add_axes([0.96, 0.12, 0.015, 0.76])
cb = fig.colorbar(im0, cax=cbar_ax, extend="neither")
cb.set_label("Fraction of records valid", fontsize=8)
cb.set_ticks([0, 0.25, 0.5, 0.75, 0.9, 1.0])
cb.ax.tick_params(labelsize=7)

# Add record-count annotation for largest days
for ci, camp in enumerate(campaigns):
    camp_rows = avail[avail["campaign"] == camp]
    for _, row in camp_rows.iterrows():
        di = int(row["flight_day"]) - 1
        nt = int(row["n_total"])
        if nt >= 30_000:
            ax0.text(di, ci, f"{nt//1000}k", ha="center", va="center",
                     fontsize=4.5, color="0.2")

fig.suptitle("Data availability by flight day — all campaigns\n"
             "(grey = no env data; colour = fraction of records with valid measurement)",
             fontsize=10, fontweight="bold")

# Legend patch for grey
grey_patch = mpatches.Patch(facecolor="0.80", label="no data")
fig.legend(handles=[grey_patch], loc="lower right", fontsize=7,
           bbox_to_anchor=(0.95, 0.01))

out = OUT_DIR / "04_data_availability.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {out}")

# ---------------------------------------------------------------------------
# 5. Summary overview: all campaigns on one Si + Tair overlay (box plots)
# ---------------------------------------------------------------------------
print("Plot 5: Summary box plots ...")
with plt.rc_context(STYLE):
    fig, (ax_si, ax_t) = plt.subplots(1, 2, figsize=(14, 5),
                                       constrained_layout=True)

    # Collect per-campaign data lists
    si_data   = [env.loc[env["Campaign"] == c, "Si"].dropna().values for c in campaigns]
    tair_data = [env.loc[env["Campaign"] == c, "Tair_C"].dropna().values for c in campaigns]
    colors_list = [COLORS[c] for c in campaigns]

    # Si box plot
    bp_si = ax_si.boxplot(
        si_data,
        vert=True,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="k", lw=1.5),
        whiskerprops=dict(lw=0.8),
        boxprops=dict(lw=0.8),
        capprops=dict(lw=0.8),
        widths=0.5,
    )
    for patch, color in zip(bp_si["boxes"], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax_si.axhline(0, color="k", lw=0.9, ls="--", alpha=0.5)
    ax_si.set_xticks(range(1, len(campaigns) + 1))
    ax_si.set_xticklabels(campaigns, rotation=40, ha="right", fontsize=8)
    ax_si.set_ylabel("Si", fontsize=9)
    ax_si.set_title("Si distribution by campaign\n(box = IQR, whiskers = 5–95th pct)", fontsize=9)
    ax_si.set_ylim(-1.2, 2.2)

    # Tair box plot
    bp_t = ax_t.boxplot(
        tair_data,
        vert=True,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="k", lw=1.5),
        whiskerprops=dict(lw=0.8),
        boxprops=dict(lw=0.8),
        capprops=dict(lw=0.8),
        widths=0.5,
    )
    for patch, color in zip(bp_t["boxes"], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax_t.axhline(0, color="k", lw=0.9, ls="--", alpha=0.5)
    ax_t.set_xticks(range(1, len(campaigns) + 1))
    ax_t.set_xticklabels(campaigns, rotation=40, ha="right", fontsize=8)
    ax_t.set_ylabel("Tair_C (°C)", fontsize=9)
    ax_t.set_title("Air temperature distribution by campaign\n(box = IQR, whiskers = 5–95th pct)", fontsize=9)

    fig.suptitle("Si and Tair_C summary across all campaigns", fontsize=11,
                 fontweight="bold")
    out = OUT_DIR / "05_summary_boxplots.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")

# ---------------------------------------------------------------------------
# 6. Si instrument coverage heatmap
#    Rows = campaigns, cols = Si_* instrument columns
#    Cell value = % of campaign records with valid data (NaN = column absent)
# ---------------------------------------------------------------------------
print("Plot 6: Si instrument coverage heatmap ...")

# All possible Si instrument columns in display order
SI_INSTRUMENT_COLS = [
    "Si_chilled_mirror",
    "Si_DLH",
    "Si_JLH",
    "Si_HWV",
    "Si_HW",
    "Si_NOAA",
    "Si_UCATS",
    "Si_MRTDL",
    "Si_ophir_tdl",
    "Si_LH_unspecified",
    "Si_frost_point",
    "Si_ALIAS",
    "Si_FISH",
]

SI_SHORT = {
    "Si_chilled_mirror":  "CM",
    "Si_DLH":             "DLH",
    "Si_JLH":             "JLH",
    "Si_HWV":             "HWV",
    "Si_HW":              "HW",
    "Si_NOAA":            "NOAA",
    "Si_UCATS":           "UCATS",
    "Si_MRTDL":           "MRTDL",
    "Si_ophir_tdl":       "TDL",
    "Si_LH_unspecified":  "LH-unspec",
    "Si_frost_point":     "FP",
    "Si_ALIAS":           "ALIAS",
    "Si_FISH":            "FISH",
}

# Only keep instrument columns that are present in this dataset
inst_cols = [c for c in SI_INSTRUMENT_COLS if c in env.columns]
labels_x  = [SI_SHORT.get(c, c) for c in inst_cols]
labels_y  = campaigns   # campaign order (already filtered)

n_camp = len(labels_y)
n_inst = len(inst_cols)

# Build matrix: rows = campaigns, cols = instruments; value = pct_valid (0–100)
mat = np.full((n_camp, n_inst), np.nan)
for ci, camp in enumerate(labels_y):
    sub = env[env["Campaign"] == camp]
    n   = len(sub)
    if n == 0:
        continue
    for ii, col in enumerate(inst_cols):
        mat[ci, ii] = sub[col].notna().sum() / n * 100

# Also build a column showing overall Si coverage for reference
si_col_vec = np.array([
    env.loc[env["Campaign"] == c, "Si"].notna().mean() * 100 if c in env["Campaign"].values else np.nan
    for c in labels_y
])

with plt.rc_context(STYLE):
    fig_h = max(3.5, 0.45 * n_camp + 1.5)
    fig, (ax_inst, ax_si) = plt.subplots(
        1, 2,
        figsize=(max(8, 0.7 * n_inst + 2.5), fig_h),
        gridspec_kw={"width_ratios": [n_inst, 1], "wspace": 0.04},
        constrained_layout=False,
    )
    plt.subplots_adjust(left=0.22, right=0.92, top=0.88, bottom=0.18, wspace=0.04)

    cmap_cov = plt.cm.RdYlGn
    norm_cov  = BoundaryNorm([0, 10, 25, 50, 75, 90, 100.01], cmap_cov.N)

    # -- Instrument matrix --
    masked = np.ma.masked_invalid(mat)
    im = ax_inst.imshow(masked, aspect="auto", interpolation="nearest",
                        cmap=cmap_cov, norm=norm_cov, origin="upper")

    # Grey out cells where column is absent (all NaN for campaign) vs truly 0%
    absent = np.isnan(mat)
    if absent.any():
        ax_inst.imshow(
            np.where(absent, 1.0, np.nan),
            aspect="auto", interpolation="nearest",
            cmap=plt.cm.Greys, vmin=0, vmax=1, origin="upper", alpha=0.25,
        )

    # Annotate cells with percentage
    for ci in range(n_camp):
        for ii in range(n_inst):
            v = mat[ci, ii]
            if np.isnan(v):
                continue
            txt = f"{v:.0f}%" if v > 0 else "0"
            color = "white" if v >= 75 else "0.2"
            ax_inst.text(ii, ci, txt, ha="center", va="center",
                         fontsize=6.5, color=color, fontweight="normal")

    ax_inst.set_xticks(range(n_inst))
    ax_inst.set_xticklabels(labels_x, rotation=45, ha="right", fontsize=8)
    ax_inst.set_yticks(range(n_camp))
    ax_inst.set_yticklabels(labels_y, fontsize=8)
    ax_inst.set_title("Per-instrument Si coverage (%)", fontsize=9, fontweight="bold", pad=6)
    ax_inst.tick_params(axis="both", length=0)

    # -- Si (best) column on the right --
    si_mat = si_col_vec.reshape(-1, 1)
    ax_si.imshow(si_mat, aspect="auto", interpolation="nearest",
                 cmap=cmap_cov, norm=norm_cov, origin="upper")
    for ci, v in enumerate(si_col_vec):
        if np.isnan(v):
            continue
        color = "white" if v >= 75 else "0.2"
        ax_si.text(0, ci, f"{v:.0f}%", ha="center", va="center",
                   fontsize=6.5, color=color)
    ax_si.set_xticks([0])
    ax_si.set_xticklabels(["Si\n(best)"], rotation=45, ha="right", fontsize=8)
    ax_si.set_yticks([])
    ax_si.set_title("", pad=6)
    ax_si.tick_params(axis="both", length=0)

    # Colorbar
    cbar = fig.colorbar(im, ax=[ax_inst, ax_si], orientation="horizontal",
                        pad=0.22, fraction=0.04, shrink=0.7)
    cbar.set_label("% of campaign records with valid Si", fontsize=8)
    cbar.set_ticks([0, 25, 50, 75, 90, 100])
    cbar.ax.tick_params(labelsize=7)

    # Legend for grey
    grey_patch = mpatches.Patch(facecolor="0.80", label="column absent / all NaN")
    fig.legend(handles=[grey_patch], loc="lower right", fontsize=7,
               bbox_to_anchor=(0.92, 0.01))

    fig.suptitle("Si instrument coverage by campaign", fontsize=11, fontweight="bold", y=0.97)

    out = OUT_DIR / "06_si_instrument_coverage.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")



# ---------------------------------------------------------------------------
# 7. P_hPa distributions — KDE per campaign
# ---------------------------------------------------------------------------
print("Plot 7: P_hPa distributions ...")
if "P_hPa" in env.columns:
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(NROWS, NCOLS, figsize=(14, NROWS * 2.6),
                                 constrained_layout=True)
        axes_flat = axes.flat

        for i, camp in enumerate(campaigns):
            ax = axes_flat[i]
            data = env.loc[env["Campaign"] == camp, "P_hPa"].dropna()
            color = COLORS[camp]

            p_lo, p_hi = 50.0, 1050.0
            x = np.linspace(p_lo, p_hi, 500)
            if len(data) > 10:
                try:
                    d = data.clip(p_lo, p_hi).values
                    y = _kde(d, x)
                    ax.fill_between(x, y, alpha=0.35, color=color)
                    ax.plot(x, y, color=color, lw=1.5)
                except Exception:
                    ax.hist(data.clip(p_lo, p_hi), bins=60, density=True,
                            color=color, alpha=0.6)
            else:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=8, color="0.5")

            ax.set_xlim(p_hi, p_lo)  # descending: surface on left, upper trop on right
            ax.set_title(camp, fontsize=8, fontweight="bold", color=color, pad=3)
            ax.set_xlabel("P_hPa (hPa)", fontsize=7)
            ax.set_ylabel("density", fontsize=7)
            ax.tick_params(labelsize=7)

            med = data.median() if len(data) > 0 else np.nan
            n_valid = len(data)
            med_str = f"{med:.0f}" if np.isfinite(med) else "—"
            ax.text(0.97, 0.92, f"n={n_valid:,}\nmed={med_str} hPa",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=6.5, color="0.35")

        hide_unused(axes_flat, len(campaigns))
        fig.suptitle(
            "Pressure distributions — all campaigns\n"
            "(x-axis descending: surface → upper troposphere)",
            fontsize=11, fontweight="bold", y=1.01,
        )
        out = OUT_DIR / "07_pressure_distributions.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out}")
else:
    print("  Skipped (P_hPa column not present in dataset)")


# ---------------------------------------------------------------------------
# 8. qv distributions — KDE per campaign
# ---------------------------------------------------------------------------
print("Plot 8: qv distributions ...")
if "qv" in env.columns:
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(NROWS, NCOLS, figsize=(14, NROWS * 2.6),
                                 constrained_layout=True)
        axes_flat = axes.flat

        for i, camp in enumerate(campaigns):
            ax = axes_flat[i]
            # Clip at 50 g/kg to exclude ground-level sensor artefacts
            data = env.loc[env["Campaign"] == camp, "qv"].dropna()
            data = data[data <= 50.0]
            color = COLORS[camp]

            qv_lo = 0.0
            p95 = float(data.quantile(0.95)) if len(data) > 0 else 10.0
            qv_hi = max(10.0, min(p95 * 1.2, 50.0))

            x = np.linspace(qv_lo, qv_hi, 500)
            if len(data) > 10:
                try:
                    d = data.clip(qv_lo, qv_hi).values
                    y = _kde(d, x)
                    ax.fill_between(x, y, alpha=0.35, color=color)
                    ax.plot(x, y, color=color, lw=1.5)
                except Exception:
                    ax.hist(data.clip(qv_lo, qv_hi), bins=60, density=True,
                            color=color, alpha=0.6)
            else:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=8, color="0.5")

            ax.set_xlim(qv_lo, qv_hi)
            ax.set_title(camp, fontsize=8, fontweight="bold", color=color, pad=3)
            ax.set_xlabel("qv (g/kg)", fontsize=7)
            ax.set_ylabel("density", fontsize=7)
            ax.tick_params(labelsize=7)

            med = data.median() if len(data) > 0 else np.nan
            n_valid = len(data)
            med_str = f"{med:.3f}" if np.isfinite(med) else "—"
            ax.text(0.97, 0.92, f"n={n_valid:,}\nmed={med_str} g/kg",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=6.5, color="0.35")

        hide_unused(axes_flat, len(campaigns))
        fig.suptitle(
            "Water vapor mixing ratio (qv) distributions — all campaigns\n"
            "(clipped at 50 g/kg to exclude sensor artifacts; per-panel x-axis auto-scaled)",
            fontsize=11, fontweight="bold", y=1.01,
        )
        out = OUT_DIR / "08_qv_distributions.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out}")
else:
    print("  Skipped (qv column not present in dataset)")


# ---------------------------------------------------------------------------
# 9. qv instrument coverage heatmap (mirrors plot 6 for Si)
# ---------------------------------------------------------------------------
print("Plot 9: qv instrument coverage heatmap ...")

QV_INSTRUMENT_COLS = [
    "qv_chilled_mirror",
    "qv_dlh",
    "qv_jlh",
    "qv_hwv",
    "qv_hw",
    "qv_noaa",
    "qv_ucats",
    "qv_mrtdl",
    "qv_ophir_tdl",
    "qv_lh_unspecified",
    "qv_frost_point",
]

QV_SHORT = {
    "qv_chilled_mirror": "CM",
    "qv_dlh":            "DLH",
    "qv_jlh":            "JLH",
    "qv_hwv":            "HWV",
    "qv_hw":             "HW",
    "qv_noaa":           "NOAA",
    "qv_ucats":          "UCATS",
    "qv_mrtdl":          "MRTDL",
    "qv_ophir_tdl":      "TDL",
    "qv_lh_unspecified": "LH-unspec",
    "qv_frost_point":    "FP",
}

qv_inst_cols = [c for c in QV_INSTRUMENT_COLS if c in env.columns]
if qv_inst_cols:
    labels_xq = [QV_SHORT.get(c, c) for c in qv_inst_cols]
    labels_yq  = campaigns

    n_camp_q = len(labels_yq)
    n_inst_q = len(qv_inst_cols)

    mat_q = np.full((n_camp_q, n_inst_q), np.nan)
    for ci, camp in enumerate(labels_yq):
        sub = env[env["Campaign"] == camp]
        nq = len(sub)
        if nq == 0:
            continue
        for ii, col in enumerate(qv_inst_cols):
            mat_q[ci, ii] = sub[col].notna().sum() / nq * 100

    qv_col_vec = np.array([
        env.loc[env["Campaign"] == c, "qv"].notna().mean() * 100
        if c in env["Campaign"].values else np.nan
        for c in labels_yq
    ])

    with plt.rc_context(STYLE):
        fig_h = max(3.5, 0.45 * n_camp_q + 1.5)
        fig, (ax_inst, ax_qv) = plt.subplots(
            1, 2,
            figsize=(max(8, 0.7 * n_inst_q + 2.5), fig_h),
            gridspec_kw={"width_ratios": [n_inst_q, 1], "wspace": 0.04},
            constrained_layout=False,
        )
        plt.subplots_adjust(left=0.22, right=0.92, top=0.88, bottom=0.18, wspace=0.04)

        cmap_cov = plt.cm.RdYlGn
        norm_cov  = BoundaryNorm([0, 10, 25, 50, 75, 90, 100.01], cmap_cov.N)

        masked_q = np.ma.masked_invalid(mat_q)
        im_q = ax_inst.imshow(masked_q, aspect="auto", interpolation="nearest",
                              cmap=cmap_cov, norm=norm_cov, origin="upper")

        absent_q = np.isnan(mat_q)
        if absent_q.any():
            ax_inst.imshow(
                np.where(absent_q, 1.0, np.nan),
                aspect="auto", interpolation="nearest",
                cmap=plt.cm.Greys, vmin=0, vmax=1, origin="upper", alpha=0.25,
            )

        for ci in range(n_camp_q):
            for ii in range(n_inst_q):
                v = mat_q[ci, ii]
                if np.isnan(v):
                    continue
                txt = f"{v:.0f}%" if v > 0 else "0"
                clr = "white" if v >= 75 else "0.2"
                ax_inst.text(ii, ci, txt, ha="center", va="center",
                             fontsize=6.5, color=clr)

        ax_inst.set_xticks(range(n_inst_q))
        ax_inst.set_xticklabels(labels_xq, rotation=45, ha="right", fontsize=8)
        ax_inst.set_yticks(range(n_camp_q))
        ax_inst.set_yticklabels(labels_yq, fontsize=8)
        ax_inst.set_title("Per-instrument qv coverage (%)", fontsize=9, fontweight="bold", pad=6)
        ax_inst.tick_params(axis="both", length=0)

        qv_mat = qv_col_vec.reshape(-1, 1)
        ax_qv.imshow(qv_mat, aspect="auto", interpolation="nearest",
                     cmap=cmap_cov, norm=norm_cov, origin="upper")
        for ci, v in enumerate(qv_col_vec):
            if np.isnan(v):
                continue
            clr = "white" if v >= 75 else "0.2"
            ax_qv.text(0, ci, f"{v:.0f}%", ha="center", va="center",
                       fontsize=6.5, color=clr)
        ax_qv.set_xticks([0])
        ax_qv.set_xticklabels(["qv\n(best)"], rotation=45, ha="right", fontsize=8)
        ax_qv.set_yticks([])
        ax_qv.set_title("", pad=6)
        ax_qv.tick_params(axis="both", length=0)

        cbar_q = fig.colorbar(im_q, ax=[ax_inst, ax_qv], orientation="horizontal",
                              pad=0.22, fraction=0.04, shrink=0.7)
        cbar_q.set_label("% of campaign records with valid qv", fontsize=8)
        cbar_q.set_ticks([0, 25, 50, 75, 90, 100])
        cbar_q.ax.tick_params(labelsize=7)

        grey_patch = mpatches.Patch(facecolor="0.80", label="column absent / all NaN")
        fig.legend(handles=[grey_patch], loc="lower right", fontsize=7,
                   bbox_to_anchor=(0.92, 0.01))

        fig.suptitle("qv instrument coverage by campaign", fontsize=11,
                     fontweight="bold", y=0.97)

        out = OUT_DIR / "09_qv_instrument_coverage.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out}")
else:
    print("  Skipped (no qv instrument columns present in dataset)")


print("\nDone. All plots saved to", OUT_DIR)
