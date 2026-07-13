#!/usr/bin/env python3
"""
Crystal Diversity Analysis
===========================
Reproduces (and fixes) ssl-cpi-analysis/notebooks/09-crystal-diversity.ipynb
as a standalone, reproducible script against the current L2 tier and current
embeddings archive.

Diversity here means: how spread out are a group's CPI-image embedding
vectors around their own mean direction? Three closely related metrics, all
computed on L2-unit-normalized embeddings:
  - mean resultant length (MRL): ||mean(unit vectors)|| -- 1.0 = identical
    vectors, ~0 = uniformly spread
  - mean cosine similarity to the group's own centroid direction (equal to
    MRL when the group's own mean is the reference)
  - von Mises-Fisher concentration kappa (`estimate_kappa`, closed-form Sra
    2011 approximation) -- higher kappa = lower diversity, on a scale
    comparable across groups of different size/dimensionality

Unlike the notebook, this only uses columns present in L2 + the embeddings
archive: no COCPIT classification labels or per-image geometric features
(width, circularity, ...), so per-class/per-particle-width diversity is out
of scope here (would require a separate labeled-data merge).

Reuses scripts/analyze_data_tiers.py::load_l2_embeddings for the L2<->
embeddings join (streamed, batch-bounded memory) rather than reimplementing
it.

Outputs (logs/analyze_crystal_diversity/<timestamp>/ and
figs/analyze_crystal_diversity/<timestamp>/, with `latest` symlinks):
  by_campaign_diversity.csv       - MRL/kappa per campaign, whole campaign
  by_temperature_bin.csv          - MRL/kappa per (campaign, Tair_C bin)
  by_pressure_bin.csv             - MRL/kappa per (campaign, P_hPa bin)
  run_config.json                 - embedding column, row counts, seed, versions
  01_diversity_by_campaign.png    - bar chart, MRL + kappa per campaign
  02_kappa_vs_temperature.png     - kappa vs Tair_C bin, one line per campaign
  03_kappa_vs_pressure.png        - kappa vs P_hPa bin, one line per campaign
  04_temp_pressure_heatmap.png    - MRL heatmap over (Tair_C bin x P_hPa bin), pooled

Usage:
    python scripts/analyze_crystal_diversity.py
    python scripts/analyze_crystal_diversity.py --embedding-col head_features
    python scripts/analyze_crystal_diversity.py --quick-test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_data_tiers import (
    CAMPAIGN_ORDER, COLORS, STYLE, RNG_SEED, EMBEDDINGS_PATH,
    load_l2_embeddings,
)
from scripts.log_paths import timestamp as _run_timestamp, update_latest

N_TEMP_BINS = 10
N_PRESSURE_BINS = 10
MIN_GROUP_N = 20  # groups smaller than this produce unstable kappa/MRL estimates


def _parse_args() -> argparse.Namespace:
    ts = _run_timestamp()
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--l2", type=Path, default=ROOT / "data" / "out" / "combined_env_data_L2.parquet")
    p.add_argument("--embeddings", type=Path, default=EMBEDDINGS_PATH)
    p.add_argument("--embedding-col", choices=["cls_features", "head_features"],
                    default="cls_features")
    p.add_argument("--out", type=Path, default=ROOT / "logs" / "analyze_crystal_diversity" / ts)
    p.add_argument("--figs", type=Path, default=ROOT / "figs" / "analyze_crystal_diversity" / ts)
    p.add_argument("--quick-test", action="store_true",
                    help="Subsample to a small number of rows per campaign for a fast smoke run")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Diversity metrics
# ---------------------------------------------------------------------------

def unit_normalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def mean_resultant_length(X_unit: np.ndarray) -> float:
    """||mean of unit vectors||. 1.0 = all vectors identical, ~0 = uniform."""
    if len(X_unit) == 0:
        return np.nan
    return float(np.linalg.norm(X_unit.mean(axis=0)))


def estimate_kappa(X_unit: np.ndarray, p: int | None = None) -> float:
    """Closed-form von Mises-Fisher concentration estimate (Sra 2011):
    kappa_hat = R_bar * (p - R_bar^2) / (1 - R_bar^2), where R_bar is the
    mean resultant length and p is the embedding dimensionality. Higher
    kappa = samples more tightly concentrated around the mean direction
    (lower diversity)."""
    if len(X_unit) < 2:
        return np.nan
    if p is None:
        p = X_unit.shape[1]
    r_bar = mean_resultant_length(X_unit)
    if not np.isfinite(r_bar) or r_bar >= 1.0:
        return np.nan
    return float(r_bar * (p - r_bar ** 2) / (1 - r_bar ** 2))


def diversity_stats(X_unit: np.ndarray) -> dict:
    p = X_unit.shape[1]
    return {
        "n": len(X_unit),
        "mrl": mean_resultant_length(X_unit),
        "kappa": estimate_kappa(X_unit, p=p),
    }


# ---------------------------------------------------------------------------
# Per-campaign / per-bin summaries
# ---------------------------------------------------------------------------

def by_campaign_diversity(df: pd.DataFrame, X_unit: np.ndarray) -> pd.DataFrame:
    rows = []
    for camp in CAMPAIGN_ORDER:
        mask = (df["Campaign"] == camp).to_numpy()
        if mask.sum() < MIN_GROUP_N:
            continue
        stats = diversity_stats(X_unit[mask])
        stats["Campaign"] = camp
        rows.append(stats)
    return pd.DataFrame(rows)[["Campaign", "n", "mrl", "kappa"]]


def _bin_edges(series: pd.Series, n_bins: int) -> np.ndarray:
    lo, hi = series.min(), series.max()
    return np.linspace(lo, hi, n_bins + 1)


def by_env_bin_diversity(df: pd.DataFrame, X_unit: np.ndarray, env_col: str,
                          n_bins: int) -> pd.DataFrame:
    edges = _bin_edges(df[env_col], n_bins)
    bin_idx = np.digitize(df[env_col].to_numpy(), edges[1:-1], right=False)
    bin_labels = [f"[{edges[i]:.1f}, {edges[i+1]:.1f})" for i in range(n_bins)]

    rows = []
    for camp in CAMPAIGN_ORDER:
        camp_mask = (df["Campaign"] == camp).to_numpy()
        if camp_mask.sum() < MIN_GROUP_N:
            continue
        for b in range(n_bins):
            mask = camp_mask & (bin_idx == b)
            if mask.sum() < MIN_GROUP_N:
                continue
            stats = diversity_stats(X_unit[mask])
            stats.update({"Campaign": camp, "bin_idx": b, "bin_label": bin_labels[b],
                          "bin_mid": (edges[b] + edges[b + 1]) / 2})
            rows.append(stats)
    cols = ["Campaign", "bin_idx", "bin_label", "bin_mid", "n", "mrl", "kappa"]
    return pd.DataFrame(rows)[cols] if rows else pd.DataFrame(columns=cols)


def temp_pressure_heatmap_data(df: pd.DataFrame, X_unit: np.ndarray) -> pd.DataFrame:
    """Pooled (all campaigns together) MRL over a (Tair_C x P_hPa) grid."""
    t_edges = _bin_edges(df["Tair_C"], N_TEMP_BINS)
    p_edges = _bin_edges(df["P_hPa"], N_PRESSURE_BINS)
    t_idx = np.digitize(df["Tair_C"].to_numpy(), t_edges[1:-1], right=False)
    p_idx = np.digitize(df["P_hPa"].to_numpy(), p_edges[1:-1], right=False)

    grid = np.full((N_TEMP_BINS, N_PRESSURE_BINS), np.nan)
    for ti in range(N_TEMP_BINS):
        for pi in range(N_PRESSURE_BINS):
            mask = (t_idx == ti) & (p_idx == pi)
            if mask.sum() >= MIN_GROUP_N:
                grid[ti, pi] = mean_resultant_length(X_unit[mask])
    return grid, t_edges, p_edges


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_by_campaign(df: pd.DataFrame, figs_dir: Path) -> None:
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
        colors = [COLORS.get(c, "#888888") for c in df["Campaign"]]
        axes[0].bar(df["Campaign"], df["mrl"], color=colors)
        axes[0].set_ylabel("mean resultant length")
        axes[0].set_title("Embedding diversity by campaign (MRL)")
        axes[0].tick_params(axis="x", rotation=60, labelsize=8)

        axes[1].bar(df["Campaign"], df["kappa"], color=colors)
        axes[1].set_ylabel("von Mises-Fisher kappa")
        axes[1].set_title("Concentration by campaign (kappa)")
        axes[1].tick_params(axis="x", rotation=60, labelsize=8)

        out_p = figs_dir / "01_diversity_by_campaign.png"
        fig.savefig(out_p, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_p}")


def plot_kappa_vs_bin(df: pd.DataFrame, env_col: str, out_name: str, figs_dir: Path) -> None:
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
        for camp, sub in df.groupby("Campaign"):
            sub = sub.sort_values("bin_mid")
            ax.plot(sub["bin_mid"], sub["kappa"], marker="o", label=camp,
                    color=COLORS.get(camp, "#888888"))
        ax.set_xlabel(env_col)
        ax.set_ylabel("von Mises-Fisher kappa")
        ax.set_title(f"Concentration (kappa) vs {env_col}, by campaign")
        ax.legend(fontsize=7, ncol=2)
        out_p = figs_dir / out_name
        fig.savefig(out_p, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_p}")


def plot_temp_pressure_heatmap(grid: np.ndarray, t_edges: np.ndarray, p_edges: np.ndarray,
                                figs_dir: Path) -> None:
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
        im = ax.imshow(grid.T, origin="lower", aspect="auto", cmap="viridis",
                        extent=[t_edges[0], t_edges[-1], p_edges[0], p_edges[-1]])
        ax.set_xlabel("Tair_C")
        ax.set_ylabel("P_hPa")
        ax.set_title("Pooled embedding diversity (MRL) over temperature x pressure")
        fig.colorbar(im, ax=ax, label="mean resultant length")
        out_p = figs_dir / "04_temp_pressure_heatmap.png"
        fig.savefig(out_p, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_p}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.figs.mkdir(parents=True, exist_ok=True)

    print(f"Loading L2 from {args.l2} ...")
    l2 = pd.read_parquet(args.l2, columns=["Campaign", "cpi_filename", "Tair_C", "P_hPa"])
    print(f"  L2: {len(l2):,} rows")

    l2_with_emb = load_l2_embeddings(l2, args.embeddings, args.embedding_col)
    del l2

    if args.quick_test:
        parts = [
            sub.sample(n=min(len(sub), 500), random_state=RNG_SEED)
            for _, sub in l2_with_emb.groupby("Campaign")
        ]
        l2_with_emb = pd.concat(parts, ignore_index=True)
        del parts
        print(f"  --quick-test: subsampled to {len(l2_with_emb):,} rows")

    l2_with_emb = l2_with_emb.dropna(subset=["Tair_C", "P_hPa"]).reset_index(drop=True)
    print(f"  {len(l2_with_emb):,} rows with embeddings + Tair_C + P_hPa")

    X = np.stack(l2_with_emb[args.embedding_col].to_numpy()).astype(np.float32)
    X_unit = unit_normalize(X)
    del X

    print("\nComputing per-campaign diversity ...")
    camp_df = by_campaign_diversity(l2_with_emb, X_unit)
    camp_path = args.out / "by_campaign_diversity.csv"
    camp_df.to_csv(camp_path, index=False)
    print(f"  Saved {camp_path}")
    plot_by_campaign(camp_df, args.figs)

    print("\nComputing per-temperature-bin diversity ...")
    temp_df = by_env_bin_diversity(l2_with_emb, X_unit, "Tair_C", N_TEMP_BINS)
    temp_path = args.out / "by_temperature_bin.csv"
    temp_df.to_csv(temp_path, index=False)
    print(f"  Saved {temp_path}")
    if not temp_df.empty:
        plot_kappa_vs_bin(temp_df, "Tair_C", "02_kappa_vs_temperature.png", args.figs)

    print("\nComputing per-pressure-bin diversity ...")
    pres_df = by_env_bin_diversity(l2_with_emb, X_unit, "P_hPa", N_PRESSURE_BINS)
    pres_path = args.out / "by_pressure_bin.csv"
    pres_df.to_csv(pres_path, index=False)
    print(f"  Saved {pres_path}")
    if not pres_df.empty:
        plot_kappa_vs_bin(pres_df, "P_hPa", "03_kappa_vs_pressure.png", args.figs)

    print("\nComputing pooled temperature x pressure heatmap ...")
    grid, t_edges, p_edges = temp_pressure_heatmap_data(l2_with_emb, X_unit)
    plot_temp_pressure_heatmap(grid, t_edges, p_edges, args.figs)

    run_config = {
        "embedding_col": args.embedding_col,
        "l2_path": str(args.l2),
        "embeddings_path": str(args.embeddings),
        "rng_seed": RNG_SEED,
        "n_temp_bins": N_TEMP_BINS,
        "n_pressure_bins": N_PRESSURE_BINS,
        "min_group_n": MIN_GROUP_N,
        "quick_test": args.quick_test,
        "n_rows_analyzed": len(l2_with_emb),
        "n_rows_per_campaign": l2_with_emb["Campaign"].value_counts().to_dict(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }
    config_path = args.out / "run_config.json"
    with open(config_path, "w") as f:
        json.dump(run_config, f, indent=2, default=str)
    print(f"  Saved {config_path}")

    update_latest(args.out.parent, args.out)
    update_latest(args.figs.parent, args.figs)
    print(f"\nLatest run: {args.out.parent / 'latest'} -> {args.out.name}")
    print(f"Latest figs: {args.figs.parent / 'latest'} -> {args.figs.name}")


if __name__ == "__main__":
    main()
