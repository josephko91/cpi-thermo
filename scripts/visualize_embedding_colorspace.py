#!/usr/bin/env python3
"""
CPI embedding color-space visualization: Si vs Tair
=====================================================
Reduces each CPI image's 384-dim `cls_features` embedding to 3 components
(PCA / UMAP / t-SNE, run as three independent variants of the same
reduction task on the same input points), maps the 3 components to an RGB
triple, converts to HSV, and renders the Hue channel through several
matplotlib colormaps as a scatter over the Si-Tair thermodynamic phase
space -- i.e. "what does the reduced-order representation of CPI
ice-particle morphology look like as a function of ambient thermodynamic
conditions?" A second, complementary view bins Si-Tair space into a 2D
grid and shows each bin's *mean* embedding-color, a spatially smoothed
version of the same signal.

Two visualization families, both starting from the same embedding-to-color
pipeline:
  1. Scatter -- one point per CPI image (a representative subsample),
     colored by its own embedding-derived color.
  2. Binned/smoothed heatmap -- Si-Tair space divided into a 2D grid;
     each bin's color is the mean of its member points' RGB triples,
     shown only for bins meeting a statistically-grounded minimum sample
     count (see MIN_BIN_FLOOR below).

Color pipeline (both families): 3 components -> min-max normalized to
[0,1] as (R,G,B) -- itself a valid "true color" rendering, shown as one
scatter panel -- then converted to HSV via `matplotlib.colors.rgb_to_hsv`
and re-rendered by passing Hue through a chosen matplotlib colormap. This
second step exists because raw RGB from 3 arbitrary reduction axes is
often low-contrast/muddy (components rarely span the full unit cube), while
Hue is a single perceptually-meaningful scalar that a well-designed
colormap can spread across the full visual range.

Compute-budget design (target: well under 3 hours on a 16-core-ish
MacBook Pro, ~17GB RAM):
  - PCA is linear and cheap, so it is fit on the FULL matched L2
    population (~1.8M rows) directly -- this cost is dominated by a
    384x384 covariance/eigendecomposition, seconds not minutes.
  - UMAP and t-SNE are fit on a single shared representative random
    subsample (SCATTER_N rows, same rows for all 3 methods, so the 3
    "variants" are directly comparable point-for-point). UMAP also
    supports `.transform()` on new points (cheaper than a fresh fit), so
    an extra UMAP_HEATMAP_EXTRA_N points are projected through the fitted
    UMAP model to give the UMAP-based binned heatmap a larger, more
    statistically robust population than the scatter alone.
  - t-SNE has no out-of-sample extension (no `.transform()`), so it is
    scatter-only here -- no t-SNE-based heatmap is attempted. t-SNE is
    also run on a 50-dim PCA pre-reduction of the subsample (standard
    practice: distance computations in 384 raw dims are much slower and
    noisier than in a 50-dim PCA-denoised subspace) rather than the raw
    384-dim embedding.
  - Representativeness of SCATTER_N: a simple-random sample of size N
    from population ~1.8M has a worst-case margin of error on any
    subgroup proportion of ~sqrt(0.25/N) (95% CI, p=0.5 is the
    conservative/widest case) -- at N=50,000 that is ~0.45%, so campaign
    mix and Tair/Si regime coverage should closely track the full
    population. This is checked empirically, not just asserted --
    see plot_representativeness_check().

Outputs (logs/visualize_embedding_colorspace/<ts>/ and
figs/visualize_embedding_colorspace/<ts>/, with `latest` symlinks):
  subsample_representativeness_stats.csv - subsample vs. full-population Tair_C/Si stats + KS test
  pca_variance_ratio.csv                 - explained variance of the 3 retained PCA components
  reduction_coords_subsample.csv         - per-method 3D coords + RGB/Hue for the shared subsample
  bin_resolution_stats.csv               - per (method, resolution): bin counts, retained %, threshold
  heatmap_bin_aggregates.csv             - per (method, resolution, bin): mean color + count
  run_config.json                        - sample sizes, seeds, timings, package versions

  01_subsample_representativeness_check.png
  02_pca_variance_explained.png
  03_colormap_hue_legend_strip.png
  04_scatter_truecolor_by_method.png
  05_scatter_hue_colormap_comparison_pca.png
  06_scatter_hue_colormap_comparison_umap.png
  07_scatter_hue_colormap_comparison_tsne.png
  08_binned_heatmap_pca_resolutions.png
  09_binned_heatmap_umap_resolutions.png

Usage:
    python scripts/visualize_embedding_colorspace.py
    python scripts/visualize_embedding_colorspace.py --quick-test   # tiny sizes, for smoke-testing
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_data_tiers import load_l2_embeddings, EMBEDDINGS_PATH, RNG_SEED, STYLE
from scripts.log_paths import timestamp as _run_timestamp, update_latest

import sklearn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

EMBEDDING_COL = "cls_features"

SCATTER_N = 50_000               # shared subsample: PCA/UMAP/t-SNE all fit + scatter on this same set
UMAP_HEATMAP_EXTRA_N = 500_000   # additional points UMAP.transform()'s (not fit) for heatmap statistics
TSNE_PCA_PREREDUCE = 50         # PCA dims fed to t-SNE (standard practice, see module docstring)
TSNE_PERPLEXITY = 30

COLORMAPS = ["hsv", "twilight", "twilight_shifted", "turbo", "viridis"]
HEATMAP_CMAP = "twilight"

MIN_BIN_FLOOR = 30              # CLT rule-of-thumb floor for a ~stable sample-mean estimate
RESOLUTION_MULTIPLIERS = {"coarse": 0.25, "medium": 0.5, "fine": 1.0}
FD_BIN_CAP = (8, 200)            # clip Freedman-Diaconis bin-count suggestion to a sane range


def _parse_args() -> argparse.Namespace:
    ts = _run_timestamp()
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--l2", type=Path, default=ROOT / "data" / "out" / "combined_env_data_L2.parquet")
    p.add_argument("--embeddings", type=Path, default=EMBEDDINGS_PATH)
    p.add_argument("--out", type=Path, default=ROOT / "logs" / "visualize_embedding_colorspace" / ts)
    p.add_argument("--figs", type=Path, default=ROOT / "figs" / "visualize_embedding_colorspace" / ts)
    p.add_argument("--scatter-n", type=int, default=SCATTER_N)
    p.add_argument("--umap-extra-n", type=int, default=UMAP_HEATMAP_EXTRA_N)
    p.add_argument("--quick-test", action="store_true",
                    help="Tiny sample sizes for a fast correctness smoke-test, not a real analysis run")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(l2_path: Path, embeddings_path: Path) -> tuple[np.ndarray, pd.DataFrame]:
    print(f"Loading L2 (Campaign, cpi_filename, Tair_C, Si only) from {l2_path} ...")
    l2 = pd.read_parquet(l2_path, columns=["Campaign", "cpi_filename", "Tair_C", "Si"])
    print(f"  L2: {len(l2):,} rows")

    print(f"Loading '{EMBEDDING_COL}' embeddings from {embeddings_path} ...")
    l2_with_emb = load_l2_embeddings(l2, embeddings_path, EMBEDDING_COL)
    del l2
    gc.collect()
    print(f"  Matched {len(l2_with_emb):,} rows")

    X = np.stack(l2_with_emb[EMBEDDING_COL].to_numpy()).astype(np.float32)
    meta = l2_with_emb[["Campaign", "Tair_C", "Si"]].reset_index(drop=True)
    del l2_with_emb
    gc.collect()
    return X, meta


def standardize(X: np.ndarray) -> np.ndarray:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    return ((X - mean) / std).astype(np.float32)


# ---------------------------------------------------------------------------
# Color pipeline: 3 components -> RGB -> HSV -> colormap(Hue)
# ---------------------------------------------------------------------------

def components_to_rgb(components: np.ndarray, ref_min: np.ndarray, ref_max: np.ndarray) -> np.ndarray:
    span = np.where(ref_max > ref_min, ref_max - ref_min, 1.0)
    return np.clip((components - ref_min) / span, 0.0, 1.0)


def rgb_to_hue(rgb: np.ndarray) -> np.ndarray:
    return mcolors.rgb_to_hsv(rgb)[:, 0]


# ---------------------------------------------------------------------------
# Reduction methods
# ---------------------------------------------------------------------------

def run_pca(Xz_full: np.ndarray, out_dir: Path, figs_dir: Path) -> tuple[np.ndarray, PCA]:
    print("  [PCA] Fitting on full population (linear, cheap even at ~1.8M rows) ...")
    t0 = time.time()
    pca = PCA(n_components=3, random_state=RNG_SEED)
    scores_full = pca.fit_transform(Xz_full).astype(np.float32)
    print(f"  [PCA] Done in {time.time() - t0:.1f}s. "
          f"Explained variance ratio: {pca.explained_variance_ratio_.round(4).tolist()}")

    var_df = pd.DataFrame({
        "component": ["PC1", "PC2", "PC3"],
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_variance_ratio": np.cumsum(pca.explained_variance_ratio_),
    })
    var_df.to_csv(out_dir / "pca_variance_ratio.csv", index=False)

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(4.5, 4), constrained_layout=True)
        ax.bar(var_df["component"], var_df["explained_variance_ratio"], color="#4c72b0")
        for i, v in enumerate(var_df["explained_variance_ratio"]):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("explained variance ratio")
        cum3 = var_df["cumulative_variance_ratio"].iloc[-1]
        ax.set_title(f"PCA (3 comps): {cum3:.1%} of 384-dim variance captured",
                     fontsize=9, fontweight="bold")
        out_p = figs_dir / "02_pca_variance_explained.png"
        fig.savefig(out_p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_p}")

    return scores_full, pca


def run_umap(X_scatter: np.ndarray, X_extra: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    print(f"  [UMAP] Fitting on {len(X_scatter):,} points ...")
    t0 = time.time()
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=3,
                         metric="euclidean", random_state=RNG_SEED)
    scores_scatter = reducer.fit_transform(X_scatter).astype(np.float32)
    print(f"  [UMAP] Fit done in {time.time() - t0:.1f}s")

    if len(X_extra) == 0:
        return scores_scatter, np.empty((0, 3), dtype=np.float32)

    print(f"  [UMAP] Transforming {len(X_extra):,} additional points for heatmap statistics ...")
    t0 = time.time()
    scores_extra = reducer.transform(X_extra).astype(np.float32)
    print(f"  [UMAP] Transform done in {time.time() - t0:.1f}s")
    return scores_scatter, scores_extra


def run_tsne(X_scatter: np.ndarray) -> np.ndarray:
    print(f"  [t-SNE] PCA pre-reducing {X_scatter.shape[1]}-dim -> {TSNE_PCA_PREREDUCE}-dim ...")
    pre = PCA(n_components=min(TSNE_PCA_PREREDUCE, X_scatter.shape[1] - 1, len(X_scatter) - 1),
              random_state=RNG_SEED).fit_transform(X_scatter)
    print(f"  [t-SNE] Fitting on {len(X_scatter):,} points (perplexity={TSNE_PERPLEXITY}) ...")
    t0 = time.time()
    tsne = TSNE(n_components=3, perplexity=min(TSNE_PERPLEXITY, len(X_scatter) - 1),
                init="pca", learning_rate="auto", random_state=RNG_SEED, n_jobs=-1)
    scores = tsne.fit_transform(pre).astype(np.float32)
    print(f"  [t-SNE] Done in {time.time() - t0:.1f}s")
    return scores


# ---------------------------------------------------------------------------
# Plot: representativeness check
# ---------------------------------------------------------------------------

def plot_representativeness_check(meta_full: pd.DataFrame, sample_idx: np.ndarray,
                                   out_dir: Path, figs_dir: Path) -> None:
    sub = meta_full.iloc[sample_idx]
    rows = []
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        for ax, col in zip(axes, ["Tair_C", "Si"]):
            full_vals = meta_full[col].to_numpy()
            sub_vals = sub[col].to_numpy()
            ax.hist(full_vals, bins=80, density=True, alpha=0.5, label=f"full (n={len(full_vals):,})", color="#4c72b0")
            ax.hist(sub_vals, bins=80, density=True, alpha=0.5, label=f"subsample (n={len(sub_vals):,})", color="#dd8452")
            ks_stat, ks_p = ks_2samp(full_vals, sub_vals)
            ax.set_title(f"{col} (KS stat={ks_stat:.4f}, p={ks_p:.3f})", fontsize=9, fontweight="bold")
            ax.legend(fontsize=7)
            rows.append({
                "variable": col, "full_mean": full_vals.mean(), "full_std": full_vals.std(),
                "subsample_mean": sub_vals.mean(), "subsample_std": sub_vals.std(),
                "ks_statistic": ks_stat, "ks_pvalue": ks_p,
            })
        fig.suptitle("Representative-subsample check: full population vs. shared reduction subsample",
                      fontsize=10, fontweight="bold")
        out_p = figs_dir / "01_subsample_representativeness_check.png"
        fig.savefig(out_p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_p}")
    pd.DataFrame(rows).to_csv(out_dir / "subsample_representativeness_stats.csv", index=False)


# ---------------------------------------------------------------------------
# Plot: colormap legend strip
# ---------------------------------------------------------------------------

def plot_colormap_legend(figs_dir: Path) -> None:
    ramp = np.linspace(0, 1, 256).reshape(1, -1)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(len(COLORMAPS), 1, figsize=(6, 0.6 * len(COLORMAPS) + 0.5),
                                  constrained_layout=True)
        for ax, cmap_name in zip(axes, COLORMAPS):
            ax.imshow(ramp, aspect="auto", cmap=cmap_name)
            ax.set_yticks([])
            ax.set_xticks([0, 255])
            ax.set_xticklabels(["hue=0", "hue=1"], fontsize=7)
            ax.set_ylabel(cmap_name, rotation=0, ha="right", va="center", fontsize=8)
        fig.suptitle("Hue -> colormap legend (applies to all colormap-comparison figures)",
                      fontsize=9, fontweight="bold")
        out_p = figs_dir / "03_colormap_hue_legend_strip.png"
        fig.savefig(out_p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_p}")


# ---------------------------------------------------------------------------
# Plot: scatter (truecolor + colormap comparison)
# ---------------------------------------------------------------------------

def plot_truecolor_by_method(tair: np.ndarray, si: np.ndarray, rgb_by_method: dict[str, np.ndarray],
                              figs_dir: Path) -> None:
    methods = list(rgb_by_method.keys())
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(methods), figsize=(5.5 * len(methods), 5), constrained_layout=True)
        if len(methods) == 1:
            axes = [axes]
        for ax, method in zip(axes, methods):
            ax.scatter(tair, si, c=rgb_by_method[method], s=6, alpha=0.7, linewidths=0)
            ax.axhline(0, color="k", linestyle="--", linewidth=0.8)
            ax.set_xlabel("Tair [°C]")
            ax.set_ylabel("Ice Supersaturation Si")
            ax.set_title(f"{method} (true RGB from 3 components)", fontsize=9, fontweight="bold")
        fig.suptitle(f"CPI embedding color by reduction method (n={len(tair):,} shared points)",
                      fontsize=11, fontweight="bold")
        out_p = figs_dir / "04_scatter_truecolor_by_method.png"
        fig.savefig(out_p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_p}")


def plot_hue_colormap_comparison(tair: np.ndarray, si: np.ndarray, hue: np.ndarray,
                                  method: str, figs_dir: Path, index_num: int) -> None:
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(COLORMAPS), figsize=(5 * len(COLORMAPS), 5), constrained_layout=True)
        for ax, cmap_name in zip(axes, COLORMAPS):
            colors = plt.get_cmap(cmap_name)(hue)
            ax.scatter(tair, si, c=colors, s=6, alpha=0.7, linewidths=0)
            ax.axhline(0, color="k", linestyle="--", linewidth=0.8)
            ax.set_xlabel("Tair [°C]")
            ax.set_ylabel("Ice Supersaturation Si")
            ax.set_title(cmap_name, fontsize=9, fontweight="bold")
        fig.suptitle(f"{method}: 3-component embedding -> RGB -> HSV Hue -> colormap "
                     f"(n={len(tair):,})", fontsize=11, fontweight="bold")
        out_p = figs_dir / f"0{index_num}_scatter_hue_colormap_comparison_{method.lower().replace('-', '')}.png"
        fig.savefig(out_p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_p}")


# ---------------------------------------------------------------------------
# Binned / smoothed heatmap
# ---------------------------------------------------------------------------

def fd_bin_count(x: np.ndarray, cap: tuple[int, int] = FD_BIN_CAP) -> int:
    """Freedman-Diaconis bin-width rule: h = 2*IQR/n^(1/3), n_bins = range/h.
    Clipped to `cap` since FD on ~1.8M points suggests very fine bins that
    would mostly fall below any reasonable minimum-sample-count threshold."""
    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25
    n = len(x)
    if iqr <= 0 or n < 2:
        return cap[0]
    h = 2 * iqr / (n ** (1 / 3))
    if h <= 0:
        return cap[0]
    n_bins = int(np.ceil((x.max() - x.min()) / h))
    return int(np.clip(n_bins, cap[0], cap[1]))


def make_edges(x: np.ndarray, n_bins: int) -> np.ndarray:
    return np.linspace(x.min(), x.max(), n_bins + 1)


def aggregate_bins(tair: np.ndarray, si: np.ndarray, components: np.ndarray,
                    tair_edges: np.ndarray, si_edges: np.ndarray) -> pd.DataFrame:
    tair_bin = pd.cut(tair, bins=tair_edges, labels=False, include_lowest=True)
    si_bin = pd.cut(si, bins=si_edges, labels=False, include_lowest=True)
    df = pd.DataFrame({
        "tair_bin": tair_bin, "si_bin": si_bin,
        "c1": components[:, 0], "c2": components[:, 1], "c3": components[:, 2],
    }).dropna(subset=["tair_bin", "si_bin"])
    grouped = df.groupby(["tair_bin", "si_bin"]).agg(
        c1=("c1", "mean"), c2=("c2", "mean"), c3=("c3", "mean"), count=("c1", "size"),
    ).reset_index()
    grouped["tair_bin"] = grouped["tair_bin"].astype(int)
    grouped["si_bin"] = grouped["si_bin"].astype(int)
    return grouped


def bin_threshold(counts: np.ndarray) -> int:
    """MIN_BIN_FLOOR (CLT rule of thumb, n>=30 for an approximately-normal
    sampling distribution of the mean) OR 10% of the median occupied-bin
    count at this resolution, whichever is larger -- adapts to how dense
    a given grid resolution naturally is while guaranteeing the statistical
    floor."""
    nonzero = counts[counts > 0]
    if len(nonzero) == 0:
        return MIN_BIN_FLOOR
    return int(max(MIN_BIN_FLOOR, np.ceil(0.1 * np.median(nonzero))))


def grid_to_rgba(agg: pd.DataFrame, n_tair: int, n_si: int, ref_min: np.ndarray, ref_max: np.ndarray,
                  min_n: int, use_hue_cmap: bool) -> np.ndarray:
    grid = np.zeros((n_si, n_tair, 4), dtype=np.float32)
    passed = agg[agg["count"] >= min_n]
    if passed.empty:
        return grid
    components = passed[["c1", "c2", "c3"]].to_numpy()
    rgb = components_to_rgb(components, ref_min, ref_max)
    if use_hue_cmap:
        hue = rgb_to_hue(rgb)
        colors = plt.get_cmap(HEATMAP_CMAP)(hue)[:, :3]
    else:
        colors = rgb
    for (si_bin, tair_bin), color in zip(passed[["si_bin", "tair_bin"]].to_numpy(), colors):
        grid[si_bin, tair_bin, :3] = color
        grid[si_bin, tair_bin, 3] = 1.0
    return grid


def build_and_plot_heatmap(method: str, tair: np.ndarray, si: np.ndarray, components: np.ndarray,
                            ref_min: np.ndarray, ref_max: np.ndarray, base_n_tair: int, base_n_si: int,
                            out_dir: Path, figs_dir: Path, index_num: int) -> None:
    tair_min, tair_max = tair.min(), tair.max()
    si_min, si_max = si.min(), si.max()

    stats_rows = []
    agg_rows = []
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, len(RESOLUTION_MULTIPLIERS), figsize=(5.5 * len(RESOLUTION_MULTIPLIERS), 9),
                                  constrained_layout=True)
        for col, (res_name, mult) in enumerate(RESOLUTION_MULTIPLIERS.items()):
            n_tair = max(FD_BIN_CAP[0], round(base_n_tair * mult))
            n_si = max(FD_BIN_CAP[0], round(base_n_si * mult))
            tair_edges = make_edges(tair, n_tair)
            si_edges = make_edges(si, n_si)
            agg = aggregate_bins(tair, si, components, tair_edges, si_edges)
            min_n = bin_threshold(agg["count"].to_numpy())
            total_bins = n_tair * n_si
            nonempty = len(agg)
            retained = int((agg["count"] >= min_n).sum())

            agg = agg.copy()
            agg["method"] = method
            agg["resolution"] = res_name
            agg_rows.append(agg)
            stats_rows.append({
                "method": method, "resolution": res_name, "n_bins_tair": n_tair, "n_bins_si": n_si,
                "total_bins": total_bins, "nonempty_bins": nonempty, "retained_bins": retained,
                "pct_retained_of_nonempty": round(100 * retained / nonempty, 1) if nonempty else 0.0,
                "pct_retained_of_total": round(100 * retained / total_bins, 1),
                "min_bin_n": min_n, "median_nonzero_count": float(np.median(agg["count"])) if nonempty else 0.0,
            })

            for row, use_hue in enumerate([False, True]):
                grid = grid_to_rgba(agg, n_tair, n_si, ref_min, ref_max, min_n, use_hue_cmap=use_hue)
                ax = axes[row, col]
                ax.set_facecolor("#d9d9d9")
                ax.imshow(grid, origin="lower", extent=[tair_min, tair_max, si_min, si_max],
                          aspect="auto", interpolation="nearest")
                ax.axhline(0, color="k", linestyle="--", linewidth=0.6)
                ax.set_xlabel("Tair [°C]")
                if col == 0:
                    ax.set_ylabel("true RGB mean" if not use_hue else f"Hue -> {HEATMAP_CMAP}",
                                  fontsize=9, fontweight="bold")
                if row == 0:
                    ax.set_title(f"{res_name} ({n_tair}x{n_si} bins, min n={min_n},\n"
                                 f"{retained}/{nonempty} occupied bins retained)", fontsize=8)
        fig.suptitle(f"{method}: binned mean embedding-color over Si-Tair space "
                     f"(n={len(tair):,} images)", fontsize=11, fontweight="bold")
        out_p = figs_dir / f"0{index_num}_binned_heatmap_{method.lower()}_resolutions.png"
        fig.savefig(out_p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_p}")

    pd.DataFrame(stats_rows).to_csv(out_dir / f"bin_resolution_stats_{method.lower()}.csv", index=False)
    pd.concat(agg_rows, ignore_index=True).to_csv(out_dir / f"heatmap_bin_aggregates_{method.lower()}.csv", index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    scatter_n = args.scatter_n
    umap_extra_n = args.umap_extra_n
    if args.quick_test:
        scatter_n = min(scatter_n, 500)
        umap_extra_n = min(umap_extra_n, 2000)
        print(f"*** --quick-test: scatter_n={scatter_n}, umap_extra_n={umap_extra_n} ***")

    args.out.mkdir(parents=True, exist_ok=True)
    args.figs.mkdir(parents=True, exist_ok=True)
    run_start = time.time()
    timings: dict[str, float] = {}

    X, meta = load_data(args.l2, args.embeddings)
    n_full = len(meta)

    print("Standardizing embeddings (full population) ...")
    Xz_full = standardize(X)
    del X
    gc.collect()

    rng = np.random.default_rng(RNG_SEED)
    sample_idx = rng.choice(n_full, size=min(scatter_n, n_full), replace=False)
    remaining = np.setdiff1d(np.arange(n_full), sample_idx, assume_unique=False)
    extra_idx = rng.choice(remaining, size=min(umap_extra_n, len(remaining)), replace=False)

    print("\n=== Representativeness check ===")
    plot_representativeness_check(meta, sample_idx, args.out, args.figs)
    plot_colormap_legend(args.figs)

    print("\n=== PCA (fit on full population) ===")
    t0 = time.time()
    pca_scores_full, _pca_model = run_pca(Xz_full, args.out, args.figs)
    timings["pca"] = time.time() - t0
    pca_scores_scatter = pca_scores_full[sample_idx]

    print("\nExtracting UMAP/t-SNE input slices, then freeing the full standardized matrix ...")
    X_scatter = Xz_full[sample_idx]
    X_extra = Xz_full[extra_idx]
    del Xz_full
    gc.collect()
    print(f"  X_scatter {X_scatter.nbytes / 1e6:.0f}MB, X_extra {X_extra.nbytes / 1e6:.0f}MB")

    print("\n=== UMAP ===")
    t0 = time.time()
    umap_scores_scatter, umap_scores_extra = run_umap(X_scatter, X_extra)
    timings["umap"] = time.time() - t0

    print("\n=== t-SNE ===")
    t0 = time.time()
    tsne_scores_scatter = run_tsne(X_scatter)
    timings["tsne"] = time.time() - t0

    del X_scatter, X_extra
    gc.collect()

    meta_scatter = meta.iloc[sample_idx].reset_index(drop=True)
    tair_scatter = meta_scatter["Tair_C"].to_numpy()
    si_scatter = meta_scatter["Si"].to_numpy()

    reductions = {
        "PCA": {"scatter": pca_scores_scatter, "heatmap_components": pca_scores_full,
                "heatmap_meta": meta},
        "UMAP": {"scatter": umap_scores_scatter,
                 "heatmap_components": np.concatenate([umap_scores_scatter, umap_scores_extra], axis=0),
                 "heatmap_meta": pd.concat([meta_scatter, meta.iloc[extra_idx].reset_index(drop=True)],
                                            ignore_index=True)},
        "t-SNE": {"scatter": tsne_scores_scatter, "heatmap_components": None, "heatmap_meta": None},
    }

    print("\n=== Scatter: truecolor by method ===")
    rgb_scatter_by_method = {}
    coords_rows = []
    for method, d in reductions.items():
        comps = d["scatter"]
        ref_min, ref_max = comps.min(axis=0), comps.max(axis=0)
        rgb = components_to_rgb(comps, ref_min, ref_max)
        hue = rgb_to_hue(rgb)
        rgb_scatter_by_method[method] = rgb
        d["hue"] = hue
        d["ref_min"], d["ref_max"] = ref_min, ref_max
        coords_rows.append(pd.DataFrame({
            "method": method, "Campaign": meta_scatter["Campaign"], "Tair_C": tair_scatter, "Si": si_scatter,
            "C1": comps[:, 0], "C2": comps[:, 1], "C3": comps[:, 2],
            "R": rgb[:, 0], "G": rgb[:, 1], "B": rgb[:, 2], "Hue": hue,
        }))
    plot_truecolor_by_method(tair_scatter, si_scatter, rgb_scatter_by_method, args.figs)
    pd.concat(coords_rows, ignore_index=True).to_csv(args.out / "reduction_coords_subsample.csv", index=False)

    print("\n=== Scatter: Hue -> colormap comparison (5 variants per method) ===")
    for i, method in enumerate(reductions.keys(), start=5):
        plot_hue_colormap_comparison(tair_scatter, si_scatter, reductions[method]["hue"], method, args.figs, i)

    print("\n=== Binned / smoothed heatmap (PCA on full population, UMAP on fit+transform) ===")
    base_n_tair = fd_bin_count(meta["Tair_C"].to_numpy())
    base_n_si = fd_bin_count(meta["Si"].to_numpy())
    print(f"  Freedman-Diaconis base resolution: {base_n_tair} Tair bins x {base_n_si} Si bins "
          f"(coarse/medium/fine = 0.25x/0.5x/1x this)")

    for i, method in enumerate(["PCA", "UMAP"], start=8):
        d = reductions[method]
        if d["heatmap_components"] is None:
            continue
        heat_meta = d["heatmap_meta"]
        build_and_plot_heatmap(
            method, heat_meta["Tair_C"].to_numpy(), heat_meta["Si"].to_numpy(), d["heatmap_components"],
            d["ref_min"], d["ref_max"], base_n_tair, base_n_si, args.out, args.figs, i,
        )

    run_config = {
        "n_full_matched": n_full,
        "scatter_n": int(len(sample_idx)),
        "umap_heatmap_extra_n": int(len(extra_idx)),
        "tsne_pca_prereduce": TSNE_PCA_PREREDUCE,
        "tsne_perplexity": TSNE_PERPLEXITY,
        "colormaps": COLORMAPS,
        "heatmap_cmap": HEATMAP_CMAP,
        "min_bin_floor": MIN_BIN_FLOOR,
        "resolution_multipliers": RESOLUTION_MULTIPLIERS,
        "fd_base_bins_tair": base_n_tair,
        "fd_base_bins_si": base_n_si,
        "rng_seed": RNG_SEED,
        "timings_seconds": {k: round(v, 1) for k, v in timings.items()},
        "total_runtime_seconds": round(time.time() - run_start, 1),
        "sklearn_version": sklearn.__version__,
        "umap_learn_version": umap.__version__,
    }
    with open(args.out / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)
    print(f"\nSaved {args.out / 'run_config.json'}")
    print(f"Total runtime: {run_config['total_runtime_seconds'] / 60:.1f} min")

    update_latest(args.out.parent, args.out)
    update_latest(args.figs.parent, args.figs)
    print(f"Latest run: {args.out.parent / 'latest'} -> {args.out.name}")
    print(f"Latest figs: {args.figs.parent / 'latest'} -> {args.figs.name}")


if __name__ == "__main__":
    main()
