#!/usr/bin/env python3
"""
L0/L1/L2 Descriptive Analysis
==============================
Descriptive statistics for every variable at each data tier, plus (for L2
only) an investigation of the relationship between CPI particle-image
embeddings and the environmental variables each image is joined to.

  L0 - data/out/combined_env_data.parquet
  L1 - data/out/combined_env_data_L1.parquet
  L2 - data/out/combined_env_data_L2.parquet

Embeddings source: data/raw/SSL-Model-v3/cpi3m_campaign_cls_head_features_
compressed.parquet -- a self-supervised model's feature vectors per CPI
image. Two are analyzed, run back-to-back on the same L2 cut:
  head_features (64-dim)  -- classification-head features, output filenames
                              have no tag (e.g. l2_embedding_pca_variance.csv)
  cls_features  (384-dim) -- backbone CLS-token features, output filenames
                              tagged "_cls_" (e.g. l2_embedding_cls_pca_variance.csv)
Both are joined to L2 on (raw CPI campaign name, filename) via the same
campaign-name mapping `parsers/cpi_timestamps.py` uses for the timestamp
archive.

PCA is implemented directly with numpy (eigendecomposition of the DxD
covariance matrix, D=64 or 384) since scipy/scikit-learn aren't installed
in this project's environment -- no new dependencies added.

Outputs (logs/analyze_data_tiers/<timestamp>/ and
figs/analyze_data_tiers/<timestamp>/, with `latest` symlinks). Below,
<emb> is empty for head_features or "cls_" for cls_features:
  <tier>_descriptive_overall.csv      - per-variable stats, whole tier
  <tier>_descriptive_by_campaign.csv  - per-campaign stats, core variables only
  <tier>_distributions.png            - histograms of core variables
  l2_embedding_<emb>pca_variance.csv       - PCA variance-explained per component (top N_PCA_COMPONENTS
                                              only -- for the scatter/correlation plots below, not the
                                              regression, which uses each embedding's full dimensionality)
  l2_embedding_<emb>env_correlations.csv   - correlation of top PCA components vs core env vars
  l2_embedding_<emb>regression_r2.csv      - held-out R^2 predicting each env var from the FULL
                                              standardized embedding (pooled; all dims, not top-N PCs --
                                              see N_PCA_COMPONENTS comment for why a fixed PC count
                                              wouldn't be a fair head_features-vs-cls_features comparison)
  l2_embedding_<emb>within_campaign_r2.csv - same, fit separately within each of the largest campaigns
                                              (isolates cross-campaign/regime confound from the pooled result)
  l2_embedding_<emb>pca_by_campaign.png    - PC1 vs PC2 scatter, colored by campaign
  l2_embedding_<emb>pca_by_env.png         - PC1 vs PC2 scatter, colored by Tair_C/Si/Alt_m/P_hPa
  l2_embedding_<emb>correlation_bars.png   - correlation magnitude bar chart

Usage:
    python scripts/analyze_data_tiers.py
    python scripts/analyze_data_tiers.py --skip-embeddings   # descriptive stats only
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from parsers.cpi_timestamps import CPI_TO_ENV_CAMPAIGN
from scripts.log_paths import timestamp as _run_timestamp, update_latest

ENV_TO_CPI_CAMPAIGN = {v: k for k, v in CPI_TO_ENV_CAMPAIGN.items()}

EMBEDDINGS_PATH = ROOT / "data" / "raw" / "SSL-Model-v3" / "cpi3m_campaign_cls_head_features_compressed.parquet"

CORE_COLS = ["Tair_C", "P_hPa", "Si", "qv", "Lat", "Lon", "Alt_m"]

CAMPAIGN_ORDER = [
    "ARM", "CRYSTAL-FACE-NASA", "CRYSTAL-FACE-UND", "MIDCIX", "MPACE",
    "AIRS-II", "ICE-L", "ISDAC", "MACPEX", "MC3E",
    "ATTREX", "IPHEX", "OLYMPEX", "POSIDON", "ESCAPE",
]

COLORS = {
    "ARM": "#1f77b4", "CRYSTAL-FACE-NASA": "#ff7f0e", "CRYSTAL-FACE-UND": "#2ca02c",
    "MIDCIX": "#d62728", "MPACE": "#9edae5", "AIRS-II": "#9467bd", "ICE-L": "#8c564b",
    "ISDAC": "#e377c2", "MACPEX": "#7f7f7f", "MC3E": "#bcbd22", "ATTREX": "#17becf",
    "IPHEX": "#aec7e8", "OLYMPEX": "#ffbb78", "POSIDON": "#98df8a", "ESCAPE": "#ff9896",
}

STYLE = {
    "figure.facecolor": "white", "axes.facecolor": "#f8f8f8", "axes.grid": True,
    "grid.color": "white", "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9,
}

N_PCA_COMPONENTS = 10   # for the variance table + PCA scatter/correlation plots only.
                        # The env-var regression uses ALL of each embedding's dimensions
                        # (64 for head_features, 384 for cls_features) via standardize(),
                        # not this truncated PC count -- comparing head_features (64-dim)
                        # against cls_features (384-dim) using the same fixed top-N PCs
                        # would use a much smaller *fraction* of cls_features' information
                        # (N/384 vs N/64), unfairly favoring head_features.
N_WITHIN_CAMPAIGN_CHECK = 4  # largest-N campaigns to re-check R^2 within (own PCA per campaign)
SCATTER_SAMPLE_N = 40_000  # points to draw in PCA scatter plots (full data used for stats)
RNG_SEED = 42


def _parse_args() -> argparse.Namespace:
    ts = _run_timestamp()
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--l0", type=Path, default=ROOT / "data" / "out" / "combined_env_data.parquet")
    p.add_argument("--l1", type=Path, default=ROOT / "data" / "out" / "combined_env_data_L1.parquet")
    p.add_argument("--l2", type=Path, default=ROOT / "data" / "out" / "combined_env_data_L2.parquet")
    p.add_argument("--embeddings", type=Path, default=EMBEDDINGS_PATH)
    p.add_argument("--skip-embeddings", action="store_true",
                    help="Skip the L2 embeddings analysis (descriptive stats only)")
    p.add_argument("--out", type=Path, default=ROOT / "logs" / "analyze_data_tiers" / ts)
    p.add_argument("--figs", type=Path, default=ROOT / "figs" / "analyze_data_tiers" / ts)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------

def descriptive_overall(df: pd.DataFrame) -> pd.DataFrame:
    """Count/mean/std/min/quartiles/max/%missing for every numeric column."""
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    rows = []
    n_total = len(df)
    for col in numeric_cols:
        s = df[col]
        n_valid = int(s.notna().sum())
        rows.append({
            "variable": col,
            "n_valid": n_valid,
            "pct_valid": round(n_valid / n_total * 100, 2) if n_total else 0.0,
            "mean": s.mean(),
            "std": s.std(),
            "min": s.min(),
            "p25": s.quantile(0.25),
            "median": s.median(),
            "p75": s.quantile(0.75),
            "max": s.max(),
        })
    return pd.DataFrame(rows)


def descriptive_by_campaign(df: pd.DataFrame) -> pd.DataFrame:
    """Core-variable count/mean/std/min/max/%valid, one row per (Campaign, variable)."""
    rows = []
    for camp, sub in df.groupby("Campaign"):
        n_total = len(sub)
        for col in CORE_COLS:
            if col not in sub.columns:
                continue
            s = sub[col]
            n_valid = int(s.notna().sum())
            rows.append({
                "Campaign": camp,
                "variable": col,
                "n_rows": n_total,
                "n_valid": n_valid,
                "pct_valid": round(n_valid / n_total * 100, 2) if n_total else 0.0,
                "mean": s.mean(),
                "std": s.std(),
                "min": s.min(),
                "max": s.max(),
            })
    return pd.DataFrame(rows)


def plot_distributions(df: pd.DataFrame, tier_name: str, figs_dir: Path) -> None:
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
        axes = axes.ravel()
        for ax, col in zip(axes, CORE_COLS):
            if col not in df.columns:
                ax.axis("off")
                continue
            vals = df[col].dropna()
            if vals.empty:
                ax.set_title(f"{col} (no data)", fontsize=9)
                ax.axis("off")
                continue
            ax.hist(vals, bins=60, color="#4c72b0", alpha=0.85)
            ax.set_title(col, fontsize=9, fontweight="bold")
            ax.set_ylabel("count")
            ax.tick_params(labelsize=7)
        for ax in axes[len(CORE_COLS):]:
            ax.axis("off")
        fig.suptitle(f"{tier_name} — core variable distributions (n={len(df):,})",
                     fontsize=11, fontweight="bold")
        out_p = figs_dir / f"{tier_name.lower()}_distributions.png"
        fig.savefig(out_p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_p}")


# ---------------------------------------------------------------------------
# Embeddings: load, join, PCA, correlation, regression
# ---------------------------------------------------------------------------

EMBEDDING_COLUMNS = ["head_features", "cls_features"]  # 64-dim and 384-dim, respectively


def load_l2_embeddings(l2: pd.DataFrame, embeddings_path: Path, embedding_col: str) -> pd.DataFrame:
    """Join L2's (Campaign, cpi_filename) against the embeddings archive's
    (campaign, filename), returning a frame with one row per L2 record plus
    a single embedding column (`embedding_col`, a numpy array per row).

    Loads only one embedding column at a time (not both head_features and
    cls_features together) -- with ~3.2M images and cls_features at 384-dim,
    holding both as Python list-objects simultaneously is too much for this
    project's 16GB development machine. Called twice (once per embedding
    column) with the intermediate frame freed in between; see main().

    Also avoids ever materializing all 3.2M embedding vectors at once:
    (campaign, filename) are read and matched first (cheap, no embeddings),
    then the embedding column is streamed in batches (`iter_batches`) and
    only the ~1.8M rows that matter for L2 are kept, one batch at a time --
    peak memory is bounded by BATCH_SIZE, not the full 3.2M-row archive.
    """
    import pyarrow.parquet as pq
    import pyarrow.compute as pc

    BATCH_SIZE = 200_000

    print(f"  Loading keys (campaign, filename) from {embeddings_path} ...")
    key_table = pq.read_table(embeddings_path, columns=["campaign", "filename"])
    print(f"  Embeddings archive: {key_table.num_rows:,} images across "
          f"{pc.count_distinct(key_table['campaign']).as_py()} campaigns")

    keys_df = key_table.to_pandas()
    keys_df["_arrow_row"] = np.arange(len(keys_df))
    del key_table

    l2_keys = l2[["Campaign", "cpi_filename"]].copy()
    l2_keys["campaign_cpi"] = l2_keys["Campaign"].map(ENV_TO_CPI_CAMPAIGN)

    merged = pd.merge(
        l2_keys.reset_index().rename(columns={"index": "_l2_row"}),
        keys_df,
        left_on=["campaign_cpi", "cpi_filename"],
        right_on=["campaign", "filename"],
        how="inner",
    )
    del keys_df
    print(f"  Matched {len(merged):,} of {len(l2):,} L2 rows to an embedding vector")

    # arrow_row -> l2_row, so vectors pulled out of arbitrary batches can be
    # placed back in the right position once streaming finishes. A dense
    # boolean mask over the full archive lets each batch just slice it
    # (O(1)) instead of re-running an isin() against a ~1.8M-element list
    # on every one of the ~16 batches.
    arrow_to_l2_row = dict(zip(merged["_arrow_row"].to_numpy(), merged["_l2_row"].to_numpy()))
    n_archive = pq.ParquetFile(embeddings_path).metadata.num_rows
    wanted_mask = np.zeros(n_archive, dtype=bool)
    wanted_mask[merged["_arrow_row"].to_numpy()] = True

    print(f"  Streaming '{embedding_col}' in batches of {BATCH_SIZE:,} "
          f"(only keeping the {int(wanted_mask.sum()):,} matched rows) ...")
    pf = pq.ParquetFile(embeddings_path)
    l2_row_to_vec: dict[int, np.ndarray] = {}
    global_row = 0
    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=[embedding_col]):
        n = batch.num_rows
        batch_mask = wanted_mask[global_row:global_row + n]
        if batch_mask.any():
            col = batch.column(0)
            hit_positions = np.nonzero(batch_mask)[0]
            values = col.take(hit_positions).to_pylist()
            for pos, vec in zip(hit_positions, values):
                arrow_row = global_row + int(pos)
                l2_row_to_vec[arrow_to_l2_row[arrow_row]] = np.asarray(vec, dtype=np.float32)
        global_row += n
    print(f"  Collected {len(l2_row_to_vec):,} embedding vectors")

    l2_with_emb = l2.loc[merged["_l2_row"]].reset_index(drop=True)
    l2_with_emb[embedding_col] = [l2_row_to_vec[r] for r in merged["_l2_row"].to_numpy()]
    return l2_with_emb


def standardize(X: np.ndarray) -> np.ndarray:
    """Z-score each column, in float32. Used to feed the *full* embedding
    (all D dims) into regression -- see the note on N_PCA_COMPONENTS above
    for why the regression doesn't truncate to a small PC count. float32
    (not float64) roughly halves memory for the largest arrays in this
    pipeline (up to ~1.8M rows x 384 dims for cls_features)."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    return ((X - mean) / std).astype(np.float32)


def pca_from_standardized(Xz: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    """PCA via eigendecomposition of the (small, DxD) covariance matrix of an
    *already-standardized* Xz -- avoids needing scipy/sklearn, and avoids
    re-standardizing (and re-allocating a second big array) on top of the
    standardize() call already done for the regression step.

    Returns (scores [n, n_components], explained_variance_ratio [n_components]).
    """
    cov = np.cov(Xz, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending order
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    explained_ratio = eigvals[:n_components] / eigvals.sum()
    scores = Xz @ eigvecs[:, :n_components]
    return scores, explained_ratio


def ols_r2(X: np.ndarray, y: np.ndarray, test_frac: float = 0.2, seed: int = RNG_SEED) -> float:
    """Multivariate OLS (with intercept) on a random train/test split;
    returns held-out R^2.

    Solved via the normal equations (X^T X) beta = X^T y -- a DxD solve
    (D <= 384 here) -- rather than np.linalg.lstsq's SVD, which needs
    O(n*D) working memory. With n up to ~1.8M rows this made lstsq's
    peak footprint several GB (on top of X itself); the normal-equations
    form never materializes anything larger than X. X's columns are
    already standardized (mean ~0), so no separate intercept column/hstack
    is needed -- y is centered instead and the mean added back at predict
    time. A small ridge term guards against collinearity among embedding
    dimensions.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.permutation(n)
    n_test = int(n * test_frac)
    test_idx, train_idx = idx[:n_test], idx[n_test:]

    # Keep the (n, D) arrays in float32 throughout -- only the DxD/(D,)
    # *outputs* of the matmuls below are upcast to float64, which is cheap
    # (D <= 384) regardless of n. Upcasting X itself would undo the memory
    # savings from standardize()'s float32 output.
    X_train, y_train = X[train_idx], y[train_idx].astype(np.float32)
    y_mean = float(y_train.mean())
    y_train_c = y_train - y_mean

    XtX = (X_train.T @ X_train).astype(np.float64)
    Xty = (X_train.T @ y_train_c).astype(np.float64)
    XtX += np.eye(XtX.shape[0]) * 1e-6
    coef = np.linalg.solve(XtX, Xty)

    y_pred = (X[test_idx] @ coef.astype(np.float32)).astype(np.float64) + y_mean
    y_test = y[test_idx].astype(np.float64)
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - y_test.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def analyze_embeddings(
    l2_with_emb: pd.DataFrame,
    out_dir: Path,
    figs_dir: Path,
    embedding_col: str = "head_features",
    tag: str = "",
) -> None:
    """Run the full PCA -> correlation -> pooled regression -> within-campaign
    regression -> figures pipeline against a single embedding column.

    `tag` namespaces output filenames/titles so multiple embedding sources
    (e.g. head_features vs cls_features) can be run in the same output
    directory without overwriting each other. `tag=""` keeps the original
    (head_features) filenames for backward compatibility with existing
    reports; pass e.g. `tag="cls"` for a second embedding source.
    """
    prefix = f"l2_embedding_{tag}_" if tag else "l2_embedding_"
    label = f"{embedding_col} ({tag})" if tag else embedding_col

    print(f"  [{label}] Standardizing + running PCA on embeddings ...")
    X = np.stack(l2_with_emb[embedding_col].to_numpy()).astype(np.float32)
    n_dims = X.shape[1]
    Xz_full = standardize(X)
    del X
    gc.collect()
    scores, explained_ratio = pca_from_standardized(Xz_full, N_PCA_COMPONENTS)
    for i in range(N_PCA_COMPONENTS):
        l2_with_emb[f"PC{i + 1}"] = scores[:, i]

    var_df = pd.DataFrame({
        "component": [f"PC{i + 1}" for i in range(N_PCA_COMPONENTS)],
        "explained_variance_ratio": explained_ratio,
        "cumulative_variance_ratio": np.cumsum(explained_ratio),
    })
    var_path = out_dir / f"{prefix}pca_variance.csv"
    var_df.to_csv(var_path, index=False)
    print(f"  Saved {var_path}")
    print(f"    [{label}] Top {N_PCA_COMPONENTS} of {n_dims} PCs explain "
          f"{explained_ratio.sum() * 100:.1f}% of embedding variance")

    # --- correlation: each PC vs each core env variable ---
    print(f"  [{label}] Computing PC <-> env variable correlations ...")
    corr_rows = []
    pc_cols = [f"PC{i + 1}" for i in range(N_PCA_COMPONENTS)]
    for var in CORE_COLS:
        if var not in l2_with_emb.columns:
            continue
        valid = l2_with_emb[var].notna()
        for pc in pc_cols:
            r = np.corrcoef(l2_with_emb.loc[valid, pc], l2_with_emb.loc[valid, var])[0, 1]
            corr_rows.append({"variable": var, "component": pc, "pearson_r": round(r, 4)})
    corr_df = pd.DataFrame(corr_rows)
    corr_path = out_dir / f"{prefix}env_correlations.csv"
    corr_df.to_csv(corr_path, index=False)
    print(f"  Saved {corr_path}")

    # --- held-out R^2: predict each env var from the FULL standardized
    # embedding (all n_dims dims, not the top-N PCs above) -- using a fixed
    # top-N PC count would use a much smaller *fraction* of cls_features'
    # 384 dims than of head_features' 64, unfairly favoring head_features.
    # Full-rank PCA is just a rotation, so this is equivalent to (and
    # simpler than) regressing on all N_PCA_COMPONENTS=n_dims PCA scores.
    print(f"  [{label}] Fitting held-out regression (full {n_dims}-dim embedding -> env variable) ...")
    r2_rows = []
    for var in CORE_COLS:
        if var not in l2_with_emb.columns:
            continue
        valid = l2_with_emb[var].notna().to_numpy()
        if valid.sum() < 100:
            continue
        r2 = ols_r2(Xz_full[valid], l2_with_emb.loc[valid, var].to_numpy())
        r2_rows.append({"variable": var, "n_components_used": n_dims,
                         "n": int(valid.sum()), "held_out_r2": round(r2, 4)})
    r2_df = pd.DataFrame(r2_rows).sort_values("held_out_r2", ascending=False)
    r2_path = out_dir / f"{prefix}regression_r2.csv"
    r2_df.to_csv(r2_path, index=False)
    print(f"  Saved {r2_path}")
    print("\n" + r2_df.to_string(index=False))

    # --- within-campaign R^2: does the pooled R^2 above just reflect between-
    # campaign/regime separation (e.g. ATTREX's stratospheric flights sitting in
    # a distinct part of embedding space), or does the relationship hold within
    # a single, more homogeneous campaign? Re-standardize per campaign (own
    # mean/std, not the pooled one, so cross-campaign structure can't leak
    # in) and regress on the full embedding, same as the pooled case above.
    print(f"  [{label}] Fitting within-campaign regression (isolates cross-campaign confound) ...")
    within_rows = []
    top_campaigns = l2_with_emb["Campaign"].value_counts().head(N_WITHIN_CAMPAIGN_CHECK).index
    for camp in top_campaigns:
        sub = l2_with_emb[l2_with_emb["Campaign"] == camp]
        Xc = np.stack(sub[embedding_col].to_numpy()).astype(np.float32)
        Xc_full = standardize(Xc)
        del Xc
        for var in CORE_COLS:
            if var not in sub.columns:
                continue
            valid = sub[var].notna().to_numpy()
            if valid.sum() < 200:
                continue
            r2 = ols_r2(Xc_full[valid], sub.loc[valid, var].to_numpy())
            within_rows.append({"Campaign": camp, "variable": var, "n": int(valid.sum()),
                                 "within_campaign_r2": round(r2, 4)})
    within_df = pd.DataFrame(within_rows)
    within_path = out_dir / f"{prefix}within_campaign_r2.csv"
    within_df.to_csv(within_path, index=False)
    print(f"  Saved {within_path}")
    print("\n" + within_df.pivot(index="variable", columns="Campaign", values="within_campaign_r2").to_string())

    # --- figures ---
    sample = l2_with_emb.sample(min(SCATTER_SAMPLE_N, len(l2_with_emb)), random_state=RNG_SEED)

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 6.5), constrained_layout=True)
        for camp in CAMPAIGN_ORDER:
            sub = sample[sample["Campaign"] == camp]
            if sub.empty:
                continue
            ax.scatter(sub["PC1"], sub["PC2"], s=3, alpha=0.35,
                       color=COLORS.get(camp, "gray"), label=camp)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"CPI embedding PCA ({label}, n={len(sample):,} sampled)\ncolored by campaign",
                     fontsize=10, fontweight="bold")
        ax.legend(fontsize=6, markerscale=3, loc="best", ncol=2)
        out_p = figs_dir / f"{prefix}pca_by_campaign.png"
        fig.savefig(out_p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_p}")

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)
        for ax, var in zip(axes, ["Tair_C", "Si", "Alt_m", "P_hPa"]):
            sub = sample.dropna(subset=[var])
            sc = ax.scatter(sub["PC1"], sub["PC2"], c=sub[var], s=3, alpha=0.5, cmap="viridis")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title(var, fontsize=9, fontweight="bold")
            fig.colorbar(sc, ax=ax, shrink=0.8)
        fig.suptitle(f"CPI embedding PCA ({label}) colored by environmental variable",
                     fontsize=11, fontweight="bold")
        out_p = figs_dir / f"{prefix}pca_by_env.png"
        fig.savefig(out_p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_p}")

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        pivot = corr_df.pivot(index="component", columns="variable", values="pearson_r").loc[pc_cols]
        pivot.plot(kind="bar", ax=ax, width=0.8)
        ax.set_ylabel("Pearson r")
        ax.set_title(f"PC <-> environmental variable correlation ({label})", fontsize=10, fontweight="bold")
        ax.axhline(0, color="k", linewidth=0.8)
        ax.legend(fontsize=7, ncol=4, loc="upper right")
        ax.tick_params(axis="x", rotation=0)
        out_p = figs_dir / f"{prefix}correlation_bars.png"
        fig.savefig(out_p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_p}")

    # Drop this embedding's PC columns before returning control, so the next
    # call (different embedding_col) doesn't inherit stale PC1..PC10 values.
    l2_with_emb.drop(columns=pc_cols, inplace=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.figs.mkdir(parents=True, exist_ok=True)

    tiers = {}
    for name, path in [("L0", args.l0), ("L1", args.l1), ("L2", args.l2)]:
        print(f"Loading {name} from {path} ...")
        tiers[name] = pd.read_parquet(path)
        print(f"  {name}: {len(tiers[name]):,} rows")

    for name, df in tiers.items():
        print(f"\n=== {name} descriptive statistics ===")
        overall = descriptive_overall(df)
        overall_path = args.out / f"{name.lower()}_descriptive_overall.csv"
        overall.to_csv(overall_path, index=False)
        print(f"  Saved {overall_path}")

        by_camp = descriptive_by_campaign(df)
        by_camp_path = args.out / f"{name.lower()}_descriptive_by_campaign.csv"
        by_camp.to_csv(by_camp_path, index=False)
        print(f"  Saved {by_camp_path}")

        plot_distributions(df, name, args.figs)

    if not args.skip_embeddings:
        l2 = tiers["L2"]
        tiers = None
        gc.collect()  # L0/L1 no longer needed; free them before the memory-heavy embeddings section

        print("\n=== L2: CPI embeddings vs environmental variables ===")

        print("\n--- head_features (64-dim) ---")
        l2_with_emb = load_l2_embeddings(l2, args.embeddings, "head_features")
        analyze_embeddings(l2_with_emb, args.out, args.figs,
                            embedding_col="head_features", tag="")
        del l2_with_emb
        gc.collect()

        print("\n--- cls_features (384-dim) ---")
        l2_with_emb = load_l2_embeddings(l2, args.embeddings, "cls_features")
        analyze_embeddings(l2_with_emb, args.out, args.figs,
                            embedding_col="cls_features", tag="cls")
        del l2_with_emb
        gc.collect()
    else:
        print("\nSkipping embeddings analysis (--skip-embeddings)")

    update_latest(args.out.parent, args.out)
    update_latest(args.figs.parent, args.figs)
    print(f"\nLatest run: {args.out.parent / 'latest'} -> {args.out.name}")
    print(f"Latest figs: {args.figs.parent / 'latest'} -> {args.figs.name}")


if __name__ == "__main__":
    main()
