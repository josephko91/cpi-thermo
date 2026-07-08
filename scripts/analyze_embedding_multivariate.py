#!/usr/bin/env python3
"""
Environment -> Morphology: multivariate analysis
===================================================
Asks the REVERSE direction from scripts/analyze_data_tiers.py's L2
embedding analysis. That script asked "how well does the CPI image
embedding predict environment?" (384 numbers -> 7 scalars). This script
asks "how well does environment predict/explain CPI image morphology?"
(7 scalars -> a genuine 384-dim multivariate response), via increasingly
sophisticated methods: PCA -> PLS regression -> Canonical Correlation
Analysis -> Random Forest w/ feature importance -> UMAP -> SHAP.

R^2 / correlation values here are NOT comparable to analyze_data_tiers.py's
numbers -- direction matters. They're only comparable *within* this
script, across methods, on the same target.

**Embedding source: `cls_features` (384-dim) ONLY.** `head_features`
(64-dim) is never loaded or referenced anywhere in this script --
analyze_data_tiers.py already established cls_features as strictly more
informative on every env-prediction target, so there is no reason to
repeat this heavier pipeline against the weaker embedding.

Reuses (does not reimplement) from scripts/analyze_data_tiers.py:
  load_l2_embeddings, standardize, pca_from_standardized, ols_r2,
  CORE_COLS, CAMPAIGN_ORDER, COLORS, STYLE, RNG_SEED, EMBEDDINGS_PATH

Needs scikit-learn, umap-learn, shap (not required by any other script in
this repo -- see requirements.txt).

Memory note (16GB dev machine, two prior OOM kills this session): the
standardized full embedding matrix (Xz_full, ~1.8M x 384 float32,
~2.76GB) is the largest object in this script. It is computed once,
used to derive everything that's needed from it (PCA spectrum + scores,
a 500k-row subsample for PLS/CCA's full-Y variant, a 50k-row subsample
for UMAP), then deleted before any of the PLS/CCA-on-PCs, Random Forest,
UMAP, or SHAP steps run -- none of those need the full 384-dim matrix.

Outputs (logs/analyze_embedding_multivariate/<ts>/ and
figs/analyze_embedding_multivariate/<ts>/, with `latest` symlinks):
  pca_variance_spectrum.csv            - full 384-component eigenvalue spectrum
  pls_full_y_component_selection.csv   - PLS (Y=full 384dim, n=500k): R^2 per n_components
  pls_full_y_loadings.csv              - PLS 2a: env-var loadings on top components
  pls_topk_pca_component_selection.csv - PLS (Y=top-50 PCs, full n): R^2 per n_components
  cca_full_y_canonical_correlations.csv- CCA (Y=full 384dim, n=500k): canonical corr per component
  cca_full_y_loadings.csv              - CCA 3a: env-var loadings on top components
  cca_topk_pca_canonical_correlations.csv - CCA (Y=top-10 PCs, full n)
  rf_target_r2.csv                     - per PC1..PC10 target: RF R^2 vs OLS baseline R^2
  rf_feature_importance.csv            - per target x per env var
  env_to_morphology_r2_comparison.csv  - unifying OLS/PLS/CCA/RF table, PC1..PC10
  shap_summary_values.csv              - per target x per env var: mean |SHAP value|
  umap_embedding_coords.csv            - 50k UMAP 2D coords + Campaign + env vars
  run_config.json                      - K's, sample sizes, seeds, package versions

Usage:
    python scripts/analyze_embedding_multivariate.py
"""

from __future__ import annotations

import argparse
import gc
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
    load_l2_embeddings, standardize, pca_from_standardized, ols_r2,
    CORE_COLS, CAMPAIGN_ORDER, COLORS, STYLE, RNG_SEED, EMBEDDINGS_PATH,
)
from scripts.log_paths import timestamp as _run_timestamp, update_latest

import sklearn
import umap
import shap
from sklearn.cross_decomposition import PLSRegression, CCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

EMBEDDING_COL = "cls_features"  # the ONLY embedding source used anywhere in this script

K_TARGET = 10          # PCA components used as RF/SHAP/CCA-3b targets (same axes as the prior report's top-10)
K_PLS_CAP = 50          # cap for PLS-2b's PCA-target variant
PLS_CCA_SAMPLE_N = 500_000   # rows for the full-384-dim-Y variants (PLS-2a, CCA-3a)
RF_SAMPLE_N = 300_000
UMAP_SAMPLE_N = 50_000
SHAP_SAMPLE_N = 500     # shap.TreeExplainer on a 300-tree, depth-16 sklearn RF has no
                        # fast native path (unlike XGBoost) and ran single-threaded at
                        # ~6.4 min/target for 10 targets at n=5000 (a first timed run was
                        # cancelled after burning ~16 min on <3 of 10 targets); this scales
                        # ~linearly with sample count, so 500/target targets <10 min total
                        # while still giving a stable mean(|SHAP|) ranking to cross-check
                        # against feature_importances_.
SHAP_TOP_N_TARGETS = 3
TEST_FRAC = 0.2


def _parse_args() -> argparse.Namespace:
    ts = _run_timestamp()
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--l2", type=Path, default=ROOT / "data" / "out" / "combined_env_data_L2.parquet")
    p.add_argument("--embeddings", type=Path, default=EMBEDDINGS_PATH)
    p.add_argument("--out", type=Path, default=ROOT / "logs" / "analyze_embedding_multivariate" / ts)
    p.add_argument("--figs", type=Path, default=ROOT / "figs" / "analyze_embedding_multivariate" / ts)
    return p.parse_args()


def make_split(n: int, test_frac: float = TEST_FRAC, seed: int = RNG_SEED) -> tuple[np.ndarray, np.ndarray]:
    """Identical 3-line permutation/split idiom to ols_r2's internal split, exposed
    separately so PLS/CCA/RF can use the exact same train/test partition (given the
    same n and seed) as an ols_r2() call on the same rows -- makes cross-method R^2
    comparisons apples-to-apples rather than differently-shuffled."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(n * test_frac)
    return idx[n_test:], idx[:n_test]  # train_idx, test_idx


# ---------------------------------------------------------------------------
# Step 1 -- PCA
# ---------------------------------------------------------------------------

def run_pca(Xz_full: np.ndarray, out_dir: Path, figs_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full eigenvalue spectrum + top-K_PLS_CAP scores (K_TARGET is a slice of this).

    Returns (scores_topK [n, K_PLS_CAP], eigvals_full [D], eigvecs_full [D, D]).
    """
    print(f"  [PCA] Computing {Xz_full.shape[1]}x{Xz_full.shape[1]} covariance + eigendecomposition ...")
    cov = np.cov(Xz_full, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    del cov

    ratio = eigvals / eigvals.sum()
    cum_ratio = np.cumsum(ratio)
    spectrum_df = pd.DataFrame({
        "component": [f"PC{i + 1}" for i in range(len(ratio))],
        "explained_variance_ratio": ratio,
        "cumulative_variance_ratio": cum_ratio,
    })
    spectrum_path = out_dir / "pca_variance_spectrum.csv"
    spectrum_df.to_csv(spectrum_path, index=False)
    print(f"  Saved {spectrum_path}")
    top10_cum = cum_ratio[9]
    print(f"  [PCA] Top-10 cumulative variance: {top10_cum:.4f} "
          f"(cross-check vs analyze_data_tiers.py's 0.2702)")
    top50_cum = cum_ratio[min(49, len(cum_ratio) - 1)]
    print(f"  [PCA] Top-{K_PLS_CAP} cumulative variance: {top50_cum:.4f}")

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
        n_show = min(100, len(ratio))
        axes[0].bar(range(1, n_show + 1), ratio[:n_show], color="#4c72b0")
        axes[0].set_xlabel("component")
        axes[0].set_ylabel("explained variance ratio")
        axes[0].set_title(f"Scree (first {n_show} of {len(ratio)})", fontsize=10, fontweight="bold")
        axes[1].plot(range(1, len(cum_ratio) + 1), cum_ratio, color="#dd8452")
        axes[1].axvline(10, color="k", linestyle="--", linewidth=0.8, label="K=10")
        axes[1].axvline(K_PLS_CAP, color="gray", linestyle=":", linewidth=0.8, label=f"K={K_PLS_CAP}")
        axes[1].set_xlabel("component")
        axes[1].set_ylabel("cumulative explained variance")
        axes[1].set_title("Cumulative variance (full spectrum)", fontsize=10, fontweight="bold")
        axes[1].legend(fontsize=8)
        out_p = figs_dir / "pca_variance_scree.png"
        fig.savefig(out_p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_p}")

    scores_topK = Xz_full @ eigvecs[:, :K_PLS_CAP]
    return scores_topK.astype(np.float32), eigvals, eigvecs


# ---------------------------------------------------------------------------
# Step 2 -- PLS
# ---------------------------------------------------------------------------

def run_pls(
    X_500k: np.ndarray, Y_500k: np.ndarray,
    env_Xz_full: np.ndarray, scores_topK_full: np.ndarray,
    out_dir: Path, figs_dir: Path,
) -> pd.DataFrame:
    print("  [PLS-2a] Sweeping n_components on full-384dim Y, n=500k ...")
    train_idx, test_idx = make_split(len(Y_500k))
    sel_rows_a = []
    for k in range(2, 8):  # PLS n_components capped at min(n_samples, n_features=7, n_targets)
        pls = PLSRegression(n_components=k)
        pls.fit(X_500k[train_idx], Y_500k[train_idx])
        r2 = pls.score(X_500k[test_idx], Y_500k[test_idx])
        sel_rows_a.append({"n_components": k, "held_out_r2": round(float(r2), 4)})
    sel_df_a = pd.DataFrame(sel_rows_a)
    sel_df_a.to_csv(out_dir / "pls_full_y_component_selection.csv", index=False)
    best_k_a = int(sel_df_a.loc[sel_df_a["held_out_r2"].idxmax(), "n_components"])
    print(f"  [PLS-2a] Best n_components={best_k_a}, R^2={sel_df_a['held_out_r2'].max():.4f}")

    pls_final_a = PLSRegression(n_components=best_k_a).fit(X_500k[train_idx], Y_500k[train_idx])
    loadings_a = pd.DataFrame(
        pls_final_a.x_loadings_,
        index=CORE_COLS,
        columns=[f"PLS{i + 1}" for i in range(best_k_a)],
    )
    loadings_a.to_csv(out_dir / "pls_full_y_loadings.csv")

    print(f"  [PLS-2b] Sweeping n_components on top-{K_PLS_CAP}-PC Y, full n ...")
    train_idx_full, test_idx_full = make_split(len(scores_topK_full))
    sel_rows_b = []
    for k in range(2, 8):
        pls = PLSRegression(n_components=k)
        pls.fit(env_Xz_full[train_idx_full], scores_topK_full[train_idx_full])
        r2 = pls.score(env_Xz_full[test_idx_full], scores_topK_full[test_idx_full])
        sel_rows_b.append({"n_components": k, "held_out_r2": round(float(r2), 4)})
    sel_df_b = pd.DataFrame(sel_rows_b)
    sel_df_b.to_csv(out_dir / "pls_topk_pca_component_selection.csv", index=False)
    print(f"  [PLS-2b] Best R^2={sel_df_b['held_out_r2'].max():.4f}")

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
        ax.plot(sel_df_a["n_components"], sel_df_a["held_out_r2"], marker="o", label=f"2a: Y=full 384dim (n={len(Y_500k):,})")
        ax.plot(sel_df_b["n_components"], sel_df_b["held_out_r2"], marker="s", label=f"2b: Y=top-{K_PLS_CAP} PCs (n={len(scores_topK_full):,})")
        ax.set_xlabel("PLS n_components")
        ax.set_ylabel("held-out R^2")
        ax.set_title("PLS regression: env -> morphology", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        out_p = figs_dir / "pls_component_r2.png"
        fig.savefig(out_p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_p}")

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        im = ax.imshow(loadings_a.values, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(best_k_a))
        ax.set_xticklabels(loadings_a.columns, fontsize=8)
        ax.set_yticks(range(len(CORE_COLS)))
        ax.set_yticklabels(CORE_COLS, fontsize=8)
        ax.set_title("PLS-2a env-variable loadings", fontsize=10, fontweight="bold")
        fig.colorbar(im, ax=ax, shrink=0.8)
        out_p = figs_dir / "pls_env_loadings_heatmap.png"
        fig.savefig(out_p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_p}")

    return sel_df_a, sel_df_b


# ---------------------------------------------------------------------------
# Step 3 -- CCA
# ---------------------------------------------------------------------------

def run_cca(
    X_500k: np.ndarray, Y_500k: np.ndarray,
    env_Xz_full: np.ndarray, scores_10_full: np.ndarray,
    out_dir: Path, figs_dir: Path,
) -> None:
    print("  [CCA-3a] Fitting on full-384dim Y, n=500k ...")
    train_idx, test_idx = make_split(len(Y_500k))
    cca_a = CCA(n_components=7).fit(X_500k[train_idx], Y_500k[train_idx])
    Xc_test, Yc_test = cca_a.transform(X_500k[test_idx], Y_500k[test_idx])
    canon_corr_a = [float(np.corrcoef(Xc_test[:, i], Yc_test[:, i])[0, 1]) for i in range(7)]
    pd.DataFrame({"component": [f"CC{i + 1}" for i in range(7)],
                  "canonical_correlation": [round(c, 4) for c in canon_corr_a]}
                 ).to_csv(out_dir / "cca_full_y_canonical_correlations.csv", index=False)
    print(f"  [CCA-3a] Canonical correlations: {[round(c, 3) for c in canon_corr_a]}")

    loadings_a = pd.DataFrame(
        cca_a.x_loadings_, index=CORE_COLS, columns=[f"CC{i + 1}" for i in range(7)],
    )
    loadings_a.to_csv(out_dir / "cca_full_y_loadings.csv")

    print(f"  [CCA-3b] Fitting on top-{K_TARGET}-PC Y, full n ...")
    train_idx_full, test_idx_full = make_split(len(scores_10_full))
    cca_b = CCA(n_components=7).fit(env_Xz_full[train_idx_full], scores_10_full[train_idx_full])
    Xc_test_b, Yc_test_b = cca_b.transform(env_Xz_full[test_idx_full], scores_10_full[test_idx_full])
    canon_corr_b = [float(np.corrcoef(Xc_test_b[:, i], Yc_test_b[:, i])[0, 1]) for i in range(7)]
    pd.DataFrame({"component": [f"CC{i + 1}" for i in range(7)],
                  "canonical_correlation": [round(c, 4) for c in canon_corr_b]}
                 ).to_csv(out_dir / "cca_topk_pca_canonical_correlations.csv", index=False)
    print(f"  [CCA-3b] Canonical correlations: {[round(c, 3) for c in canon_corr_b]}")

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
        x = np.arange(1, 8)
        ax.bar(x - 0.2, canon_corr_a, width=0.4, label="3a: Y=full 384dim")
        ax.bar(x + 0.2, canon_corr_b, width=0.4, label=f"3b: Y=top-{K_TARGET} PCs")
        ax.set_xlabel("canonical component")
        ax.set_ylabel("canonical correlation")
        ax.set_title("CCA: env <-> morphology", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        out_p = figs_dir / "cca_canonical_correlations.png"
        fig.savefig(out_p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_p}")

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        im = ax.imshow(loadings_a.values, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(7))
        ax.set_xticklabels(loadings_a.columns, fontsize=8)
        ax.set_yticks(range(len(CORE_COLS)))
        ax.set_yticklabels(CORE_COLS, fontsize=8)
        ax.set_title("CCA-3a env-variable loadings", fontsize=10, fontweight="bold")
        fig.colorbar(im, ax=ax, shrink=0.8)
        out_p = figs_dir / "cca_env_loadings_heatmap.png"
        fig.savefig(out_p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_p}")


# ---------------------------------------------------------------------------
# Step 4 -- Random Forest
# ---------------------------------------------------------------------------

def run_random_forest(
    l2_with_emb: pd.DataFrame, env_Xz_full: np.ndarray, out_dir: Path, figs_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    print(f"  [RF] Sampling {RF_SAMPLE_N:,} rows ...")
    sample = l2_with_emb.sample(min(RF_SAMPLE_N, len(l2_with_emb)), random_state=RNG_SEED)
    sample_positions = sample.index.to_numpy()  # positional row indices into l2_with_emb / env_Xz_full
    X_rf = env_Xz_full[sample_positions]
    n_rf = len(sample)
    train_idx, test_idx = make_split(n_rf)

    r2_rows = []
    importance_rows = []
    fitted_models: dict[str, RandomForestRegressor] = {}
    for i in range(K_TARGET):
        target = f"PC{i + 1}"
        y = sample[target].to_numpy()
        print(f"  [RF] Fitting target {target} ({i + 1}/{K_TARGET}) ...")
        rf = RandomForestRegressor(n_estimators=300, max_depth=16, min_samples_leaf=5,
                                    n_jobs=8, random_state=RNG_SEED)
        rf.fit(X_rf[train_idx], y[train_idx])
        y_pred = rf.predict(X_rf[test_idx])
        rf_r2 = r2_score(y[test_idx], y_pred)
        ols_baseline_r2 = ols_r2(X_rf, y)  # same (n, seed) -> identical split internally
        r2_rows.append({"target": target, "rf_r2": round(float(rf_r2), 4),
                         "ols_baseline_r2": round(float(ols_baseline_r2), 4),
                         "n_train": len(train_idx), "n_test": len(test_idx)})
        for var, imp in zip(CORE_COLS, rf.feature_importances_):
            importance_rows.append({"target": target, "env_var": var, "importance": round(float(imp), 4)})
        fitted_models[target] = rf

    r2_df = pd.DataFrame(r2_rows).sort_values("rf_r2", ascending=False)
    r2_df.to_csv(out_dir / "rf_target_r2.csv", index=False)
    print("\n" + r2_df.to_string(index=False))

    importance_df = pd.DataFrame(importance_rows)
    importance_df.to_csv(out_dir / "rf_feature_importance.csv", index=False)

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        pivot = importance_df.pivot(index="target", columns="env_var", values="importance")
        pivot = pivot.loc[r2_df["target"]]
        pivot.plot(kind="bar", ax=ax, width=0.8)
        ax.set_ylabel("feature importance")
        ax.set_title("Random Forest feature importance per PCA target", fontsize=10, fontweight="bold")
        ax.legend(fontsize=7, ncol=4)
        ax.tick_params(axis="x", rotation=0)
        out_p = figs_dir / "rf_feature_importance_bars.png"
        fig.savefig(out_p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_p}")

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
        x = np.arange(len(r2_df))
        ax.bar(x - 0.2, r2_df["ols_baseline_r2"], width=0.4, label="OLS (linear) baseline")
        ax.bar(x + 0.2, r2_df["rf_r2"], width=0.4, label="Random Forest")
        ax.set_xticks(x)
        ax.set_xticklabels(r2_df["target"], rotation=0, fontsize=8)
        ax.set_ylabel("held-out R^2")
        ax.set_title("RF vs OLS baseline, same rows/split", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        out_p = figs_dir / "rf_vs_ols_r2_comparison.png"
        fig.savefig(out_p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_p}")

    extra = {"sample_positions": sample_positions, "X_rf": X_rf,
             "train_idx": train_idx, "test_idx": test_idx}
    return r2_df, importance_df, {"models": fitted_models, **extra}


# ---------------------------------------------------------------------------
# Step 5 -- UMAP
# ---------------------------------------------------------------------------

def run_umap(
    umap_embedding_sample: np.ndarray, umap_meta: pd.DataFrame, out_dir: Path, figs_dir: Path,
) -> None:
    print(f"  [UMAP] Fitting on {len(umap_embedding_sample):,} points "
          f"(random_state fixed -> single-threaded, expect a few minutes) ...")
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2,
                         metric="euclidean", random_state=RNG_SEED)
    coords = reducer.fit_transform(umap_embedding_sample)

    out_df = umap_meta.copy()
    out_df["UMAP1"] = coords[:, 0]
    out_df["UMAP2"] = coords[:, 1]
    out_df.to_csv(out_dir / "umap_embedding_coords.csv", index=False)
    print(f"  Saved {out_dir / 'umap_embedding_coords.csv'}")

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 6.5), constrained_layout=True)
        for camp in CAMPAIGN_ORDER:
            sub = out_df[out_df["Campaign"] == camp]
            if sub.empty:
                continue
            ax.scatter(sub["UMAP1"], sub["UMAP2"], s=3, alpha=0.35,
                       color=COLORS.get(camp, "gray"), label=camp)
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.set_title(f"CPI embedding UMAP (cls_features, n={len(out_df):,})\ncolored by campaign",
                     fontsize=10, fontweight="bold")
        ax.legend(fontsize=6, markerscale=3, loc="best", ncol=2)
        out_p = figs_dir / "umap_by_campaign.png"
        fig.savefig(out_p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_p}")

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)
        for ax, var in zip(axes, ["Tair_C", "Si", "Alt_m", "P_hPa"]):
            sub = out_df.dropna(subset=[var])
            sc = ax.scatter(sub["UMAP1"], sub["UMAP2"], c=sub[var], s=3, alpha=0.5, cmap="viridis")
            ax.set_xlabel("UMAP1")
            ax.set_ylabel("UMAP2")
            ax.set_title(var, fontsize=9, fontweight="bold")
            fig.colorbar(sc, ax=ax, shrink=0.8)
        fig.suptitle("CPI embedding UMAP colored by environmental variable", fontsize=11, fontweight="bold")
        out_p = figs_dir / "umap_by_env.png"
        fig.savefig(out_p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_p}")


# ---------------------------------------------------------------------------
# Step 6 -- SHAP
# ---------------------------------------------------------------------------

def run_shap(
    rf_info: dict, r2_df: pd.DataFrame, out_dir: Path, figs_dir: Path,
) -> pd.DataFrame:
    top_targets = r2_df.sort_values("rf_r2", ascending=False)["target"].head(SHAP_TOP_N_TARGETS).tolist()
    X_rf = rf_info["X_rf"]
    test_idx = rf_info["test_idx"]
    rng = np.random.default_rng(RNG_SEED)
    n_shap = min(SHAP_SAMPLE_N, len(test_idx))
    shap_idx = rng.choice(test_idx, size=n_shap, replace=False)
    X_shap = X_rf[shap_idx]

    all_rows = []
    for i, target in enumerate(r2_df["target"] if False else [f"PC{i+1}" for i in range(K_TARGET)]):
        model = rf_info["models"][target]
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_shap)
        mean_abs = np.abs(sv).mean(axis=0)
        for var, val in zip(CORE_COLS, mean_abs):
            all_rows.append({"target": target, "env_var": var, "mean_abs_shap": round(float(val), 5)})

        if target in top_targets:
            print(f"  [SHAP] Plotting summary for {target} ...")
            fig = plt.figure(figsize=(7, 4.5))
            shap.summary_plot(sv, X_shap, feature_names=CORE_COLS, show=False)
            out_p = figs_dir / f"shap_summary_{target}.png"
            fig.savefig(out_p, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved {out_p}")

    shap_df = pd.DataFrame(all_rows)
    shap_df.to_csv(out_dir / "shap_summary_values.csv", index=False)
    print(f"  Saved {out_dir / 'shap_summary_values.csv'}")
    return shap_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.figs.mkdir(parents=True, exist_ok=True)

    print(f"Loading L2 from {args.l2} ...")
    l2 = pd.read_parquet(args.l2)
    print(f"  L2: {len(l2):,} rows")

    print(f"Loading '{EMBEDDING_COL}' embeddings ...")
    l2_with_emb = load_l2_embeddings(l2, args.embeddings, EMBEDDING_COL)
    del l2
    gc.collect()
    n = len(l2_with_emb)
    print(f"  Matched {n:,} rows")

    print("Standardizing env variables (full population) ...")
    env_X_full = l2_with_emb[CORE_COLS].to_numpy().astype(np.float32)
    env_Xz_full = standardize(env_X_full)
    del env_X_full

    print("Standardizing embeddings (full population) ...")
    X = np.stack(l2_with_emb[EMBEDDING_COL].to_numpy()).astype(np.float32)
    n_dims = X.shape[1]
    Xz_full = standardize(X)
    del X
    gc.collect()
    l2_with_emb = l2_with_emb.drop(columns=[EMBEDDING_COL])

    print("\n=== Step 1: PCA ===")
    scores_topK_full, eigvals_full, eigvecs_full = run_pca(Xz_full, args.out, args.figs)
    scores_10_full = scores_topK_full[:, :K_TARGET]
    for i in range(K_TARGET):
        l2_with_emb[f"PC{i + 1}"] = scores_10_full[:, i]
    del eigvecs_full

    print("\nDrawing shared subsamples for PLS-2a/CCA-3a (500k) and UMAP (50k) "
          "while the full embedding matrix is still in memory ...")
    rng = np.random.default_rng(RNG_SEED)
    idx_500k = rng.choice(n, size=min(PLS_CCA_SAMPLE_N, n), replace=False)
    idx_50k = rng.choice(n, size=min(UMAP_SAMPLE_N, n), replace=False)
    Y_500k = Xz_full[idx_500k]
    X_500k = env_Xz_full[idx_500k]
    umap_embedding_sample = Xz_full[idx_50k]
    umap_meta = l2_with_emb.iloc[idx_50k][["Campaign"] + CORE_COLS].reset_index(drop=True)

    del Xz_full
    gc.collect()
    print(f"  Freed full embedding matrix. Remaining large objects: "
          f"Y_500k {Y_500k.nbytes / 1e9:.2f}GB, scores_topK_full {scores_topK_full.nbytes / 1e9:.2f}GB, "
          f"umap_embedding_sample {umap_embedding_sample.nbytes / 1e9:.2f}GB")

    print("\n=== Step 2: PLS regression ===")
    run_pls(X_500k, Y_500k, env_Xz_full, scores_topK_full, args.out, args.figs)

    print("\n=== Step 3: CCA ===")
    run_cca(X_500k, Y_500k, env_Xz_full, scores_10_full, args.out, args.figs)

    del Y_500k, X_500k, scores_topK_full
    gc.collect()

    print("\n=== Step 4: Random Forest ===")
    r2_df, importance_df, rf_info = run_random_forest(l2_with_emb, env_Xz_full, args.out, args.figs)

    print("\n=== Step 5: UMAP ===")
    run_umap(umap_embedding_sample, umap_meta, args.out, args.figs)
    del umap_embedding_sample
    gc.collect()

    print("\n=== Step 6: SHAP ===")
    shap_df = run_shap(rf_info, r2_df, args.out, args.figs)

    # --- unifying comparison table ---
    print("\nBuilding unifying env_to_morphology_r2_comparison.csv ...")
    comparison = r2_df[["target", "ols_baseline_r2", "rf_r2"]].copy()
    comparison.to_csv(args.out / "env_to_morphology_r2_comparison.csv", index=False)

    run_config = {
        "embedding_col": EMBEDDING_COL,
        "n_dims": n_dims,
        "n_l2_matched": n,
        "K_TARGET": K_TARGET,
        "K_PLS_CAP": K_PLS_CAP,
        "PLS_CCA_SAMPLE_N": PLS_CCA_SAMPLE_N,
        "RF_SAMPLE_N": RF_SAMPLE_N,
        "UMAP_SAMPLE_N": UMAP_SAMPLE_N,
        "SHAP_SAMPLE_N": SHAP_SAMPLE_N,
        "RNG_SEED": RNG_SEED,
        "sklearn_version": sklearn.__version__,
        "umap_learn_version": umap.__version__,
        "shap_version": shap.__version__,
    }
    with open(args.out / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)
    print(f"  Saved {args.out / 'run_config.json'}")

    update_latest(args.out.parent, args.out)
    update_latest(args.figs.parent, args.figs)
    print(f"\nLatest run: {args.out.parent / 'latest'} -> {args.out.name}")
    print(f"Latest figs: {args.figs.parent / 'latest'} -> {args.figs.name}")


if __name__ == "__main__":
    main()
