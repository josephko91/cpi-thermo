#!/usr/bin/env python3
"""
Compare Derived-Feature Versions
=================================
The COCPIT vgg16 pipeline (github.com/vprzybylo/cocpit) has re-derived each
campaign's per-particle geometric/morphological features (equiv_d,
circularity, solidity, phi, roundness, classification probabilities, ...)
multiple times as the pipeline evolved, writing a separate CSV per
(campaign, version) under
/Users/josephko/research/cocpit/final_databases/vgg16/<version>/<CAMPAIGN>.csv.
Column names, units, and value ranges have drifted release to release (e.g.
v3.1.0 adds explicit units to column names and reports classification
columns as percentages rather than fractions; v1.3.0 only has the older
16-feature schema, no image-quality/shape-descriptor columns at all).

This script is NOT part of the main cpi-thermo pipeline -- these CSVs are
an external, unjoined data source (not currently merged into
combined_env_data*.parquet). It exists purely to answer: for campaigns with
more than one derived-feature version available, how much do the marginal
distributions of shared geometric features actually differ version to
version? That's a data-quality/provenance question (did a pipeline change
shift the numbers?), independent of any env-data fusion.

For every campaign with 2+ versions, and every geometric feature present in
all of that campaign's versions (after name/unit normalization -- see
COLUMN_ALIASES / PERCENT_TO_FRACTION_COLS below), computes summary
statistics (n, mean, std, median, IQR) per version plus a pairwise
Kolmogorov-Smirnov test between each version and the campaign's earliest
("baseline") version, and plots overlaid histograms/violin plots per
feature.

Includes frame_width/frame_height (the pre-resize crop box, in native CPI
sensor pixels -- these carry the fixed 2.3 microns/px scale, unlike
equiv_d/perim/hull_area/etc which are computed post-resize on a fixed
1000x1000 canvas and have no fixed pixel->micron scale of their own) plus a
derived equiv_d_microns column (equiv_d rescaled per-particle via
frame_width/frame_height). See
docs/decisions/2026-07-24-derived-feature-version-equiv-d-shift.md for why
this conversion does NOT reconcile the equiv_d gap between v1.4.0 and
v3.1.0 (frame_width/frame_height are themselves already close between
versions) -- it only puts equiv_d in physical units within a version.

Outputs (logs/compare_derived_feature_versions/<timestamp>/ and
figs/compare_derived_feature_versions/<timestamp>/, with `latest` symlinks):
  version_summary_stats.csv     - per (campaign, version, feature): n/mean/std/median/q25/q75
  ks_tests_vs_baseline.csv      - per (campaign, version, feature): KS stat/p-value vs earliest version
  run_config.json               - versions found, paths, row counts, package versions
  <CAMPAIGN>_<feature>_hist.png       - overlaid histograms, one line per version
  <CAMPAIGN>_violin.png               - violin plots across versions, all features for that campaign

Usage:
    python scripts/compare_derived_feature_versions.py
    python scripts/compare_derived_feature_versions.py --campaigns ARM ISDAC
    python scripts/compare_derived_feature_versions.py --quick-test
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
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.log_paths import timestamp as _run_timestamp, update_latest

DERIVED_DB_ROOT = Path("/Users/josephko/research/cocpit/final_databases/vgg16")
VERSIONS = ["v1.2.0", "v1.3.0", "v1.4.0", "v3.1.0"]  # chronological; v1.5.0 excluded (empty dir)

STYLE = {
    "figure.facecolor": "white", "axes.facecolor": "#f8f8f8", "axes.grid": True,
    "grid.color": "white", "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9,
}
VERSION_COLORS = {
    "v1.2.0": "#1f77b4", "v1.3.0": "#ff7f0e", "v1.4.0": "#2ca02c", "v3.1.0": "#d62728",
}
RNG_SEED = 42
QUICK_TEST_N = 20_000  # rows subsampled per (campaign, version) under --quick-test

# Column-name aliases -> canonical name, covering the schema drift observed
# across v1.2.0/v1.3.0/v1.4.0/v3.1.0 (see docstring). v3.1.0 column names
# carry bracketed units/percent markers; those are stripped before matching.
COLUMN_ALIASES = {
    "compact irregular": "compact_irreg",
    "compact_irreg": "compact_irreg",
    "planar_polycrystal": "planar_polycrystal",
    "plate": "planar_polycrystal",  # v1.2.0/v1.3.0 called this class "plate"
}

# v3.1.0 reports these as percentages (0-100); all other versions report
# fractions (0-1). Normalized to fraction so distributions are comparable.
PERCENT_TO_FRACTION_COLS = {
    "cutoff", "agg", "budding", "bullet", "column", "compact_irreg",
    "fragment", "planar_polycrystal", "rimed", "sphere",
}

# Geometric/shape-descriptor features of primary interest for this
# comparison (classification-probability columns are compared too, but
# these are the ones that drive the headline "did the numbers shift" plots).
FOCUS_FEATURES = [
    "particle_width_microns", "particle_height_microns", "frame_width", "frame_height",
    "equiv_d", "equiv_d_microns", "circularity", "solidity",
    "complexity", "phi", "roundness", "perim_area_ratio",
    "filled_circular_area_ratio", "extreme_points",
]

# CPI probe native resolution, applied pre-resize in process_sheets.py's
# particle_dimensions(). frame_width/frame_height are the pre-resize crop
# box (native sensor pixels); equiv_d/perim/hull_area/etc are computed on
# the *post-resize* 1000x1000 canvas and have no fixed pixel->micron scale
# of their own -- see docs/decisions/2026-07-24-derived-feature-version-equiv-d-shift.md.
CPI_MICRONS_PER_NATIVE_PX = 2.3
RESIZED_CANVAS_PX = 1000


def _parse_args() -> argparse.Namespace:
    ts = _run_timestamp()
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--derived-db-root", type=Path, default=DERIVED_DB_ROOT)
    p.add_argument("--campaigns", nargs="+", default=None,
                    help="Restrict to these campaigns (COCPIT naming, e.g. CRYSTAL_FACE_NASA). "
                         "Default: all campaigns with 2+ versions available.")
    p.add_argument("--out", type=Path,
                    default=ROOT / "logs" / "compare_derived_feature_versions" / ts)
    p.add_argument("--figs", type=Path,
                    default=ROOT / "figs" / "compare_derived_feature_versions" / ts)
    p.add_argument("--quick-test", action="store_true",
                    help="Subsample rows per (campaign, version) for a fast smoke run")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Discovery + loading
# ---------------------------------------------------------------------------

def discover_campaign_versions(db_root: Path) -> dict[str, list[str]]:
    """campaign -> sorted list of versions (chronological) that have a CSV."""
    camp_versions: dict[str, list[str]] = {}
    for v in VERSIONS:
        vdir = db_root / v
        if not vdir.is_dir():
            continue
        for f in sorted(vdir.glob("*.csv")):
            camp_versions.setdefault(f.stem, []).append(v)
    return camp_versions


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip v3.1.0-style bracketed units/percent markers, map known aliases,
    and convert percent-valued classification columns to fractions.

    Some raw CSVs (observed in v3.1.0 AIRS_II) have a literally duplicated
    block of column headers, which pandas disambiguates on read with a
    ``.1`` suffix. Drop those duplicate columns (keep the first occurrence)
    before normalizing names, or the alias-collapse below would produce two
    columns sharing one canonical name."""
    df = df.loc[:, ~df.columns.str.match(r".*\.\d+$")]

    rename = {}
    for c in df.columns:
        base = c.split(" [")[0].strip()
        base = base.replace(" ", "_")
        base = COLUMN_ALIASES.get(base, base)
        rename[c] = base
    df = df.rename(columns=rename)

    for col in PERCENT_TO_FRACTION_COLS:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            if df[col].max(skipna=True) is not None and df[col].max(skipna=True) > 1.5:
                df[col] = df[col] / 100.0
    return df


def _add_equiv_d_microns(df: pd.DataFrame) -> pd.DataFrame:
    """Per-particle equiv_d rescaled from resized-canvas pixels to microns,
    using that row's own frame_width/frame_height (the true pre-resize crop
    box, native CPI sensor pixels) as the scale reference. See
    docs/decisions/2026-07-24-derived-feature-version-equiv-d-shift.md --
    this does NOT reconcile equiv_d across versions (frame_width/height are
    already close between versions), it only converts within-version pixel
    units to physical units.

    Also aliases particle_width/particle_height to explicit
    particle_width_microns/particle_height_microns column names -- those
    raw columns are ALREADY in microns in every version (computed pre-resize
    with the fixed 2.3 microns/px CPI constant; v3.1.0 just labels the unit
    in its header, `particle width [microns]`, while v1.2.0/v1.3.0/v1.4.0
    don't), this only makes that explicit so it isn't confused with the
    canvas-pixel columns (equiv_d, perim, ...)."""
    if {"equiv_d", "frame_width", "frame_height"} <= set(df.columns):
        scale_x = df["frame_width"] * CPI_MICRONS_PER_NATIVE_PX / RESIZED_CANVAS_PX
        scale_y = df["frame_height"] * CPI_MICRONS_PER_NATIVE_PX / RESIZED_CANVAS_PX
        df["equiv_d_microns"] = df["equiv_d"] * np.sqrt(scale_x * scale_y)
    if "particle_width" in df.columns:
        df["particle_width_microns"] = df["particle_width"]
    if "particle_height" in df.columns:
        df["particle_height_microns"] = df["particle_height"]
    return df


def load_campaign_version(db_root: Path, campaign: str, version: str,
                           quick_test: bool) -> pd.DataFrame:
    path = db_root / version / f"{campaign}.csv"
    df = pd.read_csv(path, low_memory=False)
    df = _normalize_columns(df)
    df = _add_equiv_d_microns(df)
    df["_version"] = version
    if quick_test and len(df) > QUICK_TEST_N:
        df = df.sample(n=QUICK_TEST_N, random_state=RNG_SEED)
    return df


def shared_features(dfs: dict[str, pd.DataFrame]) -> list[str]:
    """Numeric columns present (and numeric) in every version's frame,
    restricted to FOCUS_FEATURES plus any classification-probability
    columns common to all versions."""
    common_cols = None
    for df in dfs.values():
        numeric_cols = set(df.select_dtypes(include=[np.number]).columns) - {"_version"}
        common_cols = numeric_cols if common_cols is None else (common_cols & numeric_cols)
    common_cols = common_cols or set()
    ordered = [c for c in FOCUS_FEATURES if c in common_cols]
    extra = sorted(common_cols - set(ordered))
    return ordered + extra


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def summary_stats(dfs: dict[str, pd.DataFrame], features: list[str], campaign: str) -> pd.DataFrame:
    rows = []
    for version, df in dfs.items():
        for feat in features:
            vals = df[feat].dropna().to_numpy()
            if len(vals) == 0:
                continue
            rows.append({
                "Campaign": campaign, "version": version, "feature": feat,
                "n": len(vals), "mean": np.mean(vals), "std": np.std(vals),
                "median": np.median(vals),
                "q25": np.percentile(vals, 25), "q75": np.percentile(vals, 75),
                "min": np.min(vals), "max": np.max(vals),
            })
    return pd.DataFrame(rows)


def ks_tests_vs_baseline(dfs: dict[str, pd.DataFrame], features: list[str],
                          campaign: str) -> pd.DataFrame:
    versions = list(dfs.keys())
    baseline_version = versions[0]
    baseline = dfs[baseline_version]
    rows = []
    for version in versions[1:]:
        df = dfs[version]
        for feat in features:
            base_vals = baseline[feat].dropna().to_numpy()
            cmp_vals = df[feat].dropna().to_numpy()
            if len(base_vals) < 2 or len(cmp_vals) < 2:
                continue
            stat, pval = sps.ks_2samp(base_vals, cmp_vals)
            rows.append({
                "Campaign": campaign, "baseline_version": baseline_version,
                "compare_version": version, "feature": feat,
                "ks_stat": stat, "p_value": pval,
                "n_baseline": len(base_vals), "n_compare": len(cmp_vals),
                "mean_diff": np.mean(cmp_vals) - np.mean(base_vals),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_feature_histograms(dfs: dict[str, pd.DataFrame], features: list[str],
                             campaign: str, figs_dir: Path) -> None:
    n_feat = len(features)
    if n_feat == 0:
        return
    n_cols = 3
    n_rows = int(np.ceil(n_feat / n_cols))
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.2 * n_rows),
                                  constrained_layout=True)
        axes = np.atleast_1d(axes).ravel()
        for i, feat in enumerate(features):
            ax = axes[i]
            for version, df in dfs.items():
                vals = df[feat].dropna().to_numpy()
                if len(vals) == 0:
                    continue
                lo, hi = np.percentile(vals, [0.5, 99.5])
                bins = np.linspace(lo, hi, 40)
                ax.hist(vals, bins=bins, density=True, histtype="step",
                        linewidth=1.5, color=VERSION_COLORS.get(version, "#888888"),
                        label=f"{version} (n={len(vals):,})")
            ax.set_title(feat, fontsize=9)
            ax.tick_params(labelsize=7)
        for j in range(n_feat, len(axes)):
            axes[j].axis("off")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(dfs),
                   bbox_to_anchor=(0.5, -0.02), fontsize=8)
        fig.suptitle(f"{campaign}: derived-feature distributions by version", fontsize=11)
        out_p = figs_dir / f"{campaign}_feature_histograms.png"
        fig.savefig(out_p, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_p}")


def plot_violin(dfs: dict[str, pd.DataFrame], features: list[str],
                 campaign: str, figs_dir: Path) -> None:
    n_feat = len(features)
    if n_feat == 0:
        return
    n_cols = 3
    n_rows = int(np.ceil(n_feat / n_cols))
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.2 * n_rows),
                                  constrained_layout=True)
        axes = np.atleast_1d(axes).ravel()
        versions = list(dfs.keys())
        for i, feat in enumerate(features):
            ax = axes[i]
            data = [dfs[v][feat].dropna().to_numpy() for v in versions]
            data_present = [(v, d) for v, d in zip(versions, data) if len(d) > 0]
            if not data_present:
                ax.axis("off")
                continue
            vs, ds = zip(*data_present)
            parts = ax.violinplot(ds, showmedians=True)
            for pc, v in zip(parts["bodies"], vs):
                pc.set_facecolor(VERSION_COLORS.get(v, "#888888"))
                pc.set_alpha(0.6)
            ax.set_xticks(range(1, len(vs) + 1))
            ax.set_xticklabels(vs, rotation=30, fontsize=7)
            ax.set_title(feat, fontsize=9)
            ax.tick_params(labelsize=7)
        for j in range(n_feat, len(axes)):
            axes[j].axis("off")
        fig.suptitle(f"{campaign}: derived-feature distributions by version (violin)", fontsize=11)
        out_p = figs_dir / f"{campaign}_violin.png"
        fig.savefig(out_p, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_p}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

STABLE_THRESH = 0.1    # KS stat below this: no meaningful shift
MAJOR_THRESH = 0.4     # KS stat above this: major shift
TOP_N_FEATURES = 3     # worst-shifted features shown per campaign


def _verdict(max_ks: float) -> str:
    if pd.isna(max_ks) or max_ks < STABLE_THRESH:
        return "Stable"
    if max_ks < MAJOR_THRESH:
        return "Some drift"
    return "Major shift"


def write_summary_report(summary_df: pd.DataFrame, ks_df: pd.DataFrame,
                          camp_versions: dict[str, list[str]], out_dir: Path) -> None:
    lines = ["# Derived-feature version comparison\n"]
    lines.append(
        "Compares the per-particle geometric/shape-classification features across "
        "COCPIT vgg16 pipeline versions, per campaign. Distributions only (not a "
        "per-particle join -- row order/count differs run to run), scored by "
        "Kolmogorov-Smirnov distance (0 = identical distributions, 1 = fully "
        "disjoint) of each version against the campaign's earliest version.\n"
    )

    # Worst KS stat per campaign, for the overview table + sort order
    worst = (
        ks_df.groupby("Campaign")["ks_stat"].max()
        if not ks_df.empty else pd.Series(dtype=float)
    )
    overview_rows = []
    for camp, versions in camp_versions.items():
        max_ks = worst.get(camp, np.nan)
        overview_rows.append((camp, versions, max_ks, _verdict(max_ks)))
    overview_rows.sort(key=lambda r: (-1 if pd.isna(r[2]) else -r[2]))

    lines.append(f"**{len(camp_versions)} campaigns compared.**\n")
    lines.append("| Campaign | Versions | Worst KS | Verdict |")
    lines.append("|---|---|---|---|")
    for camp, versions, max_ks, verdict in overview_rows:
        ks_str = f"{max_ks:.2f}" if pd.notna(max_ks) else "n/a"
        lines.append(f"| {camp} | {' -> '.join(versions)} | {ks_str} | {verdict} |")
    lines.append("")

    lines.append(
        f"\nVerdict: **Stable** = worst KS < {STABLE_THRESH}, "
        f"**Some drift** = {STABLE_THRESH}-{MAJOR_THRESH}, "
        f"**Major shift** = KS > {MAJOR_THRESH} (population of reported values "
        "changed substantially between versions).\n"
    )

    lines.append("\n## Worst-shifted features per campaign\n")
    for camp, versions, max_ks, verdict in overview_rows:
        if verdict == "Stable":
            continue
        sub = ks_df[ks_df["Campaign"] == camp]
        top = (
            sub.sort_values("ks_stat", ascending=False)
            .drop_duplicates(subset=["feature"])
            .head(TOP_N_FEATURES)
        )
        lines.append(f"**{camp}** ({verdict}, baseline `{versions[0]}`):")
        for _, r in top.iterrows():
            direction = "up" if r["mean_diff"] > 0 else "down"
            lines.append(
                f"- `{r['feature']}` shifted {direction} in `{r['compare_version']}` "
                f"(KS={r['ks_stat']:.2f}, mean {r['mean_diff']:+.3g})"
            )
        lines.append("")

    report_path = out_dir / "summary_report.md"
    report_path.write_text("\n".join(lines))
    print(f"  Saved {report_path}")


# ---------------------------------------------------------------------------
# Size-in-microns coverage (particle_width/height are always microns;
# equiv_d_microns is derived per-particle -- see _add_equiv_d_microns above
# and docs/decisions/2026-07-24-derived-feature-version-equiv-d-shift.md)
# ---------------------------------------------------------------------------

SIZE_MICRONS_COLS = ["particle_width_microns", "particle_height_microns", "equiv_d_microns"]


def size_microns_coverage(dfs: dict[str, pd.DataFrame], campaign: str) -> pd.DataFrame:
    rows = []
    for version, df in dfs.items():
        n = len(df)
        row = {"Campaign": campaign, "version": version, "n_rows": n}
        for col in SIZE_MICRONS_COLS:
            pct = 100.0 * df[col].notna().sum() / n if (n and col in df.columns) else 0.0
            row[f"{col}_pct"] = round(pct, 1)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_coverage_heatmap(coverage_df: pd.DataFrame, figs_dir: Path) -> None:
    """One row per (campaign, version), one column per size-in-microns field,
    colored by %-available -- a single-glance view of where equiv_d_microns
    is and isn't derivable, across the whole archive."""
    pct_cols = [f"{c}_pct" for c in SIZE_MICRONS_COLS]
    labels = [f"{r['Campaign']} {r['version']}" for _, r in coverage_df.iterrows()]
    grid = coverage_df[pct_cols].to_numpy(dtype=float)

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6, 0.28 * len(labels) + 1.5), constrained_layout=True)
        im = ax.imshow(grid, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
        ax.set_xticks(range(len(SIZE_MICRONS_COLS)))
        ax.set_xticklabels(SIZE_MICRONS_COLS, rotation=20, ha="right", fontsize=8)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                ax.text(j, i, f"{grid[i, j]:.0f}", ha="center", va="center", fontsize=6)
        fig.colorbar(im, ax=ax, label="% rows available", shrink=0.6)
        ax.set_title("Size-in-microns coverage by campaign/version", fontsize=11)
        out_p = figs_dir / "00_coverage_heatmap_size_microns.png"
        fig.savefig(out_p, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_p}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.figs.mkdir(parents=True, exist_ok=True)

    print(f"Discovering campaign versions under {args.derived_db_root} ...")
    camp_versions = discover_campaign_versions(args.derived_db_root)
    camp_versions = {c: vs for c, vs in camp_versions.items() if len(vs) >= 2}
    if args.campaigns:
        camp_versions = {c: vs for c, vs in camp_versions.items() if c in args.campaigns}
    if not camp_versions:
        print("No campaigns with 2+ derived-feature versions found. Nothing to do.")
        return
    print(f"  {len(camp_versions)} campaigns with multiple versions: "
          f"{', '.join(sorted(camp_versions))}")

    all_summary = []
    all_ks = []
    all_coverage = []
    row_counts = {}

    for camp, versions in sorted(camp_versions.items()):
        print(f"\n=== {camp} ({' -> '.join(versions)}) ===")
        dfs = {}
        for v in versions:
            df = load_campaign_version(args.derived_db_root, camp, v, args.quick_test)
            dfs[v] = df
            print(f"  {v}: {len(df):,} rows, {df.shape[1]} cols")
        row_counts[camp] = {v: len(df) for v, df in dfs.items()}

        features = shared_features(dfs)
        print(f"  {len(features)} shared numeric features: {features}")

        all_coverage.append(size_microns_coverage(dfs, camp))

        if not features:
            continue

        summary_df = summary_stats(dfs, features, camp)
        ks_df = ks_tests_vs_baseline(dfs, features, camp)
        all_summary.append(summary_df)
        all_ks.append(ks_df)

        plot_feature_histograms(dfs, [f for f in features if f in FOCUS_FEATURES], camp, args.figs)
        plot_violin(dfs, [f for f in features if f in FOCUS_FEATURES], camp, args.figs)

    summary_df = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()
    ks_df = pd.concat(all_ks, ignore_index=True) if all_ks else pd.DataFrame()
    coverage_df = pd.concat(all_coverage, ignore_index=True) if all_coverage else pd.DataFrame()

    summary_path = args.out / "version_summary_stats.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved {summary_path}")

    ks_path = args.out / "ks_tests_vs_baseline.csv"
    ks_df.to_csv(ks_path, index=False)
    print(f"Saved {ks_path}")

    coverage_path = args.out / "size_microns_coverage.csv"
    coverage_df.to_csv(coverage_path, index=False)
    print(f"Saved {coverage_path}")
    if not coverage_df.empty:
        plot_coverage_heatmap(coverage_df, args.figs)

    write_summary_report(summary_df, ks_df, camp_versions, args.out)

    run_config = {
        "derived_db_root": str(args.derived_db_root),
        "versions_checked": VERSIONS,
        "campaigns_compared": {c: vs for c, vs in sorted(camp_versions.items())},
        "row_counts": row_counts,
        "quick_test": args.quick_test,
        "rng_seed": RNG_SEED,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": __import__("scipy").__version__,
    }
    config_path = args.out / "run_config.json"
    with open(config_path, "w") as f:
        json.dump(run_config, f, indent=2, default=str)
    print(f"Saved {config_path}")

    update_latest(args.out.parent, args.out)
    update_latest(args.figs.parent, args.figs)
    print(f"\nLatest run: {args.out.parent / 'latest'} -> {args.out.name}")
    print(f"Latest figs: {args.figs.parent / 'latest'} -> {args.figs.name}")


if __name__ == "__main__":
    main()
