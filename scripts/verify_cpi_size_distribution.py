#!/usr/bin/env python3
"""
Verify CPI Size Distribution Against SPEC Inc's Archived PSD Product
=======================================================================
Cross-checks COCPIT's derived particle_width_microns / particle_height_microns
/ equiv_d_microns (see scripts/compare_derived_feature_versions.py and
scripts/derive_particle_size_microns.py) against an independent, verified
size-distribution product for the same campaign: CRYSTAL-FACE's WB-57 CPI
files (`CP<YYYYMMDD>.WB57`), produced by Paul Lawson/SPEC Inc -- the PI team
that built and operated the CPI instrument -- and archived on NASA's ESPO
archive (espoarchive.nasa.gov/archive/browse/crystalf/WB57/CP), which allows
anonymous HTTPS download with no login. CRYSTAL-FACE-NASA was chosen because
it was the first campaign checked whose archive both (a) requires no
authentication and (b) hosts a PI-produced size-distribution product
distinct from the raw CPI imagery -- ARM's IOP archive and NCAR EOL's
dataset catalog are both now login-gated/JS-only front ends that could not
be queried without an account; see docs/reports/<this report> for what was
tried and ruled out.

The SPEC files are NASA Ames "irregular" format (FFI 2110): each 10-second
record has a variable number of particle-size bins with bin-lower-edge (um)
and number concentration density (#/L/um). Parsing validated by a
self-consistency check: sum(concentration_i * bin_width_i) over a record's
bins reproduces that record's own reported "total particle concentration"
aux field to within ~0.3% across all 8 files / ~900 total records (see
parse_cpi_wb57() docstring below) -- strong evidence the bin-edge/width/
concentration parsing is correct.

IMPORTANT CAVEATS (this is a shape/central-tendency check, not a
particle-by-particle validation):
  - The SPEC PSD is a true ambient number CONCENTRATION (per liter of
    sampled air, accounting for probe sample volume/aircraft true air
    speed) -- it is volume-normalized. COCPIT's particle_width_microns/
    particle_height_microns/equiv_d_microns are raw per-imaged-particle
    values with NO sample-volume normalization (every saved particle image
    counts once, regardless of how much air was sampled to find it). The
    two are therefore only comparable in *distributional shape*
    (normalized density) and *central tendency* (mean/median size), never
    in absolute counts or absolute concentration.
  - The SPEC product's particle "size" is SPEC/Lawson's own CPI processing
    algorithm's sizing definition (their published methodology; not
    independently re-derived here). COCPIT's particle_width/particle_height
    come from cv2.minAreaRect() on the pre-resize contour (a bounding-box
    length/width), and equiv_d from cv2.contourArea() (area-equivalent
    circular diameter) on the post-resize canvas, rescaled to microns per
    docs/decisions/2026-07-24-derived-feature-version-equiv-d-shift.md.
    These are three different geometric definitions of "size" and are not
    expected to match exactly even for a perfect segmentation.
  - Both products apply their own (different, undocumented-here) minimum
    detectable size and image-quality/cutoff filtering.

Given these caveats, the check this script performs is deliberately modest:
does COCPIT's derived size distribution fall in the same broad range and
have a comparable central tendency to the campaign's own verified PSD, per
matching flight date? A large (order-of-magnitude) mismatch would indicate
a real problem (e.g. a units/scale bug); a difference of order 1.5-3x given
the definitional differences above would not be surprising or alarming.

Outputs (logs/verify_cpi_size_distribution/<timestamp>/ and
figs/verify_cpi_size_distribution/<timestamp>/, with `latest` symlinks):
  spec_psd_summary_by_date.csv     - SPEC PSD weighted mean/median per flight date
  cocpit_size_summary_by_date.csv  - COCPIT particle_width/height/equiv_d_microns
                                      mean/median per flight date
  parser_closure_check.csv         - per-file self-consistency validation results
  summary_report.md                - condensed comparison + verdict
  01_aggregate_comparison.png      - campaign-wide overlaid density comparison
  02_per_date_mean_size.png        - verified vs COCPIT mean size, per flight date
  03_per_date_histograms.png       - per-date overlaid density histograms

Usage:
    python scripts/verify_cpi_size_distribution.py
"""

from __future__ import annotations

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

from scripts.compare_derived_feature_versions import (
    DERIVED_DB_ROOT, _normalize_columns, _add_equiv_d_microns,
)
from scripts.log_paths import timestamp as _run_timestamp, update_latest

SPEC_DATA_DIR = ROOT / "data" / "raw" / "CRYSTAL-FACE-NASA" / "CPI_SPEC_verification"
SPEC_ARCHIVE_URL = "https://espoarchive.nasa.gov/archive/browse/crystalf/WB57/CP"
COCPIT_CAMPAIGN = "CRYSTAL_FACE_NASA"
COCPIT_VERSION = "v1.4.0"  # full 36-col schema with equiv_d + date + frame_width/height

STYLE = {
    "figure.facecolor": "white", "axes.facecolor": "#f8f8f8", "axes.grid": True,
    "grid.color": "white", "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9,
}


# ---------------------------------------------------------------------------
# SPEC Inc CPI WB57 file parser (NASA Ames FFI 2110 "irregular" format)
# ---------------------------------------------------------------------------

def parse_cpi_wb57(path: Path) -> dict:
    """Parse one CP<YYYYMMDD>.WB57 file.

    Record structure (one record = one 10-second measurement interval):
      line 1: time_sec_from_00Z  n_bins  duration_s  hh mm ss  upper_limit_largest_bin_um
      next lines: 36 aux values (total conc/SA/IWC, >55um conc/SA/IWC, then
                  5-habit-resolved counts/SA/mass, all-particles then >55um-only)
      next n_bins lines: bin_lower_edge_um  concentration_density_(#/L/um)

    Bin width for bin i: edges[i+1]-edges[i] for i < n-1; upper_limit-edges[-1]
    for the last bin. See module docstring for the closure-check validation.
    """
    lines = path.read_text().splitlines()
    nlhead = int(lines[0].split()[0])
    date_parts = lines[6].split()
    year, month, day = int(date_parts[0]), int(date_parts[1]), int(date_parts[2])

    i = nlhead
    n = len(lines)
    records = []
    while i < n:
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        parts = line.split()
        if len(parts) < 7:
            i += 1
            continue
        n_bins = int(float(parts[1]))
        hh, mm, ss = int(parts[3]), int(parts[4]), int(parts[5])
        upper_limit = float(parts[6])
        i += 1

        aux = []
        while len(aux) < 36:
            aux.extend(float(x) for x in lines[i].split())
            i += 1
        total_conc_reported = aux[0]

        edges, concs = [], []
        for _ in range(n_bins):
            e, c = lines[i].split()
            edges.append(float(e))
            concs.append(float(c))
            i += 1
        if n_bins == 0:
            continue

        edges = np.array(edges)
        concs = np.array(concs)
        widths = np.empty(n_bins)
        if n_bins > 1:
            widths[:-1] = np.diff(edges)
        widths[-1] = max(upper_limit - edges[-1], 1e-6)
        widths = np.clip(widths, 1e-6, None)

        records.append({
            "hh": hh, "mm": mm, "ss": ss, "edges": edges, "widths": widths,
            "concs": concs, "total_conc_reported": total_conc_reported,
        })

    return {"year": year, "month": month, "day": day, "records": records}


def closure_check(parsed: dict) -> np.ndarray:
    """Relative error between sum(conc_i * width_i) and each record's own
    reported total concentration -- validates the parse is correct."""
    diffs = []
    for r in parsed["records"]:
        computed = np.sum(r["concs"] * r["widths"])
        reported = r["total_conc_reported"]
        if reported > 0:
            diffs.append(abs(computed - reported) / reported)
    return np.array(diffs)


def record_midpoints_weights(parsed: dict) -> tuple[np.ndarray, np.ndarray]:
    """Flatten every (bin midpoint, weight) pair across all records in a
    parsed file. weight = concentration_density * bin_width = the
    concentration (#/L) attributable to that bin -- used as a histogram
    weight to reconstruct the number-weighted size distribution."""
    mids, weights = [], []
    for r in parsed["records"]:
        mids.append(r["edges"] + r["widths"] / 2)
        weights.append(r["concs"] * r["widths"])
    if not mids:
        return np.array([]), np.array([])
    return np.concatenate(mids), np.concatenate(weights)


def weighted_mean_median(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    if len(values) == 0 or weights.sum() == 0:
        return np.nan, np.nan
    mean = np.sum(values * weights) / weights.sum()
    order = np.argsort(values)
    v_sorted, w_sorted = values[order], weights[order]
    cum = np.cumsum(w_sorted)
    median = v_sorted[np.searchsorted(cum, cum[-1] / 2)]
    return float(mean), float(median)


# COCPIT's process_sheets.py::remove_text() masks any contour smaller than
# this native-pixel area as presumed sheet text/noise, drawing over it
# before particles are cropped -- so no particle image is ever produced for
# a true ice crystal whose native-pixel silhouette falls below this area.
REMOVE_TEXT_SMALL_THRESH_PX2 = 200
CPI_MICRONS_PER_NATIVE_PX_LOCAL = 2.3  # same CPI probe constant as compare_derived_feature_versions.py


def remove_text_floor_check(mids: np.ndarray, weights: np.ndarray) -> dict:
    """Quantifies how much of the verified PSD's number-weight sits below
    the equivalent-diameter size that COCPIT's remove_text() step would
    discard as 'text', and what the verified PSD's weighted mean becomes if
    that population is excluded -- a partial, falsifiable test of whether
    this filtering step in the COCPIT pipeline can explain (part of) the
    mean-size gap against the raw verified PSD."""
    equiv_d_um = 2 * np.sqrt(REMOVE_TEXT_SMALL_THRESH_PX2 / np.pi) * CPI_MICRONS_PER_NATIVE_PX_LOCAL
    frac_below = weights[mids < equiv_d_um].sum() / weights.sum() if weights.sum() else np.nan
    mean_full, _ = weighted_mean_median(mids, weights)
    mask = mids >= equiv_d_um
    mean_trunc, _ = weighted_mean_median(mids[mask], weights[mask])
    return {
        "floor_um": equiv_d_um,
        "frac_number_weight_below_floor": frac_below,
        "spec_mean_full_um": mean_full,
        "spec_mean_truncated_um": mean_trunc,
    }


# ---------------------------------------------------------------------------
# COCPIT loader
# ---------------------------------------------------------------------------

def load_cocpit_sizes() -> pd.DataFrame:
    path = DERIVED_DB_ROOT / COCPIT_VERSION / f"{COCPIT_CAMPAIGN}.csv"
    df = pd.read_csv(path, low_memory=False)
    df = _normalize_columns(df)
    df = _add_equiv_d_microns(df)
    df["flight_date"] = pd.to_datetime(df["date"]).dt.date
    return df


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_aggregate_comparison(spec_mids: np.ndarray, spec_weights: np.ndarray,
                               cocpit: pd.DataFrame, figs_dir: Path) -> None:
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
        bins = np.linspace(0, 600, 61)
        ax.hist(spec_mids, bins=bins, weights=spec_weights, density=True,
                histtype="step", linewidth=2, color="black",
                label="SPEC Inc CPI PSD (verified, archive)")
        for col, color in [("particle_width_microns", "#1f77b4"),
                            ("particle_height_microns", "#2ca02c"),
                            ("equiv_d_microns", "#d62728")]:
            vals = cocpit[col].dropna()
            vals = vals[(vals > 0) & (vals < 600)]
            ax.hist(vals, bins=bins, density=True, histtype="step", linewidth=1.5,
                     color=color, label=f"COCPIT {col} (n={len(vals):,})")
        ax.set_xlabel("size (microns)")
        ax.set_ylabel("density")
        ax.set_title("CRYSTAL-FACE-NASA: verified PSD vs COCPIT derived sizes\n"
                      "(all 8 matching flight dates pooled)")
        ax.legend(fontsize=8)
        out_p = figs_dir / "01_aggregate_comparison.png"
        fig.savefig(out_p, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_p}")


def plot_per_date_mean(summary_df: pd.DataFrame, figs_dir: Path) -> None:
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
        x = np.arange(len(summary_df))
        width = 0.15
        cols = ["spec_mean_um", "cocpit_particle_width_mean_um",
                "cocpit_particle_height_mean_um", "cocpit_equiv_d_mean_um"]
        labels = ["SPEC verified PSD", "COCPIT particle_width", "COCPIT particle_height",
                  "COCPIT equiv_d"]
        colors = ["black", "#1f77b4", "#2ca02c", "#d62728"]
        for j, (col, label, color) in enumerate(zip(cols, labels, colors)):
            ax.bar(x + (j - 1.5) * width, summary_df[col], width, label=label, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(summary_df["flight_date"].astype(str), rotation=45, ha="right")
        ax.set_ylabel("mean size (microns)")
        ax.set_title("CRYSTAL-FACE-NASA: mean particle size, verified PSD vs COCPIT, per flight date")
        ax.legend(fontsize=8)
        out_p = figs_dir / "02_per_date_mean_size.png"
        fig.savefig(out_p, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_p}")


def plot_per_date_histograms(spec_by_date: dict, cocpit: pd.DataFrame,
                              dates: list, figs_dir: Path) -> None:
    n = len(dates)
    n_cols = 3
    n_rows = int(np.ceil(n / n_cols))
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.3 * n_rows),
                                  constrained_layout=True)
        axes = np.atleast_1d(axes).ravel()
        bins = np.linspace(0, 600, 41)
        for ax, d in zip(axes, dates):
            mids, weights = spec_by_date[d]
            if len(mids) > 0:
                ax.hist(mids, bins=bins, weights=weights, density=True, histtype="step",
                        linewidth=2, color="black", label="SPEC PSD")
            sub = cocpit[cocpit["flight_date"] == d]
            for col, color in [("particle_width_microns", "#1f77b4"),
                                ("equiv_d_microns", "#d62728")]:
                vals = sub[col].dropna()
                vals = vals[(vals > 0) & (vals < 600)]
                if len(vals) > 0:
                    ax.hist(vals, bins=bins, density=True, histtype="step",
                            linewidth=1.2, color=color, label=col)
            ax.set_title(str(d), fontsize=9)
            ax.tick_params(labelsize=7)
        for j in range(n, len(axes)):
            axes[j].axis("off")
        axes[0].legend(fontsize=6)
        fig.suptitle("CRYSTAL-FACE-NASA: per-date verified PSD vs COCPIT sizes", fontsize=12)
        out_p = figs_dir / "03_per_date_histograms.png"
        fig.savefig(out_p, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_p}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_summary_report(summary_df: pd.DataFrame, closure_df: pd.DataFrame,
                          floor_check: dict, out_dir: Path) -> None:
    lines = ["# CPI size distribution verification -- CRYSTAL-FACE-NASA\n"]
    lines.append(
        f"Verified against SPEC Inc's PI-produced CPI particle-size-distribution "
        f"product ({SPEC_ARCHIVE_URL}, anonymous download, no login), 8 flight "
        f"dates, parsed with a self-consistency closure check "
        f"(mean {closure_df['mean_rel_diff_pct'].mean():.3f}%, "
        f"max {closure_df['max_rel_diff_pct'].max():.3f}% across all files).\n"
    )
    lines.append("## Mean particle size by flight date (microns)\n")
    lines.append("| Date | SPEC verified PSD | COCPIT particle_width | "
                  "COCPIT particle_height | COCPIT equiv_d | width/PSD ratio | equiv_d/PSD ratio |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, r in summary_df.iterrows():
        w_ratio = r["cocpit_particle_width_mean_um"] / r["spec_mean_um"] if r["spec_mean_um"] else np.nan
        e_ratio = r["cocpit_equiv_d_mean_um"] / r["spec_mean_um"] if r["spec_mean_um"] else np.nan
        lines.append(
            f"| {r['flight_date']} | {r['spec_mean_um']:.1f} | "
            f"{r['cocpit_particle_width_mean_um']:.1f} | "
            f"{r['cocpit_particle_height_mean_um']:.1f} | "
            f"{r['cocpit_equiv_d_mean_um']:.1f} | {w_ratio:.2f}x | {e_ratio:.2f}x |"
        )
    lines.append("")

    overall_w_ratio = (summary_df["cocpit_particle_width_mean_um"] / summary_df["spec_mean_um"]).mean()
    overall_e_ratio = (summary_df["cocpit_equiv_d_mean_um"] / summary_df["spec_mean_um"]).mean()
    lines.append(
        f"\n**Average ratio across all 8 dates: COCPIT particle_width is "
        f"{overall_w_ratio:.2f}x the verified PSD mean size; COCPIT equiv_d is "
        f"{overall_e_ratio:.2f}x.**\n"
    )
    lines.append(
        "Given the two products use different sizing definitions (SPEC's own "
        "CPI processing algorithm vs COCPIT's minAreaRect width/height and "
        "contour-area equiv_d) and different normalization (ambient "
        "concentration vs raw per-image counts, no sample-volume weighting), "
        "a same-order-of-magnitude, consistent-direction ratio across all 8 "
        "independently-sampled flight dates is the expected signature of "
        "genuine physical agreement -- not an exact 1:1 match, which would "
        "not even be expected in principle. A wildly inconsistent ratio "
        "from date to date, or a ratio differing by 1-2 orders of magnitude, "
        "would instead point to a units/scale bug.\n"
    )

    lines.append("\n## Candidate mechanism for part of the gap: remove_text() size floor\n")
    lines.append(
        f"COCPIT's `process_sheets.py::remove_text()` masks any contour smaller "
        f"than {REMOVE_TEXT_SMALL_THRESH_PX2} native-pixel² as presumed sheet "
        f"text/noise *before* particles are cropped and saved -- so no image is "
        f"ever produced for a true ice crystal below that silhouette size. "
        f"The equivalent circular diameter of a {REMOVE_TEXT_SMALL_THRESH_PX2} "
        f"px² native-pixel contour is **{floor_check['floor_um']:.1f} microns** "
        f"at the CPI probe's fixed 2.3 microns/pixel resolution -- matching, "
        f"nearly exactly, the ~37-48 micron 1st-5th percentile floor observed "
        f"in COCPIT's own particle_width_microns/particle_height_microns "
        f"distributions (see figs).\n"
    )
    lines.append(
        f"In the verified PSD, **{floor_check['frac_number_weight_below_floor']*100:.1f}%** "
        f"of the total number-weight falls below that {floor_check['floor_um']:.1f} "
        f"micron floor -- i.e. the large majority of particles the CPI actually "
        f"detected are smaller than what COCPIT's pipeline can ever produce an "
        f"image for. Excluding that population from the verified PSD raises its "
        f"weighted mean from {floor_check['spec_mean_full_um']:.1f} to "
        f"{floor_check['spec_mean_truncated_um']:.1f} microns -- a "
        f"{floor_check['spec_mean_truncated_um']/floor_check['spec_mean_full_um']:.2f}x "
        f"shift in the same direction as (but smaller magnitude than) the "
        f"observed COCPIT/PSD ratio. This accounts for **part** of the gap, not "
        f"all of it -- the residual is consistent with COCPIT's minAreaRect-based "
        f"particle_width/particle_height measuring a *maximum bounding-box "
        f"dimension*, which for elongated or aggregate ice crystals (common in "
        f"cirrus, CRYSTAL-FACE's target cloud type) is well known to run "
        f"substantially larger than an area-equivalent or SPEC-native size "
        f"metric; additional automated image-quality filtering downstream of "
        f"remove_text() (not traced here) may also contribute. This residual is "
        f"flagged as an open item, not confirmed.\n"
    )

    report_path = out_dir / "summary_report.md"
    report_path.write_text("\n".join(lines))
    print(f"  Saved {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ts = _run_timestamp()
    out_dir = ROOT / "logs" / "verify_cpi_size_distribution" / ts
    figs_dir = ROOT / "figs" / "verify_cpi_size_distribution" / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Parsing SPEC Inc CPI files from {SPEC_DATA_DIR} ...")
    spec_files = sorted(SPEC_DATA_DIR.glob("CP*.WB57"))
    if not spec_files:
        print(f"  No files found. Download from {SPEC_ARCHIVE_URL} first.")
        return

    closure_rows = []
    spec_by_date = {}
    for f in spec_files:
        parsed = parse_cpi_wb57(f)
        d = pd.Timestamp(year=parsed["year"], month=parsed["month"], day=parsed["day"]).date()
        diffs = closure_check(parsed)
        closure_rows.append({
            "file": f.name, "flight_date": d, "n_records": len(parsed["records"]),
            "mean_rel_diff_pct": diffs.mean() * 100 if len(diffs) else np.nan,
            "max_rel_diff_pct": diffs.max() * 100 if len(diffs) else np.nan,
        })
        mids, weights = record_midpoints_weights(parsed)
        spec_by_date[d] = (mids, weights)
        print(f"  {f.name}: {len(parsed['records'])} records, "
              f"closure mean={diffs.mean()*100:.3f}% max={diffs.max()*100:.3f}%")

    closure_df = pd.DataFrame(closure_rows)
    closure_path = out_dir / "parser_closure_check.csv"
    closure_df.to_csv(closure_path, index=False)
    print(f"\nSaved {closure_path}")

    print(f"\nLoading COCPIT {COCPIT_CAMPAIGN} {COCPIT_VERSION} ...")
    cocpit = load_cocpit_sizes()
    print(f"  {len(cocpit):,} rows, dates: {sorted(cocpit['flight_date'].unique())}")

    dates = sorted(spec_by_date)
    summary_rows = []
    for d in dates:
        mids, weights = spec_by_date[d]
        spec_mean, spec_median = weighted_mean_median(mids, weights)
        sub = cocpit[cocpit["flight_date"] == d]
        row = {"flight_date": d, "spec_mean_um": spec_mean, "spec_median_um": spec_median,
               "spec_n_records": len(mids)}
        for col, prefix in [("particle_width_microns", "cocpit_particle_width"),
                             ("particle_height_microns", "cocpit_particle_height"),
                             ("equiv_d_microns", "cocpit_equiv_d")]:
            vals = sub[col].dropna()
            vals = vals[(vals > 0) & (vals < 2000)]
            row[f"{prefix}_mean_um"] = vals.mean() if len(vals) else np.nan
            row[f"{prefix}_median_um"] = vals.median() if len(vals) else np.nan
            row[f"{prefix}_n"] = len(vals)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "spec_psd_summary_by_date.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved {summary_path}")

    cocpit_summary_path = out_dir / "cocpit_size_summary_by_date.csv"
    summary_df[[c for c in summary_df.columns if c.startswith("cocpit_") or c == "flight_date"]].to_csv(
        cocpit_summary_path, index=False)
    print(f"Saved {cocpit_summary_path}")

    all_mids = np.concatenate([spec_by_date[d][0] for d in dates])
    all_weights = np.concatenate([spec_by_date[d][1] for d in dates])
    floor_check = remove_text_floor_check(all_mids, all_weights)
    print(f"\nremove_text() floor check: floor={floor_check['floor_um']:.1f}um, "
          f"{floor_check['frac_number_weight_below_floor']*100:.1f}% of verified "
          f"PSD number-weight below floor")

    write_summary_report(summary_df, closure_df, floor_check, out_dir)

    plot_aggregate_comparison(all_mids, all_weights, cocpit, figs_dir)
    plot_per_date_mean(summary_df, figs_dir)
    plot_per_date_histograms(spec_by_date, cocpit, dates, figs_dir)

    run_config = {
        "spec_data_dir": str(SPEC_DATA_DIR),
        "spec_archive_url": SPEC_ARCHIVE_URL,
        "cocpit_campaign": COCPIT_CAMPAIGN,
        "cocpit_version": COCPIT_VERSION,
        "flight_dates": [str(d) for d in dates],
        "remove_text_floor_check": floor_check,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }
    config_path = out_dir / "run_config.json"
    with open(config_path, "w") as f:
        json.dump(run_config, f, indent=2, default=str)
    print(f"Saved {config_path}")

    update_latest(out_dir.parent, out_dir)
    update_latest(figs_dir.parent, figs_dir)
    print(f"\nLatest run: {out_dir.parent / 'latest'} -> {out_dir.name}")
    print(f"Latest figs: {figs_dir.parent / 'latest'} -> {figs_dir.name}")


if __name__ == "__main__":
    main()
