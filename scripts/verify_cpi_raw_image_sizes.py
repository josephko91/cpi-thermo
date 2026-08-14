#!/usr/bin/env python3
"""
Verify CPI Raw Image Particle Sizes (independent measurement)
================================================================
Follow-up to docs/reports/2026-07-24-cpi-size-distribution-verification.md,
which compared COCPIT's derived particle_width_microns/particle_height_microns
/equiv_d_microns against SPEC Inc's own bulk, volume-normalized PSD product
(CP<date>.WB57 files) and found a consistent 3.4x-5.2x mean-size gap --
partly (~1.7x) explained by COCPIT's remove_text() step discarding particles
below a 36.7-micron floor before they're ever imaged, with a residual ~2x
gap left open (candidate: COCPIT's bounding-box sizing running larger than
an area-equivalent metric for elongated cirrus crystals).

That comparison's limitation: the SPEC PSD is itself a *derived* product
(someone else's sizing algorithm's output), not raw imagery, so it can't
isolate whether the residual gap is a COCPIT-pipeline artifact or an
artifact of comparing two different derived products. This script closes
that gap: it measures particle sizes **directly from raw, individual CPI
particle-image crops**, run through neither SPEC's nor COCPIT's own sizing
code, to get a third, more primary reference point.

Data source: the same ESPO archive used for the SPEC PSD files
(espoarchive.nasa.gov, anonymous HTTPS, no login), a second CPI product for
CRYSTAL-FACE-NASA/WB57: gallery 1637 "CPI cloud particle images"
(archive/gallery/1637/<YYYYMMDD>), one PDF per minute-of-flight. Each PDF's
pages are montages of individually-cropped, habit-labeled (sph/col/sir/bir/
...), millisecond-timestamped CPI particle images -- confirmed by hand
(timestamps match SPEC's own PSD record times exactly). Each PAGE embeds
exactly one composite raster image at native CPI camera pixel resolution
(particle crops + background texture + text labels all baked into a single
bitmap -- confirmed empirically; NOT one embedded image per particle crop
as might be assumed from the visual layout).

Two-step pipeline:
  --download    fetch the 8 matching flight dates' PDF galleries (~401MB
                total, respects the archive's robots.txt Crawl-delay: 10 --
                takes ~70+ minutes including paginated gallery-listing
                fetches) into data/raw/CRYSTAL-FACE-NASA/CPI_raw_images_verification/
  (default)     for every page, extract its one native-resolution embedded
                image, Otsu-threshold the whole page, connected-component
                label it, and keep components that pass area/aspect-ratio/
                header-exclusion filters (segment_page() -- see the
                calibration comment above HEADER_EXCLUDE_TOP_PX for how
                these were chosen against a --debug visual spot-check).
                Each accepted component's bounding-box width/height and
                contour-area equivalent diameter (native pixels) are
                converted to microns via the CPI probe's fixed 2.3
                microns/native-pixel constant, then compared three ways
                against the SPEC verified PSD and COCPIT's derived sizes.

Outputs (logs/verify_cpi_raw_image_sizes/<timestamp>/ and
figs/verify_cpi_raw_image_sizes/<timestamp>/, with `latest` symlinks):
  raw_image_particle_measurements.csv  - every accepted particle measurement
  three_way_summary_by_date.csv        - SPEC PSD / COCPIT / raw-image mean+median per date
  extraction_diagnostics.csv           - per-date file/page/particle counts
  summary_report.md                    - condensed 3-way comparison + verdict
  01_aggregate_three_way_comparison.png
  02_per_date_three_way_mean_size.png
  debug_crops/*.png                    - (only with --debug) annotated pages, green boxes = accepted candidates

Usage:
    python scripts/verify_cpi_raw_image_sizes.py --download
    python scripts/verify_cpi_raw_image_sizes.py --debug --dates 20020709
    python scripts/verify_cpi_raw_image_sizes.py
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.compare_derived_feature_versions import (
    DERIVED_DB_ROOT, CPI_MICRONS_PER_NATIVE_PX, _normalize_columns, _add_equiv_d_microns,
)
from scripts.verify_cpi_size_distribution import (
    SPEC_DATA_DIR, parse_cpi_wb57, record_midpoints_weights, weighted_mean_median,
    REMOVE_TEXT_SMALL_THRESH_PX2,
)
from scripts.log_paths import timestamp as _run_timestamp, update_latest

RAW_IMG_DATA_DIR = ROOT / "data" / "raw" / "CRYSTAL-FACE-NASA" / "CPI_raw_images_verification"
GALLERY_BASE_URL = "https://espoarchive.nasa.gov/archive/gallery/1637"
DOWNLOAD_BASE_URL = "https://espoarchive.nasa.gov/archive/download"
CRAWL_DELAY_SEC = 10.0  # espoarchive.nasa.gov/robots.txt: Crawl-delay: 10
USER_AGENT = "Mozilla/5.0 (research; cpi-thermo verification script)"

FLIGHT_DATES = ["20020709", "20020711", "20020716", "20020719",
                 "20020721", "20020723", "20020728", "20020729"]

COCPIT_CAMPAIGN = "CRYSTAL_FACE_NASA"
COCPIT_VERSION = "v1.4.0"

STYLE = {
    "figure.facecolor": "white", "axes.facecolor": "#f8f8f8", "axes.grid": True,
    "grid.color": "white", "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9,
}

# --- Particle-vs-text/noise segmentation heuristics, calibrated against a
# --debug spot-check on CP20020709_1535_WB57.PDF page 1 before trusting the
# full run (see module docstring and the accompanying report). Each page is
# one composite raster (particle crops + background texture + text labels
# baked into a single bitmap -- confirmed empirically, not one XObject per
# particle as originally assumed). Global Otsu-thresholding the whole page
# and labeling connected components gives ~1000 components per page, but
# the distribution is extremely bimodal: median component area is ~5px^2
# (single/few-pixel noise specks from JPEG-ish compression + antialiasing),
# while genuine particle crops (visually cross-checked) sit at 300-37000
# px^2. Individual text GLYPHS (digits in timestamps, 3-letter habit codes)
# turned out to be near-square (aspect ~1.0-1.3) at this resolution -- NOT
# reliably distinguishable from small particles by aspect ratio alone, so
# the primary discriminator is area (HEADER_EXCLUDE_TOP_PX + MIN_PARTICLE_AREA_PX
# below); MAX_ASPECT_RATIO only catches multi-character runs that stayed
# connected. Residual text contamination is possible (biases toward
# smaller "particle" sizes, which -- if anything -- works against, not for,
# any apparent match with COCPIT's larger derived sizes) and is disclosed
# in the report rather than perfectly eliminated.
HEADER_EXCLUDE_TOP_PX = 20    # every page's single-line date/title header sits in this band
MIN_PARTICLE_AREA_PX = 150    # below this is noise specks or small text-glyph fragments
MAX_ASPECT_RATIO = 3.0        # rejects merged multi-character text runs


def _parse_args() -> argparse.Namespace:
    ts = _run_timestamp()
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--download", action="store_true",
                    help="Download the raw CPI image PDF galleries (~401MB, ~64 min) instead "
                         "of running the extraction/comparison pipeline")
    p.add_argument("--dates", nargs="+", default=FLIGHT_DATES,
                    help="Restrict to these YYYYMMDD dates (default: all 8 matched dates)")
    p.add_argument("--debug", action="store_true",
                    help="Save ~20 sample annotated pages (accepted candidates outlined) "
                         "per date for manual inspection instead of running the full "
                         "extraction/comparison pipeline")
    p.add_argument("--raw-img-dir", type=Path, default=RAW_IMG_DATA_DIR)
    p.add_argument("--out", type=Path,
                    default=ROOT / "logs" / "verify_cpi_raw_image_sizes" / ts)
    p.add_argument("--figs", type=Path,
                    default=ROOT / "figs" / "verify_cpi_raw_image_sizes" / ts)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Download (--download mode)
# ---------------------------------------------------------------------------

def _http_get(url: str) -> bytes:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=60) as resp:
        return resp.read()


def fetch_gallery_filenames(date: str, crawl_delay: float = CRAWL_DELAY_SEC) -> list[str]:
    """The gallery listing is paginated at 15 items/page (confirmed by
    inspecting a large date's ?page=N links) -- fetch every page until a
    page yields no new filenames."""
    filenames: set[str] = set()
    page = 0
    while True:
        url = f"{GALLERY_BASE_URL}/{date}" + (f"?page={page}" if page else "")
        html = _http_get(url).decode("utf-8", errors="ignore")
        found = set(re.findall(rf"CP{date}_\d+_WB57\.PDF", html))
        new = found - filenames
        if not new:
            break
        filenames |= new
        page += 1
        time.sleep(crawl_delay)
    return sorted(filenames)


def download_galleries(dates: list[str], out_dir: Path,
                        crawl_delay: float = CRAWL_DELAY_SEC) -> None:
    for date in dates:
        date_dir = out_dir / date
        date_dir.mkdir(parents=True, exist_ok=True)
        print(f"Listing {GALLERY_BASE_URL}/{date} ...")
        filenames = fetch_gallery_filenames(date)
        time.sleep(crawl_delay)
        print(f"  {len(filenames)} files")
        for fn in filenames:
            dest = date_dir / fn
            if dest.exists() and dest.stat().st_size > 0:
                print(f"  {fn} already downloaded, skipping")
                continue
            data = _http_get(f"{DOWNLOAD_BASE_URL}/{fn}")
            dest.write_bytes(data)
            print(f"  downloaded {fn} ({len(data):,} bytes)")
            time.sleep(crawl_delay)


# ---------------------------------------------------------------------------
# PDF embedded-image extraction
# ---------------------------------------------------------------------------

def extract_page_arrays(pdf_path: Path) -> list[np.ndarray]:
    """Each page of these montage PDFs embeds exactly ONE raster image
    (confirmed empirically: page count == embedded-image count for every
    file inspected) -- the whole page layout (particle crops + background
    texture + text labels) is baked into a single bitmap, not one XObject
    per particle crop as initially assumed. Returns one native-resolution
    grayscale array per page."""
    import fitz
    out = []
    doc = fitz.open(pdf_path)
    for page in doc:
        images = page.get_images(full=True)
        if not images:
            continue
        xref = images[0][0]
        try:
            base = doc.extract_image(xref)
            pil_img = Image.open(io.BytesIO(base["image"])).convert("L")
        except Exception:
            continue
        out.append(np.asarray(pil_img))
    doc.close()
    return out


def otsu_threshold(arr: np.ndarray) -> int:
    """Standard Otsu automatic threshold (no skimage/cv2 dependency)."""
    hist, _ = np.histogram(arr, bins=256, range=(0, 256))
    hist = hist.astype(float)
    total = hist.sum()
    sum_all = np.dot(np.arange(256), hist)
    sum_bg = weight_bg = max_var = 0.0
    threshold = 0
    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_all - sum_bg) / weight_fg
        var_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if var_between > max_var:
            max_var = var_between
            threshold = t
    return threshold


def segment_page(arr: np.ndarray) -> list[dict]:
    """Global-Otsu-thresholds one whole page image, connected-component
    labels the result, and returns one measurement dict per accepted
    component (see the calibration comment above HEADER_EXCLUDE_TOP_PX for
    how the filters were chosen). Each accepted component's bounding-box
    width/height and contour-area equivalent diameter are in native page
    pixels (not yet converted to microns -- that happens in process_date,
    since it's a single global constant regardless of page)."""
    h, w = arr.shape
    t = otsu_threshold(arr)
    mask = arr <= t
    mask[:HEADER_EXCLUDE_TOP_PX, :] = False
    if not (0 < mask.sum() < mask.size):
        return []
    labeled, n = ndimage.label(mask)
    if n == 0:
        return []
    objs = ndimage.find_objects(labeled)
    sizes = ndimage.sum(mask, labeled, index=range(1, n + 1))

    results = []
    for i in range(1, n + 1):
        area_px = float(sizes[i - 1])
        if area_px < MIN_PARTICLE_AREA_PX:
            continue
        sl = objs[i - 1]
        y0, y1 = sl[0].start, sl[0].stop - 1
        x0, x1 = sl[1].start, sl[1].stop - 1
        bbox_w, bbox_h = x1 - x0 + 1, y1 - y0 + 1
        aspect = max(bbox_w, bbox_h) / max(min(bbox_w, bbox_h), 1)
        if aspect > MAX_ASPECT_RATIO:
            continue
        touches_border = bool(y0 == 0 or x0 == 0 or y1 == h - 1 or x1 == w - 1)
        results.append({
            "bbox_width_px": bbox_w, "bbox_height_px": bbox_h,
            "area_px": area_px, "equiv_d_px": 2 * np.sqrt(area_px / np.pi),
            "aspect_ratio": aspect, "touches_border": touches_border,
            "y0": y0, "x0": x0,
        })
    return results


def process_date(date: str, raw_img_dir: Path, debug: bool,
                  debug_out: Path | None = None) -> tuple[pd.DataFrame, dict]:
    """Extracts + measures particles from every PDF for one flight date.
    Returns (measurements_df, diagnostics_dict)."""
    date_dir = raw_img_dir / date
    pdf_files = sorted(date_dir.glob("*.PDF"))
    if not pdf_files:
        return pd.DataFrame(), {"date": date, "n_files": 0}

    rows = []
    n_pages = 0
    debug_saved = 0
    for f in pdf_files:
        pages = extract_page_arrays(f)
        n_pages += len(pages)
        for page_idx, arr in enumerate(pages):
            page_results = segment_page(arr)
            for r in page_results:
                rows.append(r)

            if debug and debug_saved < 20 and debug_out is not None:
                _save_debug_page(arr, page_results,
                                  debug_out / f"{date}_{f.stem}_p{page_idx}.png")
                debug_saved += 1

    df = pd.DataFrame(rows)
    diagnostics = {
        "date": date, "n_files": len(pdf_files), "n_pages": n_pages,
        "n_particles_measured": len(df),
    }
    return df, diagnostics


def _save_debug_page(arr: np.ndarray, results: list[dict], out_path: Path) -> None:
    """Annotated whole-page image: green boxes = accepted particle
    candidates, for visual spot-checking against the source montage."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6, 8))
        ax.imshow(arr, cmap="gray")
        for r in results:
            rect = plt.Rectangle((r["x0"], r["y0"]), r["bbox_width_px"], r["bbox_height_px"],
                                  fill=False, edgecolor="lime", linewidth=1)
            ax.add_patch(rect)
        ax.set_title(f"{len(results)} accepted candidates", fontsize=8)
        ax.axis("off")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# COCPIT + SPEC loaders (reuse existing modules)
# ---------------------------------------------------------------------------

def load_cocpit_sizes() -> pd.DataFrame:
    path = DERIVED_DB_ROOT / COCPIT_VERSION / f"{COCPIT_CAMPAIGN}.csv"
    df = pd.read_csv(path, low_memory=False)
    df = _normalize_columns(df)
    df = _add_equiv_d_microns(df)
    df["flight_date"] = pd.to_datetime(df["date"]).dt.date
    return df


def load_spec_psd_by_date(dates: list[str]) -> dict:
    spec_by_date = {}
    for date_str in dates:
        d = pd.Timestamp(date_str).date()
        path = SPEC_DATA_DIR / f"CP{date_str}.WB57"
        if not path.exists():
            continue
        parsed = parse_cpi_wb57(path)
        mids, weights = record_midpoints_weights(parsed)
        spec_by_date[d] = (mids, weights)
    return spec_by_date


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_aggregate_three_way(raw_df: pd.DataFrame, spec_by_date: dict,
                              cocpit: pd.DataFrame, figs_dir: Path) -> None:
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
        bins = np.linspace(0, 600, 61)

        all_mids = np.concatenate([m for m, _ in spec_by_date.values()]) if spec_by_date else np.array([])
        all_weights = np.concatenate([w for _, w in spec_by_date.values()]) if spec_by_date else np.array([])
        if len(all_mids):
            ax.hist(all_mids, bins=bins, weights=all_weights, density=True, histtype="step",
                    linewidth=2, color="black", label="SPEC verified PSD (bulk product)")

        for col, color in [("particle_width_microns", "#1f77b4"), ("equiv_d_microns", "#d62728")]:
            vals = cocpit[col].dropna()
            vals = vals[(vals > 0) & (vals < 600)]
            ax.hist(vals, bins=bins, density=True, histtype="step", linewidth=1.5,
                    color=color, label=f"COCPIT {col} (n={len(vals):,})")

        for col, color in [("bbox_width_microns", "#2ca02c"), ("equiv_d_microns_raw", "#9467bd")]:
            vals = raw_df[col].dropna()
            vals = vals[(vals > 0) & (vals < 600)]
            ax.hist(vals, bins=bins, density=True, histtype="step", linewidth=1.5,
                    color=color, linestyle="--", label=f"Independent raw-image {col} (n={len(vals):,})")

        ax.set_xlabel("size (microns)")
        ax.set_ylabel("density")
        ax.set_title("CRYSTAL-FACE-NASA: three-way size comparison\n"
                      "SPEC verified PSD vs COCPIT derived vs independent raw-image measurement")
        ax.legend(fontsize=7)
        out_p = figs_dir / "01_aggregate_three_way_comparison.png"
        fig.savefig(out_p, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_p}")


def plot_per_date_three_way(summary_df: pd.DataFrame, figs_dir: Path) -> None:
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
        x = np.arange(len(summary_df))
        width = 0.2
        cols = ["spec_mean_um", "cocpit_particle_width_mean_um",
                "cocpit_equiv_d_mean_um", "raw_bbox_width_mean_um"]
        labels = ["SPEC verified PSD", "COCPIT particle_width", "COCPIT equiv_d",
                  "Independent raw-image bbox_width"]
        colors = ["black", "#1f77b4", "#d62728", "#2ca02c"]
        for j, (col, label, color) in enumerate(zip(cols, labels, colors)):
            ax.bar(x + (j - 1.5) * width, summary_df[col], width, label=label, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(summary_df["flight_date"].astype(str), rotation=45, ha="right")
        ax.set_ylabel("mean size (microns)")
        ax.set_title("CRYSTAL-FACE-NASA: three-way mean particle size per flight date")
        ax.legend(fontsize=8)
        out_p = figs_dir / "02_per_date_three_way_mean_size.png"
        fig.savefig(out_p, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_p}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_summary_report(summary_df: pd.DataFrame, diag_df: pd.DataFrame,
                          floor_frac: float, out_dir: Path) -> None:
    lines = ["# Independent raw-image CPI size verification -- CRYSTAL-FACE-NASA\n"]
    lines.append(
        "Three-way comparison: SPEC Inc's verified bulk PSD product vs COCPIT's "
        "derived per-particle sizes vs sizes measured directly, independently, "
        "from raw individual CPI particle-image crops (ESPO archive gallery 1637, "
        "ancestor of neither SPEC's nor COCPIT's own sizing algorithm).\n"
    )
    lines.append(f"Total particles independently measured: {int(diag_df['n_particles_measured'].sum()):,} "
                  f"across {int(diag_df['n_files'].sum())} PDF files, {len(summary_df)} flight dates.\n")

    lines.append("\n## Mean particle size by flight date (microns)\n")
    lines.append("| Date | SPEC PSD | COCPIT width | COCPIT equiv_d | Raw-image bbox width | "
                  "Raw-image equiv_d | raw/PSD ratio | COCPIT/raw ratio |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for _, r in summary_df.iterrows():
        raw_psd_ratio = r["raw_bbox_width_mean_um"] / r["spec_mean_um"] if r["spec_mean_um"] else np.nan
        cocpit_raw_ratio = r["cocpit_particle_width_mean_um"] / r["raw_bbox_width_mean_um"] if r["raw_bbox_width_mean_um"] else np.nan
        lines.append(
            f"| {r['flight_date']} | {r['spec_mean_um']:.1f} | "
            f"{r['cocpit_particle_width_mean_um']:.1f} | {r['cocpit_equiv_d_mean_um']:.1f} | "
            f"{r['raw_bbox_width_mean_um']:.1f} | {r['raw_equiv_d_mean_um']:.1f} | "
            f"{raw_psd_ratio:.2f}x | {cocpit_raw_ratio:.2f}x |"
        )
    lines.append("")

    lines.append(
        f"\nFraction of independently-measured raw particles below the 36.7-micron "
        f"remove_text() floor (see prior report): **{floor_frac*100:.1f}%**.\n"
    )

    report_path = out_dir / "summary_report.md"
    report_path.write_text("\n".join(lines))
    print(f"  Saved {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    if args.download:
        args.raw_img_dir.mkdir(parents=True, exist_ok=True)
        download_galleries(args.dates, args.raw_img_dir)
        return

    args.out.mkdir(parents=True, exist_ok=True)
    args.figs.mkdir(parents=True, exist_ok=True)
    debug_out = args.figs / "debug_crops" if args.debug else None

    all_measurements = []
    diagnostics = []
    for date in args.dates:
        print(f"Processing {date} ...")
        df, diag = process_date(date, args.raw_img_dir, args.debug, debug_out)
        diagnostics.append(diag)
        print(f"  {diag}")
        if len(df):
            df["flight_date"] = pd.Timestamp(date).date()
            all_measurements.append(df)

    diag_df = pd.DataFrame(diagnostics)
    diag_path = args.out / "extraction_diagnostics.csv"
    diag_df.to_csv(diag_path, index=False)
    print(f"\nSaved {diag_path}")

    if args.debug:
        print(f"\nDebug crops saved to {debug_out}. Inspect before running the full pipeline.")
        return

    raw_df = pd.concat(all_measurements, ignore_index=True) if all_measurements else pd.DataFrame()
    if raw_df.empty:
        print("No particles measured -- check extraction diagnostics / download status.")
        return

    raw_df["bbox_width_microns"] = raw_df["bbox_width_px"] * CPI_MICRONS_PER_NATIVE_PX
    raw_df["bbox_height_microns"] = raw_df["bbox_height_px"] * CPI_MICRONS_PER_NATIVE_PX
    raw_df["equiv_d_microns_raw"] = raw_df["equiv_d_px"] * CPI_MICRONS_PER_NATIVE_PX

    measurements_path = args.out / "raw_image_particle_measurements.csv"
    raw_df.to_csv(measurements_path, index=False)
    print(f"Saved {measurements_path}")

    print("\nLoading COCPIT derived sizes ...")
    cocpit = load_cocpit_sizes()

    print("Loading SPEC verified PSD ...")
    spec_by_date = load_spec_psd_by_date(args.dates)

    floor_frac = (raw_df["equiv_d_microns_raw"] < REMOVE_TEXT_SMALL_THRESH_PX2_TO_UM()).mean()

    summary_rows = []
    for date in args.dates:
        d = pd.Timestamp(date).date()
        row = {"flight_date": d}

        if d in spec_by_date:
            mids, weights = spec_by_date[d]
            spec_mean, _ = weighted_mean_median(mids, weights)
        else:
            spec_mean = np.nan
        row["spec_mean_um"] = spec_mean

        sub_cocpit = cocpit[cocpit["flight_date"] == d]
        for col, prefix in [("particle_width_microns", "cocpit_particle_width"),
                             ("equiv_d_microns", "cocpit_equiv_d")]:
            vals = sub_cocpit[col].dropna()
            vals = vals[(vals > 0) & (vals < 2000)]
            row[f"{prefix}_mean_um"] = vals.mean() if len(vals) else np.nan

        sub_raw = raw_df[raw_df["flight_date"] == d]
        row["raw_bbox_width_mean_um"] = sub_raw["bbox_width_microns"].mean() if len(sub_raw) else np.nan
        row["raw_equiv_d_mean_um"] = sub_raw["equiv_d_microns_raw"].mean() if len(sub_raw) else np.nan
        row["raw_n"] = len(sub_raw)

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = args.out / "three_way_summary_by_date.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved {summary_path}")

    write_summary_report(summary_df, diag_df, floor_frac, args.out)

    plot_aggregate_three_way(raw_df, spec_by_date, cocpit, args.figs)
    plot_per_date_three_way(summary_df, args.figs)

    run_config = {
        "raw_img_dir": str(args.raw_img_dir), "dates": args.dates,
        "header_exclude_top_px": HEADER_EXCLUDE_TOP_PX,
        "min_particle_area_px": MIN_PARTICLE_AREA_PX,
        "max_aspect_ratio": MAX_ASPECT_RATIO,
        "cpi_microns_per_native_px": CPI_MICRONS_PER_NATIVE_PX,
        "n_particles_measured": len(raw_df),
        "numpy_version": np.__version__, "pandas_version": pd.__version__,
    }
    config_path = args.out / "run_config.json"
    with open(config_path, "w") as f:
        json.dump(run_config, f, indent=2, default=str)
    print(f"Saved {config_path}")

    update_latest(args.out.parent, args.out)
    update_latest(args.figs.parent, args.figs)
    print(f"\nLatest run: {args.out.parent / 'latest'} -> {args.out.name}")
    print(f"Latest figs: {args.figs.parent / 'latest'} -> {args.figs.name}")


def REMOVE_TEXT_SMALL_THRESH_PX2_TO_UM() -> float:
    return 2 * np.sqrt(REMOVE_TEXT_SMALL_THRESH_PX2 / np.pi) * CPI_MICRONS_PER_NATIVE_PX


if __name__ == "__main__":
    main()
