"""
ICE-L Maycomm AC19-400 TDL dual-path diagnostic.

Attempts to compare MRTDLS_MC (short path, ~10 cm) against MRTDLL_MC
(long path, ~130 cm), colored by air temperature, and compute
R = MRTDLS_MC / MRTDLL_MC.

Key findings from data inspection:
  - MRTDLS_MC is entirely NaN in all 12 flights — short path produced no data.
  - MRTDLL_MC values are in ppmv (not g/kg as stated in the netCDF attribute);
    range ~28–21 637 ppmv across the campaign.
  - RF01 MRTDLL_MC is identically 0 for all timestamps (TDL likely inactive).
  - R cannot be computed; this script documents MRTDLL_MC coverage instead.

Run from the repo root:
    pytest scripts/tests/test_ice_l.py -v
or directly:
    python scripts/tests/test_ice_l.py
"""
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
from scripts.log_paths import timestamp as _run_timestamp, update_latest

DATA_DIR = REPO_ROOT / "data/raw/ICE-L/nav-state-microphysics"
# Timestamped so re-running this test doesn't overwrite a previous run's
# diagnostics/figures; latest symlink kept pointing at the newest one.
# Directories are created lazily inside the writing tests (not here at
# module level) so importing/collecting this file has no filesystem effects.
_RUN_TS = _run_timestamp()
DIAG_DIR = REPO_ROOT / "logs/campaign_tests/ice_l" / _RUN_TS
FIGS_DIR = REPO_ROOT / "figs/campaign_tests/ice_l" / _RUN_TS

# MRTDLL_MC is stored in ppmv despite the netCDF attribute reading "gram/kg".
# Observed range: ~3–50 000 ppmv; anything outside is treated as invalid.
_MR_PPMV_MIN = 1.0
_MR_PPMV_MAX = 50_000.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_flight(filepath: Path) -> pd.DataFrame:
    import xarray as xr

    ds = xr.open_dataset(filepath, decode_times=True, decode_timedelta=True)
    try:
        time_arr = pd.to_datetime(ds["Time"].values, utc=True, errors="coerce")

        def _extract(var: str) -> np.ndarray:
            vals = np.asarray(ds[var].values, dtype=float).reshape(-1)
            # No standard FillValue attribute; treat NaN as-is.
            return vals

        long_mr  = _extract("MRTDLL_MC")   # ppmv (mislabeled gram/kg in attrs)
        short_mr = _extract("MRTDLS_MC")   # ppmv — universally NaN in this dataset
        atx      = _extract("ATX")         # °C

        # Physical plausibility — ppmv scale for TDL channels.
        # RF01 has sub-1-ppmv startup values that are not geophysical.
        long_mr[(long_mr  < _MR_PPMV_MIN) | (long_mr  > _MR_PPMV_MAX)] = np.nan
        short_mr[(short_mr < _MR_PPMV_MIN) | (short_mr > _MR_PPMV_MAX)] = np.nan

        # Temperature plausibility
        atx[(atx < -100) | (atx > 60)] = np.nan

        rf = filepath.stem.split(".")[0]   # RF01, RF02, …

        return pd.DataFrame({
            "Time":       time_arr,
            "MRTDLL_MC":  long_mr,
            "MRTDLS_MC":  short_mr,
            "ATX_C":      atx,
            "flight":     rf,
        })
    finally:
        ds.close()


def _load_all_flights() -> pd.DataFrame:
    if not hasattr(_load_all_flights, "_cache"):
        files = sorted(DATA_DIR.glob("*.PNI.nc"))
        if not files:
            raise FileNotFoundError(f"No *.PNI.nc files in {DATA_DIR}")
        frames = [_load_flight(f) for f in files]
        df = pd.concat(frames, ignore_index=True)
        # Ratio — will be all-NaN because MRTDLS_MC is universally absent,
        # but computed here for completeness and to document the finding.
        with np.errstate(divide="ignore", invalid="ignore"):
            df["R"] = df["MRTDLS_MC"] / df["MRTDLL_MC"]
        _load_all_flights._cache = df
    return _load_all_flights._cache


def _format_exc(e: Exception) -> str:
    return f"{type(e).__name__}: {e}\n" + traceback.format_exc()


# ---------------------------------------------------------------------------
# Sanity / structural tests
# ---------------------------------------------------------------------------

def test_data_dir_exists():
    assert DATA_DIR.exists() and DATA_DIR.is_dir(), f"ICE-L data dir not found: {DATA_DIR}"


def test_nc_files_present():
    files = sorted(DATA_DIR.glob("*.PNI.nc"))
    assert len(files) > 0, f"No PNI.nc files in {DATA_DIR}"
    assert len(files) == 12, f"Expected 12 flight files, found {len(files)}"


def test_tdl_vars_present_in_all_files():
    import xarray as xr
    for f in sorted(DATA_DIR.glob("*.PNI.nc")):
        ds = xr.open_dataset(f, decode_times=False)
        for var in ("MRTDLL_MC", "MRTDLS_MC", "ATX"):
            assert var in ds.variables, f"{var} missing from {f.name}"
        ds.close()


def test_load_returns_dataframe():
    try:
        df = _load_all_flights()
    except Exception as e:
        pytest.fail(f"Loading raised: {_format_exc(e)}")
    assert isinstance(df, pd.DataFrame) and not df.empty


def test_required_columns():
    df = _load_all_flights()
    for col in ("MRTDLL_MC", "MRTDLS_MC", "ATX_C", "R", "flight", "Time"):
        assert col in df.columns, f"Column {col!r} missing"


# ---------------------------------------------------------------------------
# Data quality / availability tests
# ---------------------------------------------------------------------------

def test_short_path_universally_nan():
    """MRTDLS_MC is expected to be entirely NaN — documents the known data gap."""
    df = _load_all_flights()
    n_valid = df["MRTDLS_MC"].notna().sum()
    print(f"\n  MRTDLS_MC valid rows: {n_valid} / {len(df)}")
    # This is a known finding; assert it for documentation purposes.
    assert n_valid == 0, (
        f"MRTDLS_MC unexpectedly has {n_valid} valid values — "
        "update this test if short-path data becomes available."
    )


def test_long_path_has_data():
    """MRTDLL_MC must have valid values (in ppmv) across the campaign."""
    df = _load_all_flights()
    n_valid = df["MRTDLL_MC"].notna().sum()
    assert n_valid > 0, "No valid MRTDLL_MC (long path) values after masking"
    print(f"\n  MRTDLL_MC valid rows: {n_valid:,} / {len(df):,}")


def test_rf01_long_path_inactive():
    """RF01 MRTDLL_MC should be entirely NaN after masking (sub-1-ppmv startup values)."""
    df = _load_all_flights()
    rf01 = df[df["flight"] == "RF01"]["MRTDLL_MC"]
    n_valid = rf01.notna().sum()
    print(f"\n  RF01 MRTDLL_MC after masking: {n_valid} valid, {rf01.isna().sum()} NaN")
    assert n_valid == 0, (
        f"RF01 has {n_valid} unmasked MRTDLL_MC values — check < {_MR_PPMV_MIN} ppmv threshold"
    )


def test_long_path_coverage_per_flight():
    """Report valid data fraction per flight; at least 5 flights must have data."""
    df = _load_all_flights()
    flights_with_data = []
    for flight, grp in df.groupby("flight"):
        n = grp["MRTDLL_MC"].notna().sum()
        frac = n / len(grp)
        print(f"  {flight}: {n:>6,} valid  ({frac:.1%})")
        if n > 0:
            flights_with_data.append(str(flight))
    assert len(flights_with_data) >= 5, (
        f"Fewer than 5 flights have MRTDLL_MC data: {flights_with_data}"
    )


def test_ratio_is_all_nan():
    """R = MRTDLS_MC / MRTDLL_MC must be entirely NaN (no short-path data)."""
    df = _load_all_flights()
    n_r_valid = df["R"].notna().sum()
    assert n_r_valid == 0, (
        f"R has {n_r_valid} non-NaN values — short-path data may now be present; "
        "re-enable cross-path comparison."
    )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def test_write_diagnostics():
    """Write per-flight and overall TDL diagnostics to logs/campaign_tests/ice_l/."""
    df = _load_all_flights()

    diag: dict = {
        "instrument": "Maycomm AC19-400 TDL",
        "campaign":   "ICE-L",
        "aircraft":   "NSF/NCAR C-130Q (N130AR)",
        "variables": {
            "long_path":   "MRTDLL_MC — TDL Long Absorption Path (~130 cm)",
            "short_path":  "MRTDLS_MC — TDL Short Absorption Path (~10 cm)",
            "ratio":       "R = MRTDLS_MC / MRTDLL_MC",
            "noted_units": "ppmv (netCDF attribute incorrectly reads 'gram/kg')",
        },
        "data_quality_notes": [
            "MRTDLS_MC is entirely NaN across all 12 flights — short-path channel inactive.",
            "MRTDLL_MC for RF01 is identically 0 — TDL likely not recording on that flight.",
            "Cross-path ratio R cannot be computed; only long-path data is usable.",
        ],
        "n_total_rows": int(len(df)),
        "n_flights":    int(df["flight"].nunique()),
        "overall":      {},
        "per_flight":   {},
    }

    # Overall stats for long-path only (short-path and R are all NaN)
    s = df["MRTDLL_MC"].dropna()
    diag["overall"]["long_path_ppmv"] = {
        "n_valid": int(len(s)),
        "mean":    float(s.mean())           if len(s) else None,
        "std":     float(s.std())            if len(s) else None,
        "p05":     float(s.quantile(0.05))   if len(s) else None,
        "median":  float(s.median())         if len(s) else None,
        "p95":     float(s.quantile(0.95))   if len(s) else None,
        "min":     float(s.min())            if len(s) else None,
        "max":     float(s.max())            if len(s) else None,
    }
    diag["overall"]["short_path_ppmv"] = {"n_valid": 0, "note": "entirely NaN"}
    diag["overall"]["ratio_R"]         = {"n_valid": 0, "note": "cannot compute — no short-path data"}

    # Per-flight stats
    for flight, grp in df.groupby("flight"):
        long_s = grp["MRTDLL_MC"].dropna()
        n_zero = int((grp["MRTDLL_MC"] == 0).sum())
        diag["per_flight"][str(flight)] = {
            "long_path_ppmv": {
                "n_valid":  int(len(long_s)),
                "n_zero":   n_zero,
                "mean":     float(long_s.mean())   if len(long_s) else None,
                "median":   float(long_s.median()) if len(long_s) else None,
                "std":      float(long_s.std())    if len(long_s) else None,
                "min":      float(long_s.min())    if len(long_s) else None,
                "max":      float(long_s.max())    if len(long_s) else None,
            },
            "short_path_ppmv": {"n_valid": 0},
        }

    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    out_json = DIAG_DIR / "ice_l_maycom_tdl_diagnostics.json"
    out_json.write_text(json.dumps(diag, indent=2, default=str))
    print(f"\n  Diagnostics → {out_json}")

    out_csv = DIAG_DIR / "ice_l_maycom_tdl_diagnostics.csv"
    df[["Time", "flight", "ATX_C", "MRTDLL_MC", "MRTDLS_MC", "R"]].to_csv(
        out_csv, index=False
    )
    print(f"  Diagnostics CSV → {out_csv}")
    update_latest(DIAG_DIR.parent, DIAG_DIR)

    assert out_json.exists()
    assert out_csv.exists()


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def test_generate_figures():
    """
    Generate TDL diagnostic figures.

    Because MRTDLS_MC is entirely absent, figures focus on MRTDLL_MC alone:
      01 — All-flights scatter: MRTDLL_MC vs ATX, colored by temperature
      02 — Per-flight scatter: MRTDLL_MC vs ATX
      03 — MRTDLL_MC distribution by flight (log-scale histogram)
      04 — Per-flight MRTDLL_MC boxplot
      05 — MRTDLL_MC coverage timeline (valid-row count per flight)
    A note panel is included on each figure explaining the missing short path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, LogNorm
    from matplotlib.cm import ScalarMappable

    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    df = _load_all_flights()
    flights = sorted(df["flight"].unique())

    # Shared temperature colormap
    t_all = df["ATX_C"].dropna()
    t_norm = Normalize(
        vmin=float(np.percentile(t_all, 2)),
        vmax=float(np.percentile(t_all, 98)),
    )
    cmap_t = plt.cm.RdYlBu_r

    NOTE = (
        "Note: MRTDLS_MC (short path) is entirely NaN in all 12 flights.\n"
        "Cross-path ratio R cannot be computed. Only MRTDLL_MC is shown."
    )

    # ---- Fig 1: MRTDLL_MC vs ATX, all flights, colored by T ---------------
    sub = df[["MRTDLL_MC", "ATX_C"]].dropna()

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(
        sub["ATX_C"], sub["MRTDLL_MC"],
        c=sub["ATX_C"], cmap=cmap_t, norm=t_norm,
        s=2, alpha=0.3, rasterized=True,
    )
    ax.set_yscale("log")
    plt.colorbar(sc, ax=ax, label="ATX (°C)")
    ax.set_xlabel("ATX — Air Temperature (°C)")
    ax.set_ylabel("MRTDLL_MC (ppmv, log scale)")
    ax.set_title(f"ICE-L — TDL Long Path vs Temperature (n={len(sub):,})")
    fig.text(0.01, 0.01, NOTE, fontsize=7, color="gray",
             va="bottom", wrap=True)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(FIGS_DIR / "01_tdl_long_vs_temperature.png", dpi=150)
    plt.close(fig)

    # ---- Fig 2: per-flight scatter: MRTDLL_MC vs ATX -----------------------
    ncols = 4
    nrows = (len(flights) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3.5), squeeze=False)

    for i, flight in enumerate(flights):
        ax = axes[i // ncols][i % ncols]
        grp = df[df["flight"] == flight][["MRTDLL_MC", "ATX_C"]].dropna()
        if grp.empty:
            ax.set_title(f"{flight}\n(no data)", fontsize=9)
            ax.set_axis_off()
            continue
        sc = ax.scatter(
            grp["ATX_C"], grp["MRTDLL_MC"],
            c=grp["ATX_C"], cmap=cmap_t, norm=t_norm,
            s=3, alpha=0.5, rasterized=True,
        )
        ax.set_yscale("log")
        ax.set_title(f"{flight}  (n={len(grp):,})", fontsize=9)
        ax.set_xlabel("ATX (°C)", fontsize=7)
        ax.set_ylabel("MRTDLL_MC (ppmv)", fontsize=7)
        ax.tick_params(labelsize=7)

    for j in range(len(flights), nrows * ncols):
        axes[j // ncols][j % ncols].set_axis_off()

    fig.colorbar(
        ScalarMappable(norm=t_norm, cmap=cmap_t),
        ax=axes.ravel().tolist(),
        label="ATX (°C)", shrink=0.6,
    )
    fig.suptitle("ICE-L — MRTDLL_MC vs Temperature, per flight", fontsize=11)
    fig.savefig(FIGS_DIR / "02_tdl_long_per_flight.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- Fig 3: MRTDLL_MC log-scale histogram by flight --------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap_flights = plt.cm.tab20
    bins = np.logspace(
        np.log10(max(_MR_PPMV_MIN, 1)),
        np.log10(_MR_PPMV_MAX),
        60,
    )
    for k, flight in enumerate(flights):
        vals = df.loc[df["flight"] == flight, "MRTDLL_MC"].dropna()
        if vals.empty:
            continue
        ax.hist(vals, bins=bins, alpha=0.5, label=flight,
                color=cmap_flights(k / len(flights)), density=True, histtype="step",
                linewidth=1.2)
    ax.set_xscale("log")
    ax.set_xlabel("MRTDLL_MC (ppmv)")
    ax.set_ylabel("Density")
    ax.set_title("ICE-L — TDL Long-Path MR Distribution by Flight")
    ax.legend(fontsize=7, ncol=3)
    fig.text(0.01, 0.01, NOTE, fontsize=7, color="gray", va="bottom")
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(FIGS_DIR / "03_tdl_long_histogram_by_flight.png", dpi=150)
    plt.close(fig)

    # ---- Fig 4: per-flight MRTDLL_MC boxplot (log scale) -------------------
    plot_data  = []
    plot_labels = []
    for flight in flights:
        vals = df.loc[df["flight"] == flight, "MRTDLL_MC"].dropna()
        if not vals.empty:
            plot_data.append(np.log10(vals.values))
            plot_labels.append(flight)

    fig, ax = plt.subplots(figsize=(10, 4))
    bp = ax.boxplot(
        plot_data, tick_labels=plot_labels, patch_artist=True,
        medianprops={"color": "firebrick", "linewidth": 1.5},
        flierprops={"marker": ".", "markersize": 2, "alpha": 0.3},
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("steelblue")
        patch.set_alpha(0.6)

    # Custom y-tick labels in ppmv (fix ticks before labelling)
    yticks = ax.get_yticks()
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{10**y:.0f}" for y in yticks])
    ax.set_ylabel("MRTDLL_MC (ppmv, log scale)")
    ax.set_xlabel("Research Flight")
    ax.set_title("ICE-L — TDL Long-Path MR per Flight")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "04_tdl_long_per_flight_boxplot.png", dpi=150)
    plt.close(fig)

    # ---- Fig 5: data coverage bar chart (valid rows per flight) ------------
    coverage = {
        f: int(df.loc[df["flight"] == f, "MRTDLL_MC"].notna().sum())
        for f in flights
    }

    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ["#d62728" if v == 0 else "steelblue" for v in coverage.values()]
    ax.bar(list(coverage.keys()), list(coverage.values()), color=colors, alpha=0.8)
    ax.set_xlabel("Research Flight")
    ax.set_ylabel("Valid MRTDLL_MC rows (1 Hz)")
    ax.set_title("ICE-L — TDL Long-Path Data Coverage per Flight\n"
                 "(red = RF01 all-zero, short-path unavailable for all flights)")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "05_tdl_coverage_per_flight.png", dpi=150)
    plt.close(fig)

    png_files = sorted(FIGS_DIR.glob("*.png"))
    print(f"\n  {len(png_files)} figure(s) written to {FIGS_DIR}/")
    for f in png_files:
        print(f"    {f.name}")
    update_latest(FIGS_DIR.parent, FIGS_DIR)
    assert len(png_files) >= 5, f"Expected ≥5 figures, got {len(png_files)}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    errno = pytest.main([__file__, "-v"])
    sys.exit(errno)
