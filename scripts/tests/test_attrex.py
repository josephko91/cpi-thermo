"""
ATTREX parser tests — distribution checks, CPI overlap, diagnostics, and figures.

Run from the repo root:
    pytest scripts/tests/test_attrex.py -v
or directly:
    python scripts/tests/test_attrex.py
"""
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Paths (resolved relative to repo root regardless of cwd)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

DATA_DIR = REPO_ROOT / "data/raw/ATTREX"
CPI_CSV = REPO_ROOT / "data/raw/cpi_embeddings_timestamps.csv"
DIAG_DIR = REPO_ROOT / "logs/campaign_tests/attrex"
FIGS_DIR = REPO_ROOT / "figs/campaign_tests/attrex"

DIAG_DIR.mkdir(parents=True, exist_ok=True)
FIGS_DIR.mkdir(parents=True, exist_ok=True)

from parsers.attrex import load_attrex, extract_attrex_standard


# ---------------------------------------------------------------------------
# Module-level cache: load data once across all tests
# ---------------------------------------------------------------------------

def _load_attrex_cached() -> pd.DataFrame:
    if not hasattr(_load_attrex_cached, "_cache"):
        _load_attrex_cached._cache = load_attrex(DATA_DIR)
    return _load_attrex_cached._cache


def _load_std_cached() -> pd.DataFrame:
    if not hasattr(_load_std_cached, "_cache"):
        _load_std_cached._cache = extract_attrex_standard(_load_attrex_cached())
    return _load_std_cached._cache


def _format_exc(e: Exception) -> str:
    return f"{type(e).__name__}: {e}\n" + traceback.format_exc()


# ---------------------------------------------------------------------------
# Basic sanity tests
# ---------------------------------------------------------------------------

def test_attrex_dir_exists():
    assert DATA_DIR.exists() and DATA_DIR.is_dir(), f"ATTREX dir not found: {DATA_DIR}"


def test_instrument_subdirs_present():
    subdirs = {d.name for d in DATA_DIR.iterdir() if d.is_dir()}
    for expected in ("DLH-H2O", "NOAA-H2O", "UCATS-H2O", "MMS"):
        assert expected in subdirs, f"Expected instrument subdir '{expected}' missing in {DATA_DIR}"


def test_load_attrex_returns_dataframe():
    try:
        df = _load_attrex_cached()
    except Exception as e:
        pytest.fail(f"load_attrex raised: {_format_exc(e)}")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_required_columns_present():
    df = _load_attrex_cached()
    required = [
        "Timestamp", "T_C", "Si", "Si_best",
        "Si_DLH", "Si_NOAA", "Si_UCATS",
        "H2O_DLH_ppmv", "H2O_NOAA_ppmv", "H2O_UCATS_ppmv",
        "Campaign",
    ]
    missing = [c for c in required if c not in df.columns]
    assert not missing, f"Missing columns: {missing}"


def test_extract_standard_columns():
    std = _load_std_cached()
    expected = [
        "Timestamp", "Tair_C", "Si", "Si_best",
        "Si_DLH", "Si_NOAA", "Si_UCATS",
        "H2O_DLH_ppmv", "H2O_NOAA_ppmv", "H2O_UCATS_ppmv",
        "Lat", "Lon", "Alt_m", "Campaign", "source_file",
    ]
    missing = [c for c in expected if c not in std.columns]
    assert not missing, f"Missing standardized columns: {missing}"


def test_timestamp_dtype():
    df = _load_attrex_cached()
    assert pd.api.types.is_datetime64_any_dtype(df["Timestamp"]), "Timestamp not datetime"


def test_campaign_label():
    df = _load_attrex_cached()
    assert (df["Campaign"] == "ATTREX").all(), "Campaign column has unexpected values"


# ---------------------------------------------------------------------------
# Physical range / distribution checks
# ---------------------------------------------------------------------------

def test_temperature_physical_range():
    """Air temp at tropical tropopause should be roughly -90 to 30 °C."""
    df = _load_attrex_cached()
    t = df["T_C"].dropna()
    assert len(t) > 0, "No valid temperature values"
    assert t.min() > -100, f"T_C too cold: {t.min():.1f} °C"
    assert t.max() < 60, f"T_C too warm: {t.max():.1f} °C"
    frac_tropo = ((t >= -100) & (t <= 30)).mean()
    assert frac_tropo > 0.9, (
        f"Only {frac_tropo:.1%} of T_C in expected range (-100–30 °C)"
    )


def test_si_values_in_plausible_range():
    """Si should be in [-1, 10] after masking."""
    df = _load_attrex_cached()
    for col in ("Si_DLH", "Si_NOAA", "Si_UCATS", "Si_best"):
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if len(s) == 0:
            continue
        assert s.min() >= -1.01, f"{col} below -1: min={s.min():.4f}"
        assert s.max() <= 10.01, f"{col} above 10: max={s.max():.4f}"


def test_h2o_mixing_ratio_positive():
    """H2O mixing ratios must be positive after sanitization."""
    df = _load_attrex_cached()
    for col in ("H2O_DLH_ppmv", "H2O_NOAA_ppmv", "H2O_UCATS_ppmv"):
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if len(s) == 0:
            continue
        assert (s > 0).all(), f"{col} has non-positive values after sanitization"


def test_si_best_filled_where_any_instrument_valid():
    """Si_best must be non-NaN wherever at least one instrument Si is valid."""
    df = _load_attrex_cached()
    any_valid = df["Si_DLH"].notna() | df["Si_NOAA"].notna() | df["Si_UCATS"].notna()
    n_missed = (any_valid & df["Si_best"].isna()).sum()
    assert n_missed == 0, (
        f"{n_missed:,} rows have a valid instrument Si but Si_best is NaN"
    )


def test_ucats_h2o_data_loaded():
    """UCATS-H2O files should contribute valid measurements."""
    df = _load_attrex_cached()
    assert "H2O_UCATS_ppmv" in df.columns, "H2O_UCATS_ppmv column missing"
    n_valid = df["H2O_UCATS_ppmv"].notna().sum()
    assert n_valid > 0, "No valid UCATS H2O values — check UCATS-H2O subfolder parsing"


def test_all_three_si_instruments_have_data():
    """Verify that each water vapor instrument produced some Si values."""
    df = _load_attrex_cached()
    for col in ("Si_DLH", "Si_NOAA", "Si_UCATS"):
        n = df[col].notna().sum()
        assert n > 0, f"No valid {col} values — {col.replace('Si_', '')} instrument produced no Si"


# ---------------------------------------------------------------------------
# CPI embedding overlap
# ---------------------------------------------------------------------------

def test_cpi_embeddings_file_exists():
    assert CPI_CSV.exists(), f"CPI embeddings CSV not found: {CPI_CSV}"


def test_cpi_overlap():
    """ATTREX parser timestamps must overlap with CPI embedding timestamps."""
    if not CPI_CSV.exists():
        pytest.skip("CPI embeddings CSV not available")

    cpi = pd.read_csv(CPI_CSV, parse_dates=["datetime"])
    attrex_cpi = cpi[cpi["campaign"] == "ATTREX"].copy()
    assert not attrex_cpi.empty, "No ATTREX rows in CPI embeddings CSV"

    df = _load_attrex_cached()
    parser_minutes = (
        pd.to_datetime(df["Timestamp"], utc=True)
        .dt.tz_localize(None)
        .dt.floor("min")
        .unique()
    )
    cpi_minutes = pd.to_datetime(attrex_cpi["datetime"]).dt.floor("min").unique()
    overlap = len(set(parser_minutes) & set(cpi_minutes))
    assert overlap > 0, (
        "No temporal overlap (minute-level) between parsed ATTREX data and CPI embeddings"
    )


# ---------------------------------------------------------------------------
# Diagnostics JSON
# ---------------------------------------------------------------------------

def test_write_diagnostics():
    """Write structured diagnostic summary to logs/campaign_tests/attrex/attrex_diagnostics.json."""
    df = _load_attrex_cached()

    diag: dict = {
        "campaign": "ATTREX",
        "n_total_rows": int(len(df)),
    }

    # Per-instrument H2O and Si stats
    inst_map = {
        "DLH":   ("H2O_DLH_ppmv",   "Si_DLH"),
        "NOAA":  ("H2O_NOAA_ppmv",  "Si_NOAA"),
        "UCATS": ("H2O_UCATS_ppmv", "Si_UCATS"),
    }
    diag["instruments"] = {}
    for inst, (h2o_col, si_col) in inst_map.items():
        h2o_n = int(df[h2o_col].notna().sum()) if h2o_col in df.columns else 0
        si_s = df[si_col].dropna() if si_col in df.columns else pd.Series(dtype=float)
        diag["instruments"][inst] = {
            "h2o_valid_rows": h2o_n,
            "si_valid_rows": int(len(si_s)),
            "si_mean":   float(si_s.mean())           if len(si_s) else None,
            "si_std":    float(si_s.std())            if len(si_s) else None,
            "si_p05":    float(si_s.quantile(0.05))   if len(si_s) else None,
            "si_median": float(si_s.median())         if len(si_s) else None,
            "si_p95":    float(si_s.quantile(0.95))   if len(si_s) else None,
        }

    # Si_best summary
    sb = df["Si_best"].dropna()
    diag["si_best"] = {
        "valid_rows": int(len(sb)),
        "mean":               float(sb.mean())           if len(sb) else None,
        "std":                float(sb.std())            if len(sb) else None,
        "p05":                float(sb.quantile(0.05))   if len(sb) else None,
        "median":             float(sb.median())         if len(sb) else None,
        "p95":                float(sb.quantile(0.95))   if len(sb) else None,
        "frac_supersaturated": float((sb > 0).mean())   if len(sb) else None,
    }

    # Temperature summary
    t = df["T_C"].dropna()
    diag["temperature_C"] = {
        "valid_rows": int(len(t)),
        "mean": float(t.mean()) if len(t) else None,
        "std":  float(t.std())  if len(t) else None,
        "min":  float(t.min())  if len(t) else None,
        "max":  float(t.max())  if len(t) else None,
    }

    # Temporal coverage
    ts = pd.to_datetime(df["Timestamp"], utc=True).dt.tz_localize(None)
    diag["temporal_coverage"] = {
        "start": str(ts.min().date()),
        "end":   str(ts.max().date()),
        "n_flight_dates": int(ts.dt.date.nunique()),
    }

    # CPI overlap stats
    if CPI_CSV.exists():
        cpi = pd.read_csv(CPI_CSV, parse_dates=["datetime"])
        acpi = cpi[cpi["campaign"] == "ATTREX"]
        parser_min = ts.dt.floor("min").unique()
        cpi_min = pd.to_datetime(acpi["datetime"]).dt.floor("min").unique()
        overlap = len(set(parser_min) & set(cpi_min))
        diag["cpi_overlap"] = {
            "n_cpi_images": int(len(acpi)),
            "n_cpi_unique_minutes": int(len(cpi_min)),
            "n_parser_unique_minutes": int(len(parser_min)),
            "n_overlapping_minutes": overlap,
            "overlap_frac_of_cpi": (
                float(overlap / len(cpi_min)) if len(cpi_min) else 0.0
            ),
        }
    else:
        diag["cpi_overlap"] = None

    out = DIAG_DIR / "attrex_diagnostics.json"
    out.write_text(json.dumps(diag, indent=2, default=str))
    print(f"\n  Diagnostics → {out}")

    assert diag["n_total_rows"] > 0
    assert diag["si_best"]["valid_rows"] > 0
    assert diag["temperature_C"]["valid_rows"] > 0


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def test_generate_figures():
    """Generate and save diagnostic figures to figs/campaign_tests/attrex/."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    df = _load_attrex_cached()
    ts = pd.to_datetime(df["Timestamp"], utc=True)

    # 1. Temperature distribution ----------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    t = df["T_C"].dropna()
    ax.hist(t, bins=80, color="steelblue", edgecolor="none", alpha=0.85)
    ax.axvline(t.median(), color="firebrick", linestyle="--",
               label=f"Median {t.median():.1f} °C")
    ax.set_xlabel("Air Temperature (°C)")
    ax.set_ylabel("Count")
    ax.set_title("ATTREX — Air Temperature Distribution")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "01_temperature_distribution.png", dpi=150)
    plt.close(fig)

    # 2. Si distributions — all instruments --------------------------------
    si_specs = [
        ("Si_DLH",   "DLH-H2O",   "C0"),
        ("Si_NOAA",  "NOAA-H2O",  "C1"),
        ("Si_UCATS", "UCATS-H2O", "C2"),
        ("Si_best",  "Si best",   "k"),
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(-1, 1.5, 120)
    for col, label, color in si_specs:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if len(s) == 0:
            continue
        ax.hist(s, bins=bins, alpha=0.5, color=color, density=True,
                edgecolor="none", label=f"{label}  n={len(s):,}")
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Ice Supersaturation (Si)")
    ax.set_ylabel("Density")
    ax.set_title("ATTREX — Si Distributions by Instrument")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "02_si_distributions.png", dpi=150)
    plt.close(fig)

    # 3. Si cross-instrument scatter ----------------------------------------
    pairs = [
        ("Si_DLH",  "Si_NOAA"),
        ("Si_DLH",  "Si_UCATS"),
        ("Si_NOAA", "Si_UCATS"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    lim = (-1, 1.5)
    for ax, (xcol, ycol) in zip(axes, pairs):
        if xcol not in df.columns or ycol not in df.columns:
            ax.set_visible(False)
            continue
        sub = df[[xcol, ycol]].dropna()
        ax.scatter(sub[xcol], sub[ycol], s=1, alpha=0.25,
                   color="steelblue", rasterized=True)
        ax.plot(lim, lim, "r--", linewidth=0.8)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)
        ax.set_title(f"n={len(sub):,}")
    fig.suptitle("ATTREX — Si Cross-Instrument Comparison")
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "03_si_cross_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 4. H2O mixing ratio distributions (log scale) ------------------------
    h2o_specs = [
        ("H2O_DLH_ppmv",   "DLH-H2O",   "C0"),
        ("H2O_NOAA_ppmv",  "NOAA-H2O",  "C1"),
        ("H2O_UCATS_ppmv", "UCATS-H2O", "C2"),
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    for col, label, color in h2o_specs:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if len(s) == 0:
            continue
        ax.hist(np.log10(s.clip(lower=0.01)), bins=80, alpha=0.5, color=color,
                density=True, edgecolor="none", label=f"{label}  n={len(s):,}")
    ax.set_xlabel("log₁₀ H₂O (ppmv)")
    ax.set_ylabel("Density")
    ax.set_title("ATTREX — H₂O Mixing Ratio Distributions")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "04_h2o_distributions.png", dpi=150)
    plt.close(fig)

    # 5. Si time series by instrument --------------------------------------
    ts_utc = ts.dt.tz_localize(None)
    fig, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
    coverage_cols = [
        ("Si_DLH",   "DLH-H2O",   "C0"),
        ("Si_NOAA",  "NOAA-H2O",  "C1"),
        ("Si_UCATS", "UCATS-H2O", "C2"),
        ("Si_best",  "Si best",   "k"),
    ]
    for ax, (col, label, color) in zip(axes, coverage_cols):
        if col not in df.columns:
            ax.set_ylabel(label, fontsize=8)
            continue
        mask = df[col].notna()
        ax.scatter(ts_utc[mask], df.loc[mask, col],
                   s=0.5, alpha=0.35, color=color, rasterized=True)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.6)
        ax.set_ylabel(label, fontsize=8)
        ax.set_ylim(-1, 1.5)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate(rotation=30)
    fig.suptitle("ATTREX — Si Time Series by Instrument")
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "05_si_time_series.png", dpi=150)
    plt.close(fig)

    # 6. Geographic coverage -----------------------------------------------
    if "Lat" in df.columns and "Lon" in df.columns:
        pos = df[["Lat", "Lon", "Si_best"]].dropna(subset=["Lat", "Lon"])
        if not pos.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            sc = ax.scatter(
                pos["Lon"], pos["Lat"],
                c=pos["Si_best"].fillna(0),
                cmap="RdBu_r", vmin=-0.5, vmax=0.5,
                s=1, alpha=0.5, rasterized=True,
            )
            plt.colorbar(sc, ax=ax, label="Si_best")
            ax.set_xlabel("Longitude (°)")
            ax.set_ylabel("Latitude (°)")
            ax.set_title("ATTREX — Geographic Coverage (colored by Si_best)")
            fig.tight_layout()
            fig.savefig(FIGS_DIR / "06_geographic_coverage.png", dpi=150)
            plt.close(fig)

    # 7. CPI overlap timeline ----------------------------------------------
    if CPI_CSV.exists():
        cpi = pd.read_csv(CPI_CSV, parse_dates=["datetime"])
        acpi = cpi[cpi["campaign"] == "ATTREX"].copy()
        acpi["date"] = pd.to_datetime(acpi["datetime"]).dt.date
        daily_cpi = acpi.groupby("date").size().rename("cpi_images")

        df_dates = ts_utc.dt.date
        daily_parser = df_dates.value_counts().sort_index().rename("parser_rows")

        combined = pd.DataFrame({"cpi": daily_cpi, "parser": daily_parser / 100}).fillna(0)
        combined.index = pd.to_datetime(combined.index)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax2 = ax.twinx()
        ax.bar(combined.index, combined["cpi"], width=0.8, color="C3",
               alpha=0.7, label="CPI images")
        ax2.plot(combined.index, combined["parser"], color="steelblue",
                 alpha=0.8, linewidth=1.5, label="Parser rows (÷100)")
        ax.set_ylabel("CPI image count", color="C3")
        ax2.set_ylabel("Parser rows ÷ 100", color="steelblue")
        ax.set_xlabel("Date")
        ax.set_title("ATTREX — CPI Image Count vs Parser Coverage (daily)")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        fig.autofmt_xdate(rotation=30)
        fig.tight_layout()
        fig.savefig(FIGS_DIR / "07_cpi_overlap_timeline.png", dpi=150)
        plt.close(fig)

    png_files = sorted(FIGS_DIR.glob("*.png"))
    print(f"\n  {len(png_files)} figure(s) written to {FIGS_DIR}/")
    for f in png_files:
        print(f"    {f.name}")
    assert len(png_files) > 0, "No figures were generated"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    errno = pytest.main([__file__, "-v"])
    sys.exit(errno)
