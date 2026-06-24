"""
IPHEX water vapor instrument diagnostic test.

Compares the two water vapor sources on the UND Citation II (N555DS):

  chilled-mirror : FrostPoint column (°C), frost point from the chilled
                   mirror hygrometer. Available in ~21 of 32 flights.
                   Si_chilled_mirror via Tetens ice formula (existing parser).

  TDL (Ophir)    : MixingRatio column (ppmv), from the Ophir tunable diode
                   laser hygrometer listed in the file header. Same flight
                   coverage as FrostPoint.

  DEWPT          : Dew point temperature (°C) from a standard sensor, present
                   in ALL 32 flights. Treated as an approximate frost point
                   (valid below 0°C; used for coverage gap-filling).
                   Si_DEWPT via Tetens ice formula applied to dew point.

  CSI_M_Ratio    : Absent (all NaN) in every file — not deployed.

Key findings
------------
- FrostPoint and MixingRatio share the same flight coverage (~21 flights).
- DEWPT fills coverage for the 11 early flights that lack FrostPoint/TDL.
- CSI instrument produced no data.

Run from the repo root:
    pytest scripts/tests/test_iphex.py -v
or directly:
    python scripts/tests/test_iphex.py
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

DATA_DIR  = REPO_ROOT / "data/raw/IPHEX"
DIAG_DIR  = REPO_ROOT / "logs/diagnostics"
FIGS_DIR  = REPO_ROOT / "figs/iphex"

DIAG_DIR.mkdir(parents=True, exist_ok=True)
FIGS_DIR.mkdir(parents=True, exist_ok=True)

# Ophir TDL physical range (ppmv)
_TDL_PPMV_MIN = 1.0
_TDL_PPMV_MAX = 50_000.0

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _flight_label(filepath: Path) -> str:
    """YYYY-MM-DD HH:MM from filename YYYY_MM_DD_HH_MM_SS.iphex."""
    p = filepath.stem.split("_")
    if len(p) >= 6:
        return f"{p[0]}-{p[1]}-{p[2]} {p[3]}:{p[4]}"
    return filepath.stem


def _es_ice_tetens(temp_c: np.ndarray) -> np.ndarray:
    return 6.112 * np.exp((22.46 * temp_c) / (272.62 + temp_c))


def _si_from_frost_point(fp_c: np.ndarray, t_c: np.ndarray) -> np.ndarray:
    e  = _es_ice_tetens(fp_c)
    ei = _es_ice_tetens(t_c)
    with np.errstate(divide="ignore", invalid="ignore"):
        si = e / ei - 1.0
    si[~np.isfinite(si)] = np.nan
    return si


def _si_from_ppmv(ppmv: np.ndarray, t_c: np.ndarray, p_hpa: np.ndarray) -> np.ndarray:
    """Si w.r.t. ice from mixing ratio (ppmv), T (°C), P (hPa)."""
    t_k = t_c + 273.15
    invalid = (
        ~np.isfinite(ppmv) | (ppmv < _TDL_PPMV_MIN) | (ppmv > _TDL_PPMV_MAX)
        | ~np.isfinite(t_k)   | (t_k < 150) | (t_k > 350)
        | ~np.isfinite(p_hpa) | (p_hpa <= 0)
    )
    e_s = _es_ice_tetens(t_c)
    with np.errstate(divide="ignore", invalid="ignore"):
        e   = (ppmv / 1e6) * p_hpa
        si  = e / e_s - 1.0
    si[invalid] = np.nan
    si[~np.isfinite(si)] = np.nan
    return si


def _load_all() -> pd.DataFrame:
    """Load all IPHEX files, compute per-source Si columns, cache result."""
    if not hasattr(_load_all, "_cache"):
        from parsers.iphex import load_iphex_file, _coerce_and_mask

        files = sorted(
            f for f in DATA_DIR.glob("*.iphex")
            if f.is_file() and "Combined" not in f.name
        )
        if not files:
            raise FileNotFoundError(f"No *.iphex files found in {DATA_DIR}")

        frames = []
        for f in files:
            try:
                df = load_iphex_file(f)
            except Exception as exc:
                print(f"  Warning: {f.name}: {exc}")
                continue

            label = _flight_label(f)
            df["flight"] = label

            # --- Chilled-mirror (FrostPoint) ---------------------------------
            if "FrostPoint" in df.columns:
                fp  = df["FrostPoint"].to_numpy(dtype=float)
                t   = df["Air_Temp"].to_numpy(dtype=float)
                si  = _si_from_frost_point(fp, t)
                si[(si < -1) | (si > 5)] = np.nan
                df["Si_chilled_mirror"] = si
            else:
                df["Si_chilled_mirror"] = np.nan

            # --- TDL / Ophir (MixingRatio ppmv) ------------------------------
            if "MixingRatio" in df.columns:
                mr  = _coerce_and_mask(df["MixingRatio"]).to_numpy(dtype=float)
                t   = df["Air_Temp"].to_numpy(dtype=float)
                p   = df["STATIC_PR"].to_numpy(dtype=float)
                si  = _si_from_ppmv(mr, t, p)
                si[(si < -1) | (si > 5)] = np.nan
                df["Si_TDL"]         = si
                df["MixingRatio_ppmv"] = mr
            else:
                df["Si_TDL"]         = np.nan
                df["MixingRatio_ppmv"] = np.nan

            # --- DEWPT fallback (dew point as approximate frost point) -------
            if "DEWPT" in df.columns:
                dp  = _coerce_and_mask(df["DEWPT"]).to_numpy(dtype=float)
                t   = df["Air_Temp"].to_numpy(dtype=float)
                si  = _si_from_frost_point(dp, t)
                si[(si < -1) | (si > 5)] = np.nan
                df["Si_DEWPT"] = si
            else:
                df["Si_DEWPT"] = np.nan

            frames.append(df)

        combined = pd.concat(frames, ignore_index=True)
        _load_all._cache = combined

    return _load_all._cache


def _format_exc(e: Exception) -> str:
    return f"{type(e).__name__}: {e}\n" + traceback.format_exc()


# ---------------------------------------------------------------------------
# Sanity tests
# ---------------------------------------------------------------------------

def test_data_dir_exists():
    assert DATA_DIR.exists() and DATA_DIR.is_dir(), f"IPHEX data dir not found: {DATA_DIR}"


def test_iphex_files_present():
    files = [f for f in DATA_DIR.glob("*.iphex") if "Combined" not in f.name]
    assert len(files) > 0, f"No .iphex files in {DATA_DIR}"
    assert len(files) == 32, f"Expected 32 flight files, found {len(files)}"


def test_load_returns_dataframe():
    try:
        df = _load_all()
    except Exception as e:
        pytest.fail(f"_load_all raised: {_format_exc(e)}")
    assert isinstance(df, pd.DataFrame) and not df.empty


def test_required_columns():
    df = _load_all()
    for col in ("Air_Temp", "STATIC_PR", "DEWPT", "Si", "Timestamp", "flight",
                "Si_chilled_mirror", "Si_TDL", "Si_DEWPT"):
        assert col in df.columns, f"Column {col!r} missing"


# ---------------------------------------------------------------------------
# Instrument coverage tests
# ---------------------------------------------------------------------------

def test_dewpt_present_all_flights():
    """DEWPT must have valid data in every flight."""
    df = _load_all()
    missing = []
    for flight, grp in df.groupby("flight"):
        if grp["DEWPT"].notna().sum() == 0:
            missing.append(str(flight))
    assert not missing, f"Flights missing DEWPT data: {missing}"


def test_frostpoint_partial_coverage():
    """FrostPoint present in some but not all flights — document gap."""
    df = _load_all()
    n_flights = df["flight"].nunique()
    flights_with_fp = df.groupby("flight")["Si_chilled_mirror"].apply(
        lambda s: s.notna().sum() > 0
    ).sum()
    print(f"\n  FrostPoint (chilled mirror): {flights_with_fp}/{n_flights} flights")
    assert flights_with_fp > 0, "FrostPoint has no valid Si in any flight"
    assert flights_with_fp < n_flights, (
        "FrostPoint now in all flights — update this test if instrument coverage changed"
    )


def test_tdl_same_coverage_as_frostpoint():
    """Ophir TDL (MixingRatio) has the same flight coverage as FrostPoint."""
    df = _load_all()
    fp_flights  = set(df.groupby("flight")["Si_chilled_mirror"].apply(
        lambda g: g.notna().any()
    ).loc[lambda s: s].index)
    tdl_flights = set(df.groupby("flight")["Si_TDL"].apply(
        lambda g: g.notna().any()
    ).loc[lambda s: s].index)
    print(f"\n  FrostPoint flights : {len(fp_flights)}")
    print(f"  TDL flights        : {len(tdl_flights)}")
    # They should largely overlap; accept ≤2 flight difference
    sym_diff = fp_flights.symmetric_difference(tdl_flights)
    assert len(sym_diff) <= 2, (
        f"FrostPoint and TDL coverage differ by {len(sym_diff)} flights: {sym_diff}"
    )


def test_csi_absent():
    """CSI_M_Ratio (not deployed) must be entirely NaN."""
    df = _load_all()
    if "CSI_M_Ratio" not in df.columns:
        return
    n = df["CSI_M_Ratio"].notna().sum()
    assert n == 0, (
        f"CSI_M_Ratio unexpectedly has {n} valid values — "
        "update this test if the CSI instrument becomes active."
    )


def test_si_values_plausible():
    """All Si columns must lie in [-1, 5] after masking."""
    df = _load_all()
    for col in ("Si_chilled_mirror", "Si_TDL", "Si_DEWPT"):
        s = df[col].dropna()
        if len(s) == 0:
            continue
        assert s.min() >= -1.01, f"{col} below -1: min={s.min():.4f}"
        assert s.max() <= 5.01,  f"{col} above 5: max={s.max():.4f}"


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def test_write_diagnostics():
    """Write per-flight and overall diagnostics to logs/diagnostics/."""
    df = _load_all()

    diag: dict = {
        "campaign":   "IPHEX",
        "aircraft":   "UND Citation II (N555DS)",
        "instruments": {
            "chilled_mirror": "FrostPoint column (°C) — chilled mirror hygrometer",
            "TDL_ophir":      "MixingRatio column (ppmv) — Ophir tunable diode laser",
            "DEWPT":          "DEWPT column (°C) — dew point, all flights, approx frost point",
            "CSI_M_Ratio":    "absent — not deployed",
        },
        "n_total_rows": int(len(df)),
        "n_flights":    int(df["flight"].nunique()),
        "overall":      {},
        "per_flight":   {},
    }

    # Overall Si stats for each source
    for col, label in [
        ("Si_chilled_mirror", "chilled_mirror"),
        ("Si_TDL",            "TDL_ophir"),
        ("Si_DEWPT",          "DEWPT"),
    ]:
        s = df[col].dropna()
        diag["overall"][label] = {
            "n_valid":  int(len(s)),
            "n_flights_with_data": int(
                df.groupby("flight")[col].apply(lambda g: g.notna().any()).sum()
            ),
            "mean":    float(s.mean())           if len(s) else None,
            "std":     float(s.std())            if len(s) else None,
            "p05":     float(s.quantile(0.05))   if len(s) else None,
            "median":  float(s.median())         if len(s) else None,
            "p95":     float(s.quantile(0.95))   if len(s) else None,
            "frac_supersaturated": float((s > 0).mean()) if len(s) else None,
        }

    diag["overall"]["CSI_M_Ratio"] = {
        "n_valid": 0,
        "note": "instrument not deployed — all NaN",
    }

    # Per-flight stats
    for flight, grp in df.groupby("flight"):
        entry: dict = {}
        for col, label in [
            ("Si_chilled_mirror", "Si_chilled_mirror"),
            ("Si_TDL",            "Si_TDL"),
            ("Si_DEWPT",          "Si_DEWPT"),
            ("MixingRatio_ppmv",  "MixingRatio_ppmv"),
            ("Air_Temp",          "Air_Temp_C"),
        ]:
            s = grp[col].dropna() if col in grp.columns else pd.Series(dtype=float)
            entry[label] = {
                "n_valid": int(len(s)),
                "mean":    float(s.mean())   if len(s) else None,
                "median":  float(s.median()) if len(s) else None,
                "std":     float(s.std())    if len(s) else None,
            }
        diag["per_flight"][str(flight)] = entry

    out_json = DIAG_DIR / "iphex_diagnostics.json"
    out_json.write_text(json.dumps(diag, indent=2, default=str))
    print(f"\n  Diagnostics → {out_json}")

    out_csv = DIAG_DIR / "iphex_diagnostics.csv"
    df[[
        "Timestamp", "flight", "Air_Temp", "STATIC_PR",
        "DEWPT", "FrostPoint" if "FrostPoint" in df.columns else "DEWPT",
        "MixingRatio_ppmv",
        "Si_chilled_mirror", "Si_TDL", "Si_DEWPT", "Si",
    ]].to_csv(out_csv, index=False)
    print(f"  Diagnostics CSV → {out_csv}")

    assert out_json.exists()
    assert out_csv.exists()


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def test_generate_figures():
    """
    Generate IPHEX water vapor diagnostic figures to figs/iphex/.

      01 — Instrument coverage bar chart (valid rows per flight, per source)
      02 — Si distributions: chilled mirror vs TDL vs DEWPT
      03 — Scatter: Si_chilled_mirror vs Si_TDL, colored by temperature
      04 — Scatter: MixingRatio (ppmv) vs FrostPoint (°C), colored by temperature
      05 — Per-flight Si boxplots (chilled mirror)
      06 — Per-flight Si boxplots (TDL)
      07 — Si time series all sources
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    df   = _load_all()
    flt  = sorted(df["flight"].unique())

    t_all  = df["Air_Temp"].dropna()
    t_norm = Normalize(
        vmin=float(np.percentile(t_all, 2)),
        vmax=float(np.percentile(t_all, 98)),
    )
    cmap_t = plt.cm.RdYlBu_r

    # ---- Fig 1: coverage bar chart -----------------------------------------
    sources = {
        "DEWPT":           "Si_DEWPT",
        "FrostPoint\n(CM)": "Si_chilled_mirror",
        "MixingRatio\n(TDL Ophir)": "Si_TDL",
    }
    fig, ax = plt.subplots(figsize=(14, 4))
    x    = np.arange(len(flt))
    w    = 0.25
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for i, (src_label, col) in enumerate(sources.items()):
        counts = [
            df.loc[df["flight"] == f, col].notna().sum()
            for f in flt
        ]
        ax.bar(x + i * w, counts, w, label=src_label,
               color=colors[i], alpha=0.8)
    ax.set_xticks(x + w)
    ax.set_xticklabels([f.replace(" ", "\n") for f in flt], fontsize=6, rotation=0)
    ax.set_ylabel("Valid rows (1 Hz)")
    ax.set_xlabel("Flight")
    ax.set_title("IPHEX — Water Vapor Instrument Coverage per Flight")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "01_coverage_per_flight.png", dpi=150)
    plt.close(fig)

    # ---- Fig 2: Si distributions -------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(-1, 1.5, 100)
    specs = [
        ("Si_DEWPT",          "DEWPT (approx frost pt)", "#1f77b4"),
        ("Si_chilled_mirror", "FrostPoint (CM)",         "#ff7f0e"),
        ("Si_TDL",            "MixingRatio (TDL Ophir)", "#2ca02c"),
    ]
    for col, label, color in specs:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        ax.hist(s, bins=bins, alpha=0.5, density=True, color=color,
                edgecolor="none", label=f"{label}  n={len(s):,}")
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Ice Supersaturation (Si)")
    ax.set_ylabel("Density")
    ax.set_title("IPHEX — Si Distributions by Instrument")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "02_si_distributions.png", dpi=150)
    plt.close(fig)

    # ---- Fig 3: Si_chilled_mirror vs Si_TDL scatter -----------------------
    sub = df[["Si_chilled_mirror", "Si_TDL", "Air_Temp"]].dropna()
    fig, ax = plt.subplots(figsize=(6, 5))
    if len(sub) > 0:
        sc = ax.scatter(
            sub["Si_chilled_mirror"], sub["Si_TDL"],
            c=sub["Air_Temp"], cmap=cmap_t, norm=t_norm,
            s=2, alpha=0.4, rasterized=True,
        )
        lim = (-1, 1.5)
        ax.plot(lim, lim, "k--", linewidth=0.8, label="1:1")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        plt.colorbar(sc, ax=ax, label="Air Temp (°C)")
        ax.legend(fontsize=8)
    ax.set_xlabel("Si_chilled_mirror (FrostPoint)")
    ax.set_ylabel("Si_TDL (MixingRatio, Ophir)")
    ax.set_title(f"IPHEX — Si Cross-Instrument Comparison (n={len(sub):,})")
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "03_si_cm_vs_tdl_scatter.png", dpi=150)
    plt.close(fig)

    # ---- Fig 4: MixingRatio (ppmv) vs FrostPoint (°C) scatter -------------
    sub4 = df[["MixingRatio_ppmv", "FrostPoint", "Air_Temp"]].dropna() \
        if "FrostPoint" in df.columns \
        else pd.DataFrame()
    fig, ax = plt.subplots(figsize=(6, 5))
    if len(sub4) > 0:
        sc = ax.scatter(
            sub4["FrostPoint"], sub4["MixingRatio_ppmv"],
            c=sub4["Air_Temp"], cmap=cmap_t, norm=t_norm,
            s=2, alpha=0.4, rasterized=True,
        )
        plt.colorbar(sc, ax=ax, label="Air Temp (°C)")
    ax.set_xlabel("FrostPoint (°C)  [Chilled Mirror]")
    ax.set_ylabel("MixingRatio (ppmv)  [Ophir TDL]")
    ax.set_title(f"IPHEX — FrostPoint vs TDL MixingRatio (n={len(sub4):,})")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "04_frostpoint_vs_mixingratio.png", dpi=150)
    plt.close(fig)

    # ---- Fig 5 & 6: per-flight Si boxplots ---------------------------------
    for fignum, col, title_inst in [
        (5, "Si_chilled_mirror", "Chilled Mirror (FrostPoint)"),
        (6, "Si_TDL",            "TDL Ophir (MixingRatio ppmv)"),
    ]:
        plot_data   = []
        plot_labels = []
        for f in flt:
            vals = df.loc[df["flight"] == f, col].dropna().values
            if len(vals) > 0:
                plot_data.append(vals)
                plot_labels.append(f.replace(" ", "\n"))

        if not plot_data:
            continue

        fig, ax = plt.subplots(figsize=(max(10, len(plot_data) * 0.55), 4))
        bp = ax.boxplot(
            plot_data, tick_labels=plot_labels, patch_artist=True,
            medianprops={"color": "firebrick", "linewidth": 1.5},
            flierprops={"marker": ".", "markersize": 2, "alpha": 0.3},
        )
        for patch in bp["boxes"]:
            patch.set_facecolor("steelblue")
            patch.set_alpha(0.6)
        ax.axhline(0, color="k",    linestyle="--", linewidth=0.8)
        ax.axhline(0.1, color="gray", linestyle=":", linewidth=0.6)
        ax.set_ylabel("Si")
        ax.set_xlabel("Flight")
        ax.set_title(f"IPHEX — Si per Flight  [{title_inst}]")
        ax.set_ylim(-1.1, 1.6)
        ax.tick_params(axis="x", labelsize=6)
        fig.tight_layout()
        fig.savefig(FIGS_DIR / f"0{fignum}_si_boxplot_{col}.png", dpi=150)
        plt.close(fig)

    # ---- Fig 7: Si time series — all sources --------------------------------
    ts = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce").dt.tz_localize(None)
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    src_specs = [
        ("Si_DEWPT",          "DEWPT (approx frost pt)", "#1f77b4"),
        ("Si_chilled_mirror", "FrostPoint (CM)",         "#ff7f0e"),
        ("Si_TDL",            "MixingRatio (TDL Ophir)", "#2ca02c"),
    ]
    for ax, (col, label, color) in zip(axes, src_specs):
        mask = df[col].notna()
        ax.scatter(
            ts[mask], df.loc[mask, col],
            s=0.5, alpha=0.3, color=color, rasterized=True,
        )
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.6)
        ax.set_ylabel(label, fontsize=8)
        ax.set_ylim(-1.1, 1.6)
    import matplotlib.dates as mdates
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate(rotation=30)
    fig.suptitle("IPHEX — Si Time Series by Instrument")
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "07_si_time_series.png", dpi=150)
    plt.close(fig)

    png_files = sorted(FIGS_DIR.glob("*.png"))
    print(f"\n  {len(png_files)} figure(s) written to {FIGS_DIR}/")
    for f in png_files:
        print(f"    {f.name}")
    assert len(png_files) >= 7, f"Expected ≥7 figures, got {len(png_files)}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    errno = pytest.main([__file__, "-v"])
    sys.exit(errno)
