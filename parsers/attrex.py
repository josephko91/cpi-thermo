"""
ATTREX (Airborne Tropical TRopopause EXperiment) campaign data parser.

Campaign: ATTREX Global Hawk aircraft
- Data Source: https://espoarchive.nasa.gov/archive/browse/attrex/id4/GHawk
- Data Format: NASA ICARTT (.ict) files
- Water vapor instruments: DLH-H2O, NOAA-H2O, UCATS-H2O
- Meteorological data: MMS (Meteorological Measurement System)

Notes
-----
ATTREX data is split across multiple instruments in separate .ict files.
Temperature and pressure come from the MMS instrument, while water vapor
comes from DLH-H2O, NOAA-H2O, and/or UCATS-H2O. These must be merged
across instruments (via merge_asof on datetime_utc) before Si can be
computed.

MMS raw values for temperature and pressure are scaled by 0.01
(i.e., stored as integers; multiply by 0.01 to get Kelvin / hPa).

Per-instrument Si columns: Si_DLH, Si_NOAA, Si_UCATS.
Si_best is the first non-NaN value from the instrument ranking defined in
config.yaml (attrex.h2o_ranking). Default order: DLH > NOAA > UCATS.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Dict, List, Optional
from collections import defaultdict

from .utils import round_timestamp_to_second, si_from_ppmv, qv_from_ppmv, sw_from_si, edr_from_mms_log10kWkg


# ---------------------------------------------------------------------------
# Instrument-specific time column names
# ---------------------------------------------------------------------------
ATTREX_TIME_COLS: Dict[str, str] = {
    "DLH-H2O": "Time_UTC",
    "NOAA-H2O": "NW_UTC_s",
    "MMS": "TIME_UTC",
    "UCATS-H2O": "Start_UTC",
}

# Fallback patterns (checked in order) when the canonical name is missing
_TIME_FALLBACKS = ["Time_Start", "Time_Mid", "time"]

# Missing value flags used across ATTREX .ict files
# -99999 is used by UCATS-H2O; -9999 variants by DLH/NOAA/MMS
ATTREX_MISSING_FLAGS = [-99999, -99999.0, -9999, -9999.99, -7777, -7777.77, -8888, -8888.88]

# Default H2O instrument ranking for Si_best (overridden via config.yaml attrex.h2o_ranking)
_DEFAULT_H2O_RANKING: List[str] = ["DLH", "NOAA", "UCATS"]

# Maps ranking key → (column-detection predicate label, detection logic)
# Detection is done at runtime against the merged DataFrame's column list.
_H2O_INSTRUMENT_KEYS = {
    "DLH": "dlh",
    "NOAA": "noaa",
    "UCATS": "ucats",
}


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _replace_invalid_values(value: str) -> float:
    """Replace invalid / missing-flag values during CSV parsing."""
    try:
        val = float(value)
        if val in ATTREX_MISSING_FLAGS:
            return np.nan
        return val
    except (ValueError, TypeError):
        return np.nan


def _infer_instrument(filepath: Path) -> str:
    """
    Infer the instrument name from the parent directory or filename.

    The ATTREX archive is typically organized as::

        ATTREX/
            DLH-H2O/
                DLH-H2O_*.ict
            NOAA-H2O/
                NOAA-H2O_*.ict
            MMS/ (or MMS-1HZ/)
                MMS-1HZ_*.ict

    Returns one of the keys in ATTREX_TIME_COLS, or the raw parent name.
    """
    parent = filepath.parent.name
    stem = filepath.stem.upper()
    for key in ATTREX_TIME_COLS:
        if key.upper() in parent.upper() or key.upper().replace("-", "") in stem.replace("-", ""):
            return key
    # Check for MMS-1HZ variant
    if "MMS" in parent.upper() or "MMS" in stem:
        return "MMS"
    return parent


def _parse_ict_file(
    filepath: Path,
    time_col: str = "Time_Start",
    prefix: Optional[str] = None,
) -> pd.DataFrame:
    """
    Parse a NASA ICARTT format (.ict) file.

    Parameters
    ----------
    filepath : Path
        Path to the .ict file.
    time_col : str
        Preferred time column name for this instrument.
    prefix : str, optional
        If provided, non-time data columns are prefixed (e.g. ``"DLH-H2O_"``).

    Returns
    -------
    pd.DataFrame
        Parsed data with a ``datetime_utc`` column.
    """
    with open(filepath, "r") as f:
        first_line = f.readline().strip()
        n_header_lines = int(first_line.split(",")[0])

        f.seek(0)
        header_lines = [f.readline().strip() for _ in range(n_header_lines)]

    # --- flight date from header (ICARTT: line 7 is "YYYY, MM, DD, …") ---
    flight_date = None
    for line in header_lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            try:
                year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
                if 1990 < year < 2100 and 1 <= month <= 12 and 1 <= day <= 31:
                    flight_date = pd.Timestamp(year=year, month=month, day=day)
                    break
            except ValueError:
                continue

    if flight_date is None:
        raise ValueError(f"Could not find flight date in header of {filepath.name}")

    # --- column names from last header line ---
    col_line = header_lines[-1]
    columns = [col.strip() for col in col_line.split(",") if col.strip()]

    # --- read data ---
    df = pd.read_csv(
        filepath,
        skiprows=n_header_lines,
        names=columns,
        converters={col: _replace_invalid_values for col in columns},
        skipinitialspace=True,
        on_bad_lines="skip",
    )
    df.columns = df.columns.str.strip()

    # --- replace remaining missing-flag values ---
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].replace(ATTREX_MISSING_FLAGS, np.nan)

    # --- create datetime_utc from the time column ---
    used_time_col = None
    if time_col in df.columns:
        used_time_col = time_col
    else:
        # Try fallbacks
        for fb in _TIME_FALLBACKS:
            if fb in df.columns:
                used_time_col = fb
                break
        if used_time_col is None:
            # Try any column with 'time' or 'utc' in its name
            time_candidates = [c for c in df.columns if "time" in c.lower() or "utc" in c.lower()]
            if time_candidates:
                used_time_col = time_candidates[0]

    if used_time_col is not None:
        df["datetime_utc"] = flight_date + pd.to_timedelta(df[used_time_col], unit="s")
    else:
        # Use the first column as seconds-of-day
        df["datetime_utc"] = flight_date + pd.to_timedelta(df.iloc[:, 0], unit="s")
    df["datetime_utc"] = round_timestamp_to_second(df["datetime_utc"])

    # --- optionally prefix data columns (not datetime_utc) ---
    if prefix:
        rename_map = {}
        for c in df.columns:
            if c == "datetime_utc":
                continue
            # Don't double-prefix
            if not c.startswith(prefix):
                rename_map[c] = f"{prefix}_{c}"
        df.rename(columns=rename_map, inplace=True)

    df["source_file"] = filepath.name
    return df


# ---------------------------------------------------------------------------
# Cross-instrument merge
# ---------------------------------------------------------------------------

def _combine_ict_files(
    files: List[Path],
) -> pd.DataFrame:
    """
    Parse all .ict files, group by instrument, and merge via an exact-second
    outer join on ``datetime_utc`` -- no merge_asof tolerance. A second
    reported by only one instrument becomes a row with that instrument's
    columns filled and the rest NaN; the row grid is the union of every
    instrument's own (floored-to-the-second) timestamps.

    Parameters
    ----------
    files : list of Path
        All .ict files found under the ATTREX data directory.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with columns from all instruments and ``datetime_utc``.
    """
    # Group files by instrument
    instrument_files: Dict[str, List[Path]] = defaultdict(list)
    for f in files:
        inst = _infer_instrument(f)
        instrument_files[inst].append(f)

    # Parse and concatenate within each instrument
    instrument_dfs: Dict[str, pd.DataFrame] = {}
    for inst, inst_files in instrument_files.items():
        time_col = ATTREX_TIME_COLS.get(inst, "Time_Start")
        prefix = inst
        parsed = []
        for fp in sorted(inst_files):
            try:
                parsed.append(_parse_ict_file(fp, time_col=time_col, prefix=prefix))
            except Exception as e:
                print(f"  Warning: Could not parse {fp.name} ({inst}): {e}")
        if parsed:
            combined = pd.concat(parsed, ignore_index=True)
            combined["datetime_utc"] = round_timestamp_to_second(combined["datetime_utc"])
            # UCATS-H2O has an undocumented ~1.5s native sample interval, so
            # flooring to the second can put two originally-distinct UCATS
            # samples in the same bucket; keep_first for a deterministic
            # single value per second across every instrument, not just UCATS.
            combined = (
                combined.dropna(subset=["datetime_utc"])
                .sort_values("datetime_utc")
                .drop_duplicates(subset=["datetime_utc"], keep="first")
                .reset_index(drop=True)
            )
            instrument_dfs[inst] = combined

    if not instrument_dfs:
        raise ValueError("No ATTREX instrument files could be parsed.")

    # Start with MMS (provides T and P required for Si) if present, else the
    # instrument that starts earliest -- order no longer determines which
    # rows survive (outer join keeps every instrument's timestamps), just
    # column precedence on the rare exact-second collision below.
    if "MMS" in instrument_dfs:
        anchor = "MMS"
    else:
        anchor = min(
            instrument_dfs.keys(),
            key=lambda inst: instrument_dfs[inst]["datetime_utc"].min(),
        )
    instruments = [anchor] + [k for k in instrument_dfs if k != anchor]
    merged = instrument_dfs[anchor].copy()

    for inst in instruments[1:]:
        right = instrument_dfs[inst]
        # Drop duplicate non-key columns before merging
        overlap = set(merged.columns) & set(right.columns) - {"datetime_utc"}
        right_clean = right.drop(columns=[c for c in overlap if c != "source_file"], errors="ignore")
        if "source_file" in right_clean.columns and "source_file" in merged.columns:
            right_clean = right_clean.rename(columns={"source_file": f"source_file_{inst}"})
        merged = pd.merge(
            merged,
            right_clean,
            on="datetime_utc",
            how="outer",
        )

    return merged.sort_values("datetime_utc").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_attrex(
    data_dir: Union[str, Path],
    pattern: str = "*.ict",
    h2o_ranking: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load all ATTREX ICT files, merge across instruments, and compute Si.

    The function:
    1. Recursively finds all ``.ict`` files under *data_dir*.
    2. Groups them by instrument (DLH-H2O, NOAA-H2O, UCATS-H2O, MMS, …).
    3. Merges across instruments via ``merge_asof`` on ``datetime_utc``.
    4. Applies MMS scaling (× 0.01 for temperature and pressure).
    5. Computes per-instrument Si columns: Si_DLH, Si_NOAA, Si_UCATS.
    6. Fills Si_best from the first available instrument in *h2o_ranking*.

    Parameters
    ----------
    data_dir : str or Path
        Root directory containing ATTREX instrument sub-folders.
    pattern : str, optional
        Glob pattern for matching files (default: ``"*.ict"``).
    h2o_ranking : list of str, optional
        Ordered list of instrument keys for Si_best selection.
        Keys: ``"DLH"``, ``"NOAA"``, ``"UCATS"``.
        Defaults to ``_DEFAULT_H2O_RANKING`` (DLH > NOAA > UCATS).

    Returns
    -------
    pd.DataFrame
        Combined data with Si_DLH, Si_NOAA, Si_UCATS, Si_best, T_C, Timestamp.
    """
    data_dir = Path(data_dir)
    files = list(data_dir.rglob(pattern))

    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' found in {data_dir}")

    print(f"  Found {len(files)} .ict files in {data_dir}")

    # --- merge across instruments ---
    df = _combine_ict_files(files)

    # --- apply MMS scaling (raw integers → physical units) ---
    # Per ATTREX ICARTT documentation, MMS-1HZ stores T and P as scaled integers:
    #   T_raw  [integer units of 0.01 K]  →  T_K  = T_raw * 0.01
    #   P_raw  [integer units of 0.01 hPa] →  P_hPa = P_raw * 0.01
    # At tropopause altitude (~70–100 hPa), raw P values are ~7000–10000, so a
    # median-based threshold of > 10000 would FAIL to detect the scaling need.
    # We therefore always apply × 0.01 when an MMS column is found.

    # Temperature: look for a prefixed MMS column containing "_T" as a word boundary
    mms_t_col = next(
        (c for c in df.columns if "MMS" in c.upper() and c.upper().endswith("_T")),
        None,
    )
    # Pressure: same logic for "_P"
    mms_p_col = next(
        (c for c in df.columns if "MMS" in c.upper() and c.upper().endswith("_P")),
        None,
    )

    if mms_t_col is not None:
        # Always apply × 0.01 (documented MMS-1HZ scaling; raw ~22000–27000 → 220–270 K)
        df["T"] = df[mms_t_col] * 0.01
        print(f"  Scaled T: {mms_t_col} × 0.01  (sample median raw = {df[mms_t_col].median():.0f})")
    else:
        # Fallback: look for any column named T or starting with T_
        t_fallback = next((c for c in df.columns if c == "T" or c.startswith("T_")), None)
        if t_fallback:
            if df[t_fallback].median() > 200:
                df["T"] = df[t_fallback]
            else:
                df["T"] = df[t_fallback] + 273.15

    if mms_p_col is not None:
        # Always apply × 0.01 (documented MMS-1HZ scaling; raw ~5000–10000 → 50–100 hPa)
        df["P"] = df[mms_p_col] * 0.01
        print(f"  Scaled P: {mms_p_col} × 0.01  (sample median raw = {df[mms_p_col].median():.0f})")
    else:
        p_fallback = next((c for c in df.columns if c == "P" or c.startswith("P_")), None)
        if p_fallback:
            df["P"] = df[p_fallback]

    # --- scale MMS latitude, longitude, and altitude ---
    # Per ICARTT header: G_LAT × 0.00001 deg, G_LONG × 0.00001 deg, G_ALT × 0.1 m
    mms_lat_col = next(
        (c for c in df.columns if "MMS" in c.upper() and "G_LAT" in c.upper()),
        None,
    )
    mms_lon_col = next(
        (c for c in df.columns if "MMS" in c.upper() and "G_LONG" in c.upper()),
        None,
    )
    mms_alt_col = next(
        (c for c in df.columns if "MMS" in c.upper() and "G_ALT" in c.upper()),
        None,
    )

    if mms_lat_col is not None:
        df["Lat"] = df[mms_lat_col] * 0.00001
        df.loc[(df["Lat"] < -90) | (df["Lat"] > 90), "Lat"] = np.nan
        print(f"  Scaled Lat: {mms_lat_col} × 0.00001  (sample median raw = {df[mms_lat_col].median():.0f})")
    else:
        lat_fallback = next((c for c in df.columns if "lat" in c.lower()), None)
        if lat_fallback:
            df["Lat"] = df[lat_fallback]

    if mms_lon_col is not None:
        df["Lon"] = df[mms_lon_col] * 0.00001
        df.loc[(df["Lon"] < -180) | (df["Lon"] > 180), "Lon"] = np.nan
        print(f"  Scaled Lon: {mms_lon_col} × 0.00001  (sample median raw = {df[mms_lon_col].median():.0f})")
    else:
        lon_fallback = next((c for c in df.columns if "lon" in c.lower()), None)
        if lon_fallback:
            df["Lon"] = df[lon_fallback]

    if mms_alt_col is not None:
        df["Alt_m"] = df[mms_alt_col] * 0.1
        df.loc[(df["Alt_m"] < -500) | (df["Alt_m"] > 25000), "Alt_m"] = np.nan
        print(f"  Scaled Alt_m: {mms_alt_col} × 0.1  (sample median raw = {df[mms_alt_col].median():.0f})")

    # --- sanitize T and P: set physically impossible values to NaN ---
    # ATTREX flies near the tropical tropopause (~15-19 km altitude).
    # Physically reasonable ranges:
    #   Temperature: 150–350 K  (tropopause can be ~190 K)
    #   Pressure:    10–1100 hPa (tropopause ~70-100 hPa)
    if "T" in df.columns:
        invalid_t = (df["T"] < 150) | (df["T"] > 350) | df["T"].isna()
        n_bad_t = invalid_t.sum()
        if n_bad_t > 0:
            print(f"  Masking {n_bad_t:,} invalid T values (outside 150–350 K)")
            df.loc[invalid_t, "T"] = np.nan

    if "P" in df.columns:
        invalid_p = (df["P"] <= 0) | (df["P"] > 1100) | df["P"].isna()
        n_bad_p = invalid_p.sum()
        if n_bad_p > 0:
            print(f"  Masking {n_bad_p:,} invalid P values (outside 0–1100 hPa)")
            df.loc[invalid_p, "P"] = np.nan

    # --- detect per-instrument water vapor columns ---
    # DLH: prefixed column with "dlh" and "ppm" (e.g. DLH-H2O_H2O_ppmv)
    h2o_ppm_cols = [c for c in df.columns if "h2o" in c.lower() and "ppm" in c.lower()]
    dlh_h2o = next((c for c in h2o_ppm_cols if "dlh" in c.lower()), None)
    # NOAA: prefixed column with "noaa" or "nw_wv" and "ppm" (e.g. NOAA-H2O_NW_WV_H2O_ppm)
    noaa_h2o = next((c for c in h2o_ppm_cols if "noaa" in c.lower() or "nw_wv" in c.lower()), None)
    # UCATS: prefixed column with "ucats" and "uwv" (e.g. UCATS-H2O_H2O_UWV)
    ucats_h2o = next(
        (c for c in df.columns if "ucats" in c.lower() and "uwv" in c.lower()),
        None,
    )

    # Map instrument key → raw H2O column
    _inst_h2o_cols = {"DLH": dlh_h2o, "NOAA": noaa_h2o, "UCATS": ucats_h2o}

    # Sanitize H2O: mixing ratio must be > 0
    for inst_key, wv_col in _inst_h2o_cols.items():
        if wv_col and wv_col in df.columns:
            invalid_wv = (df[wv_col] <= 0) | df[wv_col].isna()
            n_bad_wv = invalid_wv.sum()
            if n_bad_wv > 0:
                print(f"  Masking {n_bad_wv:,} invalid {wv_col} values (≤ 0 or NaN)")
                df.loc[invalid_wv, wv_col] = np.nan

    # Clean H2O alias columns (ppmv units for all three instruments)
    if dlh_h2o:
        df["H2O_DLH_ppmv"] = df[dlh_h2o]
    if noaa_h2o:
        df["H2O_NOAA_ppmv"] = df[noaa_h2o]
    if ucats_h2o:
        df["H2O_UCATS_ppmv"] = df[ucats_h2o]

    # --- compute per-instrument Si ---
    _si_col_map = {"DLH": "Si_DLH", "NOAA": "Si_NOAA", "UCATS": "Si_UCATS"}

    if "T" not in df.columns or "P" not in df.columns:
        missing = []
        if "T" not in df.columns:
            missing.append("Temperature")
        if "P" not in df.columns:
            missing.append("Pressure")
        print(f"  WARNING: Cannot compute Si — missing columns: {', '.join(missing)}")
        print(f"  Available columns: {df.columns.tolist()}")
        for si_col in _si_col_map.values():
            df[si_col] = np.nan
    else:
        t_src = "MMS" if mms_t_col else "fallback"
        p_src = "MMS" if mms_p_col else "fallback"
        for inst_key, si_col in _si_col_map.items():
            wv_col = _inst_h2o_cols.get(inst_key)
            if wv_col is None or wv_col not in df.columns:
                df[si_col] = np.nan
                continue
            valid = df[wv_col].notna() & df["T"].notna() & df["P"].notna()
            n_valid = valid.sum()
            print(f"  {n_valid:,} rows with valid T, P, {wv_col} → {si_col} "
                  f"(T={t_src}, P={p_src})")
            df[si_col] = np.nan
            if n_valid > 0:
                df.loc[valid, si_col] = si_from_ppmv(
                    df.loc[valid, wv_col], df.loc[valid, "T"], df.loc[valid, "P"]
                )

    # Sanity check all Si columns: |Si| > 10 is physically implausible
    for si_col in _si_col_map.values():
        if si_col in df.columns:
            extreme = (df[si_col].abs() > 10) & df[si_col].notna()
            if extreme.sum() > 0:
                print(f"  Masking {extreme.sum():,} extreme {si_col} values (|Si| > 10)")
                df.loc[extreme, si_col] = np.nan

    # --- Si: fill from ranking ---
    ranking = h2o_ranking if h2o_ranking is not None else _DEFAULT_H2O_RANKING
    df["Si"] = np.nan
    for inst_key in ranking:
        si_col = _si_col_map.get(inst_key.upper().replace("-H2O", ""))
        if si_col and si_col in df.columns:
            mask = df["Si"].isna() & df[si_col].notna()
            df.loc[mask, "Si"] = df.loc[mask, si_col]
    print(f"  Si ranking: {ranking}")

    # --- temperature in Celsius ---
    if "T" in df.columns:
        df["T_C"] = df["T"] - 273.15

    # --- timestamp ---
    if "datetime_utc" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["datetime_utc"], utc=True)

    df["Campaign"] = "ATTREX"

    return df


def extract_attrex_standard(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract standardized columns from ATTREX data.

    Parameters
    ----------
    df : pd.DataFrame
        Data loaded by :func:`load_attrex`.

    Returns
    -------
    pd.DataFrame
        Standardized data with Timestamp, Tair_C, Si (=Si_best),
        Si_best, Si_DLH, Si_NOAA, Si_UCATS,
        H2O_DLH_ppmv, H2O_NOAA_ppmv, H2O_UCATS_ppmv,
        Lat, Lon, Alt_m, Campaign, source_file.
    """
    # Find position columns: prefer scaled versions (Lat, Lon) created during load,
    # then fall back to raw prefixed columns
    lat_col = "Lat" if "Lat" in df.columns else next((c for c in df.columns if "lat" in c.lower()), None)
    lon_col = "Lon" if "Lon" in df.columns else next((c for c in df.columns if "lon" in c.lower()), None)
    alt_col = "Alt_m" if "Alt_m" in df.columns else next((c for c in df.columns if "alt" in c.lower()), None)

    # Consolidate source_file columns
    src_cols = [c for c in df.columns if c.startswith("source_file")]
    if src_cols:
        source = df[src_cols[0]].astype(str)
        for sc in src_cols[1:]:
            source = source + "+" + df[sc].astype(str)
    else:
        source = ""

    def _get(col):
        return df[col] if col in df.columns else np.nan

    # MMS_TEDR fill flag: valid log10(kW/kg) readings top out around -3.2
    # (severe turbulence); a distinct cluster at ~12-16.5 (eps^(1/3) in the
    # hundreds of thousands of m^(2/3)*s^-1 once converted -- physically
    # impossible, ICAO severe threshold is 0.5) is an instrument fill flag
    # not caught by ATTREX_MISSING_FLAGS since it isn't a round sentinel in
    # log space. Mask before cube-root conversion, not after.
    _tedr_log10 = _get("MMS_TEDR") * 0.01
    if isinstance(_tedr_log10, pd.Series):
        _tedr_log10 = _tedr_log10.where(_tedr_log10 <= 0, np.nan)

    # qv from ppmv for each instrument (no P needed)
    qv_dlh  = qv_from_ppmv(_get("H2O_DLH_ppmv"))  if "H2O_DLH_ppmv"  in df.columns else np.nan
    qv_noaa = qv_from_ppmv(_get("H2O_NOAA_ppmv")) if "H2O_NOAA_ppmv" in df.columns else np.nan
    qv_ucats = qv_from_ppmv(_get("H2O_UCATS_ppmv")) if "H2O_UCATS_ppmv" in df.columns else np.nan

    # qv = best from same ranking as Si
    qv = pd.Series(np.nan, index=df.index, dtype=float)
    for src in (qv_dlh, qv_noaa, qv_ucats):
        arr = src if isinstance(src, pd.Series) else pd.Series(src, index=df.index)
        mask = qv.isna() & arr.notna()
        qv = qv.where(~mask, arr)

    # Sw from Si and T
    sw = sw_from_si(_get("Si"), _get("T_C"))

    return pd.DataFrame({
        "Timestamp": df.get("Timestamp", pd.NaT),
        "Tair_C": df.get("T_C", np.nan),
        "P_hPa": df.get("P", np.nan),
        "Si": _get("Si"),
        "Si_DLH": _get("Si_DLH"),
        "Si_NOAA": _get("Si_NOAA"),
        "Si_UCATS": _get("Si_UCATS"),
        "qv": qv,
        "qv_dlh": qv_dlh,
        "qv_noaa": qv_noaa,
        "qv_ucats": qv_ucats,
        "Sw": sw,
        "H2O_DLH_ppmv": _get("H2O_DLH_ppmv"),
        "H2O_NOAA_ppmv": _get("H2O_NOAA_ppmv"),
        "H2O_UCATS_ppmv": _get("H2O_UCATS_ppmv"),
        "Lat": df[lat_col] if lat_col else np.nan,
        "Lon": df[lon_col] if lon_col else np.nan,
        "Alt_m": df[alt_col] if alt_col else np.nan,
        # MMS-1HZ stores these as scaled integers (× 0.01), same convention
        # documented/applied for MMS T and P above -- confirmed by raw ranges
        # (e.g. Heading_deg raw maxes at exactly 36000 == 360.00 x 100).
        "Wind_U_ms": _get("MMS_U") * 0.01,
        "Wind_V_ms": _get("MMS_V") * 0.01,
        "Wind_W_ms": _get("MMS_W") * 0.01,
        "EDR_m23s1": edr_from_mms_log10kWkg(_tedr_log10),
        "Campaign": df.get("Campaign", "ATTREX"),
        "source_file": source,
    })
