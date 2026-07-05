"""
ESCAPE (Experiment of Sea breeze Convection, Aerosols, Precipitation and Environment)
campaign data parser.

Campaign: ESCAPE Learjet state measurements
Data Format: NASA ICARTT-like CSV text (.ict)

Notes
-----
- Source: NCAR EOL (https://data.eol.ucar.edu/dataset/list?project=619&children=project)
- Data files can contain repeated pseudo-headers; parser selects the most likely
  data header (with `Time_Start` and many comma-separated columns).
- water vapor: EdgeTech Chilled Mirror C-137
- Ice supersaturation (Si) is computed from dew point and ambient temperature
  using saturation vapor pressure over ice (Murphy & Koop 2005 in utils.es_ice).
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .utils import clean_column_name, si_from_frost_point, es_ice_hPa, qv_from_e_P, sw_from_si, COMMON_NA_VALUES


ESCAPE_FILE_RE = re.compile(r"ESCAPE-Page0_Learjet_(\d{8})_R\d+\.ict$", re.IGNORECASE)


def _find_data_header_index(lines: List[str]) -> Optional[int]:
    """Return best line index for the tabular data header."""
    candidates: List[Tuple[int, int]] = []
    for idx, line in enumerate(lines):
        if "Time_Start" in line:
            col_count = line.count(",") + 1
            if col_count >= 8:
                candidates.append((idx, col_count))

    if not candidates:
        return None

    # Prefer the one with most columns; if tied, later line usually is real data header.
    candidates.sort(key=lambda item: (item[1], item[0]))
    return candidates[-1][0]


def _parse_header_metadata(lines: List[str], stop_idx: int) -> Dict[str, str]:
    """Parse simple key/value metadata from header block."""
    metadata: Dict[str, str] = {}
    header_lines = lines[:stop_idx]

    for line in header_lines:
        text = line.strip()
        if not text:
            continue

        if ":" in text:
            key, value = text.split(":", 1)
            key = clean_column_name(key)
            value = value.strip()
            if key and value:
                metadata[key] = value
            continue

        # Handle "Key, Value"-style metadata rows.
        if "," in text and text.count(",") <= 3:
            parts = [part.strip() for part in text.split(",")]
            if len(parts) >= 2 and parts[0] and parts[1]:
                key = clean_column_name(parts[0])
                value = ", ".join(parts[1:]).strip()
                if key and value and "Time_Start" not in key:
                    metadata[key] = value

    return metadata


def _metadata_marks_invalid(metadata: Dict[str, str]) -> bool:
    """Conservative file-level exclusion based on metadata quality/status flags."""
    if not metadata:
        return False

    flag_key_tokens = ("flag", "qc", "quality", "status", "valid")
    bad_value_tokens = (
        "invalid", "bad", "reject", "rejected", "failed", "suspect",
        "do_not_use", "do not use", "exclude", "excluded", "not valid",
    )

    for key, value in metadata.items():
        key_l = key.lower()
        if not any(token in key_l for token in flag_key_tokens):
            continue

        value_l = value.strip().lower()
        if any(token in value_l for token in bad_value_tokens):
            return True

        # Explicit false-like validity flags.
        if "valid" in key_l and value_l in {"0", "false", "no", "f"}:
            return True

    return False


def _extract_flight_date(source_file: str, metadata: Dict[str, str]) -> Optional[datetime]:
    """Extract flight date from filename first, then metadata."""
    match = ESCAPE_FILE_RE.search(source_file)
    if match:
        return datetime.strptime(match.group(1), "%Y%m%d").replace(tzinfo=timezone.utc)

    for key in ("FlightDate", "Date", "Flight_Date"):
        val = metadata.get(key)
        if not val:
            continue

        for fmt in ("%Y/%m/%d", "%Y-%m-%d", "%Y%m%d"):
            try:
                return datetime.strptime(val[:10], fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue

    return None


def _choose_column(df: pd.DataFrame, names: Iterable[str]) -> Optional[str]:
    """Find first matching column from exact names or cleaned forms."""
    cols = list(df.columns)
    col_map = {clean_column_name(col).lower(): col for col in cols}

    for name in names:
        if name in df.columns:
            return name

    for name in names:
        key = clean_column_name(name).lower()
        if key in col_map:
            return col_map[key]

    # Soft contains fallback
    name_tokens = [clean_column_name(name).lower() for name in names]
    for col in cols:
        col_key = clean_column_name(col).lower()
        if any(token in col_key for token in name_tokens):
            return col

    return None


def _coerce_numeric(df: pd.DataFrame, columns: Iterable[Optional[str]]) -> None:
    for col in columns:
        if col and col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def _apply_row_qc_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with clear QC/flag failures when row-level flag columns exist."""
    flag_cols = [
        col for col in df.columns
        if any(token in clean_column_name(col).lower() for token in ("flag", "qc", "status"))
    ]

    if not flag_cols:
        return df

    keep_mask = pd.Series(True, index=df.index)

    for col in flag_cols:
        series = df[col]
        # Numeric convention: values >= 2 often indicate suspect/bad.
        num = pd.to_numeric(series, errors="coerce")
        numeric_mask = num.notna()
        keep_mask &= ~(numeric_mask & (num >= 2))

        # String convention: explicit bad/suspect/fail tags.
        str_vals = series.astype(str).str.lower()
        bad_tokens = str_vals.str.contains(r"bad|invalid|fail|suspect|reject", regex=True, na=False)
        keep_mask &= ~bad_tokens

    return df.loc[keep_mask].copy()


def _normalize_temperature_c(series: pd.Series) -> pd.Series:
    """Return temperature in Celsius from likely C or K input."""
    s = pd.to_numeric(series, errors="coerce")
    with np.errstate(all='ignore'):  # Suppress warnings from nanmedian on all-NaN arrays
        med = np.nanmedian(s.to_numpy(dtype=float))
    if np.isfinite(med) and med > 150:
        # Kelvin -> Celsius
        s = s - 273.15
    return s


def _normalize_altitude_m(series: pd.Series, col_name: str, metadata: Dict[str, str]) -> pd.Series:
    """Normalize altitude-like series to meters using name/metadata heuristics."""
    s = pd.to_numeric(series, errors="coerce")
    col_key = clean_column_name(col_name).lower()

    meta_text = " ".join(metadata.values()).lower()
    looks_feet = (
        "ft" in col_key
        or "feet" in col_key
        or "palt_ft" in col_key
        or ("palt" in col_key and "feet" in meta_text)
    )

    if looks_feet:
        return s * 0.3048

    # Heuristic fallback: if median altitude is extremely high for meters but plausible feet.
    with np.errstate(all='ignore'):  # Suppress warnings from nanmedian on all-NaN arrays
        med = np.nanmedian(s.to_numpy(dtype=float))
    if np.isfinite(med) and 12000 <= med <= 70000:
        return s * 0.3048

    return s


def load_escape_file(filepath: Union[str, Path]) -> Optional[pd.DataFrame]:
    """Load a single ESCAPE ICT file and compute Si from dew point."""
    filepath = Path(filepath)
    lines = filepath.read_text(errors="replace").splitlines()

    header_idx = _find_data_header_index(lines)
    if header_idx is None:
        print(f"Warning: skipping {filepath.name} (no data header found)")
        return None

    metadata = _parse_header_metadata(lines, header_idx)
    if _metadata_marks_invalid(metadata):
        print(f"Warning: excluding {filepath.name} (metadata indicates invalid/suspect)")
        return None

    table_text = "\n".join(lines[header_idx:])
    na_values = COMMON_NA_VALUES + ["-999", "-9999.99", "-7777.77", "-8888.88", "n/a", "N/A"]

    try:
        df = pd.read_csv(
            StringIO(table_text),
            sep=",",
            header=0,
            engine="python",
            na_values=na_values,
            on_bad_lines="skip",
        )
    except Exception as exc:
        print(f"Warning: read error for {filepath.name}: {exc}")
        return None

    if df.empty:
        print(f"Warning: skipping {filepath.name} (empty parsed table)")
        return None

    # Keep original names for diagnostics, but normalize whitespace.
    df.columns = [col.strip() for col in df.columns]
    df = _apply_row_qc_filters(df)

    temp_col = _choose_column(df, ["Temp", "Temperature", "Air_Temp", "T"])
    dew_col = _choose_column(df, ["Dew", "DewPoint", "Dew_Point", "Td"])
    pres_col = _choose_column(df, ["Pressure", "Pres", "P", "Static_Pressure", "P_hPa"])
    alt_col = _choose_column(df, ["Palt", "Alt", "Altitude", "GPS_Alt", "Press_Alt"])
    # Prevent the soft-contains search from matching a pressure altitude column
    # (e.g. "Palt" matches "p") as a direct static-pressure column.
    if pres_col is not None and pres_col == alt_col:
        pres_col = None
    lat_col = _choose_column(df, ["Lat", "Latitude", "GPS_Lat"])
    lon_col = _choose_column(df, ["Long", "Lon", "Longitude", "GPS_Lon"])
    time_col = _choose_column(df, ["Time_Start", "Time", "UTC", "Time_UTC"])

    _coerce_numeric(df, [temp_col, dew_col, pres_col, alt_col, lat_col, lon_col, time_col])

    # Unit normalization
    if temp_col:
        df[temp_col] = _normalize_temperature_c(df[temp_col])
    if dew_col:
        df[dew_col] = _normalize_temperature_c(df[dew_col])
    if alt_col:
        df[alt_col] = _normalize_altitude_m(df[alt_col], alt_col, metadata)

    # Physical checks
    if temp_col:
        bad_t = (df[temp_col] < -95) | (df[temp_col] > 60)
        df.loc[bad_t, temp_col] = np.nan
    if dew_col:
        bad_td = (df[dew_col] < -110) | (df[dew_col] > 40)
        df.loc[bad_td, dew_col] = np.nan
    if pres_col:
        bad_p = (df[pres_col] < 50) | (df[pres_col] > 1100)
        df.loc[bad_p, pres_col] = np.nan
    if lat_col:
        bad_lat = (df[lat_col] < -90) | (df[lat_col] > 90)
        df.loc[bad_lat, lat_col] = np.nan
    if lon_col:
        bad_lon = (df[lon_col] < -180) | (df[lon_col] > 180)
        df.loc[bad_lon, lon_col] = np.nan
    if alt_col:
        bad_alt = (df[alt_col] < -500) | (df[alt_col] > 25000)
        df.loc[bad_alt, alt_col] = np.nan

    # Temperature sensor failure: |Tair - T_ISA| > 40°C is a gross physical
    # failure at any altitude (same ISA formula/tolerance as QC2's
    # T_altitude_inconsistent check in scripts/qa_checks.py, for consistency).
    # Observed on 2022-06-10 where the Learjet's Tair sensor read +12 to
    # +16°C from ~8 km up to 17 km, corrupting Si and qv for those rows.
    # A fixed "Tair > -20°C at Alt > 10 km" threshold caught only the upper
    # portion of this same failure — e.g. +15.7°C at ~8.1 km (P_hPa≈348, dev
    # from ISA ≈ +54°C) drove qv to an unphysical ~278 g/kg but fell below
    # the old 10 km cutoff. The ISA-deviation form catches the whole failure
    # regardless of altitude.
    if temp_col and alt_col:
        h_arr = np.asarray(df[alt_col], dtype=float)
        t_isa_k = np.where(
            h_arr <= 11000,
            288.15 - 6.5e-3 * h_arr,
            np.where(h_arr <= 20000, 216.65, 216.65 - 1e-3 * (h_arr - 20000)),
        )
        t_isa_c = t_isa_k - 273.15
        sensor_failure = (
            df[temp_col].notna() & df[alt_col].notna()
            & ((df[temp_col].to_numpy(dtype=float) - t_isa_c) > 40.0)
        )
        n_fail = int(sensor_failure.sum())
        if n_fail > 0:
            df.loc[sensor_failure, temp_col] = np.nan
            if dew_col:
                df.loc[sensor_failure, dew_col] = np.nan
            print(
                f"  ESCAPE QC: nulled {n_fail:,} rows where Tair exceeded ISA by "
                f"> 40°C (temperature sensor failure — check source file)"
            )

    # Derived variables
    if dew_col and temp_col:
        df["Si_chilled_mirror"] = si_from_frost_point(df[dew_col], df[temp_col])
        # Plausibility: Si < -1 is impossible; very high values likely artifacts.
        bad_si = (df["Si_chilled_mirror"] < -1) | (df["Si_chilled_mirror"] > 5)
        df.loc[bad_si, "Si_chilled_mirror"] = np.nan
    else:
        df["Si_chilled_mirror"] = np.nan
    df["Si"] = df["Si_chilled_mirror"]

    # Timestamp from flight date + seconds
    flight_date = _extract_flight_date(filepath.name, metadata)
    if time_col and flight_date is not None:
        df["Timestamp"] = pd.to_datetime(flight_date) + pd.to_timedelta(df[time_col], unit="s")
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
    else:
        df["Timestamp"] = pd.NaT

    # Canonical convenience columns
    df["T_C"] = df[temp_col] if temp_col else np.nan

    # Pressure: use direct column if available; otherwise derive from pressure
    # altitude (Palt, already normalised to metres by _normalize_altitude_m)
    # via the ICAO standard atmosphere — exact by definition.
    if pres_col:
        df["P_hPa"] = df[pres_col].copy()
    elif alt_col and "palt" in alt_col.lower():
        h_m = np.asarray(df[alt_col], dtype=float)  # metres after normalisation
        p_icao = np.where(
            h_m <= 11000,
            1013.25 * (1.0 - 2.25577e-5 * h_m) ** 5.25588,
            226.32 * np.exp(-1.57688e-4 * (h_m - 11000)),
        )
        df["P_hPa"] = np.where(np.isfinite(h_m), p_icao, np.nan)
        # A stuck/erroneous Palt reading can pass the generic [-500, 25000] m
        # altitude bound above (line ~300) while still being unrealistic for
        # this aircraft, producing implausible ICAO pressure. Bound the
        # *derived* P_hPa the same way a direct pressure column already is
        # (line ~291) and null the offending Palt/Alt_m too, since it's the
        # source of the bad derivation.
        bad_derived_p = (df["P_hPa"] < 50) | (df["P_hPa"] > 1100)
        n_bad = int(bad_derived_p.sum())
        if n_bad > 0:
            df.loc[bad_derived_p, "P_hPa"] = np.nan
            df.loc[bad_derived_p, alt_col] = np.nan
            print(
                f"  ESCAPE QC: nulled {n_bad:,} rows where ICAO-derived P_hPa from "
                f"Palt fell outside [50, 1100] hPa (stuck/erroneous Palt sensor — "
                f"check source file)"
            )
    else:
        df["P_hPa"] = np.nan
    df["Lat"] = df[lat_col] if lat_col else np.nan
    df["Lon"] = df[lon_col] if lon_col else np.nan
    df["Alt_m"] = df[alt_col] if alt_col else np.nan
    df["source_file"] = filepath.name
    df["Campaign"] = "ESCAPE"

    return df


def load_escape(data_dir: Union[str, Path], pattern: str = "ESCAPE-Page0_Learjet_*.ict") -> pd.DataFrame:
    """Load all ESCAPE files and return combined parsed DataFrame."""
    data_dir = Path(data_dir)
    files = sorted(data_dir.rglob(pattern))

    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' found in {data_dir}")

    dfs: List[pd.DataFrame] = []
    for filepath in files:
        parsed = load_escape_file(filepath)
        if parsed is not None and not parsed.empty:
            dfs.append(parsed)

    if not dfs:
        raise ValueError(f"No valid ESCAPE files parsed from {data_dir}")

    combined = pd.concat(dfs, ignore_index=True)
    combined["Campaign"] = "ESCAPE"
    return combined


def extract_escape_standard(df: pd.DataFrame) -> pd.DataFrame:
    """Extract standardized columns for downstream campaign combination."""
    # qv_chilled_mirror from dew/frost point + pressure
    # The loader uses si_from_frost_point(dew_col, temp_col) → treat dew_col as frost point
    # Find the dew-point column via Si_chilled_mirror: recompute e = es_ice(Td)
    # We don't re-detect dew_col here; derive e from Si_cm and T instead:
    #   Si_cm = e/es_ice(T) - 1  → e = (Si_cm + 1) * es_ice(T)
    si_cm = df.get("Si_chilled_mirror")
    t_c = df.get("T_C")
    p_hpa = df.get("P_hPa")
    if si_cm is not None and t_c is not None and p_hpa is not None:
        e_cm = (np.asarray(si_cm, dtype=float) + 1.0) * es_ice_hPa(t_c)
        qv_cm = qv_from_e_P(e_cm, p_hpa)
    else:
        qv_cm = np.nan

    # Sw from Si and T
    sw = sw_from_si(df.get("Si", np.nan), df.get("T_C", np.nan))

    return pd.DataFrame({
        "Timestamp": df.get("Timestamp", pd.NaT),
        "Tair_C": df.get("T_C", np.nan),
        "P_hPa": df.get("P_hPa", np.nan),
        "Si": df.get("Si", np.nan),
        "Si_chilled_mirror": df.get("Si_chilled_mirror", np.nan),
        "qv": qv_cm,
        "qv_chilled_mirror": qv_cm,
        "Sw": sw,
        "Lat": df.get("Lat", np.nan),
        "Lon": df.get("Lon", np.nan),
        "Alt_m": df.get("Alt_m", np.nan),
        "Campaign": df.get("Campaign", "ESCAPE"),
        "source_file": df.get("source_file", ""),
    })
