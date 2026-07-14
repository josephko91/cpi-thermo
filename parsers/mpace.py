"""
MPACE (Mixed-Phase Arctic Cloud Experiment) campaign data parser.

Campaign: MPACE

Data notes:
- ARM IOP data browser: https://iop.arm.gov/2004/nsa/mpace/poellot-citation
- University of North Dakota Citation II aircraft: state parameters plus
  cloud microphysics probes (King LWC, FSSP, 1-DC, 2-DC). Barrow, AK, Oct 2004.

Data Format: NASA Ames FFI 1001 fixed-header text files with .mpace extension.
  - Header line 7 (1-indexed) is "DATE RDATE" and gives the flight date
    (YYYY MM DD) -- more reliable than the 2-digit year in the filename.
  - Column header line starts with 'Time' and contains 'Air_Temp'.
  - One units line follows the header; data starts on the line after that.
  - Filename encodes start timestamp: YY_MM_DD_HH_MM_SS.mpace (2-digit year).
  - 'Time' column is seconds since midnight UTC on the flight date. It is not
    wrapped at 86400 for flights that cross UTC midnight, so plain timedelta
    arithmetic against the header date handles day-rollover correctly.

Required variables
------------------
- Air_Temp: ambient air temperature corrected for dynamic heating (deg C)
- STATIC_PR: static pressure (mb == hPa)
- POS_Lat, POS_Lon, POS_Alt: Applanix POS aircraft position

No water-vapor instrument
--------------------------
All 14 flights in the archive (2004-09-30 through 2004-10-21) carry an
identical 48-variable header: airspeeds, static pressure/temperature,
Applanix POS navigation, winds, and cloud microphysics (King LWC, FSSP, 1-DC,
2-DC). No dew/frost-point sensor or laser hygrometer channel is present in
any file, so Si and qv cannot be derived from this dataset -- they are
reported as NaN for every record. Tair_C/P_hPa/Lat/Lon/Alt_m are still
populated and useful for fusing against the CPI particle-image archive.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from .utils import sw_from_si, wind_speed_dir_to_uv, edr_from_und_cm23s1


MPACE_INVALID_VALUES = {
    999999.9999,
    999.9999999,
    9999.999999,
    99999.99999,
    99999999999,
    9.9999e30,
    9.999e30,
}


def _coerce_and_mask(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    s = s.mask(s.isin(MPACE_INVALID_VALUES), np.nan)
    return s


def _find_data_start(lines: list[str]) -> tuple[Optional[int], Optional[int]]:
    """
    Locate the column-header line and the first data line.

    The column-header line is the first line that starts with 'Time' and
    also contains 'Air_Temp'. One units line follows it, then data.

    Returns (header_line_idx, data_start_line_idx).
    """
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("Time") and "Air_Temp" in stripped:
            return i, i + 2  # units line at i+1, data at i+2
    return None, None


def _extract_date_from_header(lines: list[str]) -> Optional[datetime]:
    """Extract the flight base date from the NASA Ames FFI 1001 DATE line."""
    if len(lines) < 7:
        return None
    parts = lines[6].split()
    if len(parts) >= 3:
        try:
            return datetime(int(parts[0]), int(parts[1]), int(parts[2]))
        except (ValueError, IndexError):
            pass
    return None


def load_mpace_file(filepath: Union[str, Path]) -> pd.DataFrame:
    """Load a single MPACE .mpace file."""
    filepath = Path(filepath)

    with open(filepath, "r") as f:
        lines = f.readlines()

    header_idx, data_start = _find_data_start(lines)
    if header_idx is None:
        raise ValueError(
            f"Could not find column header ('Time ... Air_Temp ...') in {filepath.name}"
        )

    columns = lines[header_idx].strip().split()

    na_values = [
        "999999.9999", "999.9999999", "9999.999999", "99999.99999",
        "99999999999", "9.999E+30", "9.9999E+30",
    ]

    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        skiprows=data_start,
        names=columns,
        na_values=na_values,
        engine="python",
        on_bad_lines="skip",
    )

    if df.empty:
        return df

    for required in ("Air_Temp", "STATIC_PR"):
        if required not in df.columns:
            raise KeyError(
                f"Missing required MPACE column '{required}' in {filepath.name}. "
                f"Available: {list(df.columns)}"
            )

    df["Air_Temp"] = _coerce_and_mask(df["Air_Temp"])
    df["STATIC_PR"] = _coerce_and_mask(df["STATIC_PR"])

    # Unit guards: convert Kelvin -> Celsius if median looks Kelvin-range
    t_arr = df["Air_Temp"].to_numpy(dtype=float)
    med_t = np.nan if np.all(np.isnan(t_arr)) else np.nanmedian(t_arr)
    if np.isfinite(med_t) and med_t > 150:
        df["Air_Temp"] = df["Air_Temp"] - 273.15

    # Convert Pa -> hPa if pressure looks like Pascals
    p_arr = df["STATIC_PR"].to_numpy(dtype=float)
    med_p = np.nan if np.all(np.isnan(p_arr)) else np.nanmedian(p_arr)
    if np.isfinite(med_p) and 2000 < med_p < 120000:
        df["STATIC_PR"] = df["STATIC_PR"] / 100.0

    # Physical bounds
    df.loc[(df["Air_Temp"] < -95) | (df["Air_Temp"] > 60), "Air_Temp"] = np.nan
    df.loc[(df["STATIC_PR"] < 50) | (df["STATIC_PR"] > 1100), "STATIC_PR"] = np.nan

    # Timestamps: Time column is seconds-since-midnight UTC on the flight date
    base_date = _extract_date_from_header(lines)
    if "Time" in df.columns and base_date is not None:
        time_s = pd.to_numeric(df["Time"], errors="coerce")
        df["Timestamp"] = pd.Timestamp(base_date) + pd.to_timedelta(time_s, unit="s")
        df["Timestamp"] = df["Timestamp"].dt.tz_localize("UTC")
    else:
        df["Timestamp"] = pd.NaT

    # Position columns (Applanix POS)
    for col, lo, hi in [
        ("POS_Lat", -90.0, 90.0),
        ("POS_Lon", -180.0, 180.0),
        ("POS_Alt", -500.0, 25000.0),
    ]:
        if col in df.columns:
            df[col] = _coerce_and_mask(df[col])
            df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan

    df["source_file"] = filepath.name
    df["Campaign"] = "MPACE"

    return df


def load_mpace(
    data_dir: Union[str, Path],
    pattern: str = "*.mpace",
    h2o_ranking: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load all MPACE .mpace files in data_dir. No water-vapor instrument was
    flown, so h2o_ranking is accepted for interface consistency but unused."""
    data_dir = Path(data_dir)
    files = [f for f in data_dir.glob(pattern) if f.is_file()]

    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' found in {data_dir}")

    dfs = []
    for f in sorted(files):
        try:
            dfs.append(load_mpace_file(f))
        except Exception as e:
            print(f"Warning: Could not load {f.name}: {e}")

    if not dfs:
        raise ValueError(f"No valid MPACE files were parsed in {data_dir}")

    combined = pd.concat(dfs, ignore_index=True)
    combined["Campaign"] = "MPACE"
    return combined


def extract_mpace_standard(df: pd.DataFrame) -> pd.DataFrame:
    """Return standardized columns for combined campaign output.

    No water-vapor instrument was flown on MPACE, so Si and qv are NaN for
    every record; Tair_C/P_hPa/Lat/Lon/Alt_m are populated.
    """
    si = pd.Series(np.nan, index=df.index, dtype=float)
    qv = pd.Series(np.nan, index=df.index, dtype=float)
    sw = sw_from_si(si.to_numpy(), df.get("Air_Temp", np.nan))

    # Wind_D_Nose is undocumented in this campaign's raw header; assumed
    # standard meteorological "from" convention (clockwise from north).
    wind_u, wind_v = wind_speed_dir_to_uv(
        df.get("Wind_M_Nose", np.nan), df.get("Wind_D_Nose", np.nan)
    )

    return pd.DataFrame(
        {
            "Timestamp": df.get("Timestamp", pd.NaT),
            "Tair_C": df.get("Air_Temp", np.nan),
            "P_hPa": df.get("STATIC_PR", np.nan),
            "Si": si,
            "qv": qv,
            "Sw": sw,
            "Lat": df.get("POS_Lat", np.nan),
            "Lon": df.get("POS_Lon", np.nan),
            "Alt_m": df.get("POS_Alt", np.nan),
            # MPACE's only wind vector is from the nose boom (no separate
            # wing-mounted wind channel exists in this archive's 49 columns).
            "Wind_W_ms": df.get("Wind_Z_Nose", np.nan),
            "Wind_U_ms": wind_u,
            "Wind_V_ms": wind_v,
            # EDR is the one genuinely dual-channel field here (wing primary,
            # nose kept as a campaign-local extra -- primary/wing kept only,
            # per turbulence-scope reduction).
            "EDR_m23s1": edr_from_und_cm23s1(df.get("TURB", np.nan)),
            "Campaign": df.get("Campaign", "MPACE"),
            "source_file": df.get("source_file", ""),
        }
    )
