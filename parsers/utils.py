"""
Shared utility functions for campaign data parsing.

Provides thermodynamic calculations, column name cleaning, and
common parsing helpers used across multiple campaign parsers.
"""

import re
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Optional

# Molar mass ratio water / dry air (kg/kg per mol/mol)
_EPSILON: float = 18.015 / 28.964  # ≈ 0.6220


# =============================================================================
# Thermodynamic Functions
# =============================================================================

def es_ice(T_C: np.ndarray) -> np.ndarray:
    """
    Calculate saturation vapor pressure over ice using Murphy & Koop (2005).
    
    Parameters
    ----------
    T_C : array-like
        Temperature in degrees Celsius.
        
    Returns
    -------
    np.ndarray
        Saturation vapor pressure over ice in hPa.
    """
    T_K = np.asarray(T_C) + 273.15
    return np.exp(9.550426 - 5723.265 / T_K + 3.53068 * np.log(T_K) - 0.007283 * T_K)


def si_from_frost_point(frost_point_C: np.ndarray, temperature_C: np.ndarray) -> np.ndarray:
    """
    Compute ice supersaturation (Si) from frost point and ambient temperature.
    
    Parameters
    ----------
    frost_point_C : array-like
        Frost point temperature in degrees Celsius.
    temperature_C : array-like
        Ambient air temperature in degrees Celsius.
        
    Returns
    -------
    np.ndarray
        Ice supersaturation (Si), where values > 0 indicate supersaturation.
    """
    return es_ice(frost_point_C) / es_ice(temperature_C) - 1.0


def si_from_ppmv(wv_ppmv: np.ndarray, temp_K: np.ndarray, pressure_hPa: np.ndarray) -> np.ndarray:
    """
    Compute ice supersaturation from water vapor mixing ratio (ppmv).
    
    Parameters
    ----------
    wv_ppmv : array-like
        Water vapor mixing ratio in parts per million by volume.
    temp_K : array-like
        Temperature in Kelvin.
    pressure_hPa : array-like
        Pressure in hPa.
        
    Returns
    -------
    np.ndarray
        Ice supersaturation (Si).
    """
    wv_ppmv = np.asarray(wv_ppmv, dtype=float)
    temp_K = np.asarray(temp_K, dtype=float)
    pressure_hPa = np.asarray(pressure_hPa, dtype=float)

    # Mask physically invalid inputs so they don't produce extreme Si
    invalid = (
        ~np.isfinite(wv_ppmv) | (wv_ppmv <= 0)
        | ~np.isfinite(temp_K) | (temp_K < 150) | (temp_K > 350)
        | ~np.isfinite(pressure_hPa) | (pressure_hPa <= 0)
    )

    # Calculate saturation vapor pressure over ice in hPa
    e_s = 6.112 * np.exp((22.46 * (temp_K - 273.15)) / (temp_K - 0.55))

    # Convert vapor mixing ratio (ppmv) to actual vapor pressure (e) in hPa
    e = (wv_ppmv / 1e6) * pressure_hPa

    result = (e / e_s) - 1.0

    # Replace invalid entries with NaN
    if np.ndim(result) == 0:
        return np.nan if invalid else float(result)
    result[invalid] = np.nan
    return result


def es_ice_hPa(T_C: np.ndarray) -> np.ndarray:
    """Saturation vapor pressure over ice (hPa), Murphy & Koop 2005."""
    return es_ice(T_C) / 100.0


def es_liq_hPa(T_C: np.ndarray) -> np.ndarray:
    """Saturation vapor pressure over liquid water (hPa), Murphy & Koop 2005."""
    T_K = np.asarray(T_C, dtype=float) + 273.15
    ln_e = (
        54.842763
        - 6763.22 / T_K
        - 4.21 * np.log(T_K)
        + 0.000367 * T_K
        + np.tanh(0.0415 * (T_K - 218.8))
        * (53.878 - 1331.22 / T_K - 9.44523 * np.log(T_K) + 0.014025 * T_K)
    )
    return np.exp(ln_e) / 100.0  # Pa → hPa


def qv_from_ppmv(ppmv: np.ndarray) -> np.ndarray:
    """Water vapor mass mixing ratio (g/kg) from volume mixing ratio (ppmv)."""
    return np.asarray(ppmv, dtype=float) * _EPSILON * 1e-3


def qv_from_e_P(e_hPa: np.ndarray, P_hPa: np.ndarray) -> np.ndarray:
    """Water vapor mass mixing ratio (g/kg) from vapor pressure e and total pressure P (both hPa)."""
    e = np.asarray(e_hPa, dtype=float)
    P = np.asarray(P_hPa, dtype=float)
    denom = P - e
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where((denom > 0) & np.isfinite(e) & np.isfinite(P), _EPSILON * e / denom, np.nan)
    return r * 1000.0


def sw_from_si(Si: np.ndarray, T_C: np.ndarray) -> np.ndarray:
    """Supersaturation w.r.t. liquid water from Si (w.r.t. ice) and temperature (°C)."""
    si_arr = np.asarray(Si, dtype=float)
    return (1.0 + si_arr) * (es_ice_hPa(T_C) / es_liq_hPa(T_C)) - 1.0


def si_from_rh(rh_percent: np.ndarray) -> np.ndarray:
    """
    Compute ice supersaturation from relative humidity with respect to ice.
    
    Parameters
    ----------
    rh_percent : array-like
        Relative humidity with respect to ice in percent.
        
    Returns
    -------
    np.ndarray
        Ice supersaturation (Si).
    """
    return np.asarray(rh_percent) / 100.0 - 1.0


def normalize_datetime_utc(values: pd.Series) -> pd.Series:
    """Normalize datetime values to UTC nanosecond precision.

    Uses dt.as_unit("ns") rather than astype() because in pandas 2.0+ the
    default resolution for tz-aware datetimes is microseconds, and astype()
    can silently preserve the original resolution in some build configurations.
    """
    parsed = pd.to_datetime(values, utc=True, errors="coerce")
    return parsed.dt.as_unit("ns")


def round_timestamp_to_second(series: pd.Series) -> pd.Series:
    """Floor timestamps to whole seconds for exact-key cross-instrument merges.

    .dt.round("s") uses round-half-to-even ("banker's rounding"): X.5-second
    values round toward the nearest EVEN second, not consistently up. Some
    source files sample at a fixed .5s offset, so two adjacent, physically
    distinct 1 Hz samples (e.g. :03.5 and :04.5) both round to :04 and
    collide into a spurious duplicate-timestamp row. floor() truncates
    consistently in one direction, preserving the original 1s spacing with
    no collisions. Every cross-instrument merge in this pipeline is an
    exact-key join on this floored timestamp -- no merge_asof tolerance --
    so consistent flooring across parsers is what makes the join keys
    actually collide across instruments at the same wall-clock second.
    """
    return pd.to_datetime(series, utc=True, errors="coerce").dt.floor("s")


# =============================================================================
# Wind Utilities
# =============================================================================

def wind_speed_dir_to_uv(speed, direction_deg):
    """Convert meteorological wind speed/direction (direction = FROM, degrees
    clockwise from north) to U (eastward) and V (northward) components, m/s."""
    rad = np.deg2rad(direction_deg)
    u = -speed * np.sin(rad)
    v = -speed * np.cos(rad)
    return u, v


# =============================================================================
# EDR (Eddy Dissipation Rate) Unification
# =============================================================================
# Both conversions below produce the ICAO/WMO-standard EDR quantity,
# eps^(1/3) in m^(2/3)*s^-1 -- see docs/decisions/2026-07-13-edr-unification.md.
# ARM uses edr_from_und_cm23s1 too: its raw Turbulence_eps field is
# eps^(1/3) but in the same cm^(2/3)*s^-1 house convention as the later UND
# ASCII pipeline (same aircraft/team), confirmed by value-range comparison,
# not by an explicit unit in ARM's own readme.

_CM_TO_M_23 = 100.0 ** (2.0 / 3.0)  # cm^(2/3) -> m^(2/3), ~21.544


def edr_from_und_cm23s1(cm23s1: np.ndarray) -> np.ndarray:
    """UND pipeline EDR (already eps^(1/3), cm^(2/3)*s^-1) -> m^(2/3)*s^-1."""
    return np.asarray(cm23s1, dtype=float) / _CM_TO_M_23


def edr_from_mms_log10kWkg(log10_kWkg: np.ndarray) -> np.ndarray:
    """NASA Ames MMS EDR (log10 of eps in kW/kg) -> eps^(1/3) in m^(2/3)*s^-1.

    kW/kg -> W/kg is *1000; W/kg is dimensionally m^2*s^-3 (energy per mass
    per time), so eps^(1/3) in m^(2/3)*s^-1 = (1000 * 10**log10_kWkg) ** (1/3).
    """
    eps_W_per_kg = 1000.0 * np.power(10.0, np.asarray(log10_kWkg, dtype=float))
    return np.cbrt(eps_W_per_kg)


def first_per_second(df: pd.DataFrame, ts_col: str = "Timestamp") -> pd.DataFrame:
    """Floor timestamps to whole seconds and keep the first sample per second.

    Factors out the round_timestamp_to_second -> dropna -> drop_duplicates
    idiom used when a raw source is sampled denser than 1 Hz and needs to be
    floored onto the pipeline's 1 Hz exact-second-merge grid. Takes the first
    actually-observed sample per second rather than averaging, since a mean
    across sub-second samples would synthesize a value that never existed at
    any single instant -- worse for a genuinely fluctuating turbulence
    quantity than picking one real observation.
    """
    df = df.copy()
    df[ts_col] = round_timestamp_to_second(df[ts_col])
    df = df.dropna(subset=[ts_col]).drop_duplicates(subset=[ts_col], keep="first")
    return df


# =============================================================================
# Column Name Utilities
# =============================================================================

def clean_column_name(name: str) -> str:
    """
    Clean a column name for safe DataFrame usage.
    
    Strips whitespace and replaces non-word characters with underscores.
    
    Parameters
    ----------
    name : str
        Original column name.
        
    Returns
    -------
    str
        Cleaned column name.
    """
    name = name.strip()
    name = re.sub(r"[^\w]+", "_", name)
    return name.strip("_")


def parse_columns_with_units(header_line: str) -> List[str]:
    """
    Parse column names that include units in parentheses.
    
    Handles formats like: "UT (s) T (K) RH (%)" -> ["UT_s", "T_K", "RH"]
    
    Parameters
    ----------
    header_line : str
        Header line containing column names with optional units.
        
    Returns
    -------
    list of str
        List of cleaned column names.
    """
    tokens = header_line.split()
    cols = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and re.match(r"^\(.*\)$", tokens[i + 1]):
            cols.append(f"{tokens[i]} {tokens[i + 1]}")
            i += 2
        else:
            cols.append(tokens[i])
            i += 1
    return [clean_column_name(c) for c in cols]


# =============================================================================
# Date Extraction Utilities
# =============================================================================

def extract_takeoff_date(lines: List[str]) -> datetime:
    """
    Extract the first date (YYYY MM DD) from header lines.
    
    Common format in NASA ICARTT and ESPO archive files.
    
    Parameters
    ----------
    lines : list of str
        Header lines to search.
        
    Returns
    -------
    datetime
        Extracted date.
        
    Raises
    ------
    ValueError
        If no valid date is found in the header.
    """
    for line in lines:
        match = re.match(r"\s*(\d{4})\s+(\d{2})\s+(\d{2})", line)
        if match:
            year, month, day = map(int, match.groups())
            return datetime(year, month, day)
    raise ValueError("Takeoff date not found in header.")


# =============================================================================
# Common NA Values
# =============================================================================

# Standard missing value flags used across campaigns
COMMON_NA_VALUES = [
    999999.9999,
    999.9999999,
    9999.999999,
    99999.99999,
    99999999999,
    999999,
    9.9999E+30,
    9.999E+30,
    -9999,
    -9999.99,
    -7777,
    -7777.77,
    -8888,
    -8888.88,
]
