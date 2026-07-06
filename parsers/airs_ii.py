"""
AIRS-II (Alliance Icing Research Study II) campaign data parser.

Campaign: AIRS-II
- Data Source: https://data.eol.ucar.edu/project/AIRS-II
- Data Format: NetCDF files with LRT (Low-Rate) flight-level data
- Variables: relative humidity (RHUM), air temperature (ATX), LAT, LON, ALT
- Derived: Si (from RHUM)
- Notes:
    - ATX = RAF best-estimate ambient temperature, derived from ATWH (Raw heated Rosemount air temperature) for AIRS-II
    - Relative humidity was derived from the RAF best-estimate air temperature (ATX) and primary chilled-mirror dewpoint measurement (DPXC)
    - Relative humidity (RHUM) was derived primarily from the DPBC chilled-mirror dew point sensor (reported as DPXC), which was selected as the reference humidity measurement for most flights.
    - Two collocated dew point sensors were used: DPTC (faster response but limited dynamic range) and DPBC (slower response but capable of measuring larger dew point depressions).
    - The sensors generally showed good agreement and correlation when operating normally.
    - The primary data-quality issue was water ingestion during flight, which occasionally caused sensor drift.
    - Humidity measurements were evaluated flight-by-flight, and the best-performing dew point sensor was selected when necessary.
    - Overall, the campaign reported good-quality humidity measurements, with occasional degradation during periods affected by water ingestion.
    - SpectraSensors TDL hygrometers used but only voltages provided in data. So these data were not used in this dataset.  
"""

import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
from typing import Union

from .utils import es_ice_hPa, qv_from_e_P, sw_from_si


def load_airs_ii_file(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load a single AIRS-II netCDF file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the netCDF file.
        
    Returns
    -------
    pd.DataFrame
        Parsed data with Timestamp, RHUM, ATX_C, Si, and position data.
    """
    filepath = Path(filepath)
    
    # Use xarray's default netCDF backend.
    ds = xr.open_dataset(filepath, decode_times=True, decode_timedelta=True)
    
    try:
        if "RHUM" not in ds.variables or "ATX" not in ds.variables:
            raise KeyError("RHUM or ATX missing in dataset")
        
        # Extract time coordinate
        if "Time" in ds:
            times = pd.to_datetime(ds["Time"].values)
        elif "time" in ds.coords:
            times = pd.to_datetime(ds.coords["time"].values)
        else:
            raise KeyError("No time coordinate found")
        
        rh = np.asarray(ds["RHUM"].values).ravel().astype(float)
        atx = np.asarray(ds["ATX"].values).ravel().astype(float)  # Temperature in Celsius
        
        # Filter out unrealistic RH values (> 200%)
        valid_mask = rh <= 200
        rh = rh[valid_mask]
        atx = atx[valid_mask]
        times = np.asarray(times).ravel()[valid_mask]
        
        # Extract pressure and position data if available
        pres_raw = None
        for pvar in ("PSXC", "PRES", "PSX", "P"):
            if pvar in ds.variables:
                pres_raw = np.asarray(ds[pvar].values).ravel().astype(float)
                break
        if pres_raw is None:
            pres_raw = np.full(len(times), np.nan)

        lat = np.asarray(ds.get("LAT", ds.get("LATC", [np.nan] * len(times)))).ravel()
        lon = np.asarray(ds.get("LON", ds.get("LONC", [np.nan] * len(times)))).ravel()

        # ALT (IRS Baro-Inertial Altitude) is preferred when valid, but the
        # inertial system can drop to exactly 0.0 for extended periods (up to
        # 4,413 of 33,444 samples / ~68 minutes in RF09) -- an INS
        # alignment/dropout artifact, not a real reading of sea level for
        # over an hour of flight. ds.get()'s fallback only triggers when a
        # key is entirely absent, so it never engages here since ALT is
        # present (just sometimes invalid) in every file. GPS-derived GGALT
        # is far more reliable across the whole campaign (at most 7 zero
        # samples in any single file, vs. up to 4,413 for ALT), so fall back
        # to it per-sample whenever ALT reads exactly 0.0.
        alt_irs  = np.asarray(ds.get("ALT",   [np.nan] * len(times))).ravel().astype(float)
        alt_gps  = np.asarray(ds.get("GGALT", [np.nan] * len(times))).ravel().astype(float)
        if alt_irs.size and alt_gps.size:
            alt_invalid = alt_irs == 0.0
            n_alt_invalid = int(alt_invalid.sum())
            alt = np.where(alt_invalid, alt_gps, alt_irs)
            if n_alt_invalid:
                print(f"  AIRS-II QC: fell back to GGALT for {n_alt_invalid:,} rows "
                      f"where ALT (IRS) read exactly 0.0 ({filepath.name})")
        elif alt_irs.size:
            alt = alt_irs
        else:
            alt = alt_gps
        
        # Apply same mask to position/pressure data
        if len(pres_raw) == len(valid_mask):
            pres_raw = pres_raw[valid_mask]
        if len(lat) == len(valid_mask):
            lat = lat[valid_mask]
            lon = lon[valid_mask]
            alt = alt[valid_mask]
        
        n = min(len(times), len(rh), len(atx))
        
        df = pd.DataFrame({
            "Timestamp": pd.to_datetime(times[:n], utc=True),
            "RHUM": rh[:n],
            "ATX_C": atx[:n],
            "P_hPa": pres_raw[:n] if len(pres_raw) >= n else np.nan,
            "Lat": lat[:n] if len(lat) >= n else np.nan,
            "Lon": lon[:n] if len(lon) >= n else np.nan,
            "Alt_m": alt[:n] if len(alt) >= n else np.nan,
        })

        df = df.sort_values("Timestamp").reset_index(drop=True)

        # Pressure sanity check: convert Pa → hPa if values look like Pa
        p_med = df["P_hPa"].median()
        if np.isfinite(p_med) and p_med > 2000:
            df["P_hPa"] = df["P_hPa"] / 100.0
        df.loc[(df["P_hPa"] < 50) | (df["P_hPa"] > 1100), "P_hPa"] = np.nan

        # Si from chilled mirror (RHUM derived from DPXC dew point)
        df["Si_chilled_mirror"] = df["RHUM"] / 100.0 - 1.0
        df["Si"] = df["Si_chilled_mirror"]
        
        df["source_file"] = filepath.name
        
        return df
        
    finally:
        ds.close()


def load_airs_ii(
    data_dir: Union[str, Path],
    pattern: str = "*.nc"
) -> pd.DataFrame:
    """
    Load all AIRS-II netCDF files from a directory.
    
    Parameters
    ----------
    data_dir : str or Path
        Directory containing AIRS-II netCDF files.
    pattern : str, optional
        Glob pattern for matching files (default: "*.nc").
        
    Returns
    -------
    pd.DataFrame
        Combined data from all files.
    """
    data_dir = Path(data_dir)
    files = list(data_dir.glob(pattern))
    
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' found in {data_dir}")
    
    dfs = []
    for f in sorted(files):
        try:
            dfs.append(load_airs_ii_file(f))
        except Exception as e:
            print(f"Warning: Could not load {f.name}: {e}")
    
    combined = pd.concat(dfs, ignore_index=True)
    combined["Campaign"] = "AIRS-II"
    
    return combined


def extract_airs_ii_standard(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract standardized columns from AIRS-II data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw data loaded by load_airs_ii.
        
    Returns
    -------
    pd.DataFrame
        Standardized data with Timestamp, Tair_C, Si, Lat, Lon, Alt_m, Campaign.
    """
    # qv_chilled_mirror from RHUM + T + P
    rhum = df.get("RHUM")
    t_c = df.get("ATX_C")
    p_hpa = df.get("P_hPa")
    if rhum is not None and t_c is not None and p_hpa is not None:
        e_cm = (np.asarray(rhum, dtype=float) / 100.0) * es_ice_hPa(t_c)
        qv_cm = qv_from_e_P(e_cm, p_hpa)
    else:
        qv_cm = np.nan

    # Sw from Si and T
    sw = sw_from_si(df.get("Si", np.nan), df.get("ATX_C", np.nan))

    return pd.DataFrame({
        "Timestamp": df["Timestamp"],
        "Tair_C": df.get("ATX_C", np.nan),
        "P_hPa": df.get("P_hPa", np.nan),
        "Si": df.get("Si", np.nan),
        "Si_chilled_mirror": df.get("Si_chilled_mirror", np.nan),
        "qv": qv_cm,
        "qv_chilled_mirror": qv_cm,
        "Sw": sw,
        "Lat": df.get("Lat", np.nan),
        "Lon": df.get("Lon", np.nan),
        "Alt_m": df.get("Alt_m", np.nan),
        "Campaign": df.get("Campaign", "AIRS-II"),
        "source_file": df["source_file"],
    })
