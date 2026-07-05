"""
MidCiX (Midlatitude Cirrus Experiment) campaign data parser.

Campaign: MidCiX WB-57 aircraft
Data Source: https://espoarchive.nasa.gov/archive/browse/midcix/WB57
Data Format: JLH (JPL Laser Hygrometer) text files
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
from typing import Union

from .utils import (
    parse_columns_with_units,
    extract_takeoff_date,
    si_from_rh,
    es_ice_hPa,
    qv_from_e_P,
    sw_from_si,
    COMMON_NA_VALUES,
)


def load_midcix_file(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load a single MidCiX JLH data file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the MidCiX data file.
        
    Returns
    -------
    pd.DataFrame
        Parsed data with computed Si and temperature.
    """
    filepath = Path(filepath)
    
    with open(filepath, "r") as f:
        lines = f.readlines()
    
    n_header = int(lines[0].split()[0])
    columns = parse_columns_with_units(lines[n_header - 1])
    takeoff_date = extract_takeoff_date(lines[:n_header])
    
    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        skiprows=n_header,
        names=columns,
        na_values=COMMON_NA_VALUES,
    )
    
    df["source_file"] = filepath.name

    # Canonicalize pressure column: P_mb (mb == hPa, no conversion needed)
    if "P_mb" in df.columns:
        df.rename(columns={"P_mb": "P_hPa"}, inplace=True)

    # Find UTC seconds column
    ut_col = next((c for c in df.columns if c.lower().startswith("ut")), None)
    if ut_col is None:
        raise ValueError(f"No UT seconds column found in {filepath.name}")

    # Parse UTC timestamp
    df["Timestamp"] = df[ut_col].apply(
        lambda x: takeoff_date + timedelta(seconds=float(x)) if pd.notnull(x) else pd.NaT
    )
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)

    # Calculate Si from RH — JPL Laser Hygrometer (JLH)
    if "RH" in df.columns:
        df["Si_JLH"] = si_from_rh(df["RH"])
        df["Si"] = df["Si_JLH"]

    # Convert temperature from Kelvin to Celsius
    if "T_K" in df.columns:
        df["T_C"] = df["T_K"] - 273.15
    
    return df


def load_midcix(
    data_dir: Union[str, Path],
    pattern: str = "*"
) -> pd.DataFrame:
    """
    Load all MidCiX files from a directory.
    
    Parameters
    ----------
    data_dir : str or Path
        Directory containing MidCiX JLH data files.
    pattern : str, optional
        Glob pattern for matching files (default: "*").
        
    Returns
    -------
    pd.DataFrame
        Combined data from all files.
    """
    data_dir = Path(data_dir)
    files = [f for f in data_dir.glob(pattern) if f.is_file()]
    
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' found in {data_dir}")
    
    dfs = []
    for f in sorted(files):
        try:
            dfs.append(load_midcix_file(f))
        except Exception as e:
            print(f"Warning: Could not load {f.name}: {e}")
    
    combined = pd.concat(dfs, ignore_index=True)
    combined["Campaign"] = "MIDCIX"
    
    return combined


def extract_midcix_standard(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract standardized columns from MidCiX data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw data loaded by load_midcix.
        
    Returns
    -------
    pd.DataFrame
        Standardized data with Timestamp, Tair_C, Si, Lat, Lon, Alt_m, Campaign.
    """
    # Find position columns
    lat_col = next((c for c in df.columns if "lat" in c.lower()), None)
    lon_col = next((c for c in df.columns if "lon" in c.lower()), None)
    alt_col = next((c for c in df.columns if "alt" in c.lower() or "z" in c.lower()), None)
    
    # Find pressure column (JLH ICT files often have P_hPa or similar)
    pres_col = next(
        (c for c in df.columns if c.lower() in ("p_hpa", "pressure", "pres", "static_pr", "p")),
        None,
    )
    if pres_col is None:
        pres_col = next(
            (c for c in df.columns if "press" in c.lower() or "static" in c.lower()),
            None,
        )
    p_hpa = df[pres_col] if pres_col else np.nan

    # qv_jlh: use g_kg column (JLH-computed g/kg) directly if present,
    # otherwise fall back to deriving from RH + T + P
    if "g_kg" in df.columns:
        qv_jlh = pd.to_numeric(df["g_kg"], errors="coerce")
    else:
        rh = df.get("RH")
        t_c = df.get("T_C")
        if rh is not None and t_c is not None and pres_col is not None:
            e_jlh = (np.asarray(rh, dtype=float) / 100.0) * es_ice_hPa(t_c)
            qv_jlh = qv_from_e_P(e_jlh, p_hpa)
        else:
            qv_jlh = np.nan

    # Sw from Si and T
    sw = sw_from_si(df.get("Si", np.nan), df.get("T_C", np.nan))

    return pd.DataFrame({
        "Timestamp": df["Timestamp"],
        "Tair_C": df.get("T_C", np.nan),
        "P_hPa": p_hpa,
        "Si": df.get("Si", np.nan),
        "Si_JLH": df.get("Si_JLH", np.nan),
        "qv": qv_jlh,
        "qv_jlh": qv_jlh,
        "Sw": sw,
        "Lat": df.get(lat_col, np.nan) if lat_col else np.nan,
        "Lon": df.get(lon_col, np.nan) if lon_col else np.nan,
        "Alt_m": df.get(alt_col, np.nan) if alt_col else np.nan,
        "Campaign": df.get("Campaign", "MIDCIX"),
        "source_file": df["source_file"],
    })
