"""
CRYSTAL-FACE NASA (WB-57) campaign data parser.

Campaign: CRYSTAL-FACE NASA WB-57 aircraft
Data Source: https://espoarchive.nasa.gov/archive/browse/crystalf/WB57
Data Formats:
  - JPL Laser Hygrometer (JLH) data
  - Meteorological Measurement System (MMS) data with geolocation (LAT, LONG, ALT)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Union, Optional

from .utils import (
    parse_columns_with_units,
    extract_takeoff_date,
    si_from_rh,
)


def load_mms_file(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load a single CRYSTAL-FACE MMS data file.
    
    MMS files follow ICARTT format with:
    - Header line 1: number of header lines and format version
    - Scale factors and missing values in header
    - Data columns: UT, P_ALT, LAT, LONG, TAS (with scale factors applied)
    
    Parameters
    ----------
    filepath : str or Path
        Path to the MMS data file.
        
    Returns
    -------
    pd.DataFrame
        Parsed data with UTC timestamp, Lat, Lon, Alt_m columns.
    """
    filepath = Path(filepath)
    
    with open(filepath) as f:
        lines = f.readlines()
    
    # Parse header
    n_header = int(lines[0].split()[0])
    
    # Extract takeoff date from header
    takeoff_date = extract_takeoff_date(lines[:n_header])
    
    # Parse scale factors (typically line 4)
    # Remove comments and parse only numeric values
    scale_line = lines[4].split(";")[0].split()
    scales = [float(s) for s in scale_line if s.strip()]
    
    # Parse missing value indicators (typically line 5)
    # Remove comments and parse only numeric values
    missing_line = lines[5].split(";")[0].split()
    missing_vals = [float(m) for m in missing_line if m.strip()]
    
    # Column names for MMS data
    columns = ["UT", "P_ALT", "LAT", "LONG", "TAS"]
    
    # Read data
    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        skiprows=n_header,
        names=columns,
    )
    
    if df.empty:
        return df
    
    # Replace missing values with NaN BEFORE applying scale factors
    # (missing values are in raw data units before scaling)
    for i, col in enumerate(columns):
        if i < len(missing_vals):
            df.loc[df[col] == missing_vals[i], col] = np.nan
    
    # Apply scale factors AFTER removing missing values
    for i, col in enumerate(columns):
        if i < len(scales):
            df[col] = df[col] * scales[i]
    
    # Create timestamp from UT (elapsed seconds from flight date)
    df["Timestamp"] = df["UT"].apply(
        lambda x: takeoff_date + timedelta(seconds=float(x)) if pd.notnull(x) else pd.NaT
    )
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
    
    # Rename columns to standard names
    df.rename(columns={
        "LAT": "Lat",
        "LONG": "Lon",
        "P_ALT": "Alt_m",
    }, inplace=True)
    
    return df[["Timestamp", "Lat", "Lon", "Alt_m"]]


def load_crystal_face_nasa_file(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load a single CRYSTAL-FACE NASA JLH file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the data file.
        
    Returns
    -------
    pd.DataFrame
        Parsed data with environmental measurements and computed Si.
    """
    filepath = Path(filepath)
    
    with open(filepath) as f:
        lines = f.readlines()
    
    # First line contains number of header lines
    n_header = int(lines[0].split()[0])
    
    # Parse column names from last header line
    columns = parse_columns_with_units(lines[n_header - 1])
    
    # Extract takeoff date from header
    takeoff_date = extract_takeoff_date(lines[:n_header])
    
    # Read data
    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        skiprows=n_header,
        names=columns,
    )
    
    # Find UTC seconds column
    ut_col = next((c for c in df.columns if c.lower().startswith("ut")), None)
    if ut_col is None:
        raise ValueError(f"No UT seconds column found in {filepath.name}")
    
    # Create timestamp
    df["Timestamp"] = df[ut_col].apply(
        lambda x: takeoff_date + timedelta(seconds=float(x)) if pd.notnull(x) else pd.NaT
    )
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
    
    # Calculate Si from RH (relative humidity w.r.t. ice)
    # JLH files typically have RH or %RH column
    rh_col = next((c for c in df.columns if "rh" in c.lower()), None)
    if rh_col:
        df["Si"] = si_from_rh(df[rh_col])
    
    # Convert temperature from Kelvin to Celsius if present
    temp_col = next((c for c in df.columns if c.lower() in ["t_k", "t"]), None)
    if temp_col:
        df["T_C"] = df[temp_col] - 273.15
    
    df["source_file"] = filepath.name
    
    return df


def load_crystal_face_nasa(
    data_dir: Union[str, Path], 
    mms_dir: Optional[Union[str, Path]] = None,
    pattern: str = "*"
) -> pd.DataFrame:
    """
    Load all CRYSTAL-FACE NASA files from a directory and optionally merge with MMS geolocation.
    
    Parameters
    ----------
    data_dir : str or Path
        Directory containing JLH data files.
    mms_dir : str or Path, optional
        Directory containing MMS geolocation data files. If None, looks in data_dir/MMS/.
    pattern : str, optional
        Glob pattern for matching files (default: "*").
        
    Returns
    -------
    pd.DataFrame
        Combined data from all files with optional geolocation from MMS.
    """
    data_dir = Path(data_dir)
    files = [f for f in data_dir.glob(pattern) if f.is_file()]
    
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' found in {data_dir}")
    
    # Load MMS geolocation data if available
    mms_geo = None
    if mms_dir is None:
        mms_dir = data_dir / "MMS"
    mms_dir = Path(mms_dir)
    
    if mms_dir.exists():
        mms_files = [f for f in mms_dir.glob("*") if f.is_file()]
        if mms_files:
            mms_dfs = []
            for f in sorted(mms_files):
                try:
                    mms_dfs.append(load_mms_file(f))
                except Exception as e:
                    print(f"Warning: Could not load MMS file {f.name}: {e}")
            if mms_dfs:
                mms_geo = pd.concat(mms_dfs, ignore_index=True)
    
    # Load JLH data
    dfs = []
    for f in sorted(files):
        try:
            dfs.append(load_crystal_face_nasa_file(f))
        except Exception as e:
            print(f"Warning: Could not load {f.name}: {e}")
    
    combined = pd.concat(dfs, ignore_index=True)
    
    # Merge with MMS geolocation data on nearest timestamp
    if mms_geo is not None and not mms_geo.empty:
        combined = pd.merge_asof(
            combined.sort_values("Timestamp"),
            mms_geo.sort_values("Timestamp"),
            on="Timestamp",
            direction="nearest",
            tolerance=pd.Timedelta(seconds=10),
        )
    
    combined["Campaign"] = "CRYSTAL-FACE-NASA"
    
    return combined


def extract_crystal_face_nasa_standard(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract standardized columns from CRYSTAL-FACE NASA data.
    
    If MMS data was merged, uses Lat/Lon/Alt_m from MMS.
    Otherwise, attempts to find position columns in JLH data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw data loaded by load_crystal_face_nasa.
        
    Returns
    -------
    pd.DataFrame
        Standardized data with Timestamp, Tair_C, Si, Lat, Lon, Alt_m, Campaign.
    """
    # Use Lat/Lon/Alt_m if they exist (from MMS merge), otherwise look for them
    lat_col = "Lat" if "Lat" in df.columns else next((c for c in df.columns if "lat" in c.lower()), None)
    lon_col = "Lon" if "Lon" in df.columns else next((c for c in df.columns if "lon" in c.lower()), None)
    alt_col = "Alt_m" if "Alt_m" in df.columns else next((c for c in df.columns if "alt" in c.lower() or "z" in c.lower()), None)
    
    result = pd.DataFrame({
        "Timestamp": df["Timestamp"],
        "Tair_C": df.get("T_C", np.nan),
        "Si": df.get("Si", np.nan),
        "Lat": df.get(lat_col, np.nan) if lat_col else np.nan,
        "Lon": df.get(lon_col, np.nan) if lon_col else np.nan,
        "Alt_m": df.get(alt_col, np.nan) if alt_col else np.nan,
        "Campaign": df.get("Campaign", "CRYSTAL-FACE-NASA"),
        "source_file": df.get("source_file", ""),
    })
    
    return result
