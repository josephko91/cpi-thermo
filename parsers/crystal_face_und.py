"""
CRYSTAL-FACE UND (Citation aircraft) campaign data parser.

Campaign: CRYSTAL-FACE UND Citation aircraft
Data Source: https://espoarchive.nasa.gov/archive/browse/crystalf/Citation
Data Format: 
    - ND* files (MIS.CIT humidity data, MET.CIT meteorology data)
    - frost/dew point from both chilled mirror and laser hygrometer 
    - unclear which laser hygrometer was used for UND flights (potentially JLH?)
    - 
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
from typing import Union

from .utils import (
    clean_column_name,
    extract_takeoff_date,
    si_from_rh,
    es_ice_hPa,
    qv_from_e_P,
    sw_from_si,
    COMMON_NA_VALUES,
)


def _read_mis_cit_file(filepath: Path) -> pd.DataFrame:
    """Read a MIS.CIT humidity file."""
    with open(filepath, "r") as f:
        lines = f.readlines()
    
    n_header = int(lines[0].split()[0])
    columns = [clean_column_name(c) for c in lines[n_header - 2].split()]
    takeoff_date = extract_takeoff_date(lines[:n_header])
    
    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        skiprows=n_header,
        names=columns,
        na_values=COMMON_NA_VALUES,
    )
    
    df["source_file"] = filepath.name
    
    # Parse UTC timestamp
    if "Time" in df.columns:
        df["Timestamp"] = df["Time"].apply(
            lambda x: takeoff_date + timedelta(seconds=float(x)) if pd.notnull(x) else pd.NaT
        )
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
    
    return df


def _read_met_cit_file(filepath: Path) -> pd.DataFrame:
    """Read a MET.CIT meteorology file."""
    with open(filepath, "r") as f:
        lines = f.readlines()
    
    n_header = int(lines[0].split()[0])
    columns = [clean_column_name(c) for c in lines[n_header - 2].split()]
    takeoff_date = extract_takeoff_date(lines[:n_header])
    
    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        skiprows=n_header,
        names=columns,
        na_values=COMMON_NA_VALUES,
    )
    
    df["source_file"] = filepath.name
    
    # Parse UTC timestamp
    if "Time" in df.columns:
        df["Timestamp"] = df["Time"].apply(
            lambda x: takeoff_date + timedelta(seconds=float(x)) if pd.notnull(x) else pd.NaT
        )
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
    
    return df


def load_crystal_face_und_file(filepath_mis: Union[str, Path]) -> pd.DataFrame:
    """
    Load CRYSTAL-FACE UND data from MIS.CIT file and merge with MET.CIT.
    
    Parameters
    ----------
    filepath_mis : str or Path
        Path to the MIS.CIT humidity file.
        
    Returns
    -------
    pd.DataFrame
        Merged humidity and meteorology data with computed Si.
    """
    filepath_mis = Path(filepath_mis)

    # Read MIS.CIT (humidity data)
    df_mis = _read_mis_cit_file(filepath_mis)

    # Find corresponding NAV.CIT file (POS_Lat, POS_Lon, POS_Alt — Applanix
    # POS system). This campaign has no other altitude source, and NAV.CIT
    # was never previously read, leaving Alt_m/Lat/Lon NaN for the whole
    # campaign. Same ICARTT layout as MET.CIT, so the same reader applies.
    nav_filename = filepath_mis.name.replace("MIS.CIT", "NAV.CIT")
    nav_candidates = [
        filepath_mis.parent / nav_filename,
        filepath_mis.parent.parent / "ND_NAV" / nav_filename,
    ]
    nav_path = next((p for p in nav_candidates if p.exists()), None)
    if nav_path is not None:
        df_nav = _read_met_cit_file(nav_path)
        nav_cols_to_merge = [
            c for c in ("Timestamp", "POS_Lat", "POS_Lon", "POS_Alt") if c in df_nav.columns
        ]
        if len(nav_cols_to_merge) > 1:
            df_mis = pd.merge(df_mis, df_nav[nav_cols_to_merge], on="Timestamp", how="left")

    # Find corresponding MET.CIT file.
    # When files are in instrument subdirectories (ND_MIS/ and ND_MET/), we must
    # also replace the directory name, not just the filename suffix.
    met_filename = filepath_mis.name.replace("MIS.CIT", "MET.CIT")
    met_candidates = [
        filepath_mis.parent / met_filename,                    # same directory (flat layout)
        filepath_mis.parent.parent / "ND_MET" / met_filename,  # sibling ND_MET/ subdir
    ]
    met_path = next((p for p in met_candidates if p.exists()), met_candidates[0])

    if not met_path.exists():
        df_mis["Tair"] = np.nan
        if "RH" in df_mis.columns:
            df_mis["Si_LH_unspecified"] = si_from_rh(df_mis["RH"])
            df_mis["Si"] = df_mis["Si_LH_unspecified"]
        else:
            df_mis["Si_LH_unspecified"] = np.nan
            df_mis["Si"] = np.nan
        df_mis["Si_chilled_mirror"] = np.nan
        return df_mis
    
    # Read MET.CIT (meteorology data)
    df_met = _read_met_cit_file(met_path)
    
    # Find air temperature and pressure columns in MET file
    air_temp_col = next((c for c in df_met.columns if c.startswith("Air_Temp")), None)
    pres_col_met = next(
        (c for c in df_met.columns if c.lower() in ("static_pr", "p_hpa", "pressure", "pres")),
        None,
    )

    # Build list of MET columns to bring into MIS
    met_cols_to_merge = ["Timestamp"]
    if air_temp_col:
        met_cols_to_merge.append(air_temp_col)
    if pres_col_met:
        met_cols_to_merge.append(pres_col_met)

    if len(met_cols_to_merge) > 1:
        df_mis = pd.merge(
            df_mis,
            df_met[met_cols_to_merge],
            on="Timestamp",
            how="left",
        )

    df_mis["Tair"] = df_mis[air_temp_col] if air_temp_col else np.nan
    
    # Calculate Si from RH — laser hygrometer (instrument unspecified)
    if "RH" in df_mis.columns:
        df_mis["Si_LH_unspecified"] = si_from_rh(df_mis["RH"])
        df_mis["Si"] = df_mis["Si_LH_unspecified"]
    # Chilled-mirror Si not available in this dataset
    df_mis["Si_chilled_mirror"] = np.nan

    return df_mis


def load_crystal_face_und(
    data_dir: Union[str, Path],
    pattern: str = "*MIS.CIT"
) -> pd.DataFrame:
    """
    Load all CRYSTAL-FACE UND files from a directory.
    
    Parameters
    ----------
    data_dir : str or Path
        Directory containing *MIS.CIT files.
    pattern : str, optional
        Glob pattern for MIS.CIT files (default: "*MIS.CIT").
        
    Returns
    -------
    pd.DataFrame
        Combined data from all files.
    """
    data_dir = Path(data_dir)
    # Use rglob so files inside subdirectories (e.g. ND_MIS/) are found.
    files = list(data_dir.rglob(pattern))

    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' found in {data_dir}")
    
    dfs = []
    for f in sorted(files):
        try:
            dfs.append(load_crystal_face_und_file(f))
        except Exception as e:
            print(f"Warning: Could not load {f.name}: {e}")
    
    combined = pd.concat(dfs, ignore_index=True)
    combined["Campaign"] = "CRYSTAL-FACE-UND"
    
    return combined


def extract_crystal_face_und_standard(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract standardized columns from CRYSTAL-FACE UND data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw data loaded by load_crystal_face_und.
        
    Returns
    -------
    pd.DataFrame
        Standardized data with Timestamp, Tair_C, Si, Lat, Lon, Alt_m, Campaign.
    """
    # Find position columns
    lat_col = next((c for c in df.columns if "lat" in c.lower()), None)
    lon_col = next((c for c in df.columns if "lon" in c.lower()), None)
    alt_col = next((c for c in df.columns if "alt" in c.lower()), None)
    
    # Find pressure column (MET.CIT may have STATIC_PR, P_hPa, or similar)
    pres_col = next(
        (c for c in df.columns
         if c.lower() in ("static_pr", "p_hpa", "pressure", "pres", "staticpressure")),
        None,
    )
    if pres_col is None:
        pres_col = next(
            (c for c in df.columns if "press" in c.lower() or "static" in c.lower()),
            None,
        )
    p_hpa = df[pres_col] if pres_col else np.nan

    # qv_lh_unspecified from RH + T + P
    rh = df.get("RH")
    tair = df.get("Tair")
    if rh is not None and tair is not None and pres_col is not None:
        e_lh = (np.asarray(rh, dtype=float) / 100.0) * es_ice_hPa(tair)
        qv_lhu = qv_from_e_P(e_lh, p_hpa)
    else:
        qv_lhu = np.nan

    # Sw from Si and T
    sw = sw_from_si(df.get("Si", np.nan), df.get("Tair", np.nan))

    return pd.DataFrame({
        "Timestamp": df["Timestamp"],
        "Tair_C": df.get("Tair", np.nan),
        "P_hPa": p_hpa,
        "Si": df.get("Si", np.nan),
        "Si_LH_unspecified": df.get("Si_LH_unspecified", np.nan),
        "Si_chilled_mirror": df.get("Si_chilled_mirror", np.nan),
        "qv": qv_lhu,
        "qv_lh_unspecified": qv_lhu,
        "Sw": sw,
        "Lat": df.get(lat_col, np.nan) if lat_col else np.nan,
        "Lon": df.get(lon_col, np.nan) if lon_col else np.nan,
        "Alt_m": df.get(alt_col, np.nan) if alt_col else np.nan,
        "Campaign": df.get("Campaign", "CRYSTAL-FACE-UND"),
        "source_file": df["source_file"],
    })
