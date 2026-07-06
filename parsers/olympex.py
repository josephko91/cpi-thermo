"""
OLYMPEX (Olympic Mountains Experiment) campaign data parser.

Campaign: OLYMPEX Citation aircraft
Data Source: https://www.earthdata.nasa.gov/data/catalog/ghrc-daac-gpmcmolyx-1
Data Format: UND Citation data files with predefined column structure

Note: OLYMPEX water vapor measurements may have instrument issues.
- No apparent water vapor instrument data found.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
from typing import Union

from .utils import (
    extract_takeoff_date,
    si_from_frost_point,
    es_ice_hPa,
    qv_from_e_P,
    sw_from_si,
    COMMON_NA_VALUES,
)


# Predefined OLYMPEX column names
OLYMPEX_COLUMNS = [
    "Time", "Air_Temp", "MachNo_N", "IAS", "TAS", "Press_Alt", "Pot_Temp_T1",
    "STATIC_PR", "DEWPT", "REL_HUM", "MixingRatio", "DewPoint", "FrostPoint",
    "RH", "IceMSOFreq", "TSG_Date", "POS_Roll", "POS_Pitch", "POS_Head",
    "POSZ_Acc", "POS_Lat", "POS_Lon", "POS_Alt", "POS_Spd", "POS_Trk",
    "Alpha", "Beta", "VERT_VEL", "Wind_Z", "Wind_M", "Wind_D", "TURB",
    "King_LWC_ad", "Nev_TWC", "Nev_LWCcor", "Nev_IWC", "CSI_M_Ratio",
    "CSI_CWC", "CDP_Conc", "CDP_LWC", "CDP_MenD", "CDP_VolDia", "CDP_EffRad",
    "2-DC_Conc", "2-DC_MenD", "2-DC_VolDia", "2-DC_EffRad", "Nt2DSHGT105",
    "Nt2DSH_all", "Nt2DSVGT105", "Nt2DSV_all", "Nt_HVPS3H", "Nt_HVPS3BV",
]


def load_olympex_file(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load a single OLYMPEX data file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the OLYMPEX data file.
        
    Returns
    -------
    pd.DataFrame
        Parsed data with computed Si.
    """
    filepath = Path(filepath)
    
    with open(filepath, "r") as f:
        lines = f.readlines()
    
    n_header = int(lines[0].split()[0])
    takeoff_date = extract_takeoff_date(lines[:n_header])
    
    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        skiprows=n_header,
        names=OLYMPEX_COLUMNS,
        na_values=COMMON_NA_VALUES,
    )
    
    df["source_file"] = filepath.name
    
    # Parse UTC timestamp
    if "Time" in df.columns:
        df["Timestamp"] = df["Time"].apply(
            lambda x: takeoff_date + timedelta(seconds=float(x)) if pd.notnull(x) else pd.NaT
        )
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
    
    # Calculate Si from frost point (instrument unspecified)
    if "FrostPoint" in df.columns and "Air_Temp" in df.columns:
        df["Si_frost_point"] = si_from_frost_point(df["FrostPoint"], df["Air_Temp"])
        # Plausibility bound, matching the [-1, 5] convention used for the
        # equivalent chilled-mirror Si in other campaigns (e.g. IPHEX).
        df.loc[
            (df["Si_frost_point"] < -1.0) | (df["Si_frost_point"] > 5.0),
            "Si_frost_point",
        ] = np.nan
        # Physical implausibility: FrostPoint noticeably above Air_Temp at a
        # non-cirrus (near-0C or warmer) temperature is a mirror-fault
        # signature, not real supersaturation -- e.g. flight
        # 15_12_13_19_51_41 shows FrostPoint 8.9-11.6C above Air_Temp at
        # Air_Temp = -0.2 to -2.3C (83 rows), which would require >100%
        # supersaturation w.r.t. liquid water sustained near 0C. Contrast
        # with flight 15_12_13_15_39_28 (45 rows), which shows a smaller
        # FrostPoint-Air_Temp gap (6.3-6.7C) at Air_Temp ~-44C -- cold
        # enough that chilled-mirror hysteresis can plausibly explain the
        # gap without invoking a fault, so that flight is deliberately left
        # unmasked here (see docs/decisions/2026-07-05-qc9-iphex-olympex.md).
        mirror_fault = (
            df["FrostPoint"].notna() & df["Air_Temp"].notna()
            & ((df["FrostPoint"] - df["Air_Temp"]) > 5.0) & (df["Air_Temp"] > -10.0)
        )
        n_fault = int(mirror_fault.sum())
        if n_fault > 0:
            df.loc[mirror_fault, "Si_frost_point"] = np.nan
            print(
                f"  OLYMPEX QC: nulled {n_fault:,} rows where FrostPoint exceeded "
                f"Air_Temp by >5°C at Air_Temp > -10°C (chilled-mirror fault — "
                f"check source file)"
            )
        df["Si"] = df["Si_frost_point"]
    
    return df


def load_olympex(
    data_dir: Union[str, Path],
    pattern: str = "*"
) -> pd.DataFrame:
    """
    Load all OLYMPEX files from a directory.
    
    Parameters
    ----------
    data_dir : str or Path
        Directory containing OLYMPEX data files.
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
            dfs.append(load_olympex_file(f))
        except Exception as e:
            print(f"Warning: Could not load {f.name}: {e}")
    
    combined = pd.concat(dfs, ignore_index=True)
    combined["Campaign"] = "OLYMPEX"
    
    return combined


def extract_olympex_standard(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract standardized columns from OLYMPEX data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw data loaded by load_olympex.
        
    Returns
    -------
    pd.DataFrame
        Standardized data with Timestamp, Tair_C, Si, Lat, Lon, Alt_m, Campaign.
    """
    # qv_frost_point from FrostPoint + STATIC_PR
    fp = df.get("FrostPoint")
    p_hpa = df.get("STATIC_PR")
    if fp is not None and p_hpa is not None:
        e_fp = es_ice_hPa(fp)
        qv_fp = qv_from_e_P(e_fp, p_hpa)
        # qv_fp is derived from the same FrostPoint/STATIC_PR pair as
        # Si_frost_point, which is clipped to a plausible [-1, 5] range above.
        # qv_from_e_P has no upper bound of its own, so propagate the Si
        # clip's NaN mask to keep the two consistent.
        si_fp = df.get("Si_frost_point")
        if si_fp is not None:
            qv_fp = pd.Series(qv_fp, index=df.index, dtype=float)
            qv_fp = qv_fp.where(si_fp.notna(), np.nan)
    else:
        qv_fp = np.nan

    # Sw from Si and T
    sw = sw_from_si(df.get("Si", np.nan), df.get("Air_Temp", np.nan))

    return pd.DataFrame({
        "Timestamp": df["Timestamp"],
        "Tair_C": df.get("Air_Temp", np.nan),
        "P_hPa": df.get("STATIC_PR", np.nan),
        "Si": df.get("Si", np.nan),
        "Si_frost_point": df.get("Si_frost_point", np.nan),
        "qv": qv_fp,
        "qv_frost_point": qv_fp,
        "Sw": sw,
        "Lat": df.get("POS_Lat", np.nan),
        "Lon": df.get("POS_Lon", np.nan),
        "Alt_m": df.get("POS_Alt", np.nan),
        "Campaign": df.get("Campaign", "OLYMPEX"),
        "source_file": df["source_file"],
    })
