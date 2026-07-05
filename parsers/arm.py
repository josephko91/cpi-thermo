"""
ARM (Atmospheric Radiation Measurement) campaign data parser.

Campaign: SGP 2000 Spring Cloud Campaign
- Data Source: https://iop.arm.gov/2000/sgp/cloud
- Campaign website: https://www.arm.gov/research/campaigns/sgp2000sprcloud
    - in "poellot-citation/t4" subdirectory
- Data Format: Binary .t4archive.gz files (big-endian int32)
- Variables: Date, Time_sec, Air_Temp_Rosemount_C, Frost_Point_Cryo_C, GPS_Lat_deg, GPS_Lon_deg, GPS_Alt_m
- Derived: Si (ice supersaturation)
- Notes: 
"""

import gzip
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, List, Optional

from .utils import si_from_frost_point, es_ice_hPa, qv_from_e_P, sw_from_si


# Column definitions for ARM binary files
ARM_COLUMNS = [
    "Date",
    "Time_sec",
    "True_Air_Speed_m_s",
    "Wind_Speed_m_s",
    "Wind_Direction_deg",
    "INS_Heading_deg",
    "Pitch_deg",
    "Roll_deg",
    "Static_Pressure_mb",
    "Pressure_Altitude_m",
    "Air_Temp_Rosemount_C",
    "Dew_Point_EGG_C",
    "Dew_Point_Cryo_C",
    "Frost_Point_Cryo_C",
    "King_LWC_g_m3",
    "FSSP_LWC_g_m3",
    "Vertical_Wind_m_s",
    "Turbulence_eps",
    "Ozone_ppb",
    "CN_Conc_per_cm3",
    "INS_Latitude_deg",
    "INS_Longitude_deg",
    "GPS_Lat_deg",
    "GPS_Lon_deg",
    "GPS_Alt_m",
    "GPS_Ground_Speed_m_s",
    "FSSP_Conc_per_ml",
    "FSSP_Mean_Diam_um",
    "FSSP_Mean_Vol_Diam_um",
    "DC1_Conc_per_L",
    "DC1_Mean_Diam_um",
    "DC1_Mean_Vol_Diam_um",
    "DC2_Conc_per_L",
    "DC2_Mean_Diam_um",
    "DC2_Mean_Vol_Diam_um",
    "DC2_Shadow_Or_per_L",
    "DP2_Conc_per_L",
    "DP2_Mean_Diam_um",
    "DP2_Mean_Vol_Diam_um",
]


def _decode_arm_date(d: float) -> pd.Timestamp:
    """
    Decode ARM date format (YYMMDD) to datetime.
    
    ARM Spring 2000 campaign uses year 2000 + YY format.
    
    Parameters
    ----------
    d : float
        Date as YYMMDD integer.
        
    Returns
    -------
    pd.Timestamp
        Decoded date.
    """
    d = int(d)
    year = 2000 + (d // 10000)
    month = (d % 10000) // 100
    day = d % 100
    return pd.Timestamp(year, month, day)


def load_arm_file(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load a single ARM .t4archive.gz binary file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the .t4archive.gz file.
        
    Returns
    -------
    pd.DataFrame
        Parsed data with columns for environmental and positional data,
        including computed Timestamp and Si (ice supersaturation).
    """
    filepath = Path(filepath)
    
    with gzip.open(filepath, "rb") as f:
        raw = f.read()
    
    # Each record has 39 big-endian int32 values
    dtype = ">i4"
    record_length = 39
    
    data = np.frombuffer(raw, dtype=dtype)
    N = data.size // record_length
    data = data.reshape((N, record_length))
    
    # Convert to float and apply scaling
    # First column (Date) stays as-is, rest are scaled
    data_float = data.astype(float)
    data_float[:, 1:] = data_float[:, 1:] / 1000.0 - 100.0
    
    df = pd.DataFrame(data_float, columns=ARM_COLUMNS)
    
    # Decode date and create timestamp
    df["Date"] = df["Date"].apply(_decode_arm_date)
    df["Timestamp"] = df["Date"] + pd.to_timedelta(df["Time_sec"], unit="s")
    df["Timestamp"] = df["Timestamp"].dt.tz_localize("UTC")
    
    # Calculate ice supersaturation (cryo chilled mirror)
    df["Si_chilled_mirror"] = si_from_frost_point(
        df["Frost_Point_Cryo_C"],
        df["Air_Temp_Rosemount_C"],
    )
    df["Si"] = df["Si_chilled_mirror"]

    # Water vapor mixing ratio (g/kg) from frost point + static pressure
    df["P_hPa"] = df["Static_Pressure_mb"]  # mb == hPa
    e_cm = es_ice_hPa(df["Frost_Point_Cryo_C"])
    df["qv_chilled_mirror"] = qv_from_e_P(e_cm, df["P_hPa"])
    df["qv"] = df["qv_chilled_mirror"]

    # Supersaturation w.r.t. liquid water
    df["Sw"] = sw_from_si(df["Si"], df["Air_Temp_Rosemount_C"])
    
    # Add source file tracking
    df["source_file"] = filepath.name
    
    return df


def load_arm(data_dir: Union[str, Path], pattern: str = "*.t4archive.gz") -> pd.DataFrame:
    """
    Load all ARM campaign files from a directory.
    
    Parameters
    ----------
    data_dir : str or Path
        Directory containing ARM .t4archive.gz files.
    pattern : str, optional
        Glob pattern for matching files (default: "*.t4archive.gz").
        
    Returns
    -------
    pd.DataFrame
        Combined data from all files with standardized columns.
    """
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob(pattern))
    
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' found in {data_dir}")
    
    dfs = [load_arm_file(f) for f in files]
    combined = pd.concat(dfs, ignore_index=True)
    combined["Campaign"] = "ARM"
    
    return combined


def extract_arm_standard(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract standardized environmental and positional columns from ARM data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw ARM data loaded by load_arm.
        
    Returns
    -------
    pd.DataFrame
        Standardized data with columns:
        - Timestamp: UTC time
        - Tair_C: Air temperature (Celsius)
        - Si: Ice supersaturation
        - Lat: Latitude (degrees)
        - Lon: Longitude (degrees)
        - Alt_m: Altitude (meters)
        - Campaign: Campaign name
    """
    return pd.DataFrame({
        "Timestamp": df["Timestamp"],
        "Tair_C": df["Air_Temp_Rosemount_C"],
        "P_hPa": df.get("P_hPa", np.nan),
        "Si": df["Si"],
        "Si_chilled_mirror": df.get("Si_chilled_mirror", np.nan),
        "qv": df.get("qv", np.nan),
        "qv_chilled_mirror": df.get("qv_chilled_mirror", np.nan),
        "Sw": df.get("Sw", np.nan),
        "Lat": df["GPS_Lat_deg"],
        "Lon": df["GPS_Lon_deg"],
        "Alt_m": df["GPS_Alt_m"],
        "Campaign": df.get("Campaign", "ARM"),
        "source_file": df["source_file"],
    })
