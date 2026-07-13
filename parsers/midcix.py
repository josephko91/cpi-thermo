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
    round_timestamp_to_second,
    COMMON_NA_VALUES,
)
from .crystal_face_nasa import load_mms_file


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
    df["Timestamp"] = round_timestamp_to_second(df["Timestamp"])
    df = df.dropna(subset=["Timestamp"]).drop_duplicates(subset=["Timestamp"], keep="first")

    # Calculate Si from RH — JPL Laser Hygrometer (JLH)
    if "RH" in df.columns:
        df["Si_JLH"] = si_from_rh(df["RH"])
        df["Si"] = df["Si_JLH"]

    # Convert temperature from Kelvin to Celsius
    if "T_K" in df.columns:
        df["T_C"] = df["T_K"] - 273.15

    # Merge position (Lat, Lon, Alt_m) from the corresponding FP navigation
    # file. JW files carry no position data of their own; FP*.WB57 files
    # (MMS flight path: P_ALT, LAT, LONG, TAS) share the same date-based
    # filename suffix and the same UT-seconds-from-midnight timestamp scheme.
    nav_filename = filepath.name.replace("JW", "FP", 1)
    nav_candidates = [
        filepath.parent / nav_filename,
        filepath.parent / "FP" / nav_filename,
    ]
    nav_path = next((p for p in nav_candidates if p.exists()), None)
    if nav_path is not None:
        try:
            nav_df = load_mms_file(nav_path)
            nav_df = nav_df.dropna(subset=["Timestamp"]).drop_duplicates(subset=["Timestamp"], keep="first")
            # Exact-second outer merge -- no merge_asof tolerance. A second
            # reported by only one of JW/FP becomes a row with that source's
            # columns filled and the other NaN, consistent with the rest of
            # the pipeline's no-fabricated-precision policy.
            df = pd.merge(
                df,
                nav_df[["Timestamp", "Lat", "Lon", "Alt_m", "TAS"]],
                on="Timestamp",
                how="outer",
            ).sort_values("Timestamp").reset_index(drop=True)
        except Exception as e:
            print(f"  Warning: Could not merge MIDCIX nav file {nav_path.name}: {e}")

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

    # FP/ (navigation) covers 2 more flight dates than the JW (water vapor)
    # files above (2004-04-22, 2004-04-27) -- JLH wasn't operating those
    # days, so no Tair_C/Si/qv is possible, but position data still exists
    # and was previously dropped entirely since load_midcix_file only emits
    # rows keyed to JW timestamps. Add position-only rows for those dates
    # so CPI images on them at least get Lat/Lon/Alt_m.
    fp_dir = data_dir / "FP"
    if fp_dir.exists():
        jw_dates = set(combined["Timestamp"].dropna().dt.date.unique())
        fp_files = [f for f in fp_dir.glob("*.WB57") if f.is_file()]
        fp_extra_dfs = []
        for f in sorted(fp_files):
            try:
                fp_df = load_mms_file(f)
            except Exception as e:
                print(f"Warning: Could not load FP file {f.name}: {e}")
                continue
            fp_dates = fp_df["Timestamp"].dropna().dt.date.unique()
            if len(fp_dates) and fp_dates[0] not in jw_dates:
                fp_df = fp_df.copy()
                fp_df["source_file"] = f"FP-fallback:{f.name}"
                fp_extra_dfs.append(fp_df)
        if fp_extra_dfs:
            fp_extra = pd.concat(fp_extra_dfs, ignore_index=True)
            fp_extra["Campaign"] = "MIDCIX"
            combined = pd.concat([combined, fp_extra], ignore_index=True, sort=False)
            print(f"  FP fallback: added {len(fp_extra):,} position-only rows for "
                  f"{fp_extra['Timestamp'].dt.date.nunique()} JW-absent dates")

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
        "TAS_ms": df.get("TAS", np.nan),
        "Campaign": df.get("Campaign", "MIDCIX"),
        "source_file": df["source_file"],
    })
