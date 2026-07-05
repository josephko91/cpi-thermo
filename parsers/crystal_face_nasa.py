"""
CRYSTAL-FACE NASA (WB-57) campaign data parser.

Campaign: CRYSTAL-FACE NASA WB-57 aircraft
Data Source: https://espoarchive.nasa.gov/archive/browse/crystalf/WB57
Data:
    - Water Vapor
        - JPL Laser Hygrometer (JLH) data (only ~16% coverage for water vapor)
        - ALIAS (Airborne Laser Absorption Spectrometer) data 
        - HW (Harvard Water Vapor) data (5 second intervals)
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
    si_from_ppmv,
    es_ice_hPa,
    qv_from_ppmv,
    qv_from_e_P,
    sw_from_si,
)


def _round_timestamp_to_second(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce").dt.round("s")


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
    
    # Extract takeoff date from header
    takeoff_date = extract_takeoff_date(lines)

    # Parse scale factors and missing values from the labeled header lines.
    scale_line = next((line for line in lines if "scale factors" in line.lower()), None)
    missing_line = next((line for line in lines if "missing values" in line.lower()), None)
    if scale_line is None or missing_line is None:
        raise ValueError(f"Could not locate scale/missing value lines in {filepath.name}")

    scales = [float(s) for s in scale_line.split(";")[0].split()]
    missing_vals = [float(m) for m in missing_line.split(";")[0].split()]

    # Find the line that declares the actual data columns.
    data_header_idx = next(
        (
            i
            for i, line in enumerate(lines)
            if line.strip().startswith("UT")
            and "P_ALT" in line
            and "LAT" in line
            and "LONG" in line
            and "TAS" in line
        ),
        None,
    )
    if data_header_idx is None:
        raise ValueError(f"Could not locate MMS data header in {filepath.name}")
    
    # Column names for MMS data
    columns = ["UT", "P_ALT", "LAT", "LONG", "TAS"]
    
    # Read data
    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        skiprows=data_header_idx + 1,
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
    df["Timestamp"] = _round_timestamp_to_second(df["Timestamp"])
    # record source filename for MMS rows
    df["source_file"] = filepath.name
    
    # Rename columns to standard names
    df.rename(columns={
        "LAT": "Lat",
        "LONG": "Lon",
        "P_ALT": "Alt_m",
    }, inplace=True)
    
    return df[["Timestamp", "Lat", "Lon", "Alt_m"]]


def load_hw_file(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load a CRYSTAL-FACE Harvard Water Vapor (HW) file.

    Format: ICARTT 1001, two columns — Time (UT seconds from midnight) and
    WV (water vapor volume mixing ratio, raw units; multiply by scale to get ppmv).
    Missing value: 999999 (raw).
    """
    filepath = Path(filepath)
    with open(filepath) as f:
        lines = f.readlines()

    n_header = int(lines[0].split()[0])
    takeoff_date = extract_takeoff_date(lines[:n_header])

    # Scale factors and missing values are on fixed header lines (ICARTT 1001).
    # Scale line is line index 10 (0-based), missing line is index 11.
    try:
        scale = float(lines[10].strip())
        missing_raw = float(lines[11].strip())
    except (IndexError, ValueError):
        scale, missing_raw = 0.01, 999999.0

    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        skiprows=n_header,
        names=["Time", "WV"],
    )
    df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
    df["WV"] = pd.to_numeric(df["WV"], errors="coerce")
    df.loc[df["WV"] == missing_raw, "WV"] = np.nan
    df["H2O_ppmv"] = df["WV"] * scale
    df.loc[df["H2O_ppmv"] <= 0, "H2O_ppmv"] = np.nan

    df["Timestamp"] = df["Time"].apply(
        lambda x: takeoff_date + timedelta(seconds=float(x)) if pd.notnull(x) else pd.NaT
    )
    df["Timestamp"] = _round_timestamp_to_second(df["Timestamp"])
    df["source_file"] = filepath.name
    return df[["Timestamp", "H2O_ppmv", "source_file"]]


def load_mm_met_file(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load a CRYSTAL-FACE MMS meteorology (MM) file.

    Format: ICARTT 1001, 7 columns — GMT, Psta, Tsta, Thta, U, V, W.
    Scale factors from header: Psta×0.1→hPa, Tsta×0.01→K.
    Missing value: 99999 for Psta/Tsta (raw).
    """
    filepath = Path(filepath)
    with open(filepath) as f:
        lines = f.readlines()

    n_header = int(lines[0].split()[0])
    takeoff_date = extract_takeoff_date(lines[:n_header])

    # Locate scale factor and missing value lines in the header.
    # In MM files, scale factors come first, then missing values.
    def _parse_numbers(line: str) -> list:
        # Strip inline comments (;...) before splitting
        return [float(s) for s in line.split(";")[0].split() if s]

    scales_line = next(
        (l for l in lines[:n_header] if l.strip().startswith("0.1")), None
    )
    missing_line = next(
        (l for l in lines[:n_header] if l.strip().startswith("99999")), None
    )
    scales = _parse_numbers(scales_line) if scales_line else [1.0] * 6
    missing_vals = _parse_numbers(missing_line) if missing_line else [99999.0] * 6

    data_header_idx = next(
        (i for i, l in enumerate(lines) if "GMT" in l and "Psta" in l and "Tsta" in l),
        None,
    )
    if data_header_idx is None:
        raise ValueError(f"Could not locate MM data header in {filepath.name}")

    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        skiprows=data_header_idx + 1,
        names=["GMT", "Psta", "Tsta", "Thta", "U", "V", "W"],
    )
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Replace missing values and apply scale factors for P and T
    for i, col in enumerate(["Psta", "Tsta"]):
        if i < len(missing_vals):
            df.loc[df[col] == missing_vals[i], col] = np.nan
        if i < len(scales):
            df[col] = df[col] * scales[i]

    df["P_hPa"] = df["Psta"]      # already in hPa after scaling
    df["T_K"] = df["Tsta"]        # already in K after scaling
    df.loc[(df["T_K"] < 150) | (df["T_K"] > 350), "T_K"] = np.nan
    df.loc[(df["P_hPa"] < 1) | (df["P_hPa"] > 1100), "P_hPa"] = np.nan

    df["Timestamp"] = df["GMT"].apply(
        lambda x: takeoff_date + timedelta(seconds=float(x)) if pd.notnull(x) else pd.NaT
    )
    df["Timestamp"] = _round_timestamp_to_second(df["Timestamp"])
    df["source_file"] = filepath.name
    return df[["Timestamp", "P_hPa", "T_K", "source_file"]]


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
    df["Timestamp"] = _round_timestamp_to_second(df["Timestamp"])
    
    # Calculate Si from RH (relative humidity w.r.t. ice) — JLH instrument
    rh_col = next((c for c in df.columns if "rh" in c.lower()), None)
    if rh_col:
        df["Si_JLH"] = si_from_rh(df[rh_col])
        df["RH_JLH"] = df[rh_col]  # store for qv_jlh computation later
    
    # Convert temperature from Kelvin to Celsius if present
    temp_col = next((c for c in df.columns if c.lower() in ["t_k", "t"]), None)
    if temp_col:
        df["T_C"] = df[temp_col] - 273.15
    
    df["source_file"] = filepath.name
    
    return df


def load_nm_met_file(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load a CRYSTAL-FACE NOAA Navigational Met (NM) file.

    Format: ICARTT 1001, columns — UT, Press(mb), Temp(degC), PotTemp, WindSpd, WindDir, Mach.
    Scale factors: all 1.0. Missing: 9999.999.
    Coverage: full flight at 1 Hz (vs MM which may be truncated).
    Note: labelled "unknown accuracy" — use as fallback after MM.
    """
    filepath = Path(filepath)
    with open(filepath) as f:
        lines = f.readlines()

    n_header = int(lines[0].split()[0])
    takeoff_date = extract_takeoff_date(lines[:n_header])

    data_header_idx = next(
        (i for i, l in enumerate(lines) if "UT" in l and "Press" in l and "Temp" in l),
        None,
    )
    if data_header_idx is None:
        raise ValueError(f"Could not locate NM data header in {filepath.name}")

    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        skiprows=data_header_idx + 1,
        names=["UT", "Press", "Temp", "PotTemp", "WindSpd", "WindDir", "Mach"],
    )
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.loc[df["Temp"] == 9999.999, "Temp"] = np.nan
    df.loc[df["Press"] == 9999.999, "Press"] = np.nan
    df.loc[(df["Temp"] < -100) | (df["Temp"] > 50), "Temp"] = np.nan
    df.loc[(df["Press"] < 1) | (df["Press"] > 1100), "Press"] = np.nan

    df["T_K"] = df["Temp"] + 273.15
    df["P_hPa"] = df["Press"]

    df["Timestamp"] = df["UT"].apply(
        lambda x: takeoff_date + timedelta(seconds=float(x)) if pd.notnull(x) else pd.NaT
    )
    df["Timestamp"] = _round_timestamp_to_second(df["Timestamp"])
    df["source_file"] = filepath.name
    return df[["Timestamp", "P_hPa", "T_K", "source_file"]]


def load_crystal_face_nasa(
    data_dir: Union[str, Path],
    mms_dir: Optional[Union[str, Path]] = None,
    hw_dir: Optional[Union[str, Path]] = None,
    mm_dir: Optional[Union[str, Path]] = None,
    nm_dir: Optional[Union[str, Path]] = None,
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
    # Use rglob so files inside subdirectories (e.g. JW/) are found.
    files = [f for f in data_dir.rglob(pattern) if f.is_file()]

    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' found in {data_dir}")

    # -----------------------------------------------------------------------
    # Resolve sibling instrument directories relative to data_dir (JW/).
    # data_dir.parent = CRYSTAL-FACE-NASA/
    # -----------------------------------------------------------------------
    campaign_root = data_dir.parent

    # SP/ → GPS position (lat, lon, alt) for MMS geolocation
    if mms_dir is None:
        for candidate in (data_dir / "MMS", campaign_root / "SP"):
            if candidate.exists():
                mms_dir = candidate
                break
    if mms_dir is not None:
        mms_dir = Path(mms_dir)

    # HW/ → Harvard Water Vapor (H2O ppmv, 10 s resolution)
    if hw_dir is None:
        hw_candidate = campaign_root / "HW"
        if hw_candidate.exists():
            hw_dir = hw_candidate
    if hw_dir is not None:
        hw_dir = Path(hw_dir)

    # MM/ → MMS meteorology (P hPa, T K, 1 Hz)
    if mm_dir is None:
        mm_candidate = campaign_root / "MM"
        if mm_candidate.exists():
            mm_dir = mm_candidate
    if mm_dir is not None:
        mm_dir = Path(mm_dir)

    # NM/ → NOAA navigational met (P mb, T degC, 1 Hz, full-flight fallback)
    if nm_dir is None:
        nm_candidate = campaign_root / "NM"
        if nm_candidate.exists():
            nm_dir = nm_candidate
    if nm_dir is not None:
        nm_dir = Path(nm_dir)

    # -----------------------------------------------------------------------
    # Load SP geolocation (position only)
    # -----------------------------------------------------------------------
    mms_geo = None
    if mms_dir is not None and mms_dir.exists():
        mms_files = [f for f in mms_dir.glob("*.WB57") if f.is_file()]
        if mms_files:
            mms_dfs = []
            for f in sorted(mms_files):
                try:
                    mms_dfs.append(load_mms_file(f))
                except Exception as e:
                    print(f"Warning: Could not load SP file {f.name}: {e}")
            if mms_dfs:
                mms_geo = pd.concat(mms_dfs, ignore_index=True)

    # -----------------------------------------------------------------------
    # Load MM meteorology (P and T for HW Si calculation)
    # -----------------------------------------------------------------------
    mm_met = None
    if mm_dir is not None and mm_dir.exists():
        mm_files = [f for f in mm_dir.glob("*.WB57") if f.is_file()]
        if mm_files:
            mm_dfs = []
            for f in sorted(mm_files):
                try:
                    mm_dfs.append(load_mm_met_file(f))
                except Exception as e:
                    print(f"Warning: Could not load MM file {f.name}: {e}")
            if mm_dfs:
                mm_met = pd.concat(mm_dfs, ignore_index=True).dropna(subset=["Timestamp"]).sort_values("Timestamp")
                print(f"  MM met: {len(mm_met):,} records across "
                      f"{mm_met['Timestamp'].dt.date.nunique()} dates")

    # -----------------------------------------------------------------------
    # Load NM navigational met (NOAA, full-flight fallback for T and P)
    # -----------------------------------------------------------------------
    nm_met = None
    if nm_dir is not None and nm_dir.exists():
        nm_files = [f for f in nm_dir.glob("*.WB57") if f.is_file()]
        if nm_files:
            nm_dfs = []
            for f in sorted(nm_files):
                try:
                    nm_dfs.append(load_nm_met_file(f))
                except Exception as e:
                    print(f"Warning: Could not load NM file {f.name}: {e}")
            if nm_dfs:
                nm_met = pd.concat(nm_dfs, ignore_index=True).dropna(subset=["Timestamp"]).sort_values("Timestamp")
                print(f"  NM met: {len(nm_met):,} records across "
                      f"{nm_met['Timestamp'].dt.date.nunique()} dates")

    # Build combined T/P: MM primary, NM fills gaps
    tp_met = mm_met
    if nm_met is not None:
        if tp_met is None:
            tp_met = nm_met
        else:
            # Merge NM into MM: fill MM NaN T/P from NM, and append NM rows
            # for timestamps absent from MM (e.g. Jul 29 first 3.5 h).
            tp_combined = pd.merge_asof(
                nm_met.sort_values("Timestamp"),
                mm_met[["Timestamp", "P_hPa", "T_K"]].rename(
                    columns={"P_hPa": "P_hPa_mm", "T_K": "T_K_mm"}
                ).sort_values("Timestamp"),
                on="Timestamp",
                direction="nearest",
                tolerance=pd.Timedelta(seconds=5),
            )
            # Use MM where available, NM otherwise
            tp_combined["P_hPa"] = tp_combined["P_hPa_mm"].where(
                tp_combined["P_hPa_mm"].notna(), tp_combined["P_hPa"]
            )
            tp_combined["T_K"] = tp_combined["T_K_mm"].where(
                tp_combined["T_K_mm"].notna(), tp_combined["T_K"]
            )
            tp_combined = tp_combined[["Timestamp", "P_hPa", "T_K"]].dropna(subset=["Timestamp"])
            # Also keep original MM rows (higher-quality) for complete coverage
            tp_met = pd.concat(
                [mm_met[["Timestamp", "P_hPa", "T_K"]], tp_combined],
                ignore_index=True,
            ).sort_values("Timestamp").drop_duplicates(subset=["Timestamp"], keep="first")

    # -----------------------------------------------------------------------
    # Load HW water vapour and compute Si via HW + T/P met (MM primary, NM fallback)
    # -----------------------------------------------------------------------
    hw_si = None
    if hw_dir is not None and hw_dir.exists() and tp_met is not None:
        hw_files = [f for f in hw_dir.glob("*.WB57") if f.is_file()]
        if hw_files:
            hw_dfs = []
            for f in sorted(hw_files):
                try:
                    hw_dfs.append(load_hw_file(f))
                except Exception as e:
                    print(f"Warning: Could not load HW file {f.name}: {e}")
            if hw_dfs:
                hw_all = pd.concat(hw_dfs, ignore_index=True).sort_values("Timestamp")
                # Merge HW ppmv with T/P met (nearest within 30 s — HW is 10 s resolution)
                hw_mm = pd.merge_asof(
                    hw_all,
                    tp_met[["Timestamp", "P_hPa", "T_K"]].sort_values("Timestamp"),
                    on="Timestamp",
                    direction="nearest",
                    tolerance=pd.Timedelta(seconds=30),
                )
                valid = hw_mm["H2O_ppmv"].notna() & hw_mm["T_K"].notna() & hw_mm["P_hPa"].notna()
                hw_mm["Si_HW"] = np.nan
                if valid.sum() > 0:
                    hw_mm.loc[valid, "Si_HW"] = si_from_ppmv(
                        hw_mm.loc[valid, "H2O_ppmv"],
                        hw_mm.loc[valid, "T_K"],
                        hw_mm.loc[valid, "P_hPa"],
                    )
                    hw_mm.loc[
                        (hw_mm["Si_HW"] < -1.0) | (hw_mm["Si_HW"] > 2.0), "Si_HW"
                    ] = np.nan
                # Also derive T_C from MM for HW rows
                hw_mm["T_C_HW"] = hw_mm["T_K"] - 273.15
                # qv_hw from ppmv (no pressure needed)
                hw_mm["qv_hw"] = qv_from_ppmv(hw_mm["H2O_ppmv"])
                hw_si = hw_mm[["Timestamp", "Si_HW", "T_C_HW", "qv_hw"]].dropna(subset=["Si_HW"])
                print(f"  HW Si: {len(hw_si):,} valid records across "
                      f"{hw_si['Timestamp'].dt.date.nunique()} dates")

    # -----------------------------------------------------------------------
    # Load JLH data (primary Si source)
    # -----------------------------------------------------------------------
    dfs = []
    for f in sorted(files):
        try:
            dfs.append(load_crystal_face_nasa_file(f))
        except Exception as e:
            print(f"Warning: Could not load {f.name}: {e}")

    combined = pd.concat(dfs, ignore_index=True)

    # Merge with SP geolocation on nearest timestamp
    if mms_geo is not None and not mms_geo.empty:
        combined = pd.merge_asof(
            combined.sort_values("Timestamp"),
            mms_geo.sort_values("Timestamp"),
            on="Timestamp",
            direction="nearest",
            tolerance=pd.Timedelta(seconds=10),
        )

    # -----------------------------------------------------------------------
    # Build HW-derived rows for timestamps not covered by JLH
    # and append them so Si coverage extends into JLH gaps and Jul 29.
    # -----------------------------------------------------------------------
    if hw_si is not None and not hw_si.empty:
        # Rows in HW that have no nearby JLH timestamp (>10 s gap)
        jlh_ts = combined["Timestamp"].dropna().sort_values()
        hw_si_sorted = hw_si.sort_values("Timestamp")
        hw_check = pd.merge_asof(
            hw_si_sorted,
            pd.DataFrame({"Timestamp": jlh_ts}),
            on="Timestamp",
            direction="nearest",
            tolerance=pd.Timedelta(seconds=10),
            suffixes=("", "_jlh"),
        )
        # Keep HW rows where JLH had no nearby match (i.e. JLH gap or missing date)
        hw_gap_rows = hw_check[hw_check["Timestamp"].isna() == False].copy()
        # Actually we want rows where JLH timestamp is absent after merge; since
        # merge_asof always fills right key, check using a different approach:
        # just fill all JLH-Si NaN using HW after final merge below.
        # Instead, build HW-only frame for dates absent from JLH entirely.
        jlh_dates = set(combined["Timestamp"].dropna().dt.date.unique())
        hw_extra = hw_si[~hw_si["Timestamp"].dt.date.isin(jlh_dates)].copy()
        if not hw_extra.empty:
            # These are dates (e.g. Jul 29) with no JLH data at all
            hw_extra_std = pd.DataFrame({
                "Timestamp": hw_extra["Timestamp"],
                "Si_HW": hw_extra["Si_HW"],
                "T_C": hw_extra["T_C_HW"],
                "qv_hw": hw_extra.get("qv_hw", np.nan),
                "source_file": "HW-fallback",
            })
            if mms_geo is not None and not mms_geo.empty:
                hw_extra_std = pd.merge_asof(
                    hw_extra_std.sort_values("Timestamp"),
                    mms_geo.sort_values("Timestamp"),
                    on="Timestamp",
                    direction="nearest",
                    tolerance=pd.Timedelta(seconds=10),
                )
            combined = pd.concat([combined, hw_extra_std], ignore_index=True, sort=False)
            print(f"  HW fallback: added {len(hw_extra_std):,} rows for "
                  f"{hw_extra['Timestamp'].dt.date.nunique()} JLH-absent dates")

        # Add HW rows for time gaps within shared dates.
        # JLH gaps are missing rows (not NaN), so find HW timestamps that have
        # no nearby JLH timestamp and add them as new rows.
        combined_sorted = combined.sort_values("Timestamp")
        jlh_ts_sorted = combined_sorted["Timestamp"].dropna().sort_values()
        hw_all_shared = hw_si[hw_si["Timestamp"].dt.date.isin(jlh_dates)].copy()
        if not hw_all_shared.empty:
            hw_all_shared_sorted = hw_all_shared.sort_values("Timestamp")
            jlh_ts_sorted_ns = jlh_ts_sorted.dt.as_unit("ns")
            hw_all_shared_ns = hw_all_shared_sorted.copy()
            hw_all_shared_ns["Timestamp"] = hw_all_shared_ns["Timestamp"].dt.as_unit("ns")

            check = pd.merge_asof(
                hw_all_shared_ns,
                pd.DataFrame({"JLH_ts": jlh_ts_sorted_ns}),
                left_on="Timestamp",
                right_on="JLH_ts",
                direction="nearest",
                tolerance=pd.Timedelta(seconds=10),
            )
            # HW rows with no nearby JLH timestamp = inside a JLH gap
            hw_in_gaps = check[check["JLH_ts"].isna()].drop(columns=["JLH_ts"])
            # Restore original timestamp resolution
            hw_in_gaps = hw_in_gaps.copy()
            hw_in_gaps["Timestamp"] = hw_in_gaps["Timestamp"].dt.as_unit("us")
            if not hw_in_gaps.empty:
                hw_gap_std = pd.DataFrame({
                    "Timestamp": hw_in_gaps["Timestamp"],
                    "Si_HW": hw_in_gaps["Si_HW"],
                    "T_C": hw_in_gaps["T_C_HW"],
                    "qv_hw": hw_in_gaps.get("qv_hw", np.nan),
                    "source_file": "HW-gap-fill",
                })
                if mms_geo is not None and not mms_geo.empty:
                    hw_gap_std = pd.merge_asof(
                        hw_gap_std.sort_values("Timestamp"),
                        mms_geo.sort_values("Timestamp"),
                        on="Timestamp",
                        direction="nearest",
                        tolerance=pd.Timedelta(seconds=10),
                    )
                combined = pd.concat([combined, hw_gap_std], ignore_index=True, sort=False)
                print(f"  HW gap-fill: added {len(hw_gap_std):,} rows for JLH gaps on shared dates")

    combined["Timestamp"] = _round_timestamp_to_second(combined["Timestamp"])

    # Merge Si_HW (and qv_hw) into JLH rows that don't already have it
    hw_merge_cols = ["Timestamp", "Si_HW", "qv_hw"]
    if hw_si is not None and not hw_si.empty:
        available_hw_cols = [c for c in hw_merge_cols if c in hw_si.columns]
        if "Si_HW" not in combined.columns:
            combined = pd.merge_asof(
                combined.sort_values("Timestamp"),
                hw_si[available_hw_cols].sort_values("Timestamp"),
                on="Timestamp",
                direction="nearest",
                tolerance=pd.Timedelta(seconds=30),
            )
        else:
            drop_cols = [c for c in available_hw_cols if c != "Timestamp" and c in combined.columns]
            tmp = pd.merge_asof(
                combined.sort_values("Timestamp").drop(columns=drop_cols),
                hw_si[available_hw_cols].sort_values("Timestamp"),
                on="Timestamp",
                direction="nearest",
                tolerance=pd.Timedelta(seconds=30),
            )
            combined = tmp

    # Merge P_hPa from met data into combined for qv_jlh computation
    if tp_met is not None:
        p_data = tp_met[["Timestamp", "P_hPa"]].dropna(subset=["P_hPa"]).sort_values("Timestamp")
        if not p_data.empty:
            if "P_hPa" not in combined.columns:
                combined = pd.merge_asof(
                    combined.sort_values("Timestamp"),
                    p_data,
                    on="Timestamp",
                    direction="nearest",
                    tolerance=pd.Timedelta(seconds=60),
                )
            else:
                tmp_p = pd.merge_asof(
                    combined.sort_values("Timestamp").drop(columns=["P_hPa"]),
                    p_data,
                    on="Timestamp",
                    direction="nearest",
                    tolerance=pd.Timedelta(seconds=60),
                )
                combined = tmp_p

    # ALIAS instrument not available in loaded data
    combined["Si_ALIAS"] = np.nan

    # Si = best from h2o_ranking: JLH first, then HW, then ALIAS
    combined["Si"] = np.nan
    for si_col in ("Si_JLH", "Si_HW", "Si_ALIAS"):
        if si_col in combined.columns:
            mask = combined["Si"].isna() & combined[si_col].notna()
            combined.loc[mask, "Si"] = combined.loc[mask, si_col]

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
    
    # qv_jlh from RH + T + P (all merged into combined)
    rh_jlh = df.get("RH_JLH")
    t_c = df.get("T_C")
    p_hpa = df.get("P_hPa")
    if rh_jlh is not None and t_c is not None and p_hpa is not None:
        e_jlh = (np.asarray(rh_jlh, dtype=float) / 100.0) * es_ice_hPa(t_c)
        qv_jlh = qv_from_e_P(e_jlh, p_hpa)
    else:
        qv_jlh = np.nan

    qv_hw = df.get("qv_hw", np.nan)

    # qv = best: jlh first, then hw
    qv = pd.Series(np.nan, index=df.index, dtype=float)
    for src in (qv_jlh, qv_hw):
        arr = src if isinstance(src, pd.Series) else pd.Series(src, index=df.index)
        mask = qv.isna() & arr.notna()
        qv = qv.where(~mask, arr)

    # Sw from Si and T
    si_arr = df.get("Si", np.nan)
    sw = sw_from_si(si_arr, t_c) if t_c is not None else np.nan

    result = pd.DataFrame({
        "Timestamp": df["Timestamp"],
        "Tair_C": df.get("T_C", np.nan),
        "P_hPa": df.get("P_hPa", np.nan),
        "Si": df.get("Si", np.nan),
        "Si_JLH": df.get("Si_JLH", np.nan),
        "Si_HW": df.get("Si_HW", np.nan),
        "Si_ALIAS": df.get("Si_ALIAS", np.nan),
        "qv": qv,
        "qv_jlh": qv_jlh if isinstance(qv_jlh, pd.Series) else np.nan,
        "qv_hw": df.get("qv_hw", np.nan),
        "Sw": sw if isinstance(sw, (pd.Series, np.ndarray)) else np.nan,
        "Lat": df.get(lat_col, np.nan) if lat_col else np.nan,
        "Lon": df.get(lon_col, np.nan) if lon_col else np.nan,
        "Alt_m": df.get(alt_col, np.nan) if alt_col else np.nan,
        "Campaign": df.get("Campaign", "CRYSTAL-FACE-NASA"),
        "source_file": df.get("source_file", ""),
    })

    result["Timestamp"] = _round_timestamp_to_second(result["Timestamp"])
    
    return result
