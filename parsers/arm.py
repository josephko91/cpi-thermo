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

from .utils import si_from_frost_point, es_ice_hPa, qv_from_e_P, sw_from_si, first_per_second, edr_from_und_cm23s1


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

    # ------------------------------------------------------------------ #
    # Instrument validity masking                                          #
    # ------------------------------------------------------------------ #
    # Background: the T4 binary format encodes all values as               #
    #   physical_value = raw_int32 / 1000.0 - 100.0                       #
    # so raw = 100000 decodes to 0.0 (fill/GPS-not-locked),               #
    # and raw < 0 decodes to < -100 (cryo instrument below range).        #
    #
    # Issue A — GPS fill values                                            #
    # When the GPS receiver has not yet acquired a fix, the hardware        #
    # outputs raw = 100000 for all three GPS channels, which decodes to    #
    # Lat = Lon = Alt = 0.0.  These are not valid coordinates and must be  #
    # treated as NaN to avoid the aircraft appearing to be at 0°N 0°E.    #
    # Detection: Lat ≈ 0 AND Lon ≈ 0 simultaneously.                       #
    gps_fill = (df["GPS_Lat_deg"].abs() < 0.01) & (df["GPS_Lon_deg"].abs() < 0.01)
    n_gps_fill = int(gps_fill.sum())
    if n_gps_fill:
        df.loc[gps_fill, ["GPS_Lat_deg", "GPS_Lon_deg", "GPS_Alt_m"]] = np.nan

    # Issue B — cryo hygrometer below detection limit                      #
    # The Cryo-Electric Mirror (CEM) outputs raw < 0 when the mirror is    #
    # cooled below its lower measurement limit (~-80°C frost point for     #
    # typical aircraft CEM instruments).  This decodes to physically       #
    # impossible temperatures (-100 to -150°C) which would drive           #
    # qv_from_e_P to essentially zero, creating a false "bone-dry" signal. #
    # Mask both the cryo dew-point and cryo frost-point channels.          #
    CRYO_FLOOR = -80.0  # °C; below this the CEM is out of range
    for col in ("Dew_Point_Cryo_C", "Frost_Point_Cryo_C"):
        below_range = df[col] < CRYO_FLOOR
        df.loc[below_range, col] = np.nan

    # Issue C — cryo hygrometer reads above air temperature (cloud flood)  #
    # When the aircraft flies through cloud or precipitation, liquid water  #
    # droplets coat the cryo mirror and it equilibrates to the liquid dew   #
    # point of the ambient air — which in saturated conditions can be very  #
    # close to (but not above) Tair.  However, instrument thermal lag and   #
    # droplet flooding sometimes causes the Cryo to report Td > Tair for   #
    # tens of seconds, which is thermodynamically impossible.               #
    # Empirically confirmed in citation.0317001715: Frost_Point_Cryo reads  #
    # 20–25°C while Tair = 5°C and Dew_Point_EGG reads 3.9°C (correct).   #
    #                                                                        #
    # Tolerance: use Tair + 1°C as the threshold rather than strict Tair.   #
    # In-cloud conditions near 100% RH produce Frost_Point ≈ Tair; CEM and  #
    # Rosemount temperature sensors each carry ±0.3–0.5°C precision, so a   #
    # 1°C buffer distinguishes legitimate saturated readings (which remain   #
    # within ~0.5°C of Tair) from genuine malfunctions (which exceed Tair   #
    # by 10–20°C as in the 0317001715 file).                                #
    CRYO_TAIR_MARGIN = 1.0  # °C buffer above Tair before flagging as invalid
    tair = df["Air_Temp_Rosemount_C"].values
    for col in ("Dew_Point_EGG_C", "Dew_Point_Cryo_C", "Frost_Point_Cryo_C"):
        above_tair = pd.to_numeric(df[col], errors="coerce") > tair + CRYO_TAIR_MARGIN
        df.loc[above_tair, col] = np.nan

    # Issue D — Rosemount temperature probe warm-up/frozen-reading fault     #
    # Empirically confirmed in citation.0303001655: for the first ~23        #
    # minutes after the file starts, Air_Temp_Rosemount_C is frozen at       #
    # approximately -64.5 C (fluctuating only at instrument-noise level)     #
    # while Pressure_Altitude_m varies normally around 200-1700 m -- a       #
    # deviation of up to -78 C from the ISA-predicted temperature at that    #
    # altitude, physically impossible for any real atmospheric inversion.    #
    # The reading then jumps abruptly to a plausible ~5 C and tracks         #
    # normally for the rest of the flight. Detected via ISA deviation        #
    # (same formula/threshold as qa_checks.py QC2) using Pressure_Altitude_m #
    # (always available, unlike GPS_Alt_m early in a flight). Checked the    #
    # deviation distribution across the whole campaign: excluding this      #
    # fault, the maximum |Tair - T_isa| is well under 40 C (75th percentile  #
    # is 5.4 C), so this threshold cleanly isolates the fault with no        #
    # ambiguous borderline cases.                                            #
    T_ISA_DEV_THRESHOLD = 40.0  # °C
    t_isa_c = 15.0 - 6.5e-3 * df["Pressure_Altitude_m"].values
    tair_fault = (pd.to_numeric(df["Air_Temp_Rosemount_C"], errors="coerce") - t_isa_c).abs() > T_ISA_DEV_THRESHOLD
    n_tair_fault = int(tair_fault.sum())
    if n_tair_fault:
        df.loc[tair_fault, "Air_Temp_Rosemount_C"] = np.nan
        print(f"  ARM QC: nulled {n_tair_fault:,} rows where Air_Temp_Rosemount_C "
              f"deviated from ISA by >{T_ISA_DEV_THRESHOLD:.0f}°C (probe warm-up fault, {filepath.name})")

    # ------------------------------------------------------------------ #
    # Decode date and create timestamp                                     #
    # ------------------------------------------------------------------ #
    df["Date"] = df["Date"].apply(_decode_arm_date)
    df["Timestamp"] = df["Date"] + pd.to_timedelta(df["Time_sec"], unit="s")
    df["Timestamp"] = df["Timestamp"].dt.tz_localize("UTC")

    # ARM's raw source is 4 Hz; floor onto the pipeline's 1 Hz exact-second
    # merge grid, keeping the first real sample per second (never averaged --
    # see first_per_second docstring). This changes ARM's row count from the
    # pre-existing 4 Hz-duplicated-Timestamp rows to true 1 Hz rows.
    df = first_per_second(df, ts_col="Timestamp")

    # ------------------------------------------------------------------ #
    # Derived thermodynamic quantities                                     #
    # ------------------------------------------------------------------ #
    # Calculate ice supersaturation from cryo frost point
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

    # Prefer GPS_Alt_m over Pressure_Altitude_m when the two agree (GPS is
    # more accurate), but fall back to Pressure_Altitude_m otherwise: right
    # after GPS first acquires a fix, the horizontal position (Lat/Lon) can
    # lock before the vertical/altitude solution stabilizes, producing a
    # stuck, physically-wrong GPS_Alt_m for the next 1-2 minutes (e.g.
    # citation.0313001806.t4archive.gz: GPS_Alt_m stuck at 333 m while
    # Pressure_Altitude_m correctly climbs through 6600-6900 m -- an 8,261 m
    # disagreement, confirmed by QC2's T_altitude_inconsistent check).
    # Legitimate GPS/pressure-altitude offsets (geoid vs. ICAO atmosphere)
    # are a few hundred meters at most (median 55 m across this campaign);
    # 1000 m is a generous margin that only catches genuine GPS glitches.
    ALT_AGREEMENT_TOLERANCE_M = 1000.0
    gps_alt_present = df["GPS_Alt_m"].notna() & (df["GPS_Alt_m"] > 0)
    gps_alt_agrees = (
        gps_alt_present
        & df["Pressure_Altitude_m"].notna()
        & ((df["GPS_Alt_m"] - df["Pressure_Altitude_m"]).abs() <= ALT_AGREEMENT_TOLERANCE_M)
    )

    # A frozen GPS receiver can hold the SAME whole-meter altitude for
    # minutes while the aircraft keeps climbing/descending on
    # Pressure_Altitude_m -- and can still be within the tolerance above at
    # the start of the freeze, before the two diverge enough to be caught by
    # it (e.g. citation.0312001649: GPS_Alt_m frozen at exactly 8926 m for
    # 727 samples/~3 min while Pressure_Altitude_m smoothly descends from
    # 8828 m to 7782 m -- only the back half of that run exceeded the 1000 m
    # tolerance). Detect runs of GPS_Alt_m repeating a bit-exact value for
    # >=30 consecutive samples (matching qa_checks.py QC3's own stuck-sensor
    # threshold) and treat the whole run as invalid regardless of momentary
    # agreement with Pressure_Altitude_m.
    STUCK_RUN_LENGTH = 30
    run_id = (df["GPS_Alt_m"] != df["GPS_Alt_m"].shift()).cumsum()
    run_len = run_id.map(run_id.value_counts())
    gps_alt_stuck = gps_alt_present & (run_len >= STUCK_RUN_LENGTH)

    gps_alt_valid = gps_alt_agrees & ~gps_alt_stuck
    df["Alt_m"] = np.where(gps_alt_valid, df["GPS_Alt_m"], df["Pressure_Altitude_m"])

    n_gps_disagreed = int((gps_alt_present & ~gps_alt_agrees & ~gps_alt_stuck).sum())
    n_gps_stuck = int((gps_alt_present & gps_alt_stuck).sum())
    if n_gps_disagreed:
        print(f"  ARM QC: fell back to Pressure_Altitude_m for {n_gps_disagreed:,} rows "
              f"where GPS_Alt_m disagreed by >{ALT_AGREEMENT_TOLERANCE_M:.0f} m "
              f"({filepath.name})")
    if n_gps_stuck:
        print(f"  ARM QC: fell back to Pressure_Altitude_m for {n_gps_stuck:,} rows "
              f"where GPS_Alt_m was frozen for >={STUCK_RUN_LENGTH} consecutive samples "
              f"({filepath.name})")

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
        "Alt_m": df.get("Alt_m", df["GPS_Alt_m"]),
        "Wind_W_ms": df.get("Vertical_Wind_m_s", np.nan),
        # Turbulence_eps is eps^(1/3) (data/raw/ARM/poellot-citation-t4-readme.txt
        # field 18: "Turbulence / epsilon**1/3"), but that readme's units
        # column names the quantity, not its length unit. Value-range check
        # settled it: as raw meters, median lands at 0.57 m^(2/3)*s^-1 --
        # implying HALF of all ARM records are at/above ICAO's severe-
        # turbulence threshold, physically implausible for routine flight.
        # Divided by the same cm->m factor as the later UND ASCII pipeline
        # (same UND Citation II team/aircraft, same house cm convention),
        # the median drops to 0.027 and max to 2.4 -- squarely matching the
        # UND ASCII campaigns' EDR_m23s1 range. See
        # docs/decisions/2026-07-13-edr-unification.md.
        "EDR_m23s1": edr_from_und_cm23s1(df.get("Turbulence_eps", np.nan)),
        "Campaign": df.get("Campaign", "ARM"),
        "source_file": df["source_file"],
    })
