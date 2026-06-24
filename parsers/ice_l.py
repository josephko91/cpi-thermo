"""
ICE-L (Ice in Clouds Experiment - Layer clouds) campaign data parser.

Campaign: ICE-L
Data Format: RAF Nimbus NetCDF files (.nc)
Platform: NSF/NCAR C-130Q (N130AR), October–December 2007

Source
------
https://data.eol.ucar.edu/dataset/105.004
https://doi.org/10.5065/D647485N

Water vapor instruments
-----------------------
chilled-mirror : RAF T-Electric chilled mirror (DPXC -> RHUM pre-calculated).
                 Si_chilled_mirror = RHUM / 100 - 1.
                 Note: RHUM is RH w.r.t. liquid water (CF standard_name
                 "relative_humidity"), so Si_chilled_mirror is not strictly
                 supersaturation w.r.t. ice.

MRTDL          : Maycomm AC19-400 dual-path TDL, long absorption path
                 (~130 cm), variable MRTDLL_MC (ppmv).
                 Si_MRTDL is computed from ppmv + ATX + PSXC using the
                 Magnus ice formula -- this IS supersaturation w.r.t. ice.

                 Data notes:
                   - MRTDLS_MC (short path, ~10 cm) is entirely NaN in all
                     12 flights -- short-path channel inactive.
                   - RF01 MRTDLL_MC contains sub-1-ppmv startup values,
                     masked out below _MRTDL_PPMV_MIN = 1.0.
                   - netCDF attribute incorrectly labels units "gram/kg";
                     actual values are ppmv.

h2o_ranking (config.yaml default: chilled-mirror > MRTDL)
----------------------------------------------------------
Si_best = first non-NaN value from the ranked source list.
Per-source columns Si_chilled_mirror and Si_MRTDL are always computed.

Notebook-validated variable mapping
------------------------------------
Temperature : ATX           (deg C)
Pressure    : PSXC          (hPa)
RH (liquid) : RHUM          (%) -- pre-calculated from DPXC dew point
TDL long MR : MRTDLL_MC     (ppmv, mislabeled gram/kg in netCDF attrs)
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr

from .utils import si_from_rh, si_from_ppmv


ICE_L_FILE_RE = re.compile(
    r"RF\d+\.(\d{8})\.(\d{6})_(\d{6})\.PNI\.nc$",
    re.IGNORECASE,
)

_VALID_SOURCES: List[str] = ["chilled-mirror", "MRTDL"]
_DEFAULT_H2O_RANKING: List[str] = ["chilled-mirror", "MRTDL"]

# MRTDLL_MC physical range in ppmv.
# RF01 has sub-1-ppmv startup values that are not geophysical.
_MRTDL_PPMV_MIN: float = 1.0
_MRTDL_PPMV_MAX: float = 50_000.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_float_1d(values) -> np.ndarray:
    arr = np.asarray(values).reshape(-1)
    return pd.to_numeric(pd.Series(arr), errors="coerce").to_numpy(dtype=float)


def _match_length(arr: np.ndarray, target_len: int) -> np.ndarray:
    if len(arr) == target_len:
        return arr
    if len(arr) > target_len:
        return arr[:target_len]
    out = np.full(target_len, np.nan, dtype=float)
    out[: len(arr)] = arr
    return out


def _pick_var(ds: xr.Dataset, candidates: list[str]) -> Optional[str]:
    for name in candidates:
        if name in ds.variables:
            return name
    return None


def _extract_flight_date_from_filename(filepath: Path) -> Optional[datetime]:
    match = ICE_L_FILE_RE.search(filepath.name)
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y%m%d").replace(tzinfo=timezone.utc)


def _extract_time(ds: xr.Dataset, filepath: Path) -> pd.DatetimeIndex:
    time_name = _pick_var(ds, ["Time", "time", "TIME"])
    if time_name is None:
        return pd.to_datetime(pd.Series([], dtype="datetime64[ns]"), utc=True)

    raw = np.asarray(ds[time_name].values).reshape(-1)

    # Prefer direct datetime decoding when available.
    try:
        times = pd.to_datetime(raw, utc=True, errors="coerce")
        if pd.notna(times).sum() > 0:
            return pd.DatetimeIndex(times)
    except Exception:
        pass

    # Fallback: numeric seconds from flight day inferred from filename.
    raw_num = pd.to_numeric(raw, errors="coerce")
    flight_date = _extract_flight_date_from_filename(filepath)
    if flight_date is not None and np.isfinite(raw_num).any():
        base = pd.Timestamp(flight_date)
        return pd.to_datetime(
            base + pd.to_timedelta(raw_num, unit="s"), utc=True, errors="coerce"
        )

    return pd.to_datetime(pd.Series(raw), utc=True, errors="coerce")


def _resolve_si_best(df: pd.DataFrame, h2o_ranking: List[str]) -> pd.Series:
    """Return Si_best as the first non-NaN value across ranked source columns."""
    source_col = {
        "chilled-mirror": "Si_chilled_mirror",
        "MRTDL":          "Si_MRTDL",
    }
    si_best = pd.Series(np.nan, index=df.index, dtype=float)
    for source in h2o_ranking:
        col = source_col.get(source)
        if col and col in df.columns:
            fill_mask = si_best.isna() & df[col].notna()
            si_best = si_best.where(~fill_mask, df[col])
    return si_best


# ---------------------------------------------------------------------------
# File-level loader
# ---------------------------------------------------------------------------

def load_ice_l_file(
    filepath: Union[str, Path],
    h2o_ranking: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load a single ICE-L netCDF file and compute per-source Si.

    Parameters
    ----------
    filepath : path-like
        Path to a ``*.PNI.nc`` RAF Nimbus file.
    h2o_ranking : list of str, optional
        Ordered preference list of water vapour sources.
        Valid values: ``"chilled-mirror"``, ``"MRTDL"``.
        Defaults to ``["chilled-mirror", "MRTDL"]``.

    Returns
    -------
    pd.DataFrame
        Includes ``Si_chilled_mirror``, ``Si_MRTDL``, ``Si_best`` (= ``Si``),
        ``MRTDLL_MC_ppmv``, ``RHUM``, ``ATX_C``, ``PSXC_hPa``, position
        columns, and metadata.
    """
    if h2o_ranking is None:
        h2o_ranking = _DEFAULT_H2O_RANKING

    unknown = [s for s in h2o_ranking if s not in _VALID_SOURCES]
    if unknown:
        raise ValueError(
            f"Unknown h2o_ranking source(s): {unknown}. Valid: {_VALID_SOURCES}"
        )

    filepath = Path(filepath)
    ds = xr.open_dataset(filepath, decode_times=True, decode_timedelta=True)
    try:
        temp_var = _pick_var(ds, ["ATX"])
        pres_var = _pick_var(ds, ["PSXC"])
        rh_var   = _pick_var(ds, ["RHUM"])

        if temp_var is None or pres_var is None or rh_var is None:
            missing = [
                name
                for name, var in (
                    ("ATX", temp_var), ("PSXC", pres_var), ("RHUM", rh_var)
                )
                if var is None
            ]
            raise KeyError(
                f"Missing required ICE-L variable(s) in {filepath.name}: {missing}"
            )

        times = _extract_time(ds, filepath)
        n = len(times)

        tair_c   = _match_length(_to_float_1d(ds[temp_var].values), n)
        psxc_hpa = _match_length(_to_float_1d(ds[pres_var].values), n)
        rhum     = _match_length(_to_float_1d(ds[rh_var].values),   n)

        # Unit normalization: ATX expected in Celsius for ICE-L.
        with np.errstate(all="ignore"):
            med_t = np.nanmedian(tair_c)
        if np.isfinite(med_t) and med_t > 150:
            tair_c = tair_c - 273.15

        # Physical plausibility masks
        tair_c[(tair_c < -95) | (tair_c > 60)]         = np.nan
        psxc_hpa[(psxc_hpa < 50) | (psxc_hpa > 1100)]  = np.nan
        rhum[(rhum < -20) | (rhum > 200)]               = np.nan

        # ---- Chilled-mirror path (RHUM -> Si) ------------------------------
        si_chilled = si_from_rh(rhum)
        si_chilled[(si_chilled < -1) | (si_chilled > 5)] = np.nan

        # ---- TDL long-path (MRTDLL_MC ppmv -> Si) --------------------------
        mrtdl_var = _pick_var(ds, ["MRTDLL_MC"])
        if mrtdl_var is not None:
            mrtdl_ppmv = _match_length(_to_float_1d(ds[mrtdl_var].values), n)
            # Sub-1-ppmv values (e.g. RF01 startup) are not geophysical
            mrtdl_ppmv[
                (mrtdl_ppmv < _MRTDL_PPMV_MIN) | (mrtdl_ppmv > _MRTDL_PPMV_MAX)
            ] = np.nan
        else:
            mrtdl_ppmv = np.full(n, np.nan)

        si_mrtdl = si_from_ppmv(mrtdl_ppmv, tair_c + 273.15, psxc_hpa)
        si_mrtdl[(si_mrtdl < -1) | (si_mrtdl > 5)] = np.nan

        # ---- Position ------------------------------------------------------
        lat_name = _pick_var(ds, ["LAT", "LATC", "GGLAT"])
        lon_name = _pick_var(ds, ["LON", "LONC", "GGLON"])
        alt_name = _pick_var(ds, ["PALTF", "GGALT", "ALT", "PALT"])

        lat   = _match_length(_to_float_1d(ds[lat_name].values), n) if lat_name else np.full(n, np.nan)
        lon   = _match_length(_to_float_1d(ds[lon_name].values), n) if lon_name else np.full(n, np.nan)
        alt_m = _match_length(_to_float_1d(ds[alt_name].values), n) if alt_name else np.full(n, np.nan)

        lat[(lat < -90) | (lat > 90)]           = np.nan
        lon[(lon < -180) | (lon > 180)]         = np.nan
        alt_m[(alt_m < -500) | (alt_m > 25000)] = np.nan

        # ---- Assemble ------------------------------------------------------
        df = pd.DataFrame(
            {
                "Timestamp":         pd.to_datetime(times, utc=True, errors="coerce"),
                "ATX_C":             tair_c,
                "PSXC_hPa":          psxc_hpa,
                "RHUM":              rhum,
                "MRTDLL_MC_ppmv":    mrtdl_ppmv,
                "Si_chilled_mirror": si_chilled,
                "Si_MRTDL":          si_mrtdl,
                "Lat":               lat,
                "Lon":               lon,
                "Alt_m":             alt_m,
                "source_file":       filepath.name,
                "Campaign":          "ICE-L",
            }
        )

        df["Si_best"] = _resolve_si_best(df, h2o_ranking)
        df["Si"] = df["Si_best"]

        if "Timestamp" in df.columns:
            df = df.sort_values("Timestamp", kind="stable").reset_index(drop=True)

        return df
    finally:
        ds.close()


# ---------------------------------------------------------------------------
# Campaign-level loader
# ---------------------------------------------------------------------------

def load_ice_l(
    data_dir: Union[str, Path],
    pattern: str = "*.PNI.nc",
    h2o_ranking: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load all ICE-L files and return a combined DataFrame.

    Parameters
    ----------
    data_dir : path-like
        Directory (or parent) containing ``*.PNI.nc`` files.
    pattern : str
        Glob pattern used to find files.
    h2o_ranking : list of str, optional
        Ordered water vapour source preference.
        Defaults to ``["chilled-mirror", "MRTDL"]``.
    """
    if h2o_ranking is None:
        h2o_ranking = _DEFAULT_H2O_RANKING

    unknown = [s for s in h2o_ranking if s not in _VALID_SOURCES]
    if unknown:
        raise ValueError(
            f"Unknown h2o_ranking source(s): {unknown}. Valid: {_VALID_SOURCES}"
        )

    data_dir = Path(data_dir)
    files = sorted(data_dir.rglob(pattern))

    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' found in {data_dir}")

    frames = []
    for filepath in files:
        try:
            frames.append(load_ice_l_file(filepath, h2o_ranking=h2o_ranking))
        except Exception as exc:
            print(f"Warning: Could not load {filepath.name}: {exc}")

    if not frames:
        raise ValueError(f"No valid ICE-L files parsed from {data_dir}")

    combined = pd.concat(frames, ignore_index=True)
    combined["Campaign"] = "ICE-L"
    return combined


# ---------------------------------------------------------------------------
# Standardised output
# ---------------------------------------------------------------------------

def extract_ice_l_standard(df: pd.DataFrame) -> pd.DataFrame:
    """Extract standardised columns for combined campaign output."""
    return pd.DataFrame(
        {
            "Timestamp":         df.get("Timestamp",         pd.NaT),
            "Tair_C":            df.get("ATX_C",             np.nan),
            "Si":                df.get("Si",                np.nan),
            "Si_best":           df.get("Si_best",           np.nan),
            "Si_chilled_mirror": df.get("Si_chilled_mirror", np.nan),
            "Si_MRTDL":          df.get("Si_MRTDL",          np.nan),
            "MRTDLL_MC_ppmv":    df.get("MRTDLL_MC_ppmv",   np.nan),
            "Lat":               df.get("Lat",               np.nan),
            "Lon":               df.get("Lon",               np.nan),
            "Alt_m":             df.get("Alt_m",             np.nan),
            "Campaign":          df.get("Campaign",          "ICE-L"),
            "source_file":       df.get("source_file",       ""),
        }
    )
