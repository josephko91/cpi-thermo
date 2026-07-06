#!/usr/bin/env python3
"""Diagnose missing standardized variables across campaign parsers.

This script audits campaign data in three stages:
1) Raw availability: whether candidate raw columns exist for each standardized variable.
2) Extracted availability: whether standardized fields are populated after extractor logic.
3) Cleaned availability: whether values remain after sentinel/range cleaning.

It writes CSV artifacts to help identify where NaNs originate and where parser
mappings likely need updates before combining into the final parquet output.
Outputs go to logs/campaign_missingness/<timestamp>/, with a `latest` symlink
kept pointing at the newest run.

Usage examples:
    python scripts/diagnose_campaign_missingness.py
    python scripts/diagnose_campaign_missingness.py --campaigns ATTREX MACPEX
    python scripts/diagnose_campaign_missingness.py --output-dir /tmp/custom_dir
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main import DEFAULT_CAMPAIGN_CONFIG
from parsers import CAMPAIGN_LOADERS
from scripts.log_paths import timestamp as _run_timestamp, update_latest


TARGET_VARS: List[str] = [
    "Timestamp",
    "Tair_C",
    "Si",
    "Lat",
    "Lon",
    "Alt_m",
    "source_file",
    "Tair_K",
    "Pressure_hPa",
]

SENTINELS: List[float] = [
    -9999.0,
    -9999.99,
    -99999.0,
    -999999.0,
    -9999999.0,
    -99999999.0,
    -999999999.0,
    -8888.0,
    -8888.88,
    -7777.0,
    -7777.77,
    999999.9999,
    999.9999999,
    9999.999999,
    99999.99999,
    99999999999.0,
    999999.0,
]

PHYSICAL_RANGES: Dict[str, Tuple[float, float]] = {
    "Lat": (-90.0, 90.0),
    "Lon": (-180.0, 180.0),
    "Alt_m": (-1000.0, 40000.0),
}

RAW_PATTERNS: Dict[str, List[str]] = {
    "Timestamp": ["timestamp", "time", "utc", "ut", "datetime"],
    "Tair_C": ["tair", "air_temp", "temp", "atx", "t_c", "temperature"],
    "Si": ["si", "rh", "rhi", "frost", "h2o", "wv", "mixing"],
    "Lat": ["lat", "latitude", "gps_lat", "pos_lat"],
    "Lon": ["lon", "longitude", "gps_lon", "pos_lon"],
    "Alt_m": ["alt", "altitude", "gps_alt", "prealt", "z"],
    "source_file": ["source_file", "filename", "file"],
    "Tair_K": ["tair_k", "t_k", "temp_k", "temperature_k"],
    "Pressure_hPa": ["pressure", "press", "hpa", "pres_hpa", "p_hpa"],
}

PARSER_METADATA: Dict[str, Dict[str, str]] = {
    "ARM": {"file": "parsers/arm.py", "function": "extract_arm_standard"},
    "CRYSTAL-FACE-NASA": {
        "file": "parsers/crystal_face_nasa.py",
        "function": "extract_crystal_face_nasa_standard",
    },
    "CRYSTAL-FACE-UND": {
        "file": "parsers/crystal_face_und.py",
        "function": "extract_crystal_face_und_standard",
    },
    "MC3E": {"file": "parsers/mc3e.py", "function": "extract_mc3e_standard"},
    "MIDCIX": {"file": "parsers/midcix.py", "function": "extract_midcix_standard"},
    "OLYMPEX": {"file": "parsers/olympex.py", "function": "extract_olympex_standard"},
    "AIRS-II": {"file": "parsers/airs_ii.py", "function": "extract_airs_ii_standard"},
    "ATTREX": {"file": "parsers/attrex.py", "function": "extract_attrex_standard"},
    "IPHEX": {"file": "parsers/iphex.py", "function": "extract_iphex_standard"},
    "ISDAC": {"file": "parsers/isdac.py", "function": "extract_isdac_standard"},
    "POSIDON": {"file": "parsers/posidon.py", "function": "extract_posidon_standard"},
    "ESCAPE": {"file": "parsers/escape.py", "function": "extract_escape_standard"},
    "ICE-L": {"file": "parsers/ice_l.py", "function": "extract_ice_l_standard"},
    "MACPEX": {"file": "parsers/macpex.py", "function": "extract_macpex_standard"},
}

RISK_HINTS: Dict[str, str] = {
    "ATTREX": "Multi-stream merge_asof can leave position variables empty if time alignment fails.",
    "MACPEX": "Multi-stream MMS + chemistry merge can reduce positional coverage.",
    "POSIDON": "MMS merge/scaling logic may suppress positional fields when joins are sparse.",
    "OLYMPEX": "Extractor uses hardcoded POS_* names; raw naming mismatches can force NaNs.",
    "IPHEX": "Extractor uses hardcoded POS_* names; raw naming mismatches can force NaNs.",
    "ISDAC": "Hardcoded positional columns can fail when source naming varies.",
}


@dataclass
class MatchResult:
    variable: str
    best_column: Optional[str]
    score: float
    reason: str


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def score_column_match(column_name: str, variable: str, patterns: Dict[str, List[str]]) -> Tuple[float, str]:
    """Return (score, reason) for how well a raw column matches a standardized variable."""
    col = _normalize_name(column_name)
    aliases = patterns.get(variable, [])

    if col == _normalize_name(variable):
        return 1.0, "exact variable name"

    for alias in aliases:
        alias_norm = _normalize_name(alias)
        if col == alias_norm:
            return 0.95, f"exact alias: {alias}"

    for alias in aliases:
        alias_norm = _normalize_name(alias)
        if col.startswith(alias_norm):
            return 0.8, f"prefix alias: {alias}"

    for alias in aliases:
        alias_norm = _normalize_name(alias)
        if alias_norm in col:
            return 0.65, f"contains alias: {alias}"

    # Weak heuristic: split tokens and check overlap
    tokens = set(col.split("_"))
    for alias in aliases:
        alias_tokens = set(_normalize_name(alias).split("_"))
        if alias_tokens & tokens:
            return 0.4, f"token overlap alias: {alias}"

    return 0.0, "no match"


def find_best_raw_column(columns: Iterable[str], variable: str) -> MatchResult:
    """Pick the strongest matching raw column for a standardized variable."""
    best_col: Optional[str] = None
    best_score = 0.0
    best_reason = "no match"

    for col in columns:
        score, reason = score_column_match(col, variable, RAW_PATTERNS)
        if score > best_score:
            best_col = col
            best_score = score
            best_reason = reason

    return MatchResult(variable=variable, best_column=best_col, score=best_score, reason=best_reason)


def coerce_numeric(series: pd.Series) -> pd.Series:
    """Convert a series to numeric where possible without raising."""
    return pd.to_numeric(series, errors="coerce")


def clean_series(variable: str, series: pd.Series) -> pd.Series:
    """Apply sentinel and physical-range cleaning for diagnostics."""
    if series.empty:
        return series

    if is_numeric_dtype(series):
        cleaned = series.mask(series.isin(SENTINELS))
    else:
        cleaned = series.copy()

    if variable in PHYSICAL_RANGES:
        lo, hi = PHYSICAL_RANGES[variable]
        num = coerce_numeric(cleaned)
        cleaned = num.mask((num < lo) | (num > hi))

    if variable == "Timestamp":
        return pd.to_datetime(cleaned, errors="coerce", utc=True)

    return cleaned


def classify_gap(raw_match: MatchResult, std_exists: bool, std_null_rate_cleaned: float) -> str:
    """Classify root-cause bucket for missing values."""
    raw_available = raw_match.best_column is not None and raw_match.score >= 0.4

    if not raw_available:
        return "unavailable_in_raw"
    if not std_exists:
        return "dropped_by_extractor"
    if std_null_rate_cleaned >= 0.95:
        return "mostly_invalid_or_missing_post_extraction"
    if std_null_rate_cleaned >= 0.5:
        return "partially_missing_post_extraction"
    return "covered"


def null_fraction(series: pd.Series) -> float:
    if len(series) == 0:
        return math.nan
    return float(series.isna().sum() / len(series))


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _safe_extract(extractor: Callable[[pd.DataFrame], pd.DataFrame], df_raw: pd.DataFrame) -> pd.DataFrame:
    out = extractor(df_raw)
    if not isinstance(out, pd.DataFrame):
        raise TypeError(f"Extractor returned {type(out)} instead of pandas.DataFrame")
    return out


def diagnose_campaign(
    campaign: str,
    config: Dict[str, Any],
    max_raw_rows: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Diagnose one campaign and return detail, schema, and recommendation rows."""
    loader = CAMPAIGN_LOADERS.get(campaign)
    extractor = config.get("extractor")
    data_dir = Path(config["path"])
    pattern = config.get("pattern", "*")

    if loader is None:
        raise ValueError(f"No loader registered for {campaign}")
    if extractor is None:
        raise ValueError(f"No extractor registered for {campaign}")

    df_raw = loader(data_dir, pattern)
    if not isinstance(df_raw, pd.DataFrame):
        raise TypeError(f"Loader for {campaign} returned {type(df_raw)}")

    if max_raw_rows is not None and len(df_raw) > max_raw_rows:
        df_raw_for_extract = df_raw.iloc[:max_raw_rows].copy()
    else:
        df_raw_for_extract = df_raw

    df_std = _safe_extract(extractor, df_raw_for_extract)

    detail_rows: List[Dict[str, Any]] = []
    schema_rows: List[Dict[str, Any]] = []
    rec_rows: List[Dict[str, Any]] = []

    raw_cols = list(df_raw.columns)
    std_cols = list(df_std.columns)

    for col in raw_cols:
        schema_rows.append(
            {
                "campaign": campaign,
                "stage": "raw",
                "column": col,
                "dtype": str(df_raw[col].dtype),
                "non_null_count": int(df_raw[col].notna().sum()),
                "null_count": int(df_raw[col].isna().sum()),
            }
        )

    for col in std_cols:
        schema_rows.append(
            {
                "campaign": campaign,
                "stage": "standardized",
                "column": col,
                "dtype": str(df_std[col].dtype),
                "non_null_count": int(df_std[col].notna().sum()),
                "null_count": int(df_std[col].isna().sum()),
            }
        )

    parser_meta = PARSER_METADATA.get(campaign, {"file": "unknown", "function": "unknown"})

    for variable in TARGET_VARS:
        raw_match = find_best_raw_column(raw_cols, variable)
        std_exists = variable in df_std.columns

        if std_exists:
            std_series = df_std[variable]
            std_cleaned = clean_series(variable, std_series)
            raw_null_rate = (
                null_fraction(df_raw[raw_match.best_column])
                if raw_match.best_column in df_raw.columns
                else math.nan
            )
            std_null_rate = null_fraction(std_series)
            std_null_rate_cleaned = null_fraction(std_cleaned)
            sentinel_rate = (
                float(coerce_numeric(std_series).isin(SENTINELS).sum() / len(std_series))
                if len(std_series) > 0
                else math.nan
            )
        else:
            raw_null_rate = (
                null_fraction(df_raw[raw_match.best_column])
                if raw_match.best_column in df_raw.columns
                else math.nan
            )
            std_null_rate = 1.0
            std_null_rate_cleaned = 1.0
            sentinel_rate = math.nan

        gap_class = classify_gap(raw_match, std_exists, std_null_rate_cleaned)

        detail_rows.append(
            {
                "campaign": campaign,
                "variable": variable,
                "raw_best_column": raw_match.best_column,
                "raw_match_score": round(raw_match.score, 3),
                "raw_match_reason": raw_match.reason,
                "std_column_exists": std_exists,
                "raw_null_fraction": raw_null_rate,
                "std_null_fraction": std_null_rate,
                "std_null_fraction_cleaned": std_null_rate_cleaned,
                "sentinel_fraction_std": sentinel_rate,
                "gap_class": gap_class,
                "loader": loader.__name__,
                "extractor": extractor.__name__,
                "data_path": str(data_dir),
                "pattern": pattern,
                "n_raw_rows": int(len(df_raw)),
                "n_std_rows": int(len(df_std)),
                "parser_file": parser_meta["file"],
                "parser_function": parser_meta["function"],
                "risk_hint": RISK_HINTS.get(campaign, ""),
            }
        )

        if gap_class != "covered":
            rec_rows.append(
                {
                    "campaign": campaign,
                    "variable": variable,
                    "gap_class": gap_class,
                    "suggested_raw_column": raw_match.best_column,
                    "match_score": round(raw_match.score, 3),
                    "suggestion_confidence": (
                        "high"
                        if raw_match.score >= 0.8
                        else "medium"
                        if raw_match.score >= 0.5
                        else "low"
                    ),
                    "recommended_action": (
                        "Add/adjust extractor mapping for this variable"
                        if raw_match.best_column is not None
                        else "Confirm variable availability in raw source or alternative instrument stream"
                    ),
                    "parser_file": parser_meta["file"],
                    "parser_function": parser_meta["function"],
                    "risk_hint": RISK_HINTS.get(campaign, ""),
                }
            )

    return pd.DataFrame(detail_rows), pd.DataFrame(schema_rows), pd.DataFrame(rec_rows)


def diagnose_all_campaigns(
    campaigns: List[str],
    output_dir: Path,
    max_raw_rows: Optional[int] = None,
) -> Dict[str, Path]:
    """Run diagnostics for all campaigns and persist CSV outputs."""
    detail_frames: List[pd.DataFrame] = []
    schema_frames: List[pd.DataFrame] = []
    rec_frames: List[pd.DataFrame] = []
    failures: List[Dict[str, str]] = []

    for campaign in campaigns:
        config = DEFAULT_CAMPAIGN_CONFIG.get(campaign)
        if config is None:
            failures.append({"campaign": campaign, "error": "campaign_missing_in_config"})
            continue

        try:
            detail_df, schema_df, rec_df = diagnose_campaign(campaign, config, max_raw_rows=max_raw_rows)
            detail_frames.append(detail_df)
            schema_frames.append(schema_df)
            rec_frames.append(rec_df)
        except Exception as exc:
            failures.append({"campaign": campaign, "error": str(exc)})

    detail_all = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    schema_all = pd.concat(schema_frames, ignore_index=True) if schema_frames else pd.DataFrame()
    rec_all = pd.concat(rec_frames, ignore_index=True) if rec_frames else pd.DataFrame()
    fail_all = pd.DataFrame(failures)

    summary_all = pd.DataFrame()
    if not detail_all.empty:
        summary_all = (
            detail_all.groupby(["campaign", "variable", "gap_class"], as_index=False)
            .agg(
                raw_match_score=("raw_match_score", "max"),
                std_null_fraction_cleaned=("std_null_fraction_cleaned", "mean"),
                n_raw_rows=("n_raw_rows", "max"),
                parser_file=("parser_file", "first"),
                parser_function=("parser_function", "first"),
            )
            .sort_values(["campaign", "variable"])
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_paths = {
        "detail": output_dir / "campaign_variable_detail.csv",
        "summary": output_dir / "campaign_variable_summary.csv",
        "schema": output_dir / "campaign_schema_inventory.csv",
        "recommendations": output_dir / "campaign_parser_recommendations.csv",
        "failures": output_dir / "campaign_diagnostic_failures.csv",
    }

    write_csv(detail_all, out_paths["detail"])
    write_csv(summary_all, out_paths["summary"])
    write_csv(schema_all, out_paths["schema"])
    write_csv(rec_all, out_paths["recommendations"])
    write_csv(fail_all, out_paths["failures"])

    return out_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose campaign missingness and mapping gaps")
    parser.add_argument(
        "--campaigns",
        nargs="+",
        default=list(DEFAULT_CAMPAIGN_CONFIG.keys()),
        help="Campaign names to audit (default: all configured campaigns)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "logs" / "campaign_missingness" / _run_timestamp(),
        help="Directory for diagnostics CSV artifacts (default: "
             "logs/campaign_missingness/<timestamp>/, with "
             "logs/campaign_missingness/latest kept pointing at the newest run)",
    )
    parser.add_argument(
        "--max-raw-rows",
        type=int,
        default=None,
        help="Optional row cap per campaign before extractor diagnostics (faster, less complete)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_paths = diagnose_all_campaigns(
        campaigns=args.campaigns,
        output_dir=args.output_dir,
        max_raw_rows=args.max_raw_rows,
    )

    print("Diagnostics complete. Generated files:")
    for key, path in out_paths.items():
        print(f"- {key}: {path}")

    update_latest(args.output_dir.parent, args.output_dir)
    print(f"\nLatest run: {args.output_dir.parent / 'latest'} -> {args.output_dir.name}")


if __name__ == "__main__":
    main()
