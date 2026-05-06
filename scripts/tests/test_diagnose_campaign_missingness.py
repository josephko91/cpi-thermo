#!/usr/bin/env python3
"""Unit tests for campaign missingness diagnostics helpers."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Ensure scripts package can be imported in test context
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.diagnose_campaign_missingness import (
    classify_gap,
    clean_series,
    find_best_raw_column,
)


def test_find_best_raw_column_prefers_exact_alias():
    cols = ["GPS_Lat_deg", "Temperature", "source_file"]
    res = find_best_raw_column(cols, "Lat")
    assert res.best_column == "GPS_Lat_deg"
    assert res.score >= 0.65


def test_clean_series_applies_physical_range_for_lat():
    series = pd.Series([42.0, -95.0, 91.0, np.nan])
    cleaned = clean_series("Lat", series)
    assert pd.isna(cleaned.iloc[1])
    assert pd.isna(cleaned.iloc[2])
    assert cleaned.iloc[0] == 42.0


def test_clean_series_masks_sentinel_values():
    series = pd.Series([1.0, -9999.0, -7777.77])
    cleaned = clean_series("Tair_C", series)
    assert pd.isna(cleaned.iloc[1])
    assert pd.isna(cleaned.iloc[2])


def test_classify_gap_paths():
    from scripts.diagnose_campaign_missingness import MatchResult

    unavailable = MatchResult(variable="Lon", best_column=None, score=0.0, reason="no match")
    assert classify_gap(unavailable, std_exists=False, std_null_rate_cleaned=1.0) == "unavailable_in_raw"

    available_dropped = MatchResult(variable="Lon", best_column="Longitude", score=0.8, reason="contains")
    assert classify_gap(available_dropped, std_exists=False, std_null_rate_cleaned=1.0) == "dropped_by_extractor"

    mostly_missing = MatchResult(variable="Lon", best_column="Longitude", score=0.8, reason="contains")
    assert (
        classify_gap(mostly_missing, std_exists=True, std_null_rate_cleaned=0.99)
        == "mostly_invalid_or_missing_post_extraction"
    )
