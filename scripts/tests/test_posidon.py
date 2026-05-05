import traceback
from pathlib import Path

import pytest
import pandas as pd

from parsers.posidon import load_posidon, extract_posidon_standard


DATA_DIR = Path("data/raw/POSIDON")


def _format_exc(e: Exception) -> str:
    return f"{type(e).__name__}: {e}\n" + traceback.format_exc()


def test_posidon_dir_exists():
    assert DATA_DIR.exists() and DATA_DIR.is_dir(), f"POSIDON dir not found at {DATA_DIR}"


def test_load_posidon():
    try:
        df = load_posidon(DATA_DIR)
    except Exception as e:
        pytest.fail(f"load_posidon raised: {_format_exc(e)}")

    assert isinstance(df, pd.DataFrame), "load_posidon did not return a DataFrame"
    assert not df.empty, "POSIDON combined DataFrame is empty"
    assert "datetime_utc" in df.columns or "Timestamp" in df.columns, (
        "POSIDON output missing datetime index/column"
    )


def test_extract_posidon_standard():
    try:
        df = load_posidon(DATA_DIR)
    except Exception as e:
        pytest.fail(f"load_posidon raised: {_format_exc(e)}")

    std = extract_posidon_standard(df)
    expected = ["Timestamp", "Tair_K", "Tair_C", "Pressure_hPa", "Si", "Lat", "Lon", "Alt_m", "Campaign"]
    missing = [c for c in expected if c not in std.columns]
    assert not missing, f"Missing standardized columns: {missing}"


if __name__ == "__main__":
    import sys
    errno = pytest.main([__file__])
    sys.exit(errno)
