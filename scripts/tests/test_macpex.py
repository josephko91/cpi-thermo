import traceback
from pathlib import Path

import pytest
import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/test_macpex.log"),
        logging.StreamHandler()
    ]
)

from parsers.macpex import load_macpex, extract_macpex_standard


DATA_DIR = Path("data/raw/MACPEX")


def _format_exc(e: Exception) -> str:
    return f"{type(e).__name__}: {e}\n" + traceback.format_exc()


def test_macpex_dir_exists():
    assert DATA_DIR.exists() and DATA_DIR.is_dir(), f"MACPEX dir not found at {DATA_DIR}"


def test_load_macpex():
    try:
        df = load_macpex(DATA_DIR)
    except Exception as e:
        pytest.fail(f"load_macpex raised: {_format_exc(e)}")

    assert isinstance(df, pd.DataFrame), "load_macpex did not return a DataFrame"
    assert not df.empty, "MACPEX combined DataFrame is empty"
    assert "datetime_utc" in df.columns or "Timestamp" in df.columns, (
        "MACPEX output missing datetime index/column"
    )


def test_extract_macpex_standard():
    try:
        df = load_macpex(DATA_DIR)
    except Exception as e:
        pytest.fail(f"load_macpex raised: {_format_exc(e)}")

    std = extract_macpex_standard(df)
    expected = ["Timestamp", "Tair_C", "Si", "Lat", "Lon", "Alt_m", "Campaign", "source_file"]
    missing = [c for c in expected if c not in std.columns]
    assert not missing, f"Missing standardized columns: {missing}"


if __name__ == "__main__":
    import sys
    logging.info("Starting MACPEX tests")
    errno = pytest.main([__file__])
    logging.info("MACPEX tests completed")
    sys.exit(errno)
