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
        logging.FileHandler("logs/test_attrex.log"),
        logging.StreamHandler()
    ]
)

from parsers.attrex import load_attrex, extract_attrex_standard


DATA_DIR = Path("data/raw/ATTREX")


def _format_exc(e: Exception) -> str:
    return f"{type(e).__name__}: {e}\n" + traceback.format_exc()


def test_attrex_dir_exists():
    assert DATA_DIR.exists() and DATA_DIR.is_dir(), f"ATTREX dir not found at {DATA_DIR}"


def test_load_attrex():
    try:
        df = load_attrex(DATA_DIR)
    except Exception as e:
        pytest.fail(f"load_attrex raised: {_format_exc(e)}")

    assert isinstance(df, pd.DataFrame), "load_attrex did not return a DataFrame"
    assert not df.empty, "ATTREX combined DataFrame is empty"
    assert "Timestamp" in df.columns, "Timestamp column missing in ATTREX output"
    assert pd.api.types.is_datetime64_any_dtype(df["Timestamp"]), "Timestamp is not datetime dtype"


def test_extract_attrex_standard():
    try:
        df = load_attrex(DATA_DIR)
    except Exception as e:
        pytest.fail(f"load_attrex raised: {_format_exc(e)}")

    std = extract_attrex_standard(df)
    expected = ["Timestamp", "Tair_C", "Si", "Lat", "Lon", "Alt_m", "Campaign", "source_file"]
    missing = [c for c in expected if c not in std.columns]
    assert not missing, f"Missing standardized columns: {missing}"


if __name__ == "__main__":
    import sys
    logging.info("Starting ATTREX tests")
    errno = pytest.main([__file__])
    logging.info("ATTREX tests completed")
    sys.exit(errno)
