import traceback
from pathlib import Path

import pytest
import pandas as pd

from parsers.airs_ii import (
    load_airs_ii,
    load_airs_ii_file,
    extract_airs_ii_standard,
)


DATA_DIR = Path("data/raw/AIRS-II")


def _format_exc(e: Exception) -> str:
    return f"{type(e).__name__}: {e}\n" + traceback.format_exc()


def test_airs_dir_exists():
    assert DATA_DIR.exists() and DATA_DIR.is_dir(), (
        f"AIRS-II directory not found at {DATA_DIR}."
    )


def test_files_present():
    files = list(DATA_DIR.glob("*.PNI.nc"))
    assert files, f"No .PNI.nc files found in {DATA_DIR}"


def test_load_first_file():
    files = sorted(DATA_DIR.glob("*.PNI.nc"))
    assert files, f"No files to test in {DATA_DIR}"
    f = files[0]
    try:
        df = load_airs_ii_file(f)
    except Exception as e:
        pytest.fail(f"load_airs_ii_file raised for {f.name}: {_format_exc(e)}")

    assert isinstance(df, pd.DataFrame), "Loader did not return a DataFrame"
    assert not df.empty, f"DataFrame empty for file {f.name}"
    assert "Timestamp" in df.columns, "Timestamp column missing"
    assert pd.api.types.is_datetime64_any_dtype(df["Timestamp"]), (
        "Timestamp column is not datetime dtype"
    )


def test_load_all_files_combined():
    try:
        combined = load_airs_ii(DATA_DIR)
    except Exception as e:
        pytest.fail(f"load_airs_ii raised: {_format_exc(e)}")

    assert isinstance(combined, pd.DataFrame), "Combined loader did not return DataFrame"
    assert not combined.empty, "Combined DataFrame is empty"
    assert "Campaign" in combined.columns and (combined["Campaign"] == "AIRS-II").all(), (
        "Combined DataFrame missing Campaign or incorrect values"
    )


def test_extract_standard_columns():
    try:
        combined = load_airs_ii(DATA_DIR)
    except Exception as e:
        pytest.fail(f"load_airs_ii raised: {_format_exc(e)}")

    std = extract_airs_ii_standard(combined)
    expected = ["Timestamp", "Tair_C", "Si", "Lat", "Lon", "Alt_m", "Campaign", "source_file"]
    missing = [c for c in expected if c not in std.columns]
    assert not missing, f"Missing standardized columns: {missing}"


if __name__ == "__main__":
    # Allow running the tests directly for quick diagnostics
    import sys
    errno = pytest.main([__file__])
    sys.exit(errno)
