"""
Tests for the dataset-wide Si plausibility range (parsers/utils.py SI_MIN/SI_MAX).

Unit tests cover mask_si_out_of_range; the parquet-invariant tests check the
built dataset tiers and are skipped when a tier has not been built.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from parsers.utils import SI_MAX, SI_MIN, mask_si_out_of_range  # noqa: E402

OUT_DIR = REPO_ROOT / "data" / "out"
TIERS = [
    "combined_env_data.parquet",
    "combined_env_data_L1.parquet",
    "combined_env_data_L2.parquet",
    "combined_env_data_L1_cocpit.parquet",
    "combined_env_data_L2_cocpit.parquet",
]


def test_range_is_minus_one_to_two():
    assert (SI_MIN, SI_MAX) == (-1.0, 2.0)


def test_out_of_range_becomes_nan_not_clamped():
    out = mask_si_out_of_range(np.array([-3.0, -1.0001, 5.0, 2.0001, 100.0]))
    assert np.isnan(out).all()


def test_in_range_and_exact_bounds_kept():
    vals = np.array([-1.0, -0.5, 0.0, 1.5, 2.0])
    np.testing.assert_array_equal(mask_si_out_of_range(vals), vals)


def test_nan_passes_through_and_inf_is_masked():
    out = mask_si_out_of_range(np.array([np.nan, np.inf, -np.inf, 0.3]))
    assert np.isnan(out[:3]).all() and out[3] == 0.3


def test_series_keeps_index_and_dtype():
    s = pd.Series([0.1, 7.0, -2.0, np.nan], index=list("abcd"))
    out = mask_si_out_of_range(s)
    assert isinstance(out, pd.Series) and list(out.index) == list("abcd")
    assert out["a"] == 0.1 and out[["b", "c", "d"]].isna().all()
    assert out.dtype == float


def test_scalar_input():
    assert mask_si_out_of_range(1.0) == 1.0
    assert np.isnan(mask_si_out_of_range(3.0))


@pytest.mark.parametrize("fname", TIERS)
def test_built_tier_respects_si_range_and_qv_floor(fname):
    path = OUT_DIR / fname
    if not path.exists():
        pytest.skip(f"{fname} not built")
    import pyarrow.parquet as pq

    cols = pq.ParquetFile(path).schema.names
    si_cols = [c for c in cols if c == "Si" or c.startswith("Si_")]
    qv_cols = [c for c in cols if c == "qv" or c.startswith("qv_")]
    df = pd.read_parquet(path, columns=si_cols + qv_cols)
    for c in si_cols:
        s = df[c].dropna()
        assert s.empty or (s.min() >= SI_MIN and s.max() <= SI_MAX), (
            f"{fname}:{c} outside [{SI_MIN}, {SI_MAX}]: min={s.min()}, max={s.max()}"
        )
    for c in qv_cols:
        s = df[c].dropna()
        assert s.empty or s.min() >= 0, f"{fname}:{c} has negative values (min={s.min()})"
