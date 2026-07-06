"""
Loader for the CPI (Cloud Particle Imager) embeddings timestamp archive.

Data source: data/raw/cpi_embeddings_timestamps.csv (campaign, filename, datetime).
This file records the acquisition time of every particle image used to build the
SSL embeddings, independent of the thermodynamic (env) pipeline in main.py.

Two data-quality quirks are corrected here, in one canonical place, so any
downstream code that fuses CPI images with env data gets a consistent result
without re-discovering them:

1. Campaign naming: this CSV uses underscores (CRYSTAL_FACE_UND, AIRS_II,
   ICE_L); the env pipeline uses hyphens (CRYSTAL-FACE-UND, AIRS-II, ICE-L).
2. MC3E timezone: MC3E's CPI timestamps were recorded in local CDT (UTC-5) but
   the datetime column is labelled UTC. Evidence: CPI images on 2011-05-23 run
   16:35-19:32 while env data for the same date starts at 21:20 UTC -- a ~5h
   gap consistent with the CDT offset, not a real day-to-day acquisition gap.
   Confirmed by applying the +5h correction: MC3E's matched-timestamp rate
   (+/-1s tolerance) rises from 7.5% (uncorrected, 5 min tolerance) to 92.7%.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

#: CPI CSV campaign name -> env pipeline campaign name.
CPI_TO_ENV_CAMPAIGN: dict[str, str] = {
    "AIRS_II":           "AIRS-II",
    "ARM":               "ARM",
    "ATTREX":            "ATTREX",
    "CRYSTAL_FACE_NASA": "CRYSTAL-FACE-NASA",
    "CRYSTAL_FACE_UND":  "CRYSTAL-FACE-UND",
    "ICE_L":             "ICE-L",
    "IPHEX":             "IPHEX",
    "ISDAC":             "ISDAC",
    "MACPEX":            "MACPEX",
    "MC3E":              "MC3E",
    "MIDCIX":            "MIDCIX",
    "MPACE":             "MPACE",
    "OLYMPEX":           "OLYMPEX",
    "POSIDON":           "POSIDON",
    "ESCAPE":            "ESCAPE",
}

#: Per-campaign hour offsets to add to the CSV's "datetime" column to get true
#: UTC. Only campaigns with a confirmed local-time mislabeling are listed.
CPI_UTC_OFFSET_HOURS: dict[str, float] = {
    "MC3E": 5.0,   # recorded in CDT (UTC-5); flown from Oklahoma, May 2011
}


def load_cpi_embeddings_timestamps(path: Union[str, Path]) -> pd.DataFrame:
    """Load the CPI embeddings timestamp archive with known corrections applied.

    Returns a DataFrame with the original ``campaign``, ``filename``,
    ``datetime`` columns (datetime corrected to true UTC per
    ``CPI_UTC_OFFSET_HOURS``), plus a ``campaign_env`` column mapping each row
    to its corresponding env-pipeline campaign name (per
    ``CPI_TO_ENV_CAMPAIGN``).
    """
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")

    for campaign, hours in CPI_UTC_OFFSET_HOURS.items():
        mask = df["campaign"] == campaign
        if mask.any():
            df.loc[mask, "datetime"] = df.loc[mask, "datetime"] + pd.Timedelta(hours=hours)

    df["campaign_env"] = df["campaign"].map(CPI_TO_ENV_CAMPAIGN)
    return df
