# Dataset Summary — L0 / L1 / L2 — 2026-08-28

Descriptive reference for the current `combined_env_data*.parquet` tiers.
Computed directly against `data/out/combined_env_data.parquet` /
`_L1.parquet` / `_L2.parquet`, already rebuilt and validated this session
(`docs/reports/2026-08-28-dataset-validation.md`) — no rebuild performed for
this report. See CLAUDE.md's "Data tiers" section for tier definitions.

## 1. Overview

| Tier | Rows | Columns | Campaigns |
|---|---:|---:|---:|
| L0 | 4,572,581 | 46 | 15 |
| L1 | 2,997,447 | 47 | 12 |
| L2 | 1,828,818 | 47 | 11 |

L1/L2 have one extra column (`cpi_filename`) vs L0. Campaign drop-off:
OLYMPEX, POSIDON, ESCAPE have no CPI imagery so contribute 0 rows at L1/L2
(15 → 12 campaigns). MPACE has CPI imagery and reaches L1 (99.9% of its CPI
images matched) but flew no water-vapor instrument, so every row fails L2's
core-variable-complete filter (12 → 11 campaigns at L2).

## 2. Variables per tier

L0 has 46 columns, L1/L2 have 47 (add `cpi_filename`). Same set otherwise,
grouped by category:

**Identity / metadata (4, +1 at L1/L2):** `Timestamp`, `Campaign`,
`source_file`, and (L1/L2 only) `cpi_filename`.

**Core thermodynamic (6):** `Tair_C`, `Tair_K`, `P_hPa`, `Si`, `qv`, `Sw`.

**Per-instrument Si fallbacks (13):** `Si_chilled_mirror`, `Si_JLH`,
`Si_HW`, `Si_ALIAS`, `Si_LH_unspecified`, `Si_DLH`, `Si_frost_point`,
`Si_NOAA`, `Si_UCATS`, `Si_ophir_tdl`, `Si_MRTDL`, `Si_HWV`, `Si_FISH`.

**Per-instrument qv fallbacks (12):** `qv_chilled_mirror`, `qv_jlh`,
`qv_hw`, `qv_alias`, `qv_lh_unspecified`, `qv_dlh`, `qv_frost_point`,
`qv_noaa`, `qv_ucats`, `qv_ophir_tdl`, `qv_mrtdl`, `qv_hwv`.

**Raw mixing-ratio intermediates (5):** `H2O_DLH_ppmv`, `H2O_NOAA_ppmv`,
`H2O_UCATS_ppmv`, `MixingRatio_ppmv`, `MRTDLL_MC_ppmv`.

**Position (3):** `Lat`, `Lon`, `Alt_m`.

**Turbulence (4):** `Wind_U_ms`, `Wind_V_ms`, `Wind_W_ms`, `EDR_m23s1`
(unified across all reporting families — see
`docs/decisions/2026-07-13-edr-unification.md`).

`Si`/`qv` are each campaign's "best available" reading per
`config.yaml`'s `h2o_ranking`; the per-instrument Si/qv columns and raw
`*_ppmv` intermediates are the fallback sources behind that ranking —
present (non-NaN) only for campaigns that flew that specific instrument.

## 3. Per-variable completeness (% non-null)

### L0 (4,572,581 rows, 15 campaigns)

| Variable | % | Variable | % | Variable | % |
|---|---:|---|---:|---|---:|
| Timestamp | 100.0 | Campaign | 100.0 | source_file | 61.4 |
| Tair_C | 75.6 | Tair_K | 7.5 | P_hPa | 80.7 |
| Si | 59.0 | qv | 74.5 | Sw | 59.0 |
| Lat | 80.5 | Lon | 80.3 | Alt_m | 81.0 |
| Wind_U_ms | 68.1 | Wind_V_ms | 68.1 | Wind_W_ms | 72.1 |
| EDR_m23s1 | 42.9 | | | | |
| Si_chilled_mirror | 25.8 | qv_chilled_mirror | 25.8 | Si_JLH | 6.4 |
| Si_HW | 0.6 | Si_ALIAS | 0.3 | Si_LH_unspecified | 2.9 |
| Si_DLH | 19.5 | Si_frost_point | 2.7 | Si_NOAA | 8.7 |
| Si_UCATS | 6.5 | Si_ophir_tdl | 4.3 | Si_MRTDL | 4.0 |
| Si_HWV | 1.7 | Si_FISH | 0.0 | qv_jlh | 3.2 |
| qv_hw | 0.6 | qv_alias | 0.3 | qv_lh_unspecified | 2.9 |
| qv_dlh | 34.8 | qv_frost_point | 2.7 | qv_noaa | 15.5 |
| qv_ucats | 6.5 | qv_ophir_tdl | 4.3 | qv_mrtdl | 4.0 |
| qv_hwv | 1.7 | H2O_DLH_ppmv | 25.4 | H2O_NOAA_ppmv | 15.5 |
| H2O_UCATS_ppmv | 6.5 | MixingRatio_ppmv | 4.3 | MRTDLL_MC_ppmv | 4.0 |

`Si_FISH` is 0.0% — declared in `config.yaml` but never loaded (MACPEX;
FISH instrument data not available in this pipeline's raw inputs).
`Tair_K` (7.5%) is a raw-Kelvin passthrough kept for one campaign path, not
a general duplicate of `Tair_C` (75.6%).

### L1 (2,997,447 rows, 12 campaigns)

| Variable | % | Variable | % | Variable | % |
|---|---:|---|---:|---|---:|
| cpi_filename | 100.0 | Timestamp | 100.0 | Campaign | 100.0 |
| source_file | 93.7 | Tair_C | 92.9 | Tair_K | 0.0 |
| P_hPa | 97.6 | Si | 61.1 | qv | 61.0 |
| Sw | 61.0 | Lat | 99.8 | Lon | 99.8 |
| Alt_m | 99.9 | Wind_U_ms | 56.0 | Wind_V_ms | 56.0 |
| Wind_W_ms | 62.3 | EDR_m23s1 | 68.0 | |
| Si_chilled_mirror | 21.1 | qv_chilled_mirror | 21.0 | Si_JLH | 2.1 |
| Si_HW | 0.3 | Si_ALIAS | 0.0 | Si_LH_unspecified | 28.3 |
| Si_DLH | 8.7 | Si_frost_point | 0.0 | Si_NOAA | 2.7 |
| Si_UCATS | 1.7 | Si_ophir_tdl | 0.9 | Si_MRTDL | 1.5 |
| Si_HWV | 0.8 | Si_FISH | 0.0 | qv_jlh | 1.6 |
| qv_hw | 0.3 | qv_alias | 0.0 | qv_lh_unspecified | 28.3 |
| qv_dlh | 8.7 | qv_frost_point | 0.0 | qv_noaa | 2.7 |
| qv_ucats | 1.7 | qv_ophir_tdl | 0.9 | qv_mrtdl | 1.5 |
| qv_hwv | 0.8 | H2O_DLH_ppmv | 3.9 | H2O_NOAA_ppmv | 2.7 |
| H2O_UCATS_ppmv | 1.7 | MixingRatio_ppmv | 0.9 | MRTDLL_MC_ppmv | 1.5 |

`Tair_K` drops to 0.0% at L1 — its one source campaign has no CPI imagery
overlap for that raw field. `Si_frost_point`/`qv_frost_point` similarly
drop to 0.0% — OLYMPEX (their only source) has zero CPI images.

### L2 (1,828,818 rows, 11 campaigns)

| Variable | % | Variable | % | Variable | % |
|---|---:|---|---:|---|---:|
| cpi_filename | 100.0 | Timestamp | 100.0 | Campaign | 100.0 |
| source_file | 93.3 | Tair_C | 100.0 | Tair_K | 0.0 |
| P_hPa | 100.0 | Si | 100.0 | qv | 100.0 |
| Sw | 100.0 | Lat | 100.0 | Lon | 100.0 |
| Alt_m | 100.0 | Wind_U_ms | 70.8 | Wind_V_ms | 70.8 |
| Wind_W_ms | 72.1 | EDR_m23s1 | 61.5 | |
| Si_chilled_mirror | 34.5 | qv_chilled_mirror | 34.5 | Si_JLH | 3.4 |
| Si_HW | 0.4 | Si_ALIAS | 0.0 | Si_LH_unspecified | 46.4 |
| Si_DLH | 14.3 | Si_frost_point | 0.0 | Si_NOAA | 4.5 |
| Si_UCATS | 2.8 | Si_ophir_tdl | 1.5 | Si_MRTDL | 2.5 |
| Si_HWV | 1.4 | Si_FISH | 0.0 | qv_jlh | 2.7 |
| qv_hw | 0.4 | qv_alias | 0.0 | qv_lh_unspecified | 46.4 |
| qv_dlh | 14.3 | qv_frost_point | 0.0 | qv_noaa | 4.5 |
| qv_ucats | 2.8 | qv_ophir_tdl | 1.5 | qv_mrtdl | 2.5 |
| qv_hwv | 1.4 | H2O_DLH_ppmv | 6.5 | H2O_NOAA_ppmv | 4.5 |
| H2O_UCATS_ppmv | 2.8 | MixingRatio_ppmv | 1.5 | MRTDLL_MC_ppmv | 2.5 |

Core 7 variables (`Tair_C`, `P_hPa`, `Si`, `qv`, `Sw`, `Lat`, `Lon`,
`Alt_m` — `CORE_COLS` in `scripts/build_data_tiers.py`) are 100% by
construction; turbulence and per-instrument columns are ungated and simply
reflect whichever rows happen to also have that field.

## 4. Aggregation by campaign

### Row counts, all tiers

| Campaign | L0 | L1 | L2 |
|---|---:|---:|---:|
| AIRS-II | 312,792 | 92,168 | 92,168 |
| ARM | 141,940 | 230,029 | 64,706 |
| ATTREX | 1,316,204 | 122,050 | 120,595 |
| CRYSTAL-FACE-NASA | 323,310 | 78,151 | 20,441 |
| CRYSTAL-FACE-UND | 200,864 | 1,608,674 | 848,940 |
| ESCAPE | 67,380 | 0 | 0 |
| ICE-L | 210,561 | 46,203 | 46,202 |
| IPHEX | 287,600 | 38,697 | 28,189 |
| ISDAC | 357,071 | 400,805 | 399,668 |
| MACPEX | 307,780 | 80,240 | 51,747 |
| MC3E | 165,431 | 173,766 | 137,272 |
| MIDCIX | 181,459 | 90,667 | 18,890 |
| MPACE | 139,360 | 35,997 | 0 |
| OLYMPEX | 209,321 | 0 | 0 |
| POSIDON | 351,508 | 0 | 0 |
| **Total** | **4,572,581** | **2,997,447** | **1,828,818** |

L1 > L0 for ARM (141,940 → 230,029) and CRYSTAL-FACE-UND (200,864 →
1,608,674): L1 is one row per CPI *image*, and a single L0 second can hold
multiple CPI images — these two campaigns have especially high per-second
image rates.

### Per-campaign completeness, core variables (L0)

| Campaign | Tair_C | P_hPa | Si | qv | Sw | Lat | Lon | Alt_m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AIRS-II | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| ARM | 99.0 | 100.0 | 36.4 | 36.4 | 36.4 | 91.7 | 91.7 | 100.0 |
| ATTREX | 44.0 | 44.0 | 37.6 | 91.4 | 37.6 | 43.9 | 43.9 | 43.9 |
| CRYSTAL-FACE-NASA | 50.3 | 100.0 | 50.3 | 50.3 | 50.3 | 99.9 | 99.9 | 99.9 |
| CRYSTAL-FACE-UND | 98.3 | 99.5 | 66.2 | 65.9 | 65.9 | 99.9 | 99.9 | 100.0 |
| ESCAPE | 86.9 | 98.4 | 86.5 | 86.4 | 86.5 | 100.0 | 100.0 | 98.4 |
| ICE-L | 100.0 | 100.0 | 99.7 | 99.8 | 99.7 | 100.0 | 100.0 | 99.7 |
| IPHEX | 99.6 | 100.0 | 68.4 | 68.8 | 68.4 | 99.4 | 99.4 | 99.4 |
| ISDAC | 99.9 | 99.4 | 100.0 | 99.3 | 99.9 | 100.0 | 97.5 | 100.0 |
| MACPEX | 90.7 | 90.7 | 63.4 | 63.4 | 63.4 | 90.7 | 90.7 | 90.7 |
| MC3E | 98.6 | 100.0 | 88.2 | 88.2 | 88.2 | 90.7 | 90.7 | 100.0 |
| MIDCIX | 41.8 | 41.8 | 41.8 | 41.8 | 41.8 | 100.0 | 100.0 | 98.2 |
| MPACE | 58.9 | 100.0 | 0.0 | 0.0 | 0.0 | 89.4 | 89.4 | 89.4 |
| OLYMPEX | 100.0 | 100.0 | 58.4 | 58.4 | 58.4 | 100.0 | 100.0 | 100.0 |
| POSIDON | 97.2 | 98.2 | 52.0 | 53.1 | 52.0 | 76.6 | 76.6 | 76.6 |

At L2 all 8 core variables read 100% for every one of the 11 campaigns
present, by construction (the filter). L1's per-campaign core-variable
completeness sits between the L0 and L2 figures above for each campaign,
generally higher than L0 (CPI-image timestamps concentrate on
instrumented flight segments) — see
`docs/reports/2026-07-13-turbulence-phase2-3-dataset-report.md` for a
worked example (CRYSTAL-FACE-NASA's core vars jump from ~50-100% at L0 to
99-100% at L1).

Full per-campaign × per-column breakdown (all 46/47 variables, all three
tiers) is not reproduced here — a 46-column × 15-campaign table isn't
readable in markdown. Regenerate the underlying CSVs directly:

```bash
conda activate cpi-thermo
python scripts/build_data_tiers.py             # logs/build_data_tiers/latest/tier_summary.csv (row counts)
python scripts/diagnose_turbulence_coverage.py  # logs/diagnose_turbulence_coverage/latest/coverage_by_campaign.csv (wind/EDR)
python scripts/diagnose_cpi_fusion.py           # logs/cpi_fusion/latest/cpi_fusion_report.txt (per-campaign, per-variable, L1 match rates)
```

## 5. Basic stats, core physical variables

| Variable | Tier | Mean | Std | Min | Max |
|---|---|---:|---:|---:|---:|
| Tair_C | L0 | -34.071 | 32.126 | -88.850 | 41.700 |
| | L1 | -28.383 | 18.595 | -86.480 | 28.944 |
| | L2 | -26.821 | 20.232 | -86.480 | 28.944 |
| P_hPa | L0 | 439.741 | 297.008 | 50.027 | 1042.194 |
| | L1 | 417.288 | 212.197 | 77.700 | 1023.151 |
| | L2 | 448.921 | 221.327 | 77.700 | 1020.366 |
| Si | L0 | -0.275 | 0.359 | -1.000 | 2.000 |
| | L1 | -0.037 | 0.133 | -1.000 | 1.855 |
| | L2 | -0.037 | 0.133 | -1.000 | 1.855 |
| qv (g/kg) | L0 | 1.372 | 3.298 | 0.000 | 100.790 |
| | L1 | 1.581 | 1.653 | 0.000 | 17.514 |
| | L2 | 1.582 | 1.653 | 0.000 | 17.514 |
| Sw | L0 | -0.454 | 0.293 | -1.000 | 4.852 |
| | L1 | -0.244 | 0.146 | -1.000 | 0.931 |
| | L2 | -0.244 | 0.146 | -1.000 | 0.931 |
| Alt_m | L0 | 8351.058 | 5829.384 | -97.000 | 20572.000 |
| | L1 | 7926.079 | 3612.505 | -52.000 | 17788.100 |
| | L2 | 7358.898 | 3845.754 | -41.000 | 17788.100 |

L0's wider ranges (e.g. qv max 100.8 g/kg, Alt_m min -97 m) reflect raw,
unfiltered per-second env data across all 15 campaigns, including
low-altitude/high-moisture segments (e.g. ESCAPE ground ops) that don't
survive the CPI-image join to L1/L2. L1 and L2 have nearly identical
stats for Si/qv/Sw (both already conditioned on having a CPI image at that
second) — L2 differs mainly by which campaigns/rows are dropped for
missing core variables, not by a different underlying distribution.
