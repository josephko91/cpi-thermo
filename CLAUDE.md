# CPI Thermo — Claude Code Context

## What this project does

Combines atmospheric aircraft campaign data (15 campaigns, ~3.84M records) into a single
parquet for thermodynamic analysis — primarily ice supersaturation (Si), water vapor (qv),
and temperature vs altitude. Parsers normalize each campaign's raw format to a standard
column schema; `main.py` runs all parsers and writes `data/out/combined_env_data.parquet`.

## Key files

| File | Role |
|------|------|
| `main.py` | Top-level pipeline — loads all campaigns, writes parquet |
| `config.yaml` | Per-campaign settings (h2o_ranking, file paths) |
| `parsers/<campaign>.py` | One parser per campaign; each has `load_*()` + `extract_*_standard()` |
| `parsers/utils.py` | Thermodynamic utilities: `es_ice_hPa`, `es_liq_hPa`, `qv_from_e_P`, `si_from_frost_point` |
| `scripts/qa_checks.py` | 9 QC check functions; writes CSVs to `logs/qaqc/<timestamp>/` |
| `data/out/combined_env_data.parquet` | Main output (gitignored) |
| `parsers/cpi_timestamps.py` | Canonical loader for `data/raw/cpi_embeddings_timestamps.csv` (CPI particle-image timestamps); normalizes campaign names and known UTC-offset bugs (e.g. MC3E) |
| `scripts/diagnose_cpi_fusion.py` | Cross-references CPI image timestamps against `combined_env_data.parquet`; writes `logs/cpi_fusion/<timestamp>/cpi_fusion_report.txt` |
| `scripts/log_paths.py` | Shared helper: every diagnostic script writes to `<logs\|figs>/<script>/<timestamp>/` and refreshes a `latest` symlink — see "Logs & figs layout" below |

## Standard output schema

Every `extract_*_standard()` returns these columns:

```
Timestamp, Tair_C, P_hPa, Si, Si_chilled_mirror, Si_<instrument>, qv, qv_chilled_mirror,
qv_<instrument>, Sw, Lat, Lon, Alt_m, Campaign, source_file
```

## Conventions

- Temperature in **Celsius** (`Tair_C`), pressure in **hPa**, altitude in **meters**
- All timestamps UTC (`tz_localize("UTC")` or `tz_convert("UTC")`)
- Si = ice supersaturation (dimensionless); Sw = liquid supersaturation; qv in g/kg
- Fill values → NaN before returning from `load_*()`, not after
- Never commit parquet files, plots, or `logs/` — all gitignored

## Campaigns

ARM, AIRS-II, ATTREX, CRYSTAL-FACE-NASA, CRYSTAL-FACE-UND, ESCAPE, IPHEX, ICE-L,
ISDAC, MACPEX, MC3E, MIDCIX, MPACE, OLYMPEX, POSIDON

## Known issues / active investigations

See `docs/decisions/` for per-investigation records and `docs/sessions/` for
session-by-session summaries. Key items:

- **ARM qv NaN**: 63.6% NaN (real data sparsity in dry upper troposphere, not parser bug)
- **ARM 2000-03-13 CPI timestamp anomaly**: CPI has images at 00:00-01:xx UTC with no
  corresponding env data that date (env only covers 18:07-22:29). Investigated —
  raw archive is complete (12 files = campaign's official "12 IOP flights" exactly;
  filename encoding matches every file's actual data to the minute). Most likely a
  CPI-side ground-test/calibration session or clock fault, not a missing raw file.
  See `docs/decisions/2026-07-05-arm-cpi-timestamp-investigation.md`.
- **IPHEX/OLYMPEX cold-regime Si flags** (1,391 / 45 rows, IPHEX 2014-06-13 + 2014-05-19
  flights and OLYMPEX 15_39_28 flight): chilled-mirror hysteresis at extreme cold can
  amplify small errors into large fractional Si swings, but Si up to ~1.5-1.7 is also
  physically documented for real cirrus near the homogeneous-freezing threshold — kept
  in the data, flagged via `cold_regime_amplification_candidate` in QC9's
  `09_lwc_crossval.csv`. Needs an independent cross-check (e.g. Ophir TDL) to resolve.
- **MIDCIX Alt_m** at 96.7%, not 100%: navigation (`FP`) files cover 2 more flight
  dates than the water-vapor (JW) files that `load_midcix()` keys rows off of.
- **MPACE**: `parsers/mpace.py` added 2026-07-06, loading the UND Citation NASA Ames
  files from `data/raw/MPACE` (15 flights, 2004-09-30 to 2004-10-21, Barrow AK).
  No water-vapor instrument was flown on this platform, so Si/qv are NaN for every
  record — only Tair_C/P_hPa/Lat/Lon/Alt_m are populated. CPI fusion: 99.9% of
  ~36k CPI images get a matched timestamp, 76.3% get Tair_C.
- **CPI/env fusion** (as of 2026-07-06, post-MPACE): 91.3% of 3.2M CPI images have a
  matched thermo timestamp (±1s, up from 89.1% pre-MPACE); 57.7% have both Tair_C and
  Si (unchanged — MPACE contributes matches but no Si/qv). Run
  `python scripts/diagnose_cpi_fusion.py` for the full per-campaign breakdown.
- Resolved 2026-07-05: ESCAPE P_hPa<50 residual, ESCAPE 2022-06-10 sensor-failure mask
  gap, IPHEX/OLYMPEX qv/Si bound asymmetry, POSIDON P_hPa sentinel bug, OLYMPEX
  19_51_41-flight chilled-mirror fault masking, Alt_m recovery for CRYSTAL-FACE-NASA,
  CRYSTAL-FACE-UND, MACPEX, and MIDCIX (79.2% → 98.2% overall Alt_m coverage), an
  MC3E CPI timestamp timezone mislabeling, CRYSTAL-FACE-NASA's ALIAS Si fallback, and
  CRYSTAL-FACE-UND's missing 2002-07-11 flight segment — see
  `docs/decisions/2026-07-05-open-issues-resolved.md`,
  `docs/decisions/2026-07-05-qc9-iphex-olympex.md`, and
  `docs/decisions/2026-07-05-cpi-fusion-gap-fixes.md`.

## Running the pipeline

```bash
python main.py                          # rebuild parquet + diagnostics + figures
python scripts/qa_checks.py             # run all 9 QC checks
python scripts/diagnose_cpi_fusion.py   # cross-check CPI images vs env data
```

## Logs & figs layout

Every diagnostic entry point writes to its own `logs/<script>/<YYYYMMDD_HHMMSS>/`
and/or `figs/<script>/<YYYYMMDD_HHMMSS>/` directory (never overwriting a prior
run) and refreshes a `<script>/latest` symlink, via the shared helper in
`scripts/log_paths.py`:

| Script | Logs | Figs |
|--------|------|------|
| `main.py` | `logs/pipeline/<ts>/` (output.log + campaign/Si/qv coverage CSVs) | `figs/all-campaigns/<ts>/` (via `plot_all_campaigns.py`, plots 01–12) |
| `scripts/qa_checks.py` | `logs/qaqc/<ts>/` (00–09 CSVs) | `figs/qaqc/<ts>/` |
| `scripts/diagnose_cpi_fusion.py` | `logs/cpi_fusion/<ts>/` | `figs/cpi_fusion/<ts>/` |
| `scripts/diagnose_campaign_missingness.py` | `logs/campaign_missingness/<ts>/` | — |
| `scripts/summarize_parser_recommendations.py` | reads `logs/campaign_missingness/latest/` by default | — |
| `scripts/full_diagnostic.py` | — (console-only: variable stats, availability table, known-issue checks) | — |
| `scripts/tests/test_{attrex,ice_l,iphex}.py` | `logs/campaign_tests/<campaign>/<ts>/` | `figs/campaign_tests/<campaign>/<ts>/` |

`scripts/full_diagnostic.py` used to also write its own `figs/full_diagnostic/`
figures, but 4 of its 6 distributions plus its Si-vs-Tair scatter duplicated
`plot_all_campaigns.py`'s plots 01/02/03/07/08; the 3 genuinely unique ones
(Alt_m distribution, Sw distribution, Alt_m-vs-Tair scatter) were folded into
`plot_all_campaigns.py` as plots 10–12 on 2026-07-06, and `full_diagnostic.py`
is now console-output only.

Pre-reorg content (from before 2026-07-06) was preserved under `logs/archive/`
and `figs/archive/`, and old date-only snapshots (`qaqc_20260630/`, etc.) were
migrated into the `<script>/<ts>/` layout above. `logs/` and `figs/` are both
gitignored in full, so none of this is version-controlled.
