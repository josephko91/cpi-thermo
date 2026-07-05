# CPI Thermo — Claude Code Context

## What this project does

Combines atmospheric aircraft campaign data (14 campaigns, ~3.6M records) into a single
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
| `scripts/qa_checks.py` | 8 QC check functions; writes CSVs to `logs/qaqc_<YYYYMMDD>/` |
| `data/out/combined_env_data.parquet` | Main output (gitignored) |

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
ISDAC, MACPEX, MC3E, MIDCIX, OLYMPEX, POSIDON

## Known issues / active investigations

See `docs/decisions/` for per-investigation records. Key items:

- **ESCAPE residual**: 1,104 rows with P_hPa < 50 hPa from stuck/erroneous Palt driving ICAO formula
- **IPHEX severe qv_exceeds_saturation**: 11,931 rows (4.1%) — possible rain contamination on chilled mirror
- **OLYMPEX severe flags**: 1,341 rows — marine precipitation, LWC flag needed
- **ARM qv NaN**: 63.6% NaN (real data sparsity in dry upper troposphere, not parser bug)

## Running the pipeline

```bash
python main.py                          # rebuild parquet
python scripts/qa_checks.py \
  --env data/out/combined_env_data.parquet \
  --out logs/qaqc_$(date +%Y%m%d)      # run all 8 QC checks
```
