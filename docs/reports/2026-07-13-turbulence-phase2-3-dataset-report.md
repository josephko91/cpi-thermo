# Turbulence Phase 2+3 Completion — Dataset Variable Availability Report

Date: 2026-07-13
Status: Phase 1 (PR #16, merged) + Phase 2 + Phase 3 of the turbulence
measurements plan now all implemented (see `docs/dataset-changelog.md`'s
2026-07-13 entry). Full pipeline rebuilt and re-diagnosed after the change.

**Superseded, same day:** the wide column set below (`Roll_deg`, `Pitch_deg`,
`Heading_deg`, `TAS_ms`, etc.) was deliberately dropped later on 2026-07-13
— see `docs/reports/2026-07-13-turbulence-scope-reduction-diagnostics.md`
for the final, authoritative schema (`Wind_U/V/W_ms` + unified `EDR_m23s1`
only) and its corrected CRYSTAL-FACE-NASA coverage figure. This report is
kept for its historical record of the Phase 2/3 loader changes, not as a
current schema reference.

## What changed this session

Implemented the two remaining phases from the turbulence plan:

- **Phase 2** (upstream loader fixes): `crystal_face_und.py`, `crystal_face_nasa.py`,
  `midcix.py`, `escape.py` — extended merge column lists / added `_choose_column`
  lookups to surface wind, attitude, EDR, and airspeed fields already sitting
  unused in each loader's raw read.
- **Phase 3** (new read paths): `ice_l.py`, `airs_ii.py` — extended/added
  `_pick_var` fallback lookups for the NCAR/RAF-Nimbus Family-C variable set
  (wind, pitch/roll/heading, AoA/sideslip, TAS).
- `config.yaml`'s `output.standardized_columns` doc block updated; `qa_checks.py`
  `HARD_BOUNDS` already covered the new columns from Phase 1 (no change needed).
- Excluded per plan: ESCAPE's NRC-Convair source, CRYSTAL-FACE-NASA's
  MG/FP/FT/PT subdirs, ISDAC's 5 Hz wolde-convair source, TKE derivation.

Full pipeline rerun: `main.py` (L0) → `build_data_tiers.py` (L1/L2) →
`qa_checks.py` (9 checks) → `diagnose_turbulence_coverage.py`.

## Regression check

L0 row count unchanged: **4,572,581** rows, 15 campaigns (matches pre-change
baseline in CLAUDE.md — Phase 2/3 only add columns, no row-count effect, as
expected since new fields ride existing merges/timestamps).

L1: 2,997,447 rows. L2: 1,828,818 rows. Both match expected pattern (OLYMPEX/
POSIDON/ESCAPE = 0 rows at L1/L2, MPACE = 0 rows at L2 — see CLAUDE.md's
"Campaigns" section for why).

QC (`qa_checks.py`): all 9 checks ran clean, flag rates in line with known
baseline (QC9's 1,436 severe-Si rows = the already-documented IPHEX/OLYMPEX
cold-regime issue, not new).

Turbulence coverage diagnostic (`diagnose_turbulence_coverage.py`): level-flight
Roll/Pitch medians all within a few degrees of 0 for every campaign (no sign
inversion), EDR-family histograms show no unit mixups (Family A log10 vs
Family B linear ranges stayed distinct, as required by
`docs/decisions/2026-07-13-turbulence-schema.md`).

## Variable availability — overall (% non-null), by tier

| Variable | L0 (4.57M) | L1 (3.00M) | L2 (1.83M) |
|---|---|---|---|
| Tair_C | 75.6 | 92.9 | 100.0 |
| P_hPa | 80.7 | 97.6 | 100.0 |
| Si | 59.0 | 61.1 | 100.0 |
| Si_chilled_mirror | 25.8 | 21.1 | 34.5 |
| qv | 74.5 | 61.0 | 100.0 |
| qv_chilled_mirror | 25.8 | 21.0 | 34.5 |
| Sw | 59.0 | 61.0 | 100.0 |
| Lat / Lon | 80.5 / 80.3 | 99.8 / 99.8 | 100.0 / 100.0 |
| Alt_m | 81.0 | 99.9 | 100.0 |
| Wind_U_ms / Wind_V_ms | 32.0 | 9.3 | 10.5 |
| Wind_W_ms | 72.6 | 62.3 | 72.1 |
| WindSpeed_ms / WindDir_deg | 43.8 | 49.2 | 61.4 |
| Roll_deg / Pitch_deg | 67.0 | 85.1 | 90.4 / 90.5 |
| Heading_deg | 68.5 | 85.1 | 90.5 |
| AngleOfAttack_deg / Sideslip_deg | 34.8 / 34.7 | 24.9 | 37.5 |
| VertVel_ms | 16.8 | 8.1 | 9.0 |
| Accel_Vert_ms2 | 28.3 | 12.8 | 16.6 |
| TAS_ms | 67.2 | 25.8 | 30.2 |
| IAS_ms / IAS_ms_nose | 16.5 / 2.8 | 7.1 / 1.2 | 8.5 / 0.0 |
| MachNo | 21.4 | 8.6 | 9.6 |
| DriftAngle_deg / TrackAngle_deg | 7.7 / 18.5 | 13.3 / 68.3 | 21.8 / 67.2 |
| EDR_mms_log10kWkg (Family A) | 20.0 | 4.1 | 6.6 |
| EDR_und_cm23s1 (Family B) | 19.9 | 56.3 | 51.4 |
| EDR_und_cm23s1_nose (MPACE) | 1.7 | 0.9 | 0.0 |
| EDR_arm (ARM) | 3.1 | 7.7 | 3.5 |
| REYN_mms | 20.1 | 4.1 | 6.6 |

Note: L1/L2 % is computed over the campaigns that reach that tier (OLYMPEX/
POSIDON/ESCAPE drop out entirely at L1/L2 — see CLAUDE.md); L2's higher core
variable % is by construction (`CORE_COLS` filter). Turbulence columns are
NOT in `CORE_COLS`, so their % at L2 reflects whichever rows happened to also
have wind/attitude/EDR filled, not a completeness guarantee.

## Per-campaign availability by tier

Full CSVs (all 33 columns × campaign, one file per tier):
`logs/diagnose_turbulence_coverage/latest/coverage_by_campaign.csv` (L0-specific,
turbulence-focused) plus ad hoc L0/L1/L2 breakdowns saved to
`logs/build_data_tiers/latest/tier_summary.csv` (row counts) — this report's
full per-campaign × per-tier % table is reproduced below.

### L0 (all 15 campaigns, 4,572,581 rows)

| Campaign | Tair_C | Si | qv | Wind_W_ms | WindSpeed | Roll/Pitch | Heading | AoA/Sideslip | TAS_ms | EDR (family) |
|---|---|---|---|---|---|---|---|---|---|---|
| AIRS-II | 100.0 | 100.0 | 100.0 | 96.9 | 96.9 | 100.0 | 100.0 | 98.7/98.8 | 100.0 | — |
| ARM | 99.0 | 36.4 | 36.4 | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | EDR_arm 100.0 |
| ATTREX | 44.0 | 37.6 | 91.4 | 43.9 | 0.0 | 43.9 | 43.9 | 0.0 | 44.0 | EDR_A 43.9 |
| CRYSTAL-FACE-NASA | 50.3 | 50.3 | 50.3 | 89.1 | 100.0 | 99.9 | 99.9 | 0.0 | 99.9 | — |
| CRYSTAL-FACE-UND | 98.3 | 66.2 | 65.9 | 77.7 | 77.7 | 100.0 | 100.0 | 0.0 | 6.4 | EDR_B 93.1 |
| ESCAPE | 86.9 | 86.5 | 86.4 | 100.0 | 0.0 | 0.0 | 100.0 | 0.0 | 100.0 | — |
| ICE-L | 100.0 | 99.7 | 99.8 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0/100.0 | 100.0 | — |
| IPHEX | 99.6 | 68.4 | 68.8 | 88.4 | 88.5 | 99.4 | 99.4 | 99.6/99.6 | 99.4 | EDR_B 99.3 |
| ISDAC | 99.9 | 100.0 | 99.3 | 90.7 | 98.6 | 90.6/90.7 | 90.7 | 93.7/93.7 | 0.0 | — |
| MACPEX | 90.7 | 63.4 | 63.4 | 83.1 | 0.0 | 0.0 | 0.0 | 0.0 | 90.7 | — |
| MC3E | 98.6 | 88.2 | 88.2 | 77.7 | 77.7 | 90.7 | 90.7 | 87.2/87.4 | 95.8 | EDR_B 93.6 |
| MIDCIX | 41.8 | 41.8 | 41.8 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 88.0 | — |
| MPACE | 58.9 | 0.0 | 0.0 | 47.5 | 47.5 | 89.4 | 89.4 | 88.5/83.7 | 95.1 | EDR_B 51.2 / nose 57.4 |
| OLYMPEX | 100.0 | 58.4 | 58.4 | 99.1 | 99.1 | 100.0 | 100.0 | 88.4/88.2 | 100.0 | EDR_B 100.0 |
| POSIDON | 97.2 | 52.0 | 53.1 | 96.4 | 0.0 | 98.2 | 98.2 | 0.0 | 97.2 | EDR_A 96.4 |

New this session (previously all-zero for these columns): **AIRS-II**
(Wind/Roll/Pitch/Heading/AoA/Sideslip/TAS: 0→97-100%), **CRYSTAL-FACE-NASA**
(Wind/Speed/Dir/Roll/Pitch/Heading/Mach/TrackAngle: 0→89-100%),
**CRYSTAL-FACE-UND** (Wind_W/Speed/Dir/Roll/Pitch/Heading/TrackAngle/EDR_B:
0→6-100%), **MIDCIX** (TAS_ms: 0→88.0%), **ESCAPE** (Wind_W/Heading/TAS:
0→100%), **ICE-L** (Wind_W/Speed/Dir/Roll/Pitch/Heading/AoA/Sideslip/Accel/TAS:
0→100%).

### L1 (12 campaigns with CPI images, 2,997,447 rows)

Same pattern, generally higher % for env core vars (CPI-image join biases
toward instrumented flight segments) — see full table in
`/tmp/L1_by_campaign_pct.csv` region reproduced in this report's companion
CSV export (regenerable via the commands in "Reproduce" below). Headline:
CRYSTAL-FACE-NASA's Wind/Roll/Pitch/Heading jump to 99-100% at L1 (was 89-100%
at L0 — CPI images concentrated during instrumented legs); AIRS-II and ICE-L
stay at 97-100%.

### L2 (10 campaigns with any L2 rows, 1,828,818 rows)

Core vars are 100% by construction. Turbulence-column % generally rises
further at L2 for the same reason (more selective row filter concentrates on
well-instrumented segments): e.g. CRYSTAL-FACE-NASA's Wind_U/V/W and TAS all
reach 100%; ATTREX's Wind/Roll/Pitch/Heading/EDR_A/REYN reach 100%.

## Reproduce

```bash
conda activate cpi-thermo
python main.py --all
python scripts/build_data_tiers.py
python scripts/qa_checks.py
python scripts/diagnose_turbulence_coverage.py
```

Full per-campaign × per-column % tables (all 33 standard+turbulence columns,
all 3 tiers) available by rerunning the groupby shown in this report's
generation — not checked in as CSV since `logs/`/`data/out/` are gitignored;
regenerate with:

```python
import pandas as pd
cols = [...]  # see CLAUDE.md's "Standard output schema" section for the full column list
for tier, path in [("L0","data/out/combined_env_data.parquet"),
                    ("L1","data/out/combined_env_data_L1.parquet"),
                    ("L2","data/out/combined_env_data_L2.parquet")]:
    df = pd.read_parquet(path)
    df.groupby("Campaign")[cols].apply(lambda g: g.notna().mean()*100).round(1)
```

## Known follow-ups (unchanged, still open)

Same list as CLAUDE.md's "Phase 1 turbulence columns" known-issues entry
(ARM missing several already-available raw fields, ATTREX/POSIDON EDR/REYN
scale unconfirmed independently, fill-value masking not yet applied to new
IPHEX/MPACE/MC3E/OLYMPEX turbulence columns, single global Roll/Pitch
`HARD_BOUNDS` can't catch a per-family sign inversion) — none of that was in
scope for Phase 2/3 and remains a separate follow-up.
