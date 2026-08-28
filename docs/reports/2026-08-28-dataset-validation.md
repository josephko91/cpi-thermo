# Dataset Validation Report — 2026-08-28

Post repo-condensation rebuild and validation of L0/L1/L2. **No pipeline
code changed** (`parsers/`, `main.py`, `config.yaml`, `scripts/build_data_tiers.py`
all unchanged since the last parser commit, `9c02cf0`, 2026-07-13). This run
exists to confirm the dataset reproduces exactly from current code after the
docs/notes cleanup described in `docs/dataset-changelog.md`'s 2026-08-28
entry, and to catch any drift.

## Reproduce

```bash
conda activate cpi-thermo
python main.py --all
python scripts/build_data_tiers.py
python scripts/qa_checks.py
python scripts/diagnose_cpi_fusion.py
python scripts/diagnose_turbulence_coverage.py
```

## Row/campaign counts — L0/L1/L2

| Tier | Rows | vs. last known baseline |
|---|---:|---|
| L0 | 4,572,581 | Exact match (`logs/pipeline/20260729_143622/`, per-campaign row counts identical) |
| L1 | 2,997,447 | Exact match (`logs/build_data_tiers/20260713_211259/`) |
| L2 | 1,828,818 | Exact match |

15 campaigns at L0; 12 with any CPI imagery (3,200,351 images); 10 with any
L2 rows (OLYMPEX/POSIDON/ESCAPE have no CPI imagery, MPACE has no Si/qv —
see CLAUDE.md's "Campaigns" section). Per-campaign L0 row counts (this run)
match `logs/pipeline/20260729_143622/campaign_summary.csv` exactly, all 15
campaigns.

## QA checks (`scripts/qa_checks.py`)

| Check | Name | Flags | vs. baseline |
|---|---|---:|---|
| QC1 | Physical range checks | 6 | match |
| QC2 | Internal consistency | 80,648 | match |
| QC3 | Stuck-sensor / temporal continuity | 365 | match |
| QC4 | Fill/sentinel value detection | 0 | match |
| QC5 | Inter-instrument cross-validation | 0 | match |
| QC6 | Per-flight coverage audit | 67 | match |
| QC7 | Duplicate/out-of-order timestamps | 2 | match |
| QC8 | Vertical profile plausibility | 6 | match |
| QC9 | LWC cross-check (severe Si flags) | 1,436 | match |

Baseline: `docs/reports/2026-07-13-turbulence-scope-reduction-diagnostics.md`.
All 9 checks reproduce exactly.

## CPI/env timestamp fusion

| Metric | Count | % | vs. baseline |
|---|---:|---:|---|
| Matched env timestamp | 2,997,447 / 3,200,351 | 93.7% | match |
| Both `Tair_C` and `Si` present | 1,829,607 / 3,200,351 | 57.2% | match |

## Turbulence coverage (`scripts/diagnose_turbulence_coverage.py`)

Current schema: `Wind_U_ms`/`Wind_V_ms`/`Wind_W_ms` + unified `EDR_m23s1`
(14 of 15 campaigns, all but MIDCIX — see CLAUDE.md "Standard output
schema"). All campaigns reproduce the
`docs/reports/2026-07-13-turbulence-scope-reduction-diagnostics.md`
baseline **except one documentation correction found during this
validation**:

- **CRYSTAL-FACE-NASA Wind_U/V/W: 81.5%**, not the 89.1% recorded in that
  report and copied into `docs/reports/2026-07-13-dataset-diagnostics-summary.md`
  and `docs/reports/2026-07-13-turbulence-phase2-3-dataset-report.md`. Root
  cause: those reports' turbulence-coverage table was never regenerated
  after commit `9c02cf0` (same-day, later) applied a missing-value mask to
  CRYSTAL-FACE-NASA's wind columns — the log-path citation was updated to
  point at the post-fix run, but the table's numbers were not. 81.5% is the
  correct, current, reproducible figure (confirmed by two independent runs
  of `diagnose_turbulence_coverage.py` against unchanged code, this session).
  **This is a documentation-accuracy fix, not a dataset change** — the
  parquet's `Wind_U_ms`/`Wind_V_ms`/`Wind_W_ms` values for CRYSTAL-FACE-NASA
  are unchanged; only the historical reports' reported percentage was wrong.
  All three affected report files have been corrected in place with a note
  pointing here.

All other campaigns' Wind_U/V/W and EDR_m23s1 coverage reproduce their
2026-07-13 baseline values exactly (AIRS-II 96.9%, ARM 0%/100%/100%,
ATTREX 43.9%/43.6%, CRYSTAL-FACE-UND 77.7%/93.1%, ESCAPE 0%/100%, ICE-L
~100%, IPHEX 88.4%/99.3%, ISDAC 90.7–98.6%, MACPEX 83.1–83.3%, MC3E
77.7%/93.6%, MIDCIX 0%, MPACE 47.5%/51.2%, OLYMPEX 99.1%/100%, POSIDON
96.1–96.6%/96.1%).

## L0 data availability by campaign (% non-null)

| Campaign | n_rows | Tair_C | P_hPa | Si | qv | Sw | Lat | Lon | Alt_m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AIRS-II | 312,792 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| ARM | 141,940 | 99.0 | 100.0 | 36.4 | 36.4 | 36.4 | 91.7 | 91.7 | 100.0 |
| ATTREX | 1,316,204 | 44.0 | 44.0 | 37.6 | 91.4 | 37.6 | 43.9 | 43.9 | 43.9 |
| CRYSTAL-FACE-NASA | 323,310 | 50.3 | 100.0 | 50.3 | 50.3 | 50.3 | 99.9 | 99.9 | 99.9 |
| CRYSTAL-FACE-UND | 200,864 | 98.3 | 99.5 | 66.2 | 65.9 | 65.9 | 99.9 | 99.9 | 100.0 |
| ESCAPE | 67,380 | 86.9 | 98.4 | 86.5 | 86.4 | 86.5 | 100.0 | 100.0 | 98.4 |
| ICE-L | 210,561 | 100.0 | 100.0 | 99.7 | 99.8 | 99.7 | 100.0 | 100.0 | 99.7 |
| IPHEX | 287,600 | 99.6 | 100.0 | 68.4 | 68.8 | 68.4 | 99.4 | 99.4 | 99.4 |
| ISDAC | 357,071 | 99.9 | 99.4 | 100.0 | 99.3 | 99.9 | 100.0 | 97.5 | 100.0 |
| MACPEX | 307,780 | 90.7 | 90.7 | 63.4 | 63.4 | 63.4 | 90.7 | 90.7 | 90.7 |
| MC3E | 165,431 | 98.6 | 100.0 | 88.2 | 88.2 | 88.2 | 90.7 | 90.7 | 100.0 |
| MIDCIX | 181,459 | 41.8 | 41.8 | 41.8 | 41.8 | 41.8 | 100.0 | 100.0 | 98.2 |
| MPACE | 139,360 | 58.9 | 100.0 | 0.0 | 0.0 | 0.0 | 89.4 | 89.4 | 89.4 |
| OLYMPEX | 209,321 | 100.0 | 100.0 | 58.4 | 58.4 | 58.4 | 100.0 | 100.0 | 100.0 |
| POSIDON | 351,508 | 97.2 | 98.2 | 52.0 | 53.1 | 52.0 | 76.6 | 76.6 | 76.6 |

MPACE Si/qv/Sw = 0% by design (no water-vapor instrument flown). Reproduces
this session's earlier pre-condensation snapshot exactly, and the campaign
summary in `logs/pipeline/latest/campaign_summary.csv`.

## Known issues / open caveats

Unchanged by this validation — see CLAUDE.md's "Known issues / active
investigations" section, which remains the single source of truth: ARM qv
NaN sparsity, CPI/env unmatched images (~6.34%, ISDAC/ARM-dominated),
IPHEX/OLYMPEX cold-regime Si flags (1,436 rows, pending independent TDL
cross-check), MIDCIX Alt_m at 96.7%.

## Conclusion

Repo condensation (deleted `docs/sessions/`, stale `docs/todo/` plan, one
redundant report snapshot, a stale notes file, a scratch script, and
superseded dated/per-campaign parquet files; fixed 4 broken cross-references;
corrected a stale schema description and a stale turbulence-follow-ups list
in `CLAUDE.md`; corrected a stale coverage percentage across 3 historical
reports) made **no changes to the dataset itself**. L0/L1/L2 row counts, all
9 QC checks, and CPI/env fusion rates reproduce their last-known baselines
exactly from unchanged code. The one discrepancy found (CRYSTAL-FACE-NASA
turbulence coverage) was a stale number in historical documentation, not a
data or code issue, and has been corrected at its source.
