# Dataset Diagnostics Summary — 2026-07-13

Snapshot of the combined environmental dataset (L0/L1/L2) after the
turbulence-schema scope reduction + EDR unification
(`ea206e2`, see `docs/decisions/2026-07-13-edr-unification.md`).

## Data tiers

| Tier | Rows | Definition |
|---|---:|---|
| L0 | 4,572,581 | Every whole second any instrument reported anything, 15 campaigns |
| L1 | 2,997,447 | One row per CPI particle image, joined to its exact-second L0 record |
| L2 | 1,828,818 | L1 filtered to rows with all 7 core variables present (`Tair_C, P_hPa, Si, qv, Lat, Lon, Alt_m`) |

L2 retains **61.0%** of L1 and **40.0%** of L0's CPI-matched images.

### Per-campaign funnel

| Campaign | CPI images | L0 rows | L1 rows | L2 rows | % images matched (L1) | % L2 of L1 | L2 EDR_m23s1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| AIRS-II | 92,201 | 312,792 | 92,168 | 92,168 | 99.96% | 100.00% | 0.00% |
| ARM | 295,703 | 141,940 | 230,029 | 64,706 | 77.79% | 28.13% | 100.00% |
| ATTREX | 129,128 | 1,316,204 | 122,050 | 120,595 | 94.52% | 98.81% | 99.84% |
| CRYSTAL-FACE-NASA | 78,152 | 323,310 | 78,151 | 20,441 | 100.00% | 26.16% | 0.00% |
| CRYSTAL-FACE-UND | 1,617,826 | 200,864 | 1,608,674 | 848,940 | 99.43% | 52.77% | 94.46% |
| ESCAPE | 0 | 67,380 | 0 | 0 | — | — | — |
| ICE-L | 46,236 | 210,561 | 46,203 | 46,202 | 99.93% | 100.00% | 0.00% |
| IPHEX | 40,692 | 287,600 | 38,697 | 28,189 | 95.10% | 72.85% | 100.00% |
| ISDAC | 505,812 | 357,071 | 400,805 | 399,668 | 79.24% | 99.72% | 0.00% |
| MACPEX | 80,240 | 307,780 | 80,240 | 51,747 | 100.00% | 64.49% | 0.00% |
| MC3E | 187,558 | 165,431 | 173,766 | 137,272 | 92.65% | 79.00% | 79.82% |
| MIDCIX | 90,761 | 181,459 | 90,667 | 18,890 | 99.90% | 20.83% | 0.00% |
| MPACE | 36,042 | 139,360 | 35,997 | 0 | 99.88% | 0.00% | — |
| OLYMPEX | 0 | 209,321 | 0 | 0 | — | — | — |
| POSIDON | 0 | 351,508 | 0 | 0 | — | — | — |

`EDR_m23s1` is an ungated column (not part of L2's `CORE_COLS` filter —
see CLAUDE.md), so its L2 coverage tracks each campaign's raw EDR
availability, not the L2 selection itself. Campaigns with 0.00% simply
have no EDR source at all (AIRS-II, CRYSTAL-FACE-NASA, ICE-L, ISDAC,
MACPEX, MIDCIX). ARM now shows 100% — it was folded into the unified
column after its instrument-team README confirmed its native units (see
`docs/decisions/2026-07-13-edr-unification.md`).

OLYMPEX, POSIDON, ESCAPE have no CPI imagery in this pipeline's inputs
(L0-only, env/thermodynamic analysis only). MPACE flew no water-vapor
instrument, so Si/qv are NaN for every row — zero L2 rows despite 99.9%
L1 match.

## CPI/env timestamp fusion

`logs/cpi_fusion/latest/` — 3,200,351 CPI images across 12 campaigns.

| Metric | Count | % |
|---|---:|---:|
| Matched env timestamp | 2,997,447 | 93.7% |
| Both `Tair_C` and `Si` present | 1,829,607 | 57.2% |
| All 7 core variables present | 1,828,818 | 57.1% |

Weakest campaigns: CRYSTAL-FACE-NASA and MIDCIX (~21–26% qv coverage —
missing water-vapor instrument on many flight dates), MPACE (0% — no
water-vapor instrument at all), ARM (21.9% — dry upper-troposphere qv
sparsity, not a parser bug).

## QA checks (`scripts/qa_checks.py`)

`logs/qaqc/latest/`

| Check | Name | Flags | % of dataset | Campaigns affected |
|---|---|---:|---:|---:|
| QC1 | Physical range checks | 6 | 0.000% | 1 |
| QC2 | Internal consistency | 80,648 | 1.764% | 12 |
| QC3 | Stuck-sensor / temporal continuity | 365 | 0.008% | 6 |
| QC4 | Fill/sentinel value detection | 0 | 0.000% | 0 |
| QC5 | Inter-instrument cross-validation | 0 | 0.000% | 5 |
| QC6 | Per-flight coverage audit | 67 | 0.000% | 10 |
| QC7 | Duplicate/out-of-order timestamps | 2 | 0.000% | 1 |
| QC8 | Vertical profile plausibility | 6 | 0.000% | 3 |
| QC9 | LWC cross-check (severe Si flags) | 1,436 | 0.031% | 2 |

QC4 dropped to 0 after fixing a CRYSTAL-FACE-NASA wind-scaling bug (its
ICARTT file's 0.1 scale factor and 9999 missing-value sentinel were never
applied to `Wind_U_ms`/`Wind_V_ms`/`Wind_W_ms` — see
`docs/reports/2026-07-13-turbulence-scope-reduction-diagnostics.md`). QC2's
80,648 flags are 74.2% mild (in-cloud qv exceeding saturation by up
to 1.05×, expected instrument physics) vs. 25.8% severe. QC9's 1,436 severe
Si flags (IPHEX/OLYMPEX cold-regime) remain flagged-not-masked pending an
independent TDL cross-check — see CLAUDE.md "Known issues."

## Turbulence coverage (`scripts/diagnose_turbulence_coverage.py`)

`logs/diagnose_turbulence_coverage/latest/` — schema now wind U/V/W + a
single unified `EDR_m23s1` (MMS, UND-ASCII, and ARM/UND-binary all merged;
see `2026-07-13-turbulence-scope-reduction-diagnostics.md` for the
unification work and the source-family overlap check).

| Campaign | Wind_U/V/W | EDR_m23s1 |
|---|---:|---:|
| AIRS-II | 96.9% | — |
| ARM | 0% U/V, 100% W | 100% |
| ATTREX | 43.9% | 43.6% |
| CRYSTAL-FACE-NASA | 89.1% | — |
| CRYSTAL-FACE-UND | 77.7% | 93.1% |
| ESCAPE | 0% U/V, 100% W | — |
| ICE-L | ~100% | — |
| IPHEX | 88.4% | 99.3% |
| ISDAC | 90.7–98.6% | — |
| MACPEX | 83.1–83.3% | — |
| MC3E | 77.7% | 93.6% |
| MIDCIX | 0% | — |
| MPACE | 47.5% | 51.2% |
| OLYMPEX | 99.1% | 100% |
| POSIDON | 96.1–96.6% | 96.1% |

## Si / qv instrument coverage (from pipeline run)

| Metric | Mean | Median | % valid |
|---|---:|---:|---:|
| Si | −0.275 | −0.210 | 59.0% |
| qv | 1.372 g/kg | 0.060 g/kg | 74.5% |

Best-covered campaigns: AIRS-II, ISDAC, ICE-L (~100% Si/qv). Weakest:
MPACE (0% Si/qv, no water-vapor instrument), MIDCIX (42%), CRYSTAL-FACE-NASA
(50%).

## Open items (unchanged from CLAUDE.md)

- ARM qv 63.6% NaN — real dry-upper-troposphere sparsity, not a bug.
- 6.34% of CPI images (202,904 / 3,200,351) unmatched to env data —
  physical camera-power-on-vs-env-recording gaps, dominated by ISDAC and
  ARM; documented in `docs/decisions/2026-07-07-cpi-env-unmatched-images-investigation.md`.
- IPHEX/OLYMPEX cold-regime Si flags (1,436 rows) — flagged, not masked,
  pending independent TDL cross-check.
- MIDCIX Alt_m at 96.7% — navigation files cover 2 more flight dates than
  the water-vapor files rows are keyed off.
