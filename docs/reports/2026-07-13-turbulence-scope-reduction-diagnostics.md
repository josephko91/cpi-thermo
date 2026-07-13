# Turbulence Schema: Scope Reduction + EDR Unification — Diagnostic Report

**Date:** 2026-07-13

## Summary

Two changes to the turbulence/wind schema, run and validated together:

1. **Scope reduction**: kept only wind components (`Wind_U_ms`, `Wind_V_ms`,
   `Wind_W_ms`) and EDR; dropped `WindSpeed_ms`, `WindDir_deg`, `Roll_deg`,
   `Pitch_deg`, `Heading_deg`, `AngleOfAttack_deg`, `Sideslip_deg`,
   `VertVel_ms`, `Accel_Vert_ms2`, `TAS_ms`, `IAS_ms`, `IAS_ms_nose`,
   `MachNo`, `DriftAngle_deg`, `TrackAngle_deg`, `REYN_mms` from parsers,
   `config.yaml`, `qa_checks.py`, and `diagnose_turbulence_coverage.py`.
2. **EDR unification**: replaced the separate `EDR_mms_log10kWkg` (ATTREX,
   POSIDON) and `EDR_und_cm23s1` (IPHEX, MC3E, MPACE, OLYMPEX,
   CRYSTAL-FACE-UND) columns with a single **`EDR_m23s1`** column in the
   ICAO/WMO-standard unit, eps^(1/3) in m^(2/3)·s⁻¹. `EDR_arm` (ARM) stays
   separate — its raw archive units are unconfirmed and were **not**
   folded into the unified column. Full reasoning:
   `docs/decisions/2026-07-13-edr-unification.md`.

A bug surfaced *by* the unification was fixed before the final run: MMS's
`TEDR` field carries an undocumented fill-flag cluster around
log10(kW/kg) ≈ 12–16.5 (valid readings top out ≈ −3.2) that was invisible
as a flat log-space number but exploded into physically impossible
hundred-thousand-range `EDR_m23s1` values once cubed. Now masked to NaN
pre-conversion in `parsers/attrex.py` and `parsers/posidon.py`.

## Pipeline rebuild

`python main.py --all` — `logs/pipeline/20260713_183828/`

- **15 campaigns, 4,572,581 rows** — unchanged (row count is driven by
  Si/qv/Tair coverage, not the turbulence columns).

## QA checks (`scripts/qa_checks.py`)

`logs/qaqc/20260713_184051/`

| Check | Name | Flags | % of dataset | Campaigns affected |
|---|---|---:|---:|---:|
| QC1 | Physical range checks | 6 | 0.000% | 1 |
| QC2 | Internal consistency | 80,648 | 1.764% | 12 |
| QC3 | Stuck-sensor / temporal continuity | 365 | 0.008% | 6 |
| QC4 | Fill/sentinel value detection | 73,566 | 1.609% | 1 |
| QC5 | Inter-instrument cross-validation | 0 | 0.000% | 5 |
| QC6 | Per-flight coverage audit | 67 | 0.000% | 10 |
| QC7 | Duplicate/out-of-order timestamps | 2 | 0.000% | 1 |
| QC8 | Vertical profile plausibility | 6 | 0.000% | 3 |
| QC9 | LWC cross-check (severe Si flags) | 1,436 | 0.031% | 2 |

Matches the pre-change baseline exactly. (An intermediate run — before the
TEDR fill-flag mask — briefly pushed QC4 to 73,569 flags across 3 campaigns
as the exploded `EDR_m23s1` values collided with sentinel numbers like
99999; that's gone after the mask.)

## Turbulence coverage (`scripts/diagnose_turbulence_coverage.py`)

`logs/diagnose_turbulence_coverage/20260713_184147/`

| Campaign | Wind_U/V/W | EDR_m23s1 | EDR_arm |
|---|---:|---:|---:|
| AIRS-II | 96.9% | — | — |
| ARM | 0% U/V, 100% W | — | 100% |
| ATTREX | 43.9% | 43.6% | — |
| CRYSTAL-FACE-NASA | 89.1% | — | — |
| CRYSTAL-FACE-UND | 77.7% | 93.1% | — |
| ESCAPE | 0% U/V, 100% W | — | — |
| ICE-L | ~100% | — | — |
| IPHEX | 88.4% | 99.3% | — |
| ISDAC | 90.7–98.6% | — | — |
| MACPEX | 83.1–83.3% | — | — |
| MC3E | 77.7% | 93.6% | — |
| MIDCIX | 0% | — | — |
| MPACE | 47.5% | 51.2% | — |
| OLYMPEX | 99.1% | 100% | — |
| POSIDON | 96.1–96.6% | 96.1% | — |

**EDR_m23s1 value ranges by source family** (the key unification sanity
check — do the two independently-converted families land in the same
physical range?):

| Source | Campaigns | Min | Median | Max |
|---|---|---:|---:|---:|
| NASA Ames MMS (from log10 kW/kg) | ATTREX, POSIDON | 0.00001 | 0.006 | 0.66–0.88 |
| UND pipeline (from cm^(2/3)s⁻¹) | IPHEX, MC3E, MPACE, OLYMPEX, CRYSTAL-FACE-UND | 0.003–0.006 | 0.02–0.06 | 0.42–1.21 |

Both families overlap cleanly in the 0–1.2 m^(2/3)·s⁻¹ range, consistent
with ICAO turbulence-severity bands (moderate 0.3–0.5, severe ≥0.5) — no
disjoint sub-range that would indicate a leftover unit-scale error.
`EDR_arm` (ARM, not unified) ranges 0–52 in its raw, unconfirmed units —
excluded from the comparison since it isn't m^(2/3)·s⁻¹.

## Conclusion

Scope reduction is clean. EDR unification is verified by physical-range
overlap between the two converted source families, matching the ICAO
standard EDR band; ARM was deliberately kept out. QA flags match baseline
after fixing the MMS TEDR fill-flag bug the conversion exposed. Safe to
commit.
