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
2. **EDR unification**: replaced `EDR_mms_log10kWkg` (ATTREX, POSIDON),
   `EDR_und_cm23s1` (IPHEX, MC3E, MPACE, OLYMPEX, CRYSTAL-FACE-UND), and
   `EDR_arm` (ARM) with a single **`EDR_m23s1`** column in the ICAO/WMO
   standard unit, eps^(1/3) in m^(2/3)·s⁻¹. ARM was initially left out
   pending unit confirmation, then folded in via
   `data/raw/ARM/poellot-citation-t4-readme.txt` (field 18:
   `Turbulence — epsilon**1/3`) — ARM flew the same UND Citation II
   aircraft/team as the other UND-sourced campaigns, just an older binary
   archive. Full reasoning: `docs/decisions/2026-07-13-edr-unification.md`.

Two bugs surfaced *by* the unification, both caught by post-hoc plots/checks
rather than assumed correct on the first pass:

- MMS's `TEDR` field carries an undocumented fill-flag cluster around
  log10(kW/kg) ≈ 12–16.5 (valid readings top out ≈ −3.2) that was invisible
  as a flat log-space number but exploded into physically impossible
  hundred-thousand-range `EDR_m23s1` values once cubed. Masked to NaN
  pre-conversion in `parsers/attrex.py` and `parsers/posidon.py`.
- ARM's README labels field 18's *quantity* (`epsilon**1/3`) but never
  states its *length unit*, unlike every other row (explicit m/s, mb, °C).
  First pass assumed meters (no "cm" anywhere in the README) and used the
  raw value as-is; a distribution plot
  (`figs/all-campaigns/*/13_edr_distributions.png`) showed ARM wildly
  skewed high vs. every other campaign — median 0.57 m^(2/3)·s⁻¹ implied
  half of all ARM records sat at/above ICAO's *severe* turbulence
  threshold, physically implausible as a routine flight condition.
  Applying the same cm→m conversion used for the later UND ASCII pipeline
  (ARM is the same aircraft/team, just an older archive with the same
  house cm convention) dropped the median to 0.027 and max from an
  impossible 51.9 to a plausible 2.4 — landing squarely inside the other
  UND campaigns' range. Fixed in `parsers/arm.py`.

## Pipeline rebuild

`python main.py --all` — `logs/pipeline/20260713_205407/`

- **15 campaigns, 4,572,581 rows** — unchanged (row count is driven by
  Si/qv/Tair coverage, not the turbulence columns).

## QA checks (`scripts/qa_checks.py`)

`logs/qaqc/20260713_205854/`

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

`logs/diagnose_turbulence_coverage/20260713_210050/`

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

**EDR_m23s1 value ranges by source family** (the key unification sanity
check — do all three independently-converted families land in the same
physical range?):

| Source | Campaigns | Min | Median | Max |
|---|---|---:|---:|---:|
| NASA Ames MMS (from log10 kW/kg) | ATTREX, POSIDON | 0.00001 | 0.006 | 0.88 |
| UND ASCII pipeline (from cm^(2/3)s⁻¹) | IPHEX, MC3E, MPACE, OLYMPEX, CRYSTAL-FACE-UND | 0.0003 | 0.037 | 1.21 |
| ARM / UND Citation binary (from cm^(2/3)s⁻¹, same conversion) | ARM | 0.0 | 0.027 | 2.41 |

All three families now overlap cleanly in the 0–2.4 m^(2/3)·s⁻¹ range,
consistent with ICAO turbulence-severity bands (moderate 0.3–0.5, severe
≥0.5) — most of every campaign sits well below the moderate threshold
(routine flight), with tails extending into moderate/severe territory
(storm penetrations). ARM's max (2.4) is the highest of the three, plausible
for its Spring 2000 IOP's intentional storm penetrations, and its median
(0.027) now sits with the other UND-family campaigns rather than standing
alone. See `docs/decisions/2026-07-13-edr-unification.md` for the earlier,
wrong assumption (ARM in native meters) this replaced and how the skewed
`13_edr_distributions.png` plot caught it.

## Conclusion

Scope reduction is clean. EDR unification is verified by physical-range
overlap across all three converted source families, matching the ICAO
standard EDR band — including a correction after the fact once ARM's
value-range turned out implausible under the initial (wrong) unit
assumption. QA flags match baseline after fixing the MMS TEDR fill-flag
bug the conversion exposed. Safe to commit.
