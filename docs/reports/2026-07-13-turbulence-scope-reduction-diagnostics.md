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

## Third bug found: CRYSTAL-FACE-NASA wind scale factor never applied

Separate from the EDR unification work, the new `14_wind_uvw_distributions.png`
plot showed CRYSTAL-FACE-NASA's U/V wind KDE badly right-skewed compared to
every other campaign. Root cause, in `parsers/crystal_face_nasa.py::load_mm_met_file`:
the raw ICARTT file's header (`data/raw/CRYSTAL-FACE-NASA/MM/MM20020721.WB57`,
lines 11-18) declares scale factors and missing-value sentinels for **all
six** primary variables (`Psta, Tsta, Thta, U, V, W`):

```
0.1  0.01  0.01  0.1  0.1  0.1     ;scale factors
99999 99999 99999 9999 9999 9999  ;missing values
```

but the parser only ever applied them to `Psta`/`Tsta` — `U`, `V`, `W` were
passed straight through raw: unscaled (10x too large) and with the 9999
missing-value sentinel sitting unmasked in the data (mean 820 m/s, max
exactly 9999.0). Fixed by extending the existing scale/missing-value loop
to all 6 columns. After the fix: `Wind_U_ms` min/median/max for
CRYSTAL-FACE-NASA go from (-294, -37, 9999) to (-29.4, -4.6, 35.3) —
physically sane aircraft-scale wind. This also fully explains QC4's prior
73,566-flag count: it was *entirely* this one unmasked sentinel (QC4 now
reads 0 flags, 0 campaigns — see below).

## Pipeline rebuild

`python main.py --all` — `logs/pipeline/20260713_210614/`

- **15 campaigns, 4,572,581 rows** — unchanged (row count is driven by
  Si/qv/Tair coverage, not the turbulence columns).

## QA checks (`scripts/qa_checks.py`)

`logs/qaqc/20260713_211134/`

| Check | Name | Flags | % of dataset | Campaigns affected |
|---|---|---:|---:|---:|
| QC1 | Physical range checks | 6 | 0.000% | 1 |
| QC2 | Internal consistency | 80,648 | 1.764% | 12 |
| QC3 | Stuck-sensor / temporal continuity | 365 | 0.008% | 6 |
| QC4 | Fill/sentinel value detection | **0** | 0.000% | 0 |
| QC5 | Inter-instrument cross-validation | 0 | 0.000% | 5 |
| QC6 | Per-flight coverage audit | 67 | 0.000% | 10 |
| QC7 | Duplicate/out-of-order timestamps | 2 | 0.000% | 1 |
| QC8 | Vertical profile plausibility | 6 | 0.000% | 3 |
| QC9 | LWC cross-check (severe Si flags) | 1,436 | 0.031% | 2 |

QC4 dropped from the long-standing baseline of 73,566 flags (1 campaign) to
**0** — that entire flag category turned out to be exactly the
CRYSTAL-FACE-NASA wind sentinel bug above, not 12 separate scattered
issues. Every other check is unchanged. (An earlier intermediate run —
before the MMS TEDR fill-flag mask — briefly pushed QC4 to 73,569 flags
across 3 campaigns as the exploded `EDR_m23s1` values collided with
sentinel numbers like 99999; that was already gone before this final run.)

## Turbulence coverage (`scripts/diagnose_turbulence_coverage.py`)

`logs/diagnose_turbulence_coverage/20260713_211257/`

**Correction (found during 2026-08-28 repo validation):** the log-path
above was bumped to the post-fix rerun when this section was last edited
(commit `9c02cf0`), but the CRYSTAL-FACE-NASA row below was not
regenerated from that rerun — it's still the pre-missing-value-mask
number. The correct value, confirmed by rerunning
`diagnose_turbulence_coverage.py` against current code, is **81.5%**, not
89.1% — masking the previously-unmasked `9999` sentinel legitimately
removes some rows that had been miscounted as valid. See
`docs/reports/2026-08-28-dataset-validation.md` for the current figure.

| Campaign | Wind_U/V/W | EDR_m23s1 |
|---|---:|---:|
| AIRS-II | 96.9% | — |
| ARM | 0% U/V, 100% W | 100% |
| ATTREX | 43.9% | 43.6% |
| CRYSTAL-FACE-NASA | ~~89.1%~~ 81.5% (see correction above) | — |
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
assumption. The new per-campaign distribution plots did real work here:
they caught both the ARM unit error and an unrelated, pre-existing
CRYSTAL-FACE-NASA wind-scaling bug that QC4 had been silently flagging at
1.6% of the whole dataset for the entire life of this schema. QC4 is now
clean (0 flags). Safe to commit.
