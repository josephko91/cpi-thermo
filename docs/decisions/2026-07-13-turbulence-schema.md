# 2026-07-13 — Turbulence/wind/attitude schema: why EDR is NOT unified

**Superseded in part by `2026-07-13-edr-unification.md` (EDR unification) —
that doc reverses this one's "keep EDR separate forever" conclusion.** This
doc's reasoning for why wind/attitude columns *are* safely unified still
applies and is cited elsewhere; read it for that, not for EDR's final
treatment.

## Question

Phase 1 of the turbulence-measurements implementation plan (2026-07-13, now
shipped — see `docs/dataset-changelog.md`'s 2026-07-13 entry) adds wind,
attitude, angle-of-attack/sideslip, and true-airspeed columns to the L0
output. Physically-equivalent quantities (wind components, roll/pitch/
heading, TAS) are unified into single columns across campaigns once
unit-converted, the same way `Tair_C`/`P_hPa` are already unified across
differently-sensored campaigns. EDR (eddy dissipation rate) is the one
turbulence quantity that is **not** unified this way, even though every
family reports "an EDR." A future contributor might reasonably look at
three separate `EDR_*` columns and be tempted to merge them into one
`EDR` column the same way Si/qv have a ranked "best" column. This doc
records why that would be wrong, so it doesn't happen by "cleanup."

## Decision

Keep three (four, counting the MPACE nose/wing split as one family)
explicitly-tagged, never-merged EDR columns:

- `EDR_mms_log10kWkg` — NASA Ames MMS instrument family (ATTREX, POSIDON),
  **log10 kW/kg**.
- `EDR_und_cm23s1` (+ `EDR_und_cm23s1_nose` for MPACE's nose-boom channel) —
  UND Citation ground-processing pipeline (IPHEX, MC3E, MPACE, OLYMPEX),
  **linear cm^(2/3)·s⁻¹**.
- `EDR_arm` — ARM's binary `.t4archive.gz` archive, units unconfirmed.
- `EDR_isdac_tdreddy` (Phase 3, not yet implemented) — ISDAC's proprietary
  `TDREDDY` coefficient from the unused `wolde-convair` 5 Hz source.

**Do not merge these into one `EDR` column, and do not add a conversion
helper between them.** Unlike wind/attitude/TAS — where, once
unit-converted, m/s is m/s and degrees are degrees regardless of which gust
probe measured them, so `Campaign` alone tells the reader the provenance —
these are different published quantities from different processing
pipelines, not just different units of the same physical measurement. ARM
and the UND pipeline are both institutionally "eddy dissipation rate," but
this repo has not verified a conversion between ARM's `Turbulence_eps` and
the UND pipeline's `TURB`, and fabricating an equivalence between them
would repeat exactly the mistake `docs/decisions/2026-07-07-exact-second-merge-rewrite.md`
already fixed elsewhere in this codebase (silently claiming precision/
equivalence the data doesn't actually support).

## What would justify changing this

A verified, documented conversion factor between two of these EDR products
(e.g. from an instrument-team publication or cross-calibration flight),
confirmed against real overlapping data — not visual similarity of value
ranges. Until then, treat the three-way split as a hard constraint, not a
schema smell.

## Related

- `docs/reports/2026-07-13-turbulence-measurements-survey.md` — the research
  survey that found the per-family variable/unit inventory.
- `docs/decisions/2026-07-07-exact-second-merge-rewrite.md` — prior instance
  of the same "don't manufacture precision the data doesn't have" principle.
