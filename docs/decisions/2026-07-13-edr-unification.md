# 2026-07-13 — EDR unification: all three families merged into EDR_m23s1

## Context

`docs/decisions/2026-07-13-turbulence-schema.md` originally kept three EDR
columns permanently separate, reasoning that "ARM and the UND pipeline are
both institutionally 'eddy dissipation rate,' but this repo has not verified
a conversion between [them]." That doc's own escape hatch: "a verified,
documented conversion factor... from an instrument-team publication...
would justify changing this."

User requested a single unified EDR column. Research (below) supplied that
verification for all three families — the first pass merged MMS+UND and
left ARM separate pending its own documentation; ARM's format was then
located and confirmed, and folded in too.

## What was verified

- **EDR is internationally standardized** by ICAO/WMO as eps^(1/3) (cube
  root of turbulent kinetic energy dissipation rate), in **m^(2/3)*s^-1**.
  This is the quantity aviation turbulence-severity thresholds are defined
  in (moderate ~0.3-0.5, severe >=0.5).
- **NASA Ames MMS `TEDR`** (ATTREX, POSIDON): confirmed via the MMS
  instrument team's own `.ict` file header (NASA ESPO archive,
  `MMS-GpsTurb_WB57_*.ict`) to be **log10 of eps in kW/kg** — i.e. the raw
  dissipation rate eps, not its cube root, and in kW not W. This repo's
  former `EDR_mms_log10kWkg` column was already this exact quantity
  (`MMS_TEDR * 0.01`, the standard MMS integer-scale factor).
- **UND Citation ASCII pipeline `TURB`** (IPHEX, MC3E, MPACE, OLYMPEX,
  CRYSTAL-FACE-UND): already eps^(1/3), just in cm^(2/3)*s^-1 instead of
  m^(2/3)*s^-1 — a pure length-unit conversion, not a processing-pipeline
  assumption.
- **ARM `Turbulence_eps`**: confirmed via
  `data/raw/ARM/poellot-citation-t4-readme.txt` (the instrument team's own
  data-format README for this file), field 18: `Turbulence — epsilon**1/3`.
  ARM's data comes from the **same UND Citation II aircraft/team** as the
  IPHEX/MC3E/MPACE/OLYMPEX/CRYSTAL-FACE-UND campaigns, just an older binary
  (`.t4archive.gz`) processing generation instead of the later ASCII
  pipeline. Every other variable in that same README is explicit SI
  (meters/sec, millibars, Celsius — no centimeters anywhere), so
  `Turbulence_eps` is eps^(1/3) already in **meters**, not centimeters —
  no length-unit conversion needed beyond the file's generic
  `raw/1000 - 100` integer decode (already applied to all 39 fields).
  Confirmed further by value distribution: ARM's raw values (median 0.57,
  IQR 0.29-1.18) land exactly in the ICAO moderate/severe eps^(1/3) band,
  with no discontinuous sentinel cluster (unlike the MMS TEDR fill-flag bug
  found below).

## Decision

`parsers/utils.py::edr_from_mms_log10kWkg` and `edr_from_und_cm23s1`
produce eps^(1/3) in m^(2/3)*s^-1:

```
edr_from_mms_log10kWkg(x) = cbrt(1000 * 10**x)     # kW/kg -> W/kg=m^2/s^3, then cube root
edr_from_und_cm23s1(x)    = x / 100**(2/3)          # cm^(2/3) -> m^(2/3)
```

ARM needs no conversion function — `Turbulence_eps` is used as-is.

`EDR_mms_log10kWkg`, `EDR_und_cm23s1`, and `EDR_arm` are all replaced by a
single **`EDR_m23s1`** column (ATTREX, POSIDON, IPHEX, MC3E, MPACE,
OLYMPEX, CRYSTAL-FACE-UND, ARM).

## Bug found along the way

Unifying MMS exposed an undocumented `TEDR` fill-flag cluster (raw
log10(kW/kg) in ~12-16.5, valid data tops out ~-3.2) that cubed into
physically impossible EDR values. Masked to NaN pre-conversion in
`parsers/attrex.py` / `parsers/posidon.py`.

## Verification

`scripts/diagnose_turbulence_coverage.py` plots `EDR_m23s1` split by all
three source families on one histogram — if any family didn't actually
land in the same physical range after conversion, it would show as a
disjoint sub-range instead of an overlap.

## Related

- Fully supersedes `docs/decisions/2026-07-13-turbulence-schema.md`'s
  "keep three EDR columns forever" stance — that doc's reasoning was sound
  given the information available at the time (no ARM documentation had
  been located), but the escape hatch it defined ("a verified, documented
  conversion factor... from an instrument-team publication") has now been
  met for all three families.
