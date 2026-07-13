# 2026-07-13 — EDR unification: MMS + UND merged, ARM stays separate

## Context

`docs/decisions/2026-07-13-turbulence-schema.md` originally kept three EDR
columns permanently separate, reasoning that "ARM and the UND pipeline are
both institutionally 'eddy dissipation rate,' but this repo has not verified
a conversion between [them]." That doc's own escape hatch: "a verified,
documented conversion factor... from an instrument-team publication...
would justify changing this."

User requested a single unified EDR column. Research (below) supplied that
verification for two of the three families.

## What was verified

- **EDR is internationally standardized** by ICAO/WMO as eps^(1/3) (cube
  root of turbulent kinetic energy dissipation rate), in **m^(2/3)*s^-1**.
  This is the quantity aviation turbulence-severity thresholds are defined
  in (moderate ~0.3-0.5, severe >=0.5).
- **NASA Ames MMS `TEDR`** (ATTREX, POSIDON): confirmed via the MMS
  instrument team's own `.ict` file header (NASA ESPO archive,
  `MMS-GpsTurb_WB57_*.ict`) to be **log10 of eps in kW/kg** — i.e. the raw
  dissipation rate eps, not its cube root, and in kW not W. This repo's
  existing `EDR_mms_log10kWkg` column was already this exact quantity
  (`MMS_TEDR * 0.01`, the standard MMS integer-scale factor).
- **UND Citation pipeline `TURB`** (IPHEX, MC3E, MPACE, OLYMPEX,
  CRYSTAL-FACE-UND): already eps^(1/3), just in cm^(2/3)*s^-1 instead of
  m^(2/3)*s^-1 — a pure length-unit conversion, not a processing-pipeline
  assumption.
- **ARM `Turbulence_eps`**: no documentation found for the `.t4archive.gz`
  binary format's scale or units (UND's own public data-format page covers
  their ASCII Citation format, not this binary archive). Still unconfirmed.

## Decision

Add `parsers/utils.py::edr_from_mms_log10kWkg` and `edr_from_und_cm23s1`,
both producing eps^(1/3) in m^(2/3)*s^-1:

```
edr_from_mms_log10kWkg(x) = cbrt(1000 * 10**x)     # kW/kg -> W/kg=m^2/s^3, then cube root
edr_from_und_cm23s1(x)    = x / 100**(2/3)          # cm^(2/3) -> m^(2/3)
```

Replace `EDR_mms_log10kWkg` and `EDR_und_cm23s1` with a single
**`EDR_m23s1`** column (ATTREX, POSIDON, IPHEX, MC3E, MPACE, OLYMPEX,
CRYSTAL-FACE-UND).

**`EDR_arm` is kept separate and is NOT part of `EDR_m23s1`.** Its units
are still unverified; folding it in would repeat the exact mistake the
original decision doc was written to prevent. If ARM's format is ever
documented, revisit.

## Verification

`scripts/diagnose_turbulence_coverage.py` now plots `EDR_m23s1` split by
source family (MMS-derived vs. UND-derived) on one histogram — if the two
families didn't actually land in the same physical range after conversion,
this would show as a disjoint sub-range instead of an overlap.

## Related

- Supersedes the "keep three EDR columns forever" stance of
  `docs/decisions/2026-07-13-turbulence-schema.md` for the MMS/UND pair;
  that doc's ARM reasoning still applies unchanged.
