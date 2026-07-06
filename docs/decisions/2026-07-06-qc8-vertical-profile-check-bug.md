# 2026-07-06 — QC8 vertical profile plausibility check-logic bug

## Fix applied in commit: fad5479 (qa_checks.py check_08_vertical_profiles)

## What we found

QC8's `qv_exceeds_saturation` sub-check flagged 47 pressure bins across 11
campaigns with zero prior investigation. Unlike every other gap investigated
this session, the root cause here was in the **check itself**, not the data
or a parser.

`check_08_vertical_profiles()` computed each bin's saturation reference
(`qv_sat_gkg`) from the **ISA (ICAO standard atmosphere) theoretical
temperature** for that pressure level, using the **ice** saturation formula
(`es_ice_hPa`) — regardless of the bin's actual observed temperature or
whether conditions were above freezing. This is inconsistent with QC2's
row-level `qv_exceeds_saturation` check, which correctly uses each row's own
**observed** `Tair_C` and the **liquid** formula (`es_liq_hPa`).

Two compounding effects made the reference systematically too low at
low-altitude/high-pressure bins:
1. Real near-surface/lower-troposphere air in these mid-latitude and
   tropical/summer campaigns is routinely warmer than the ISA standard
   atmosphere assumes at the same pressure level.
2. Ice saturation vapor pressure is always <= liquid saturation vapor
   pressure at the same temperature (the basis of the Wegener-Bergeron-
   Findeisen process) — using the ice formula for warm, liquid-dominated
   near-surface layers further underestimates the true reference.

Recomputed all 47 flagged bins using each bin's own `T_mean_C` (already
present in the profile output) and `es_liq_hPa` instead: **41 of 47 (87%) no
longer exceed saturation at all.** The magnitude of the artifact was large —
e.g. ESCAPE's 987.5 hPa bin (23,213 records at ~25°C) showed `qv_mean` of
22.7 g/kg against a wrongly-computed `qv_sat` of 11.4 g/kg (a 100% "excess"),
versus a correctly-computed liquid-at-observed-temperature saturation of 31.7
g/kg (well within bounds, no exceedance at all).

Also found ~10 lines of dead code: a `T_isa_ref`/`qv_sat_ref` precomputation
block whose outputs were never referenced — the per-bin loop already
recomputed everything locally. Removed.

## Fix details

`scripts/qa_checks.py`, `check_08_vertical_profiles()`: the saturation
reference for the `qv_exceeds_saturation` sub-check now uses the bin's own
`T_mean` (observed) and `es_liq_hPa` (liquid), matching QC2. The separate
`T_deviation_from_ISA` sub-check, and the `T_isa_C`/`T_dev_C` profile
columns, are unchanged and still correctly compare against the ISA reference
— that check is explicitly about temperature-vs-ISA plausibility, where ISA
is the intended comparison point. The plot-only ISA-based saturation
envelope curve in the QC8b figure is also unchanged (a smooth theoretical
reference curve for visual context is a reasonable, different use case from
a per-bin pass/fail decision).

Before → after:
- QC8 `qv_exceeds_saturation`: 47 bins (11 campaigns) → 6 bins (3 campaigns)
- Remaining 6 bins (MC3E 887.5/775 hPa, IPHEX 887.5/250 hPa, OLYMPEX 350/250
  hPa): small residual mean exceedances (4-25% above the corrected liquid
  reference) in campaigns already in `IN_CLOUD_CAMPAIGNS` — consistent with
  the same expected in-cloud (warm bins) and cirrus ice-supersaturation
  (cold, high-altitude bins) physics already documented in
  `2026-07-05-qc2-severity-tiers.md`. Not further investigated; accepted.

## How to detect

Compare a check's saturation/plausibility reference computation against a
theoretical/model value (ISA temperature, standard atmosphere) rather than
the actual observed data for the same bin/row — a red flag whenever a
mean-aggregated check (bins with thousands of records) shows a much higher
false-positive rate than an equivalent row-level check on the same physical
quantity (QC2 here).

## What remains

None for the check-logic bug itself. The 6 residual bins are genuine,
expected in-cloud/cirrus behavior, consistent with already-accepted
phenomena elsewhere in this project — not tracked as a new open issue.
