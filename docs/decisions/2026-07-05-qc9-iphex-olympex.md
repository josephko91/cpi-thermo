# 2026-07-05 — POSIDON pressure sentinel fix + IPHEX/OLYMPEX QC9 flag resolution

Follow-up to `docs/decisions/2026-07-05-open-issues-resolved.md`, which left two
items open in its final summary. Both investigated and resolved here.

## 1. POSIDON P_hPa ≈ -999.99 (379 rows)

**Root cause:** `parsers/posidon.py:64-68` (`_MISSING_FLAGS`) didn't include
`-99999.0`, the documented ICARTT fill code for the `P` (pressure), `U`, `V`, `W`,
and `PALT` variables in the raw MMS-1HZ files — confirmed identical across all 18
POSIDON MMS `.ict` files. The unmasked `-99999` was scaled by `COEF_PRESSURE = 0.01`
into exactly `-999.99`. Also missing: `-999.0` (REYN), `-9999999.0` (LAT),
`-99999999.0` (LONG) — same bug class, not yet observed causing visible damage
downstream but latent.

Isolated to two flight dates (2016-10-28: 199 rows, 2016-10-14: 180 rows) out of 11
POSIDON flights — a genuine sensor/transmission dropout on those two flights, not a
campaign-wide issue. `Tair_C`/`Alt_m`/`Si` were already NaN on these rows via other
already-covered sentinels or downstream bound checks; `qv` survived because it's
DLH-derived (doesn't depend on pressure).

**Fix:** Added all 4 missing sentinel codes to `_MISSING_FLAGS`
(`parsers/posidon.py`).

**Result:** POSIDON `P_hPa` rows `< -900`: 379 → 0. POSIDON `P_hPa` coverage: 99.1%.
`Lat`/`Lon` ranges post-fix are physically sane (-1.4°N to 21.0°N, 130.7°E to
167.7°E — consistent with the Guam-based western Pacific POSIDON mission).

## 2. IPHEX/OLYMPEX QC9 sensor-error-candidate flags (1,488 rows)

**Diagnosis:** the 1,488 flags concentrate in exactly 4 flights, with two distinct
mechanisms:

| Flight | Rows | Mechanism |
|---|---|---|
| IPHEX `2014_06_13_18_51_49.iphex` | 1,248 | Extreme-cold cruise leg (Air_Temp median -55.2°C). Chilled-mirror hysteresis is known to amplify small errors into large fractional Si swings via the Clausius-Clapeyron exponential at these temperatures — but Si up to ~1.5 is *also* physically documented for real cirrus near the homogeneous-freezing threshold. **Genuinely ambiguous.** |
| IPHEX `2014_05_19_06_07_13.iphex` | 143 | Same mechanism, milder cold (-21 to -29°C). Same ambiguity. |
| OLYMPEX `15_12_13_19_51_41.olympex` | 83 | FrostPoint exceeds Air_Temp by 8.9–11.6°C at Air_Temp = -0.2 to -2.3°C — physically implausible (would require e > ~2× ice-saturation vapor pressure at near-0°C, far beyond what a liquid cloud at that temperature can produce even accounting for the liquid/ice saturation-ratio difference). **Clear mirror-fault signature.** |
| OLYMPEX `15_12_13_15_39_28.olympex` | 45 | FrostPoint exceeds Air_Temp by 6.3–6.7°C at Air_Temp ≈ -44°C — same ambiguous class as the IPHEX cases. |

**Decisions (user-confirmed):**
- IPHEX ambiguous cold-regime flights → flag only, do not mask. Real resolution
  needs an independent cross-check (e.g. Ophir TDL for that flight) not available
  this session.
- OLYMPEX `19_51_41` → mask. `FrostPoint > Air_Temp` at non-cirrus temperature is a
  physical impossibility, not a severity judgment call.

**Fix:**
1. `parsers/olympex.py`: added `(FrostPoint - Air_Temp > 5.0) & (Air_Temp > -10.0)`
   → null `Si_frost_point` (propagates to `Si`, `Sw`, `qv_frost_point`, `qv` via the
   existing NaN-propagation fix from earlier today). Verified this threshold cleanly
   separates the two OLYMPEX flights: `19_51_41` (Air_Temp -0.2 to -2.3°C) triggers
   it; `15_39_28` (Air_Temp -43.5 to -44.0°C) does not.
2. `scripts/qa_checks.py`: QC9 now merges `Tair_C` and adds a
   `cold_regime_amplification_candidate` boolean column
   (`likely_cause == "sensor_error_candidate" & Tair_C < -10°C`) — visible in
   `09_lwc_crossval.csv`, flag-only, no masking.

**Side finding:** the OLYMPEX `19_51_41` mask also removed 31 rows that the original
LWC-threshold heuristic had classified as "plausible in-cloud/precip" (elevated
LWC). On reflection this is a *correction*, not collateral damage: at Air_Temp -0.2
to -2.3°C, even a genuine liquid cloud cannot produce ice-supersaturation (Si) above
~1.05 by a wide margin — the liquid/ice saturation-vapor-pressure ratio at that
temperature is only ~1.1×, nowhere near the >2× required to explain the observed
values. The physically-grounded FrostPoint-vs-Air_Temp criterion is more
discriminating here than the LWC-threshold heuristic: it's more likely that rain
simultaneously contaminated both the chilled-mirror and the LWC probe on this flight
segment than that this was a real, sustained, order-of-magnitude ice-supersaturation
event at near-0°C.

**Result:** OLYMPEX Si>1.05 rows: 128 → 45 (only the ambiguous, unmasked flight
remains). OLYMPEX qv coverage: 58.6% → 58.4% (marginal, as expected for 83 rows out
of 209,321). QC9 now reports 1,436 severe rows total (IPHEX 1,391 unchanged +
OLYMPEX 45), all still tagged `sensor_error_candidate`, and all 1,436 now also
tagged `cold_regime_amplification_candidate` — i.e. everything remaining in QC9's
severe-flag output is, by construction, the flag-only/ambiguous bucket; the
unambiguous fault case has already been removed from the shipped data.

## Verification

`python main.py --all` → `python scripts/qa_checks.py` (all 9 checks). QC1 POSIDON
flags: 379 → 0. QC1 overall: down to 6 residual flags (pre-existing ESCAPE noise,
unrelated to this session). OLYMPEX `19_51_41` Si/qv confirmed NaN; `15_39_28` and
both IPHEX flights confirmed unchanged in the parquet but flagged in QC9's new
column.
