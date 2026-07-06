# 2026-07-05 — Resolution of the four open items from the full diagnostic

Follow-up to `docs/decisions/2026-07-05-full-diagnostic.md`. All four listed
open items were investigated and fixed; one turned out to be misattributed
in the original doc, and fixing it surfaced a related gap that's also fixed
here.

## 1. ESCAPE P_hPa < 50 hPa (1,104 rows)

**Root cause:** `parsers/escape.py` derives `P_hPa` from `Palt` via the ICAO
barometric formula when no static-pressure column exists (ESCAPE never has
one). The derived pressure had no plausibility bound, so a stuck/erroneous
Palt reading on the 2022-06-10 flight (21:11–23:29 UTC — the same flight
already known for the temperature-sensor failure) produced P_hPa as low as
28.7 hPa.

**Fix:** Added a `[50, 1100]` hPa plausibility bound on the ICAO-derived
`P_hPa`, matching the bound already applied to direct pressure columns.
Nulls both `P_hPa` and the offending `Alt_m`/`Palt` reading.

**Result:** 0 rows flagged (was 1,104).

## 2. IPHEX/OLYMPEX qv unbounded — and the real source of "qv = 279 g/kg"

**What we found:** The original doc attributed a 279 g/kg qv outlier to
IPHEX. That did not reproduce — current IPHEX qv max was 17.5 g/kg before
any fix. The actual code bug (real, but not yet causing IPHEX/OLYMPEX
damage): `qv_from_e_P` has no upper bound, while `Si` computed from the same
inputs is clipped to `[-1, 5]`, so a row with implausible Si could still
carry unbounded qv. Fixed defensively in both `parsers/iphex.py` (propagate
the `Si_chilled_mirror` clip mask to `qv_chilled_mirror`) and
`parsers/olympex.py` (which had **no** Si bound at all — added one matching
the `[-1, 5]` convention, then propagated it to `qv_frost_point`).

**The actual 278.5 g/kg value** turned out to be in **ESCAPE**, on the same
2022-06-10 flight, and is addressed below.

## 2b. ESCAPE temperature-sensor-failure mask was too narrow

**Root cause:** The existing 2022-06-10 fix (commit `6503c22`) masked
`Tair_C > -20°C at Alt_m > 10,000 m`. The same sensor failure also produced
+15.7°C at ~8.1 km (P_hPa≈348, ~54°C above ISA) — below the 10 km cutoff, so
it wasn't masked, and drove `qv_chilled_mirror` to 278.5 g/kg.

**Fix:** Replaced the fixed altitude cutoff with the same ISA-deviation
formula/tolerance already used by QC2's `T_altitude_inconsistent` check
(`scripts/qa_checks.py`): flag `Tair_C - T_ISA(Alt_m) > 40°C` at any
altitude, not just above 10 km.

**Result:** Dataset-wide qv max dropped from 278.5 → 100.8 g/kg. The
remaining max (ESCAPE, 2022-06-08, Tair=23.3°C at ~2.3 km — a plausible
near-surface reading, not a sensor failure) sits just above QC1's existing
`qv ∈ [0, 100]` hard bound and will surface there for future investigation;
no further parser change made here to avoid over-fitting one data point.

## 3. IPHEX/OLYMPEX severe Si>1.05 flags — LWC cross-check (QC-only)

Per project decision, LWC stays out of the standard schema. Added QC9 to
`scripts/qa_checks.py`, which re-reads raw King/Nevzorov/CDP LWC values for
already-flagged `Si > 1.05` rows and splits them into plausible
in-cloud/precip vs. sensor-error candidates. Output:
`logs/qaqc_<date>/09_lwc_crossval.csv`.

**Finding:** Of 1,519 severe Si>1.05 rows (IPHEX + OLYMPEX), only 31 (2%)
have elevated LWC (> 0.05 g/m³). The other 1,488 have near-zero or missing
LWC — i.e. the "rain contamination" hypothesis in the original diagnostic
doc looks wrong for most of these rows; they look like genuine sensor
errors rather than in-cloud/precip effects. Worth a closer look before any
future masking decision.

## 4. Altitude recovery (4 campaigns with 0% Alt_m)

| Campaign | Outcome |
|---|---|
| CRYSTAL-FACE-NASA | **Fixed.** Loader pointed at `SP/` (SPP-100 particle-probe data, not navigation — always failed silently). Added `load_np_file()` reading `NP/` (WB-57F navigational data, barometric+GPS altitude, all 19 flights). Coverage: 0% → 100.0%. |
| CRYSTAL-FACE-UND | **Fixed.** `ND_NAV/*NAV.CIT` (Applanix POS altitude/lat/lon) was never read. Now merged the same way `MET.CIT` already is. Coverage: 0% → 100.0%. |
| MACPEX | **Fixed** (revised — see below). |
| MIDCIX | **Fixed** (revised — see below). |

### Revision: MACPEX/MIDCIX nav data does exist at ESPO

The initial pass (above) declared MACPEX/MIDCIX altitude an accepted
limitation, since the position files referenced by other instrument
headers weren't present in the local `data/raw/` holdings. The user
pointed to the live ESPO archives
(`https://espoarchive.nasa.gov/archive/browse/{macpex,midcix}/WB57`) and
asked to re-check — the referenced files exist there and are public,
unauthenticated downloads. Downloaded and wired in:

- **MACPEX**: `MMS-FlightPath_WB57_*.ict` (16 files) → `data/raw/MACPEX/MMS-FlightPath/`.
  Standard ICARTT-1001, comma-delimited, columns `P_ALT, LAT, LONG, TAS`
  with scale factors `1.0, 0.001, 0.001, 0.1`. MACPEX's parser
  (`parsers/macpex.py`) already had a generic multi-instrument ICARTT
  reader (`_parse_ict_file` + `_load_and_merge`) that picks up any new
  instrument subfolder automatically; added the missing scale-factor
  application for LAT/LONG (`Lat_deg`/`Lon_deg`, ×0.001) and P_ALT→`Alt_m`
  (×1.0, already metres), plus the three MMS-FlightPath-specific
  missing-value codes (`-99999, -999999, -999`) to `MACPEX_MISSING_FLAGS`.
  Coverage: 0% → **100.0%**.
- **MIDCIX**: `FP*.WB57` (9 files) → `data/raw/MidCix/FP/`. Exact same
  column layout already handled by `load_mms_file()` in
  `parsers/crystal_face_nasa.py`, reused directly. `load_midcix_file` now
  merges the corresponding `FP<date>.WB57` file via `merge_asof` on
  `Timestamp` (nearest, 2 s tolerance). Coverage: 0% → **96.7%** (gaps
  where the FP file's flight-time coverage is narrower than JW's).

**Bug found and fixed along the way:** `load_mms_file()` had a pre-existing
off-by-one indexing bug — `scales`/`missing_vals` cover only the 4
*dependent* variables (`P_ALT, LAT, LONG, TAS`), but the code indexed them
against the 5-element `columns` list (which includes the independent `UT`
variable at index 0), shifting every scale factor by one column. This
silently scaled `P_ALT` by LAT's factor and `LONG` by TAS's factor. It was
never caught before because CRYSTAL-FACE-NASA's `SP/` files (the only
prior caller) always failed at the header-matching step before reaching
this code — this MIDCIX work was the first time `load_mms_file()` actually
executed end-to-end. Fixed by indexing against `columns[1:]` instead.

**Overall Alt_m coverage: 79.2% → 98.2%.**

## Final state (2026-07-05, post-fix)

| Check | Before | After |
|---|---|---|
| ESCAPE P_hPa < 50 hPa | 1,104 rows | 0 |
| Dataset-wide qv max | 279 g/kg (ESCAPE, misattributed to IPHEX) | 100.8 g/kg |
| CRYSTAL-FACE-NASA Alt_m | 0% | 100.0% |
| CRYSTAL-FACE-UND Alt_m | 0% | 100.0% |
| MACPEX Alt_m | 0% | 100.0% |
| MIDCIX Alt_m | 0% | 96.7% |
| Overall Alt_m coverage | 79.2% | 98.2% |

Verified via `python main.py --all`, `python scripts/qa_checks.py` (all 9
checks), and `python scripts/full_diagnostic.py`.
