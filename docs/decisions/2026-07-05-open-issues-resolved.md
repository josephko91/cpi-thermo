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
| MACPEX | **Confirmed dead end**, not a parser bug. Instrument headers reference `MMS-FlightPath_*.ict`/`MMS-GpsTurb_*.ict`/`MMS-Attitude_*.ict` for position, but none of those files exist anywhere in `data/raw/MACPEX/` — never downloaded. One instrument header (HWV) claims "Lat, Lon, Alt included in the data records," but the actual columns are only `Time_UTC, H2O, +Unc, -Unc` — boilerplate text, verified false. Double-checked directly against raw files per user request. Recovery requires re-acquiring files from NASA ESPO. |
| MIDCIX | **Confirmed dead end.** Raw archive has exactly 7 `JW*.WB57` files with no lat/lon/alt columns and no other files present. Same as MACPEX. |

Per user decision, no synthetic ICAO-inverse altitude was added for
MACPEX/MIDCIX — `Alt_m` stays NaN, documented as an accepted limitation
(same treatment as ARM's qv NaN).

**Overall Alt_m coverage: 79.2% → 88.5%.**

## Final state (2026-07-05, post-fix)

| Check | Before | After |
|---|---|---|
| ESCAPE P_hPa < 50 hPa | 1,104 rows | 0 |
| Dataset-wide qv max | 279 g/kg (ESCAPE, misattributed to IPHEX) | 100.8 g/kg |
| CRYSTAL-FACE-NASA Alt_m | 0% | 100.0% |
| CRYSTAL-FACE-UND Alt_m | 0% | 100.0% |
| MACPEX / MIDCIX Alt_m | 0% | 0% (confirmed limitation, documented) |
| Overall Alt_m coverage | 79.2% | 88.5% |

Verified via `python main.py --all`, `python scripts/qa_checks.py` (all 9
checks), and `python scripts/full_diagnostic.py`.
