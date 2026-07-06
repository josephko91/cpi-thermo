# 2026-07-06 — Working through the CPI fusion report's remaining issues

Went through all 8 items in `logs/diagnostics/cpi_fusion_report.txt`'s
"REMAINING KNOWN ISSUES" section to see which could actually be resolved.

## Fixed

**#7 — MIDCIX timestamp match (~26% → ~65%).** JLH (water vapor) files
genuinely don't exist for 2004-04-22 and 2004-04-27 (confirmed — no fix
possible for Tair_C/Si/qv on those dates), but `FP/` (navigation) files
*do* exist for both. `load_midcix()` previously only emitted rows keyed to
JW file timestamps, so that position data was silently dropped entirely.
Added a fallback pass (mirroring the HW/ALIAS fallback pattern in
`crystal_face_nasa.py`): for any `FP*.WB57` file whose date isn't already
covered by a JW file, add position-only rows (`Lat`/`Lon`/`Alt_m`, all
other fields NaN). Recovers 42,178 rows; MIDCIX CPI timestamp match rate
64.6% (was 25.9%); Tair_C/Si/qv correctly stay at 25.9% (genuinely
unavailable). Overall CPI fusion match rate: 89.1% → 90.2%.

**#4 — stale parquet.** `data/raw/combined_env_si_airtemp_01.parquet`
(18.7 MB, dated 2026-01-16) used the old `Tair` column name and covered
only 6 campaigns — a footgun for anything that referenced it by accident.
Grepped the repo first to confirm nothing depends on it (only the
diagnostic report's own text mentioned it). Deleted, per user confirmation
since it predates this session.

## Investigated, confirmed not fixable

**#5 — CRYSTAL_FACE_UND's 3 dead-RH days (07/07, 07/09, 07/11).** Checked
every other instrument folder in the raw archive (`PD`, `CR`, `EC`, `CV`,
`IN`, `RE`, `CP`, plus the `ND_*` set) for an alternative humidity/
supersaturation source. Found one — `IN/` (ice nuclei counter) has direct
`SSw`/`SSi` (supersaturation w.r.t. water/ice) columns — but it only has
files for 07/18, 19, 21, 23, 25, 28, 29; **not** for the 3 dead-RH dates.
Confirmed: no alternative water-vapor source exists for those 3 dates in
this archive. (Note: `IN`'s `SSw`/`SSi` is a genuinely unused, independent
Si source for the *other* 7 dates it does cover — worth a future look as a
cross-validation source, separate from this specific gap.)

## Left as-is (no action needed)

- **#1** campaign name mismatch — already handled centrally in
  `parsers/cpi_timestamps.py`.
- **#2** MPACE has no env parser — confirmed no raw MPACE data exists
  anywhere in this repo; would need a from-scratch parser built from data
  we don't have. Out of scope without new data acquisition.
- **#3** OLYMPEX/ESCAPE have env data but no CPI images — likely
  intentional (CPI wasn't deployed on those platforms), not an issue.
- **#6** MC3E timezone — already resolved (2026-07-05 session).
- **#8** CRYSTAL_FACE_NASA's ~44% ceiling — root cause (JLH/HW/ALIAS gaps
  anti-correlate with peak CPI activity) already investigated and
  partially addressed (ALIAS fallback, 2026-07-05). Closing it further
  needs relaxing the merge tolerance beyond ±1s or higher-cadence raw data
  — both are trade-offs, not bugs, and not made unilaterally here.

## Verification

`python main.py --all` → `python scripts/qa_checks.py` (all 9 checks, no
new flags beyond the expected +2 QC6 flight-days-with-zero-Si, matching the
2 new MIDCIX position-only dates) → `python scripts/diagnose_cpi_fusion.py`.
