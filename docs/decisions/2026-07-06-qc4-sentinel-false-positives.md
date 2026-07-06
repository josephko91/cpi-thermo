# 2026-07-06 — QC4 sentinel value check-logic bug

## Fix applied in commit: (pending — qa_checks.py check_04_sentinel_values)

## What we found

QC4 flagged 682 rows across 34 (campaign, variable, sentinel) combinations
in 15 campaigns, with zero prior investigation. As with QC8
(`2026-07-06-qc8-vertical-profile-check-bug.md`), the root cause was in the
**check itself**: `1000.0` and `9999.0` (plus `-1000.0`) are physically
plausible values for `Alt_m` (meters), `P_hPa`, and `*_ppmv` concentration
columns, which routinely pass through these magnitudes as real data:

- `Alt_m` crosses 1000 m or 9999 m during any climb/descent through that
  altitude.
- `P_hPa` is commonly close to 1000 hPa near sea level.
- Water-vapor mixing ratios of 1000-10000 ppmv are unremarkable in the
  lower troposphere.

Every one of the 34 flagged combinations was checked directly against the
combined parquet:

- Most (e.g. `CRYSTAL-FACE-UND Alt_m` near 9999, `ICE-L MRTDLL_MC_ppmv` near
  1000, `IPHEX MixingRatio_ppmv` near 9999) showed many distinct floating-
  point values spanning the tolerance band (16-175 unique values for
  16-175 matches) — unambiguously real, continuously-varying physical data,
  not a repeated fill code.
- The largest single hit, `ARM Alt_m == 1000.0` (256 matches, all exactly
  `1000.0`), looked suspicious at first (zero variance) but was confirmed
  real by reading the raw GPS stream directly: `GPS_Alt_m` is whole-meter-
  quantized (a 1 Hz GPS update logged onto a 4 Hz file), so a genuine,
  smoothly-descending aircraft (`...1001, 1001, 1001, 1000, 1000, 1000,
  999, 999...`) produces short, separate multi-sample bursts of the exact
  integer `1000` at many different points in the flight — not a single
  stuck fill value.
- No sentinel match anywhere in the dataset used any of the other 8
  (unambiguous) codes in `SENTINELS` (`-9999`, `-9999.99`, `-999`,
  `-999.99`, `-8888`, `-7777`, `99999`) — only `1000.0`/`9999.0` produced
  false positives, and they produced *only* false positives.

## Fix details

`scripts/qa_checks.py`, `check_04_sentinel_values()`: added a
`_sentinel_applies(col, sent)` guard that skips checking `1000.0`,
`-1000.0`, and `9999.0` specifically against `Alt_m`, `P_hPa`, and any
column ending in `_ppmv` — these three (magnitude, variable-type)
combinations are where the ambiguity lives. All other sentinel/variable
combinations are unchanged (e.g. `1000.0`/`9999.0` are still checked against
`Tair_C`, `Si`, `qv`, where they would be unambiguous). Also fixed a latent
edge case this exposed: `sentinel_df = pd.DataFrame(sentinel_rows)` produced
a zero-column DataFrame when the list was empty, crashing the QC4b plot's
`sentinel_df["variable"]` lookup — now constructed with an explicit column
list so an empty result renders cleanly.

Before → after: QC4 flags 682 rows (15 campaigns, 34 combinations) → 0.

## How to detect

For a "sentinel exceeds tolerance" check on a continuous physical variable:
pull the actual matched values and check how many *distinct* values are
present. Real data passing through the sentinel's neighborhood produces many
distinct values (or, for a quantized/rounded source like whole-meter GPS
altitude, several separate short bursts of a repeated integer at different
times) — a genuine un-converted fill code produces the exact same value
every time it appears, typically as one contiguous run or a small, fixed set
of encoded values.

## What remains

None. This check-logic fix required no parser changes — the underlying data
was correct throughout; only the diagnostic was miscalibrated for these
three (magnitude, variable) combinations.
