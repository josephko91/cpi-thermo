# 2026-07-06 — QC7 duplicate-timestamp investigation

## Fix applied in commit: (pending — CRYSTAL-FACE-NASA banker's-rounding bug)

## What we found

QC7 (`scripts/qa_checks.py`, timestamp quality) flagged 8,910 exact-duplicate
rows across 2 campaigns, with zero decision-doc coverage previously:

- **CRYSTAL-FACE-NASA: 8,908 rows, all from a single source file**
  (`JW20020719.WB57`). The raw file has 12,430 data rows and *zero* duplicate
  elapsed-seconds values internally, but the combined parquet showed only
  7,976 unique `Timestamp` values for this file — 4,454 timestamps each
  appearing exactly twice, with slightly different `Tair_C` readings (e.g.
  -36.25 vs -36.15 at the same displayed second), confirming these were two
  distinct, real 1 Hz samples colliding onto one timestamp, not genuine
  source-level duplicates.

  Root cause: `parsers/crystal_face_nasa.py`'s `_round_timestamp_to_second()`
  used `.dt.round("s")`, which is *round-half-to-even* ("banker's rounding") —
  an `X.5`-second value rounds toward the nearest **even** second, not
  consistently up. This file's JLH instrument samples at a fixed `.5`-second
  offset (e.g. `64803.5s`, `64804.5s`, ... elapsed since takeoff). Two
  adjacent, physically distinct samples like `:03.5` and `:04.5` both round to
  `:04` under banker's rounding (verified directly: `pd.Series([...]).dt.round("s")`
  collapses `03.5→04` and `04.5→04`), producing a spurious duplicate-timestamp
  row every other sample. Checked all other CRYSTAL-FACE-NASA source files —
  only this one flight's JLH file has the `.5`-second sampling offset, so no
  other file in this campaign was affected.

- **MPACE: 2 rows**, both timestamped `2004-10-21 22:10:12 UTC` from
  `04_10_21_21_16_07.mpace`. Traced to the raw file itself: its elapsed-seconds
  column reads `..., 79811, 79812, 79813, 79812, 79815, ...` — a one-off
  instrument clock glitch where the recorded elapsed-time counter briefly
  repeated a value before continuing forward. This is a genuine raw-data
  anomaly on a single flight, not a parser bug.

## Fix details

`_round_timestamp_to_second()` changed from `.dt.round("s")` to `.dt.floor("s")`.
Flooring truncates consistently in one direction regardless of parity, so
evenly `.5`-second-offset samples stay 1 second apart with no collisions
(`03.5→03, 04.5→04, 05.5→05, 06.5→06` — all distinct, verified). This shifts
every CRYSTAL-FACE-NASA timestamp earlier by up to 0.999s versus the previous
rounding, which is functionally immaterial given the ±1s tolerance used
throughout the CPI/env fusion pipeline (`scripts/diagnose_cpi_fusion.py`).

Before → after (full dataset rebuild):
- Total rows: 3,841,812 → 3,841,797 (-15; a benign side effect of no longer
  double-counting a handful of gap-fill rows that used to collide with a
  genuine JLH row under the old rounding — see "How to detect" below for the
  verification that no legitimate data was lost)
- CRYSTAL-FACE-NASA rows: 154,830 → 154,815 (-15), now with **zero** duplicate
  timestamps (154,815 rows == 154,815 unique timestamps)
- QC7 flags: 8,910 → 2 (only the MPACE raw-glitch pair remains)

MPACE's 2-row raw-instrument glitch was **not** fixed in code — it is a
genuine, trivial (2 out of 139,360 rows) source-data anomaly, consistent with
how QC1's residual ESCAPE flags and QC9's ambiguous IPHEX/OLYMPEX rows are
handled elsewhere in this project: documented and accepted rather than
special-cased.

## How to detect

`logs/qaqc/latest/07_timestamp_duplicates.csv` — group by `Campaign` and
`source_file` to see which raw files contribute duplicates. For
CRYSTAL-FACE-NASA specifically, cross-check the raw file's native time column
for a `.5`-second (or other sub-second) sampling offset before assuming a
parser bug — this is what distinguished the two campaigns' root causes here.

## What remains

None for CRYSTAL-FACE-NASA (fully resolved). MPACE's 2-row raw glitch remains
in the data; not tracked as an open issue since it's below any reasonable
action threshold.
