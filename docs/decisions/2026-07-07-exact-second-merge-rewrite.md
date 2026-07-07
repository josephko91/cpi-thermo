# 2026-07-07 — Exact-second merge rewrite + L0/L1/L2 data tiers

## Question

GitHub issue #10 (QC3 stuck-sensor runs) traced repeated values in
CRYSTAL-FACE-NASA/ESCAPE/ISDAC/MIDCIX to `pd.merge_asof(direction=
"nearest", tolerance=...)` calls combining instruments at different native
sample rates. A closer audit (issue #12) found the tolerances used across
several parsers were wider than the reference source's real update
interval (e.g. Harvard Water Vapor documented at 10s cadence merged with a
30s tolerance; MM/NM meteorology merged with a 60s tolerance for the final
`P_hPa` assembly) -- meaning a slow source's stale value was being reused
across many rows of a denser timeline, silently claiming higher time
resolution than the data actually has.

**Decision**: stop using merge tolerance entirely, repo-wide. Round every
instrument's own timestamp to the nearest second (floor), then join by
exact second. If an instrument has no reading at a given second, that cell
is NaN -- no nearest-neighbor reuse of a value from a different second,
ever. On top of this, build three explicit output tiers.

## What changed

**New shared helper**: `parsers/utils.py::round_timestamp_to_second()` --
a verbatim, renamed move of `crystal_face_nasa.py`'s pre-existing
`_round_timestamp_to_second` (floor-based, not `.dt.round()`, to avoid
banker's-rounding collisions between adjacent 1Hz samples at a fixed .5s
offset). All rewritten parsers now import this shared version.

**Per-parser merge rewrite** (every `merge_asof(direction="nearest",
tolerance=...)` replaced with an exact-key `pd.merge(..., how="outer")`,
so the row grid becomes the union of every instrument's own timestamps
rather than being anchored to one "primary" instrument):

- `parsers/crystal_face_nasa.py` -- 16 call sites across T/P consolidation
  (MM/NM), HW/ALIAS Si computation, position (mms_geo), and final JLH
  assembly. The ~150-line "gap-fill" block that manually re-derived "is
  there a nearby JLH timestamp" to decide whether to append HW/ALIAS-only
  rows was deleted entirely -- an outer merge makes it structurally
  unnecessary (gap rows are automatically present with JLH columns NaN).
- `parsers/midcix.py` -- JW+FP position merge (P_hPa/Tair_C come pre-baked
  into the raw JW file itself from the original archive, untouched by
  this change -- see "Out of scope" below).
- `parsers/macpex.py`, `parsers/attrex.py`, `parsers/posidon.py` -- their
  `time_tolerance` parameter (previously threaded through from
  `load_*()` down to the merge call, default `"1s"`) is removed entirely;
  the multi-instrument merge is now a plain exact-second outer join.
- `parsers/crystal_face_und.py` -- already used exact-key `pd.merge` (not
  `merge_asof`); changed `how="left"` to `how="outer"` for grid
  consistency with everything else.
- `scripts/diagnose_cpi_fusion.py` -- `merge_campaign()`'s
  `MATCH_TOLERANCE_S = 1` / `merge_asof` replaced with an exact-key merge
  on floored timestamps, consistent with the rest of the pipeline.

**New `scripts/build_data_tiers.py`** derives L1/L2 from the (now
corrected) L0 combined parquet:

- **L0** = `data/out/combined_env_data.parquet` (no rename) -- every whole
  second where any instrument in a campaign reported anything.
- **L1** = `data/out/combined_env_data_L1.parquet` -- L0 filtered to only
  seconds with a CPI image for that campaign (exact match against
  `parsers/cpi_timestamps.py`, joined per campaign to avoid cross-campaign
  timestamp collisions).
- **L2** = `data/out/combined_env_data_L2.parquet` -- L1 filtered to rows
  with every core variable present (`Tair_C, P_hPa, Si, qv, Lat, Lon,
  Alt_m`; `Sw` excluded as derived, not independent).

## Results

Verified via `scripts/qa_checks.py` (QC3) and `logs/pipeline/<ts>/
campaign_summary.csv` before/after every step (baseline captured before
any change).

**Untouched campaigns (ARM, AIRS-II, ESCAPE, ICE-L, IPHEX, ISDAC, MC3E,
MPACE, OLYMPEX) show exactly zero diff** in row count or Si mean/std --
confirms the rewrite is properly scoped.

**Touched campaigns** (row count grows because the union grid now
includes seconds where only a secondary instrument reported; coverage %
drops because it's no longer fabricated by tolerance-based reuse):

| Campaign | Rows before -> after | Tair_C valid | Si valid |
|---|---|---|---|
| CRYSTAL-FACE-NASA | 154,815 -> 323,310 | 100.0% -> 50.3% | 100.0% -> 50.3% |
| MIDCIX | 118,105 -> 181,459 | 64.3% -> 41.8% | 64.3% -> 41.8% |
| MACPEX | 279,073 -> 307,780 | 100.0% -> 90.7% | 69.9% -> 63.4% |
| ATTREX | 581,370 -> 1,316,204 | 99.7% -> 44.0% | 86.3% -> 37.6% |
| POSIDON | 190,418 -> 351,508 | 97.9% -> 97.2% | 96.0% -> 52.0% |
| CRYSTAL-FACE-UND | 200,740 -> 200,864 | 98.4% -> 98.3% | 66.2% -> 66.2% |

In every case, Si mean/std for the actually-computed values are unchanged
(e.g. CRYSTAL-FACE-NASA Si -0.2386/0.4591 in both versions) -- confirming
the Si/qv computation itself was never wrong, only which rows got a
(sometimes fabricated) value.

**Two genuine pre-existing bugs surfaced as a side effect** of switching
from an anchor-based `merge_asof` to a proper union outer-join:

- **ATTREX**: `_combine_ict_files` always started `merged` as
  `instrument_dfs["MMS"].copy()` when MMS was present, and `merge_asof`
  can only add columns to that row index, never new rows. MMS files only
  exist for the 2013-2014 deployment (10 files); DLH-H2O has 17 files
  going back to 2011. The entire 2011 deployment (and any other
  MMS-absent date) was silently dropped from every prior run. Date range
  recovered: 2013-12-20..2014-03-15 -> 2011-10-28..2014-03-15.
- **POSIDON**: `_combine_ict_files` started `combined` as
  `merged_instruments[inst_names[0]]` (alphabetically first instrument),
  with the same structural limitation. Date range recovered:
  2016-10-12..2016-10-31 -> 2016-09-13..2016-11-02.

**QC3 stuck-sensor check**: identical before/after for every
campaign/variable except CRYSTAL-FACE-NASA `P_hPa` (22 -> 78 runs).
Verified directly against the raw MM meteorology file (`MM20020514.WB57`):
pressure genuinely holds `192.4 mb` for 84 consecutive real 1-second
samples while temperature and wind columns keep changing every row --
0.1 mb sensor precision during stable-altitude cruise flight, not a merge
artifact. This is the same category of finding as the CRYSTAL-FACE-NASA
`Tair_C` / MIDCIX `P_hPa` cases already documented as raw-archive-baked
(see below) -- more of this genuine, previously-hidden precision-limited
data is now visible because `P_hPa` used to only reach the ~46% of rows
anchored to JLH; now it reaches the full MM/NM union.

**CPI fusion coverage** (issue #9): re-derived via
`scripts/diagnose_cpi_fusion.py` with exact-second matching. Position
(Lat/Lon/Alt_m) coverage is unaffected (100% for CRYSTAL-FACE-NASA, still
matched independently via MMS). Tair_C/Si coverage for
CRYSTAL-FACE-NASA drops from the previously-reported ~44% to the real,
honest **26.2%** -- the old ~44% included seconds where a stale HW/MM/NM
value from a different second (up to 60s away) had been reused. MIDCIX
Tair_C/Si similarly changes from ~26% to **20.8%**. Both numbers and the
hardcoded narrative text in `diagnose_cpi_fusion.py`'s report were
updated to reflect this.

## Out of scope (documented, not fixed)

Repetition already baked into decades-old raw archive files themselves
cannot be fixed by changing merge logic, since the file already contains
the repeated value before our pipeline touches it:

- **MIDCIX `P_hPa`, CRYSTAL-FACE-NASA `Tair_C`**: both raw JW hygrometer
  files self-describe these columns as "from MMS" in their own header --
  i.e. the original data providers (c. 2002-2004) already merged a
  coarser-cadence instrument into the file before archiving it.
- **ESCAPE, ISDAC**: similar repetition confirmed present in the raw
  files, but with no documented secondary-source explanation in their
  headers.

These are unchanged by this rewrite and remain documented in
`docs/decisions/2026-07-06-issue10-qc3-merge-granularity-diagnostic.md`.
No masking/flag column was added for them (kept as real data, per the
project's existing QC philosophy of flagging rather than silently
dropping ambiguous-but-plausible readings).

## How to reproduce

```bash
/Users/josephko/miniconda3/envs/cpi-thermo/bin/python main.py
/Users/josephko/miniconda3/envs/cpi-thermo/bin/python scripts/qa_checks.py
/Users/josephko/miniconda3/envs/cpi-thermo/bin/python scripts/diagnose_cpi_fusion.py
/Users/josephko/miniconda3/envs/cpi-thermo/bin/python scripts/build_data_tiers.py
```
