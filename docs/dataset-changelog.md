# Dataset Changelog

Reverse-chronological log of changes that affect `data/out/combined_env_data.parquet`'s
rows, columns, or coverage — new/removed campaigns, schema changes, and bugfixes that
measurably move row counts or coverage %. **Not** a commit log (see `git log`) and
**not** a per-pipeline-run log (see `logs/pipeline/<ts>/`) — those already capture
every change at the code/run level. An entry here is warranted only when a change
moves the campaign count, the row count by more than a trivial amount, a coverage
percentage, or the schema (column added/removed/renamed). Tooling/logging/refactor
changes that don't alter the parquet's actual contents do not get an entry.

Concrete check for future sessions: diff `logs/pipeline/<ts>/campaign_summary.csv`
between the last entry below and the current run — if campaign count or row counts
moved, add an entry.

See `CLAUDE.md`'s "Known issues" section for current-state facts, `docs/decisions/`
for per-investigation root-cause writeups.

---

## 2026-08-28 — Repo docs condensation; L0/L1/L2 regenerated and validated (no dataset change)

**See:** `docs/reports/2026-08-28-dataset-validation.md`.

Removed `docs/sessions/` (2 files, fully redundant with `docs/decisions/` +
this changelog), the stale `docs/todo/2026-07-13-turbulence-measurements-plan.md`
(marked "not yet implemented" but shipped same day it was written),
`docs/reports/20260706_170726-dataset-report.md` (redundant re-snapshot),
`notes/missing_data_notes.txt` (fully stale), and `scripts/diagnose_timestamps.py`
(scratch debug script); deleted superseded dated/per-campaign parquet
snapshots under `data/out/` (gitignored, regenerable). Corrected a stale
CRYSTAL-FACE-NASA turbulence-coverage figure (89.1% → 81.5%) across 3
2026-07-13 reports, whose tables were never regenerated after that day's
later missing-value-mask fix (`9c02cf0`) — a documentation-accuracy fix, not
a dataset change. Also corrected `CLAUDE.md`'s "Standard output schema" and
"Known issues" sections, which still described the wider turbulence column
set (`Roll_deg`, `TAS_ms`, etc.) dropped the same day it was added, and
3-separate-EDR-columns, both superseded by `2026-07-13-edr-unification.md`
and the scope-reduction commit `ea206e2`.

Full L0/L1/L2 rebuild + all QC checks + CPI fusion diagnostic reproduced
their last-known baselines exactly (row counts, QC flag counts, fusion %
all unchanged) — confirms the pipeline is reproducible from current code.
No parser or pipeline code changed this session.

---

## 2026-07-13 — Phase 1 turbulence columns added; ARM L0 now floored to 1 Hz (was native 4 Hz)

**See:** `docs/decisions/2026-07-13-turbulence-schema.md`,
`docs/decisions/2026-07-13-edr-unification.md`.
**Campaigns:** 15 (no change). **Schema:** +22 columns (wind, attitude,
angle-of-attack/sideslip, true/indicated airspeed, EDR). **Rows:** total L0
row count drops from ~5.0M to 4,572,581 — entirely attributable to the ARM
row-count change below; every other campaign's row count is unchanged.

Adds wind vector, aircraft attitude, angle-of-attack/sideslip, true/indicated
airspeed, Mach number, and EDR (eddy dissipation rate) columns for 9
campaigns (ARM, ATTREX, POSIDON, MACPEX, IPHEX, MC3E, MPACE, OLYMPEX, ISDAC).
These fields were already being read into each parser's intermediate
DataFrame but dropped at the final `extract_*_standard()` step. See
`config.yaml`'s `output.standardized_columns` for the full list and
per-campaign provenance.

EDR is deliberately kept as separate, never-unified columns
(`EDR_mms_log10kWkg`, `EDR_und_cm23s1` [+ `_nose` for MPACE],
`EDR_arm`) rather than one merged column — different instruments/pipelines,
not just different units of one quantity. See
`docs/decisions/2026-07-13-turbulence-schema.md` for the full rationale.

**ARM L0 row-count change (supersedes part of the 2026-07-07 entry below):**
the 2026-07-07 fix explicitly deduped ARM to 1 Hz *only inside `build_l1()`*,
leaving L0 at ARM's native 4 Hz. This change moves that flooring into
`load_arm_file()` itself (matching the plan's explicit instruction to apply
it "to the whole load_arm_file output, not just new turbulence columns"),
so **L0 itself is no longer untouched for ARM**: 567,760 rows (native 4 Hz,
confirmed by re-running `main.py --campaigns ARM` against the prior commit)
→ 141,940 rows (floored 1 Hz, first real sample per second, never averaged).
Side effect worth flagging: ARM's pre-existing GPS-altitude stuck-run
detector (`STUCK_RUN_LENGTH = 30` samples) now operates on the 1 Hz stream
instead of 4 Hz, so its effective detection window changes from ~7.5s to
30s — the code's own comment already claimed 30s was the intent ("matching
qa_checks.py QC3's own stuck-sensor threshold"), so this is a correction of
a pre-existing miscalibration rather than a new bug, but it does mean a
7.5–30s GPS freeze that previously triggered a fallback to
`Pressure_Altitude_m` no longer does.

**Known follow-ups (not yet fixed, tracked from code review):**
- ARM is missing `WindSpeed_ms`/`WindDir_deg`/`TAS_ms`/`Roll_deg`/`Pitch_deg`/`Heading_deg`
  even though the raw binary already has ready-named columns for all of them
  (`ARM_COLUMNS` in `parsers/arm.py`) — only `Wind_W_ms`/`EDR_arm` were wired up.
- ATTREX/POSIDON's `EDR_mms_log10kWkg`/`REYN_mms` use the same x0.01 MMS
  integer-scale as T/P/Heading, confirmed for Heading (raw maxes at exactly
  36000) but not independently confirmed for EDR/REYN specifically — values
  are physically plausible but the scale is asserted by analogy, not a
  header citation.
- New turbulence columns in IPHEX/MPACE (and likely MC3E/OLYMPEX) bypass
  the existing fill-value sentinel masking (`-9999`/`-7777`/`-8888` family)
  applied to other columns in the same files — no corruption found in the
  current build, but the path is unguarded.
- `HARD_BOUNDS` in `scripts/qa_checks.py` uses one global, campaign-agnostic
  Roll/Pitch range, which can't catch a per-family sign-convention inversion.

Phase 2 (`crystal_face_und.py`, `crystal_face_nasa.py`, `midcix.py`,
`escape.py` — needs upstream loader/merge-list changes) and Phase 3
(`ice_l.py`, `airs_ii.py`, ISDAC's unused 5 Hz source — new NetCDF read
paths) are not yet implemented.

---

## 2026-07-07 — L1 fixed to one row per CPI image (was collapsing ~30 images/second into 1)

**See:** `scripts/build_data_tiers.py::build_l1()`.
**Campaigns:** 15 (no change), but `combined_env_data_L1.parquet` /
`_L2.parquet` grow substantially.

`build_l1()` was a semi-join (filtered L0 rows to seconds present in a
campaign's CPI-image set) rather than a proper one-to-many merge. Since
CPI images average ~30 per matched second (up to 644 in one second),
every image beyond the first at a given second was silently discarded.
Rewrote it as a per-campaign inner merge -- one row per CPI image, env
columns duplicated across images sharing a second, plus a new
`cpi_filename` column so every row traces back to its source image.

Surfaced a related issue while fixing this: L0 is not always unique per
`(Campaign, Timestamp)` once floored to the second -- ARM is a genuine
native 4Hz stream (0.25s intervals), so naively joining against its
floored timestamp fanned each image out across up to 4 sub-second L0
rows. Fixed by deduping L0 to one row per floored second (keep first)
*inside `build_l1()` only* -- L0 itself and its native sub-second
resolution are untouched.

| Campaign | n_L1 before -> after |
|---|---|
| ARM | 15,452 -> 230,029 |
| CRYSTAL-FACE-NASA | 2,716 -> 78,151 |
| CRYSTAL-FACE-UND | 28,609 -> 1,608,674 |
| MIDCIX | 5,200 -> 90,667 |
| MPACE | 5,806 -> 35,997 |
| AIRS-II | 4,709 -> 92,168 |
| ICE-L | 8,407 -> 46,203 |
| ISDAC | 6,986 -> 400,805 |
| MACPEX | 486 -> 80,240 |
| MC3E | 15,586 -> 173,766 |
| ATTREX | 1,000 -> 122,050 |
| IPHEX | 17,406 -> 38,697 |
| **TOTAL** | **112,363 -> 2,997,447** |

L2 grows correspondingly (72,809 -> 1,828,818). 93.66% of all 3,200,351
CPI images across the 12 campaigns with any CPI archive coverage now have
a matching L1 row (the remaining ~6.3% have no L0 second at all -- no
instrument reported anything at that exact time).

---

## 2026-07-07 — Exact-second merge rewrite; L0/L1/L2 data tiers introduced

**See:** `docs/decisions/2026-07-07-exact-second-merge-rewrite.md`,
GitHub issues #10, #12.
**Campaigns:** 15 (no change), but row counts change substantially for 6
of them, and `data/out/combined_env_data_L1.parquet` /
`combined_env_data_L2.parquet` are new outputs.

Replaced every `pd.merge_asof(direction="nearest", tolerance=...)` call in
the pipeline with an exact-second outer-join (round every instrument's own
timestamp to the nearest second, then join by exact second; no reading at
that second means NaN, never a nearest-neighbor value fabricated from a
different second). Row counts grow (union-of-all-instrument-timestamps
row grid, vs. previously being anchored to one primary instrument) while
coverage % drops (previously-fabricated values via wide merge tolerances,
up to 60s in one case, no longer counted):

| Campaign | Rows before -> after | Tair_C valid before -> after | Si valid before -> after |
|---|---|---|---|
| CRYSTAL-FACE-NASA | 154,815 -> 323,310 | 100.0% -> 50.3% | 100.0% -> 50.3% |
| MIDCIX | 118,105 -> 181,459 | 64.3% -> 41.8% | 64.3% -> 41.8% |
| MACPEX | 279,073 -> 307,780 | 100.0% -> 90.7% | 69.9% -> 63.4% |
| ATTREX | 581,370 -> 1,316,204 | 99.7% -> 44.0% | 86.3% -> 37.6% |
| POSIDON | 190,418 -> 351,508 | 97.9% -> 97.2% | 96.0% -> 52.0% |
| CRYSTAL-FACE-UND | 200,740 -> 200,864 | 98.4% -> 98.3% | 66.2% -> 66.2% |

All other campaigns (ARM, AIRS-II, ESCAPE, ICE-L, IPHEX, ISDAC, MC3E,
MPACE, OLYMPEX) show exactly zero diff. Si mean/std for actually-computed
values are unchanged everywhere -- confirms this was a coverage-honesty
fix, not a correctness fix to the Si/qv computation itself.

Two genuine pre-existing bugs (not previously known) surfaced as a side
effect of the anchor -> union-join change: ATTREX's entire 2011 deployment
(pre-MMS, no MMS files) was silently dropped by the old anchor-based
merge; POSIDON was similarly restricted to its alphabetically-first
instrument's date range. Both are now recovered. CRYSTAL-FACE-NASA's
CPI-fusion Tair_C/Si coverage (issue #9) changes from the previously
reported ~44% to a real, honest 26.2%.

---

## 2026-07-06 — Systematic QA/QC flag investigation (7 fixes, ARM/AIRS-II/CRYSTAL-FACE-NASA corrected)

**Commits:** `ee1a933`, `c36bd12`, `18f19e3`, `fad5479`, `60a92b5`, `6853008`
(parser/check fixes); see `docs/decisions/2026-07-06-qc{2,3,4,7,8}-*.md` and
`2026-07-06-qc6-zero-si-flight-days.md` for the full per-check writeups.
**Campaigns:** 15 (no change)

Cross-referenced every QC1-QC9 flag against existing documentation and
found 7 gaps with zero prior investigation, worked through in priority
order. Row count barely moves (this was a data-*correctness* pass, not a
coverage-recovery one — most fixes correct wrong values in place or
suppress false-positive diagnostics rather than add/remove rows), but QC
health improved substantially:

| Check | Before | After |
|---|---|---|
| QC2 Internal consistency | 89,318 flags | 80,645 flags |
| QC3 Stuck-sensor | 1,567 runs (7 campaigns) | 309 runs (4 campaigns) |
| QC4 Sentinel values | 682 flags (15 campaigns) | 0 |
| QC7 Duplicate timestamps | 8,910 rows | 2 rows |
| QC8 Vertical profile plausibility | 47 bins (11 campaigns) | 6 bins (3 campaigns) |

Real bugs fixed (not just check recalibration):
- **CRYSTAL-FACE-NASA**: `_round_timestamp_to_second()` used banker's
  rounding, colliding two adjacent half-second-sampled JLH readings onto one
  timestamp (4,454 spurious duplicate rows in `JW20020719.WB57` alone).
  Switched to `floor()`.
- **ARM**: two independent GPS altitude failure modes corrected via
  fallback to `Pressure_Altitude_m` — a lock-acquisition glitch (GPS stuck
  at a bogus value right after first fix, ~12,853 rows across 4 files) and a
  mid-flight freeze (GPS holds a stale value for minutes while
  `Pressure_Altitude_m` keeps changing, ~70,000 rows across nearly every
  flight file). Also masked a Rosemount probe warm-up fault (frozen at
  -64.5°C for the first 23 minutes of one flight, 5,668 rows) — ARM Tair_C
  coverage: 100% → 99.0%.
- **AIRS-II**: `ALT` (IRS Baro-Inertial Altitude) reads exactly `0.0` for
  extended stretches (up to 4,413 of 33,444 samples in one flight) — now
  falls back to GPS-derived `GGALT` per-sample.

Check-logic bugs fixed (no parser/data changes, the check itself was wrong):
- **QC8**: computed the qv-saturation reference from the *ISA theoretical*
  temperature using the *ice* formula, instead of the bin's own *observed*
  mean temperature and the *liquid* formula (which QC2 already used
  correctly) — 41 of 47 flagged bins were false positives once corrected.
- **QC4**: `1000.0`/`9999.0` are physically plausible values for `Alt_m`,
  `P_hPa`, and `*_ppmv` columns, producing false "sentinel" matches on real
  continuously-varying data; excluded those combinations.
- **QC2**: AIRS-II was missing from `IN_CLOUD_CAMPAIGNS` despite using the
  same chilled-mirror instrument as its documented siblings.

Investigated and accepted as genuine data limitations, no fix: MACPEX's
independent Si-vs-qv instrument fallback (77 rows), ATTREX's low-absolute-
humidity measurement noise (13 rows), a MPACE-style raw-instrument clock
glitch, and QC6's remaining zero-Si flight-days (ARM/IPHEX/OLYMPEX/ATTREX/
MACPEX — all traced to instruments genuinely not operating that day/flight).

| Metric | Before | After |
|---|---|---|
| Rows | 3,841,812 | 3,841,797 (-15, from the CRYSTAL-FACE-NASA dedup) |
| Campaigns | 15 | 15 |
| ARM Tair_C coverage | 100.0% | 99.0% |
| CPI/env fusion (matched) | 91.3% | 91.3% (unchanged) |
| CPI/env fusion (Tair_C + Si) | 57.7% | 57.7% (unchanged) |

## 2026-07-06 — MIDCIX fallback fix + MPACE campaign added

**Commits:** `e2ac25b` (MIDCIX), `135d35d` (MPACE)
**Campaigns:** 14 → 15 (added MPACE)

Two changes landed in the same session before the next fusion measurement was taken:

- **MIDCIX**: `load_midcix()` now adds position-only fallback rows for FP navigation
  files whose date has no corresponding JW (water vapor) file (2004-04-22,
  2004-04-27) — JLH genuinely wasn't operating those days, so `Tair_C`/`Si`/`qv` stay
  NaN, but `Lat`/`Lon`/`Alt_m` no longer get dropped entirely. MIDCIX's own CPI
  timestamp match rate: 25.9% → 64.6%.
- **MPACE**: added `parsers/mpace.py`, loading UND Citation NASA Ames files from
  `data/raw/MPACE` (15 flights, 2004-09-30 to 2004-10-21, Barrow AK). No water-vapor
  instrument was flown on this platform, so `Si`/`qv` are NaN for every MPACE record;
  only `Tair_C`/`P_hPa`/`Lat`/`Lon`/`Alt_m` are populated.

Because no diagnostic run happened between the two commits, the measured
89.1%→91.3% overall fusion jump below reflects **both changes together, not MPACE
alone** — they can't be cleanly separated from the numbers on hand.

| Metric | Before (end of 2026-07-05) | After |
|---|---|---|
| Rows | 3,660,274 | 3,841,812 (+181,538) |
| Campaigns | 14 | 15 |
| CPI/env fusion (matched) | 89.1% | 91.3% |
| CPI/env fusion (Tair_C + Si) | 57.7% | 57.7% (unchanged — MPACE adds matches but no Si/qv) |

Same-day addendum: commit `9c1026f` fixed MPACE being missed from two hardcoded
campaign-list constants (`plot_all_campaigns.py`, `qa_checks.py`) on first add — a
correction to how this same addition was wired up, not a new dataset event, so it
isn't given its own entry.

---

## 2026-07-05 — Coverage recovery sweep

**Commits:** `8c9d8de`, `096704a`, `ac6a2c2`, `af92140`
**Campaigns:** 14 (no change)

Rollup of same-day fixes: ESCAPE P_hPa/temperature masking (stuck-Palt sensor
failure), IPHEX/OLYMPEX `qv` upper-bound fix (propagated the `Si` clip mask to
co-derived `qv`), POSIDON pressure-sentinel fix, OLYMPEX chilled-mirror
physical-impossibility fault masking, Alt_m recovery for CRYSTAL-FACE-NASA (0%→100%,
wrong geolocation source), CRYSTAL-FACE-UND (0%→100%, unread `ND_NAV` files, plus a
recovered 2002-07-11 flight segment), MACPEX/MIDCIX (0%→100%/96.7%, downloaded real
navigation data + fixed a scale-factor bug in `load_mms_file`), CRYSTAL-FACE-NASA's
`ALIAS` instrument wired up as a third Si fallback, and a centralized fix for an MC3E
CPI-timestamp timezone mislabeling (`parsers/cpi_timestamps.py`).

| Metric | Before (start of day) | After (end of day) |
|---|---|---|
| Rows | — | 3,660,274 |
| Campaigns | 14 | 14 |
| Alt_m coverage | 79.2% | 98.2% |
| CPI/env fusion (matched) | — | 89.1% |
| CPI/env fusion (Tair_C + Si) | — | 57.7% |

See `docs/decisions/2026-07-05-open-issues-resolved.md`,
`docs/decisions/2026-07-05-qc9-iphex-olympex.md`, and
`docs/decisions/2026-07-05-cpi-fusion-gap-fixes.md` for the full investigations.
