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
for per-investigation root-cause writeups, and `docs/sessions/` for session-by-session
engineering logs.

---

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

**Commits:** `8c9d8de`, `096704a`, `ac6a2c2`, `af92140` (see
`docs/sessions/2026-07-05-cpi-fusion-and-remaining-fixes.md` for the full commit
range and table)
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
