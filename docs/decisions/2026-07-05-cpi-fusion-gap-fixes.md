# 2026-07-05 — CRYSTAL_FACE_NASA Si recovery attempt + CRYSTAL_FACE_UND L2 segment fix

Follow-up to the CPI/env fusion investigation (ARM, CRYSTAL_FACE_NASA,
CRYSTAL_FACE_UND). Two concrete actions taken.

## 1. CRYSTAL_FACE_NASA — wired up the ALIAS instrument (third Si fallback)

**Finding:** `config.yaml`'s `h2o_ranking` for this campaign already listed
`JLH, HW, ALIAS` — but `parsers/crystal_face_nasa.py` hardcoded
`Si_ALIAS = NaN` unconditionally; ALIAS was never actually loaded, even
though 9 raw `AL*.WB57` files exist in `data/raw/CRYSTAL-FACE-NASA/ALIAS/`
(covering 2002-07-07, 09, 11, 13, 16, 23, 26, 28, 29 — 4 of the 5 lowest-
match flight days from the earlier investigation).

**Fix:** Added `load_alias_file()` (same generic ICARTT-1001 positional
parser used for `load_np_file`; ALIAS's header doesn't label its scale-
factor/missing-value lines with text). Wired into `load_crystal_face_nasa()`
following the exact same three-stage pattern already used for HW: (1) merge
ALIAS ppmv with T/P met and compute `Si_ALIAS` (clipped `[-1, 2]`, same as
HW), (2) add ALIAS-only rows for dates with no JLH/HW coverage at all, (3)
add ALIAS rows inside remaining JLH+HW gaps (5 s tolerance, matching
ALIAS's ~2 s native resolution). `extract_crystal_face_nasa_standard` now
also resolves `qv_alias` into the `qv` fallback chain (JLH → HW → ALIAS).

**Measured impact — honest accounting:** ALIAS produced 11,556 valid Si
records across 9 dates and is now cross-referenced onto 36,999 existing
rows, but **only 1 row in the entire campaign ends up with Si populated
exclusively from ALIAS** (JLH and HW both NaN there) — HW's existing gap-
fill mechanism already achieves near-complete Si coverage for whatever
timestamp rows exist in the combined frame. ALIAS's real, measurable
contribution is **107 net-new rows** at timestamps that previously had no
env row at all (in JLH/HW gaps). This is a genuine, if small, improvement
worth having (completes the config's originally-intended fallback chain),
but it does **not** meaningfully move CRYSTAL_FACE_NASA's CPI-fusion match
rate (still 43.8%, unchanged to 1 decimal place).

**Why the bottleneck remains:** the earlier investigation's per-date
sampling-cadence check showed most days already have ~1s median row
spacing — the shortfall isn't overall row density, it's that JLH's
per-instance gaps (and now HW/ALIAS's gaps too) occur in the same specific
seconds CPI is busiest (cloud penetrations). Closing this fully would
require either relaxing the CPI/env merge tolerance beyond ±1s (a data-
quality trade-off flagged in the existing report) or genuinely
higher-cadence raw water-vapor data that doesn't exist in this campaign's
archive.

## 2. CRYSTAL_FACE_UND — recovered the missing 2002-07-11 flight segment

**Finding:** `data/raw/CRYSTAL-FACE-UND/ND_MIS/` contains
`ND20020711__MIS_L2.CIT` — a **second, non-overlapping flight segment**
(17:59:58–22:15:17 UTC) alongside the primary `ND20020711__MIS.CIT`
(14:19:21–16:52:20 UTC). The same `_L2` pattern exists in parallel across
*every* instrument for that date (`MET_L2`, `NAV_L2`, `CLD_L2`, `FSS_L2`,
etc.) — this looks like a mid-flight recording restart that NASA's archive
split into two files. The parser's glob pattern (`*MIS.CIT`) doesn't match
`*_L2.CIT`, so the second segment was silently dropped entirely (not just
for RH — for Tair, pressure, and position too).

**Fix:**
- `config.yaml` and `main.py`'s `DEFAULT_CAMPAIGN_CONFIG` pattern changed
  from `"*MIS.CIT"` to `"*MIS*.CIT"` (verified this doesn't pick up
  anything unintended — only the 14 expected `ND_MIS/` files match).
  **Note:** `main.py --all` reads `DEFAULT_CAMPAIGN_CONFIG` directly, not
  `config.yaml` — `config.yaml` only applies when `--config` is passed
  explicitly. Both were updated to keep them consistent.
- `parsers/crystal_face_und.py`: the MET/NAV companion-file lookup used
  `.replace("MIS.CIT", "MET.CIT")`, which doesn't match `..._MIS_L2.CIT`
  filenames. Changed to a bare-token replace (`.replace("MIS", "MET")` /
  `.replace("MIS", "NAV")`), which correctly derives `MET_L2.CIT`/`NAV_L2.CIT`
  companions for the L2 file too.

**Result:** 2002-07-11 now spans 14:19–22:15 UTC (was cut off at 16:52).
+15,320 rows recovered. RH is NaN throughout the L2 segment too (confirmed
directly — the sensor was down for the whole day, matching the existing
"Jul 7, 9, 11 RH entirely fill values" finding), so Si coverage for this
date doesn't improve, but Tair_C/P_hPa/Lat/Lon/Alt_m now do.

## Final diagnostics (`logs/diagnostics/cpi_fusion_report.txt`, ±1s tolerance)

| Metric | Before | After |
|---|---|---|
| Total env records | 3,644,847 | 3,660,274 |
| CPI images matched (any campaign) | 88.0% | 89.1% |
| CPI images with both Tair_C & Si | 57.7% | 57.7% (unchanged) |
| CRYSTAL_FACE_UND matched% | 97.3% | **99.4%** |
| CRYSTAL_FACE_UND Tair% | 94.1% | **96.3%** |
| CRYSTAL_FACE_UND Si% | 52.5% | 52.5% (unchanged — RH still down) |
| CRYSTAL_FACE_NASA matched%/Si% | 43.8% | 43.8% (unchanged to 1dp; +107 rows, +1 Si-only-via-ALIAS row) |

ARM and MIDCIX unaffected by this round (both are genuine raw-data
acquisition gaps per the prior investigation, not addressable without new
source files).

Verified via `python main.py --all` → `python scripts/diagnose_cpi_fusion.py`.
