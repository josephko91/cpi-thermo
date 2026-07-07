# 2026-07-07 — CPI/env unmatched-images investigation (6.34% of CPI images)

Follow-up to the L1 image-level fix (`docs/dataset-changelog.md`,
2026-07-07 entry). After L1 was corrected to one row per CPI image,
93.66% of the 3,200,351 CPI images across the 12 campaigns with any CPI
archive coverage have a matching L0 env record; 202,904 (6.34%) do not.
This doc investigates why, campaign by campaign, and separately checks
whether the CPI-timestamp timezone assumption in
`parsers/cpi_timestamps.py` could account for any of it. No code changes
— investigation only.

## Per-campaign breakdown

| Campaign | Unmatched | % of campaign | Root cause |
|---|---:|---:|---|
| ISDAC | 105,007 | 20.76% | Mixed: 2 dates with a missing raw env file, 2 dates with a pre-flight CPI gap |
| ARM | 65,674 | 22.21% | Already documented (`2026-07-05-arm-cpi-timestamp-investigation.md`) — one anomalous date, 2000-03-13 |
| MC3E | 13,792 | 7.35% | Pre-flight CPI gap on one date, 2011-06-02 |
| CRYSTAL-FACE-UND | 9,152 | 0.57% | Scattered small within-flight instrument gaps |
| ATTREX | 7,078 | 5.48% | Post-flight CPI gap on one date, 2014-03-05 |
| IPHEX | 1,995 | 4.90% | Pre-flight CPI gap on one date, 2014-05-23 |
| Remaining 6 campaigns | ~206 | <0.15% each | Negligible edge effects |

ISDAC and ARM alone account for 84% of all unmatched images.

## Detail per campaign

**ARM (65,674 unmatched, 98.9% on 2000-03-13):** Already investigated in
`docs/decisions/2026-07-05-arm-cpi-timestamp-investigation.md`. CPI
captured images 00:00–01:xx UTC that date with zero corresponding env
data (env starts at 18:07). Confirmed via raw archive: exactly 12 files
match the campaign's "12 IOP flights" — no missing file. Most likely a
ground-test/calibration session or CPI-side clock fault.

**ISDAC (105,007 unmatched — newly diagnosed):**
- **2008-04-06 and 2008-04-15**: no raw env file exists for these dates
  at all (checked every `Flight Date:` header across all 27 raw files
  under `data/raw/ISDAC/strapp-convair_bulk/CommaDelimited/`), yet CPI
  has images. Same category as ARM's anomaly.
- **2008-04-25**: raw env file exists but starts at 02:16:13 UTC; CPI
  images that day run 00:00:18–01:20:15 — entirely before the flight's
  recorded start. Zero overlap, 100% unmatched.
- **2008-04-27**: env starts at 04:32:06 UTC; CPI runs
  00:34:20–06:32:15 — the first ~4 hours of CPI images that day predate
  the flight, matching the observed 78% unmatched rate.

**MC3E (13,792 unmatched, 93% on 2011-06-02):** CPI runs 01:10–17:31 UTC
(after the existing +5h CDT correction); env only starts at 14:29:59 — a
~13.3-hour pre-flight gap where CPI was evidently running (ground
checkout) well before the science flight began.

**ATTREX (7,078 unmatched, 100% on 2014-03-05):** Trailing edge instead
of leading — CPI continues to 06:09 UTC but env recording stops at
05:38, a ~30-minute post-flight/landing gap.

**IPHEX (1,995 unmatched, 98% on 2014-05-23):** Same pre-flight pattern
— CPI starts at 13:29 UTC, env at 18:08, a ~4.6-hour gap.

**CRYSTAL-FACE-UND (9,152 unmatched, spread across many dates):**
Different shape — e.g. on 2002-07-07 the env data's overall time range
(17:51–21:43) fully spans the CPI window (18:13–21:25), but there are
second-level gaps inside that range (brief instrument dropouts), not a
boundary mismatch. Given this campaign has 1.6M images total, the 0.57%
impact is minor.

## Could a CPI timestamp/timezone error explain any of this?

Crawled `github.com/vprzybylo/cocpit` (the tool that produced these CPI
archive filenames/timestamps) to check.

**Naming convention — confirmed correct for 11/12 campaigns, one
outlier.** `cocpit/add_date.py` derives the timestamp directly from the
filename: `pd.to_datetime(date_str, format="%Y_%m%d_%H%M%S")`. Checked
every campaign's actual filename in `data/raw/cpi_embeddings_timestamps.csv`
against this — 11 of 12 match exactly (e.g. ARM:
`2000_0306_202027_531_20.png`). **AIRS-II is the one outlier**: its
filenames are `MMDD-HHMMSS_ms_index.png` (e.g. `1114-115708_753_18.png`)
— no year at all, and a dash instead of cocpit's expected underscore
structure. Cocpit's own generic parser would fail to parse this format.
AIRS-II's timestamps in our archive must come from a different/special-
cased process not visible in the cocpit code. This doesn't explain any of
the current gap, though: AIRS-II matches at 99.96% (only 33 unmatched
images), so whatever assigned that year got it right.

Traced further: `cocpit/process_sheets.py` shows individual chip
filenames are the raw CPI probe "sheet" filename plus an appended
particle index — cocpit doesn't invent or reinterpret the timestamp, it
inherits it verbatim from the CPI hardware/vendor software (SPEC Inc.'s
`cpiview`) at acquisition time.

**Timezone — cocpit provides no independent validation, but empirically
ruled out as the cause here.** Searched the entire cocpit codebase, full
commit history, and all issues/PRs for "UTC," "timezone," "local time,"
"GMT," "offset" — zero hits anywhere. `add_date.py`'s `pd.to_datetime()`
produces a naive datetime with no timezone correction of any kind;
cocpit simply trusts whatever the CPI probe's onboard clock wrote into
the filename. This means our own pipeline's "UTC for every campaign
except MC3E" assumption (`parsers/cpi_timestamps.py`,
`CPI_UTC_OFFSET_HOURS`) isn't confirmed by any authoritative source
either — it was reverse-engineered from a specific forensic gap analysis
for MC3E, not verified against ground truth.

Tested this directly against the three largest unmatched-image clusters,
checking whether a plausible local-time offset for each flight's
location would shift the CPI window into alignment with the actual
env-recorded flight window:

| Campaign/date | CPI window (as UTC) | ENV window (UTC) | Gap | Plausible local offset | Resolves? |
|---|---|---|---|---|---|
| ARM 2000-03-13 | 00:00–01:20 | 18:07–22:29 | ~17–23h | CST = UTC-6 (SGP, Oklahoma) | No — shifting by 6h lands at 06:00–07:20, nowhere near 18:07–22:29 |
| ISDAC 2008-04-25 | 00:00–01:20 | 02:16–05:53 | ~1–2h | AKDT = UTC-8 (Fairbanks) | No — shifting by 8h lands at 08:00–09:20, past the env window |
| MC3E 2011-06-02 | 01:10–17:31 (already CDT-corrected) | 14:30–18:20 | ~13h | Already corrected; no further shift bridges 13h | No |

None resolve with a clean timezone shift — the gaps are either far
larger than any standard UTC offset (max ±14h) or the direction of any
plausible shift moves the wrong way. This rules out mislabeled timezone
as the driver and supports the "genuine ground-test/pre-flight CPI
operation" explanation for each cluster.

## Conclusion

This is not a pipeline bug. The unmatched 6.34% of CPI images reflects a
real, physical fact: at each of these flight-dates, the CPI camera was
capturing images before (or, for ATTREX, slightly after) the aircraft's
environmental instruments were recording — most plausibly ground
tests/instrument checkouts/calibration sessions, or in ISDAC's two
fully-missing dates, a genuinely absent raw env file. No fix is possible
from the data available; closing the last ~2 dates (ISDAC 04-06/04-15)
would need archive access to check for a missing raw file, mirroring
ARM's situation.

Left as a secondary, low-priority open item: AIRS-II's CPI filenames use
a different, year-less naming convention than every other campaign and
than cocpit's own documented parser — harmless today (99.96% match
rate) but worth understanding if AIRS-II CPI data is ever reprocessed.

## How to reproduce

```bash
/Users/josephko/miniconda3/envs/cpi-thermo/bin/python -c "
import pandas as pd, sys
sys.path.insert(0, '.')
from parsers.cpi_timestamps import load_cpi_embeddings_timestamps
from parsers.utils import round_timestamp_to_second

l0 = pd.read_parquet('data/out/combined_env_data.parquet', columns=['Campaign','Timestamp'])
l0['Timestamp'] = round_timestamp_to_second(l0['Timestamp'])
l0 = l0.drop_duplicates(subset=['Campaign','Timestamp'])
cpi = load_cpi_embeddings_timestamps('data/raw/cpi_embeddings_timestamps.csv')
cpi['datetime'] = round_timestamp_to_second(cpi['datetime'])

for camp, sub in cpi.groupby('campaign_env'):
    l0_seconds = set(l0[l0['Campaign']==camp]['Timestamp'])
    sub = sub.copy()
    sub['matched'] = sub['datetime'].isin(l0_seconds)
    unmatched = (~sub['matched']).sum()
    print(camp, unmatched, 'unmatched of', len(sub))
"
```
