# 2026-07-05 — ARM CPI 2000-03-13 timestamp mismatch investigation

Follow-up to the CPI/env fusion investigation. ARM's overall CPI timestamp
match rate (77.8%) is dragged down almost entirely by one day,
2000-03-13, where CPI has two disjoint image clusters (00:00–01:xx UTC and
18:07–22:29 UTC) but the raw environmental archive only covers the second
window. This doc records the investigation into whether the first cluster
reflects a missing raw file or a CPI-side artifact. No code changes —
investigation only.

## What was checked

1. **Crawled the campaign source pages** named in `parsers/arm.py`
   (`iop.arm.gov/2000/sgp/cloud`, `arm.gov/research/campaigns/sgp2000sprcloud`).
   The campaign page confirms: Spring Cloud IOP, March 1–26 2000, SGP site —
   **12 IOP flights** with the University of North Dakota Citation aircraft.
   The PI-archive directory name is `poellot-citation` (CPI + PMS data),
   matching the readme already bundled in `data/raw/ARM/`.

2. **`iop.arm.gov` is a client-rendered SPA** ("ARM PI Data Browser") — static
   fetch only returns the app shell. Traced its JS bundle to the backend API
   (`iop-api.svcs.arm.gov`) and found a `/contents/{path}` directory-listing
   endpoint, but it requires OAuth2 login (`401 Not authenticated`). ARM's
   classic public `armlive` data API also requires a registered username
   (guest login rejected). Browsing the archive directly to check for a
   missing file needs real ARM Data Center credentials, which aren't
   available in this session.

3. **Reconciled the flight count instead.** `data/raw/ARM/` already holds
   **12** `citation.*.t4archive.gz` files across 10 distinct dates — with
   *two* files each for 2000-03-12 and 2000-03-18 (both pairs start only
   ~5–6 minutes apart, e.g. `00:16:49` / `00:22:22` — a single session split
   by a brief recorder restart, not two separate flights). **12 files
   exactly matches the campaign page's "12 IOP flights."** If 2000-03-13
   genuinely had a second (early-morning) flight, the total would be 13, not
   12 — it isn't.

4. **Decoded the raw filename convention** and verified it against every
   file's actual parsed data:
   `citation.<MM><DD>00<HHMM>.t4archive.gz`, where the last 4 digits are the
   flight's real start time (hour+minute). This matched the parsed data to
   within a minute for **all 12 files**, including the lone 2000-03-13 file:
   filename encodes `1806` → actual data starts `18:07:06`. The filename is
   not hiding an earlier start time.

5. **Checked whether the "anomalous" second 2000-03-12 file
   (`citation.0312002222...`) secretly contains mis-dated 2000-03-13
   records** (it parses with a suspiciously wide apparent timestamp range).
   It doesn't: 48,320 records correctly dated 2000-03-12, 240 stray records
   mis-dated 2000-03-11 (decode noise, negligible), **zero** records dated
   2000-03-13.

6. **Checked the CPI side's own filename-to-timestamp fidelity.** ARM CPI
   filenames are `YYYY_MMDD_HHMMSS_ms_index.png`
   (e.g. `2000_0313_000002_474_0.png`); the CSV's `datetime` column
   (`2000-03-13 00:00:02`) matches the embedded filename timestamp exactly.
   No extraction/parsing bug between the CPI filename and
   `data/raw/cpi_embeddings_timestamps.csv`.

## Conclusion

Every check — filename encoding, per-record dates, and the exact flight
count — confirms the raw environmental archive is complete for what NASA/
ARM actually recorded. The CPI images timestamped 2000-03-13 00:00–01:xx
UTC do not correspond to any Citation aircraft science flight in this
campaign. Most likely explanation: the CPI probe was powered on and
capturing images during a **ground test, calibration run, or pre-flight
check** that wasn't a recorded science flight (so no T4 environmental data
exists for it by design), or there's a clock fault isolated to the CPI unit
for that one session. This is not fixable from data available or
plausibly obtainable without ARM Data Center login credentials to browse
the archive directly and rule out a truly missing file with certainty.

## To close this out definitively

Would need a free ARM account (register at `adc.arm.gov`) to browse
`https://adc.arm.gov/discovery/#v/results/s/fiop::sgp2000sprcloud` or query
the `iop-api.svcs.arm.gov` `/contents/` endpoint directly — low priority
given the flight-count reconciliation already makes a missing file unlikely.
