# 2026-07-06 — QC6 zero-Si flight-day investigation

## Investigation only — no fix applied (all genuine data limitations)

## What we found

QC6 flagged 37 flight-days with 0% valid `Si` across 8 campaigns.
MPACE (15 days), MIDCIX (2 days), and CRYSTAL-FACE-UND (3 days) were already
documented (`docs/dataset-changelog.md`, `docs/decisions/2026-07-06-cpi-fusion-report-cleanup.md`).
The remaining 5 campaigns — ARM (4), IPHEX (6), OLYMPEX (4), ATTREX (2),
MACPEX (1) — had no prior investigation. Checked each; **all are genuine
instrument/raw-data limitations, not parser bugs.**

**ARM (2000-03-03, 03-05, 03-19, 03-21) — cryo hygrometer below detection
floor for the entire flight.** Read the raw `.t4archive.gz` bytes directly
(bypassing the parser): `Frost_Point_Cryo_C` is -141 to -174°C across all
four full flight-days — far below the existing `CRYO_FLOOR = -80.0°C`
threshold documented in `docs/decisions/2026-07-05-arm-cryo-masking.md`.
The already-existing masking logic correctly nulls every row; these are
simply flights where the CEM sensor never came into range at all (as
opposed to only losing range briefly, which is the typical case elsewhere
in the campaign).

**IPHEX (2014-03-06, 04-16, 04-29, 05-09, 05-10, 05-12) and OLYMPEX
(2015-11-12, 11-13, 11-14, 12-02) — chilled-mirror/TDL not operating that
flight.** Read each flagged file directly: IPHEX's `FrostPoint` and
`MixingRatio` columns are the fill value `999999.9999` for literally every
row of each flagged file (e.g. all 4,352 rows of `2014_03_06_17_45_37.iphex`);
OLYMPEX's `FrostPoint` is the same fill value for all 11,900 rows of
`15_11_12_18_49_13.olympex`. This matches the campaigns' documented
instrument coverage (IPHEX: chilled-mirror on 24/32 flights, Ophir TDL on
21/32; OLYMPEX: frost-point sensor not always operating) — these are simply
among the flights without it.

**ATTREX (2013-12-21, 2014-03-15) — no water-vapor raw file covers that
window.** 2013-12-21's flagged rows (00:00-00:57 UTC) are the tail end of a
single flight that took off 2013-12-20 and crossed midnight UTC — Tair_C is
72% valid there (from `MMS`), but all three H2O instrument columns
(`Si_DLH`/`Si_NOAA`/`Si_UCATS`) are 0% valid because `UCATS-H2O`'s own raw
file ends at 23:33:42 UTC (its last data row), well before `MMS`'s 00:57:52
UTC end, and no `DLH-H2O`/`NOAA-H2O` files exist for that flight at all.
2014-03-15 is the campaign's last flight day: no `DLH-H2O`, `NOAA-H2O`, or
`UCATS-H2O` raw file exists for that date at all (checked
`data/raw/ATTREX/*/` directly) — likely a return/ferry leg.

**MACPEX (2011-03-27) — first flight day, no water-vapor instrument
deployed.** Only `MMS-FlightPath` and `MMS-MetData` raw files exist for this
date (checked `data/raw/MACPEX/`) — no `HWV`, `DLH`, or `JLH` file at all.
Consistent with a ferry/test flight before science instruments came online
(campaign span is 2011-03-27 to 04-26, so this is day one).

## Fix details

None. Every case traces to the raw data itself (instrument not operating,
sensor out of range, or no file for that window) rather than a parsing
defect — masking or "fixing" any of these would mean fabricating Si values
that were never measured.

## How to detect

For a flagged zero-Si flight-day: check whether *any* Si instrument column
(not just the resolved `Si`) has valid data that day; if all are zero, read
the raw file(s) for that date directly and check whether the water-vapor
column(s) are entirely a known fill value, or whether a raw file exists for
that instrument/date combination at all before suspecting a parser bug.

## What remains

None actionable — these are genuine, permanent raw-data limitations. Not
tracked as new open issues in `CLAUDE.md` given they don't represent
anything fixable, consistent with how MPACE/MIDCIX/CRYSTAL-FACE-UND's
equivalent gaps are already handled.
