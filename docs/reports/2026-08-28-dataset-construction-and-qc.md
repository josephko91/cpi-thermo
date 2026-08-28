# Dataset Construction and Quality Control — Source Material — 2026-08-28

Source material for a Nature Scientific Data paper's "Dataset construction
and quality control" methods subsection. Organized prose and tables for
drafting from, not a finished paper section — every claim cites the repo
file it comes from so it can be verified or expanded. Companion report:
`docs/reports/2026-08-28-data-harmonization.md` (standardization + final
tier construction).

## 1. Data sources and retrieval

Fifteen aircraft field campaigns, spanning 2000–2022, were retrieved from
three public archive systems: NASA's Earth Science Project Office (ESPO)
archive, NCAR's Earth Observing Laboratory (EOL) data archive, and the DOE
ARM (Atmospheric Radiation Measurement) Data Center. Each campaign's raw
files were downloaded directly from its official archive in whatever
native format that archive publishes (ICARTT `.ict`, UND Citation
FFI1001-derived text, NCAR-RAF/Nimbus NetCDF, or campaign-specific binary)
— no third-party or reprocessed intermediate source was used. Full
per-campaign source links: `parsers/README.md`.

| Campaign | Archive | Raw format | Date range | n_flight_days | Source |
|---|---|---|---|---:|---|
| ARM | ARM Data Center (SGP 2000 Spring Cloud) | Binary `.t4archive.gz` (Citation aircraft) | 2000-03-03 to 2000-03-21 | 11 | archive.arm.gov/data/sgp2000sprcloud/citation/ |
| AIRS-II | NCAR EOL Archive | NetCDF `.nc` (NCAR-RAF/Nimbus) | 2003-11-05 to 2003-12-06 | 16 | data.eol.ucar.edu/project/AIRS-II |
| ATTREX | NASA ESPO Archive | ICARTT `.ict` | 2011-10-28 to 2014-03-15 | 36 | espoarchive.nasa.gov/archive/browse/attrex |
| CRYSTAL-FACE-NASA | NASA ESPO Archive | Custom `.WB57` text | 2002-05-09 to 2002-07-31 | 20 | espoarchive.nasa.gov/archive/browse/crystal_face |
| CRYSTAL-FACE-UND | UND Atmos. Dept. archive | UND `.CIT` text | 2002-07-03 to 2002-07-29 | 13 | atmos.und.edu/und_cloud_aerosol/crystalface.html |
| ESCAPE | NASA ESPO Archive | ICARTT-like `.ict` (Learjet state) | 2022-06-02 to 2022-06-17 | 7 | espoarchive.nasa.gov/archive/browse/escape |
| ICE-L | NCAR EOL Archive | NetCDF `.PNI.nc` (NCAR-RAF/Nimbus) | 2007-11-07 to 2007-12-16 | 13 | data.eol.ucar.edu/dataset/346.1 |
| IPHEX | NASA ESPO Archive (GHRC) | Custom `.iphex` text | 2014-03-06 to 2014-06-13 | 24 | espoarchive.nasa.gov/archive/browse/iphex |
| ISDAC | ARM Data Center | Comma-delimited STRAPP bulk `.txt` | 2008-03-31 to 2008-04-30 | 19 | archive.arm.gov/data/isdac2008/strapp-convair_bulk/CommaDelimited/ |
| MACPEX | NASA ESPO Archive | ICARTT `.ict` | 2011-03-27 to 2011-04-26 | 17 | espoarchive.nasa.gov/archive/browse/macpex |
| MC3E | NASA ESPO Archive | Custom `.mc3e` text | 2011-04-22 to 2011-06-02 | 16 | espoarchive.nasa.gov/archive/browse/mc3e |
| MIDCIX | NASA ESPO Archive | Custom `.WB57`/`FP` text | 2004-04-17 to 2004-05-06 | 9 | espoarchive.nasa.gov/archive/browse/midcix |
| MPACE | ARM IOP Data Browser | Custom `.mpace` text (FFI1001) | 2004-09-30 to 2004-10-22 | 15 | iop.arm.gov/2004/nsa/mpace/poellot-citation |
| OLYMPEX | NASA ESPO Archive | Custom `.olympex` text | 2015-11-12 to 2015-12-19 | 17 | espoarchive.nasa.gov/archive/browse/olympex |
| POSIDON | NASA ESPO Archive | ICARTT `.ict` | 2016-09-13 to 2016-11-02 | 21 | espoarchive.nasa.gov/archive/browse/posidon |

Instrument-level provenance (PI/team, manufacturer where known) for every
water-vapor, temperature, position, and turbulence sensor behind these
campaigns is in `docs/reports/2026-08-28-instrument-inventory.md` (32
instruments cataloged).

## 2. Pre-processing pipeline

Each campaign has a dedicated parser (`parsers/<campaign>.py`) implementing
a `load_*()` (raw file → intermediate DataFrame) and `extract_*_standard()`
(intermediate → standardized schema) pair, orchestrated by `main.py`. Every
parser follows the same processing sequence:

1. **Raw file discovery and parsing** — glob the campaign's raw directory
   per its configured file pattern (`config.yaml`), parse the
   campaign-specific header/column format (fixed-width ICARTT, whitespace-
   delimited UND text, or NetCDF variable arrays).
2. **Column/unit standardization** — rename raw instrument fields to the
   standard schema, apply documented scale factors (e.g. ICARTT header
   scale-factor lines), and convert to standard units (Celsius, hPa,
   meters, g/kg — see harmonization report §1).
3. **Missing-value/sentinel masking** — replace campaign-specific fill
   values (a shared list in `parsers/utils.py::COMMON_NA_VALUES`, e.g.
   `-9999`, `-7777`, `9.9999E+30`, plus campaign-specific sentinels found
   during QC4 investigations) with NaN before any downstream calculation,
   never after (per-project convention, stated in CLAUDE.md).
4. **Physical-plausibility clipping** — applied uniformly in `main.py`
   after every parser returns: `Si`/`Si_*` clipped to [-1, 2] (values
   outside are physically impossible or near-certain instrument
   artifacts), `qv`/`qv_*` floored at 0 (negative mixing ratios are
   impossible) and set to NaN below it.
5. **Timestamp flooring** — every timestamp floored (not rounded) to the
   nearest whole second (`parsers/utils.py::round_timestamp_to_second`),
   the merge key for every cross-instrument join within a campaign (see
   harmonization report §3 for why floor rather than round).
6. **Per-campaign cross-instrument merge** — each campaign's own
   instrument files (temperature, pressure, humidity, position,
   turbulence) are combined via an outer merge on the floored timestamp —
   never a `merge_asof` tolerance (repo-wide policy since
   `docs/decisions/2026-07-07-exact-second-merge-rewrite.md`, GitHub issue
   #12).

### Campaign-specific fixes applied during pre-processing

Beyond the generic pipeline above, ~23 campaign-specific bugs/edge cases
were found and fixed during development, documented individually in
`docs/decisions/`. Grouped by category (full detail in the cited docs):

- **Unit-conversion bugs**: ATTREX altitude ×0.1 scale factor not applied
  (`2026-07-05-altitude-unit-bugs.md`), ICE-L altitude in feet not
  converted to meters (same doc), CRYSTAL-FACE-NASA wind U/V/W never
  scaled or sentinel-masked (`2026-07-13-turbulence-scope-reduction-diagnostics.md`
  — this single bug was QC4's entire 73,566-flag baseline), a latent
  off-by-one scale-factor bug in `load_mms_file` (`2026-07-05-cpi-fusion-and-remaining-fixes.md`
  commit history, now folded into the changelog).
- **Sensor-failure masking**: ARM cryo-hygrometer below-range/cloud-flooding
  readings (`2026-07-05-arm-cryo-masking.md`), ESCAPE temperature-sensor
  failure at altitude on the 2022-06-10 flight (`2026-07-05-escape-temp-sensor-failure.md`),
  OLYMPEX chilled-mirror physical-impossibility fault on one specific
  flight (`2026-07-05-qc9-iphex-olympex.md`), POSIDON pressure-sentinel
  bug (same doc).
- **Geolocation-source fixes**: CRYSTAL-FACE-NASA's geolocation loader
  pointed at the wrong raw subdirectory (0%→100% Alt_m), CRYSTAL-FACE-UND's
  navigation files never read plus a missing flight segment recovered
  (+15,320 rows), MACPEX/MIDCIX altitude recovered from the public NASA
  ESPO archive after initially being declared unavailable (all three:
  `2026-07-05-open-issues-resolved.md`).
- **Timestamp/timezone fixes**: MC3E's CPI image timestamps were recorded
  in CDT but labeled UTC — found and fixed, centralized in
  `parsers/cpi_timestamps.py` (`2026-07-05-cpi-fusion-gap-fixes.md`).
- **Architecture-level fix**: the entire pipeline's merge strategy was
  rewritten from `merge_asof(tolerance=...)` to exact-second joins
  repo-wide — grew the L0 row count substantially since a
  previously-silent tolerance had been fabricating time resolution the
  data didn't have (`2026-07-07-exact-second-merge-rewrite.md`).
- **Turbulence schema evolution**: wind/attitude/EDR columns were added
  (2026-07-13), then deliberately narrowed to just wind components + a
  single unified EDR column after further analysis
  (`docs/decisions/2026-07-13-edr-unification.md`,
  `docs/reports/2026-07-13-turbulence-scope-reduction-diagnostics.md`).

Full chronological log with row/coverage-percentage impact of every
dataset-affecting change: `docs/dataset-changelog.md`.

## 3. Quality control

Nine automated QC checks (`scripts/qa_checks.py`) run against every L0
build. Each check writes a CSV report; a `00_qaqc_summary.csv` aggregates
flag counts.

| # | Check | Method |
|---|---|---|
| QC1 | Physical range checks | Flags any value outside absolute hard bounds: `Tair_C` [-95, 60] °C, `P_hPa` [50, 1050] hPa, `Si` [-1, 2], `qv`/`qv_*` [0, 100] g/kg, `Alt_m` [-500, 25000] m, `Lat` [-90, 90], `Lon` [-180, 180]. Wind/EDR deliberately unbounded — storm-severity turbulence is legitimately unbounded, a hard bound would false-flag real severe-weather events. |
| QC2 | Internal consistency | Cross-validates physically coupled variables: `qv` against saturation, `Tair_C` against `Alt_m` (lapse-rate plausibility). Flags split into mild (in-cloud `qv` exceeding saturation by up to 1.05×, expected instrument physics) and severe tiers. |
| QC3 | Stuck-sensor / temporal continuity | Flags any of `Tair_C`, `P_hPa`, `Alt_m`, `qv`, `Si` frozen at an identical value for ≥30 consecutive readings (matching the 30 s QC threshold at 1 Hz), plus intra-file timestamp gaps >2 hours. |
| QC4 | Fill/sentinel value detection | Checks for un-converted fill/sentinel magnitudes (`-9999`, `-999`, `-8888`, `-7777`, `9999`, `99999`, `±1000`) surviving in the combined dataset — i.e. a masking step that should have caught them but didn't. |
| QC5 | Inter-instrument cross-validation | For campaigns with 2+ independent instruments measuring the same quantity (e.g. MACPEX's HWV/DLH/JLH hygrometers, ATTREX's DLH/NOAA/UCATS, IPHEX's chilled-mirror/Ophir TDL), compares simultaneous readings for systematic offset. |
| QC6 | Per-flight coverage audit | Characterizes data coverage (`Si`, `Tair_C`, `P_hPa`, `qv`, `Alt_m`) at the individual-flight-date level, catching whole-flight gaps a campaign-level aggregate would hide. |
| QC7 | Timestamp quality | Detects duplicate, near-duplicate, and out-of-order timestamps after the exact-second flooring. |
| QC8 | Vertical profile plausibility | Bins `Tair_C`/`qv` by pressure level (11 standard levels, 1050→0 hPa) and compares against the ICAO standard atmosphere temperature profile and saturation-vapor-pressure bounds. |
| QC9 | LWC cross-check (severe Si flags) | Re-reads raw liquid-water-content data for IPHEX/OLYMPEX's severe `Si > 1.05` flags (LWC isn't part of the standard schema) to distinguish real in-cloud/precipitation contamination from likely sensor error. |

**Latest results** (2026-08-28, reproduced exactly from current code — see
`docs/reports/2026-08-28-dataset-validation.md`):

| Check | Flags | % of dataset | Campaigns affected |
|---|---:|---:|---:|
| QC1 | 6 | 0.000% | 1 |
| QC2 | 80,648 | 1.764% | 12 |
| QC3 | 365 | 0.008% | 6 |
| QC4 | 0 | 0.000% | 0 |
| QC5 | 0 | 0.000% | 5 |
| QC6 | 67 | 0.000% | 10 |
| QC7 | 2 | 0.000% | 1 |
| QC8 | 6 | 0.000% | 3 |
| QC9 | 1,436 | 0.031% | 2 |

QC9's 1,436 flags are IPHEX/OLYMPEX cold-regime Si values — genuinely
ambiguous between chilled-mirror hysteresis and real cirrus supersaturation
near the homogeneous-freezing threshold; kept in the data, flagged not
masked, pending an independent instrument cross-check (see CLAUDE.md
"Known issues"). No other check has an unresolved flag category.

## 4. Uncertainty characterization

A dedicated survey (`docs/reports/2026-07-08-raw-data-uncertainty-metadata-survey.md`)
inspected every raw file's header/metadata plus companion documentation
(readmes, PDFs) for traceable numeric measurement uncertainty. Headline
finding: **uncertainty availability tracks the raw-file format family, not
instrument or campaign quality.**

- Every water-vapor instrument shipped as an ICARTT `.ict` file states a
  quantitative uncertainty (typically "X% of reading" or "X% + absolute
  floor") — ATTREX's 3 hygrometers, MACPEX's 3 hygrometers, POSIDON's DLH,
  CRYSTAL-FACE-NASA's Harvard Water instrument.
- Every MMS-branded meteorology file (ATTREX, MACPEX, POSIDON — same NASA
  Ames team) reports an identical ±0.3 hPa / ±0.3 K figure.
- Every UND Citation-format campaign (CRYSTAL-FACE-UND, IPHEX, MC3E,
  MPACE, OLYMPEX) has **no** uncertainty field in any instrument, ever —
  confirmed to be a property of that data provider's export format
  specification, not an omission by any one file.
- NetCDF (AIRS-II, ICE-L) and ARM's raw binary carry no uncertainty
  channel at all in the files as delivered.
- Two companion documents found by a directory-wide sweep substantially
  changed the picture: ESCAPE's `.ict` header defers to "contact PI," but
  a companion PDF gives a complete per-variable accuracy table (Rosemount
  temperature ±0.5°C, EdgeTech chilled-mirror dew point ±1°C, Aventech GPS
  ±10 m, RVSM altitude ±60 ft); ISDAC's companion readme gives a real
  position-uncertainty figure (~8 km typical).

Full per-campaign, per-instrument table (which raw files have a traceable
number and which don't, with exact citations): see the survey report and
`docs/reports/2026-08-28-instrument-inventory.csv`'s `accuracy` column.
