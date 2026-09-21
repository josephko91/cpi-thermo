# Navigation / GPS Position Accuracy Verification — 2026-08-31

What navigation/GPS/INS hardware actually produced each campaign's
`Lat`/`Lon`/`Alt_m` columns, and what accuracy — if any — is documented
for it, either in this dataset's own raw archive or in external
literature for the identified system. Companion to
`docs/reports/2026-08-28-instrument-inventory.md` (which covers every
instrument, position included, at lower resolution) and reuses
`docs/reports/2026-07-08-raw-data-uncertainty-metadata-survey.md`'s
uncertainty-presence findings directly rather than re-deriving them.

## Method

Two sources, combined:

1. **This repo's raw archive** — for the 9 campaigns whose nav-system
   identity wasn't yet established this session (ARM, AIRS-II, ATTREX,
   ICE-L, IPHEX, MC3E, MIDCIX, OLYMPEX, POSIDON), representative raw
   files and companion documentation were read directly this turn,
   specifically searching for an instrument/manufacturer name near the
   Lat/Lon/Alt fields (not just an uncertainty number, which the existing
   survey already covers). The other 6 campaigns' nav-system identity was
   already established earlier this session (ESCAPE, ISDAC, MACPEX,
   CRYSTAL-FACE-NASA, CRYSTAL-FACE-UND, MPACE).
2. **Web search** — for the systems newly identified by name this turn
   (Applanix POS, Litton LN-100G, C-MIGITS-II), typical published
   accuracy specifications were looked up. Every externally-sourced
   figure is marked **external** below; it describes the general
   product/system, not a measurement independently confirmed for this
   dataset's specific deployment.

## Key finding: 4 named hardware clusters, not 15 independent systems

Despite spanning 15 separate field campaigns and multiple aircraft
(WB-57, UND Citation II, NSF/NCAR C-130, NRC Convair-580, Learjet, Global
Hawk), the actual navigation hardware behind this dataset clusters into
just a few reused systems:

- **Applanix POS (5 campaigns)**: CRYSTAL-FACE-UND, IPHEX, MC3E, MPACE,
  OLYMPEX — all UND Citation II campaigns, all explicitly name "the
  Applanix Position and Orientation System (POS)" in their raw file
  headers (confirmed by direct raw-header read this turn for IPHEX, MC3E,
  and OLYMPEX; CRYSTAL-FACE-UND and MPACE were already known from parser
  comments). Same aircraft/instrumentation team (Poellot/Delene, UND)
  across all 5, consistent with the existing uncertainty survey's finding
  that these 5 share one raw-format family.
- **Litton LN-100G (WB-57 MMS platform, 3+ campaigns)**: ATTREX's raw
  header explicitly states `G_LAT`/`G_LONG`/`G_ALT` are "GPS Latitude/
  Longitude/Altitude from LN100g INS." MIDCIX's raw `FP*.WB57` files name
  "LITTON 100-G INU" directly as the primary system, with a documented
  fallback to "C-MIGITS-II INU" during two partial failures. POSIDON's
  outage-period files reference a "CMIGITS INU" backup in the same role
  (primary system not separately named in POSIDON's files, but the
  platform — WB-57 MMS — and fallback hardware match MIDCIX exactly).
  MACPEX also flies WB-57 MMS (MMS-FlightPath file, same instrument
  family/PI team per the existing uncertainty survey) and CRYSTAL-FACE-
  NASA's `NP` file is WB-57 navigational data too — both are **plausibly**
  the same LN-100G lineage given the shared platform, but neither file
  independently names "LN100g" the way ATTREX's does, so this link is
  noted as plausible, not confirmed.
- **NovaTel GPS + Litton INS fallback (ISDAC)**: established earlier this
  session — primary GPS lost lock often enough that a Litton INS dead-
  reckoning fallback (interpolated across gaps) was needed.
- **Aventech AIMMS-20 + West Star RVSM altimetry (ESCAPE)**: established
  earlier this session — the dataset's single most completely documented
  navigation system.
- **Unnamed generic "GPS"/"IRS" (3 campaigns)**: ARM, AIRS-II, ICE-L. All
  three raw archives use only generic labels ("GPS Latitude," "Inertial
  Latitude," "GPS-Corrected Inertial Latitude") with no manufacturer or
  model named anywhere in the data file, variable attributes, or
  companion documentation checked.

## Per-campaign table

| Campaign | Nav system (as named in raw archive) | Horizontal accuracy | Vertical accuracy | Source |
|---|---|---|---|---|
| ARM | Unnamed on-board GPS + INS (`GPS_Lat_deg`/`GPS_Lon_deg`/`GPS_Alt_m`, fallback `INS_Latitude_deg`/`INS_Longitude_deg`) | Not documented | Not documented | `parsers/arm.py:47-48`; `data/raw/ARM/poellot-citation-t4-readme.txt` (checked this turn, no instrument name or accuracy found) |
| AIRS-II | Unnamed "IRS"/"GPS" (`LAT`/`LON` primary, `LATC`/`LONC` GPS-corrected fallback, `GGLAT`/`GGLON` pure GPS; altitude `ALT`/`GGALT`) | Not documented | Not documented | `parsers/airs_ii.py:93-118`; netCDF variable `long_name` attributes + `README-info-201_001.txt` (checked this turn — Instruments list has no nav hardware entry) |
| ATTREX | **Litton LN-100G INS** (raw header: "GPS Latitude/Longitude/Altitude from LN100g INS") | Not documented in this dataset's raw archive | Not documented in this dataset's raw archive | `data/raw/ATTREX/MMS/*.ict` (checked this turn); P/T/wind ARE documented for this same MMS file (±0.3 hPa/±0.3K/±1.0 m/s, per the existing uncertainty survey) but position is not |
| CRYSTAL-FACE-NASA | "INU + GPS" (`NP` file) — plausibly the same WB-57 LN-100G family as ATTREX/MIDCIX/POSIDON, not independently confirmed | Explicitly disclaimed: raw header states **"unknown accuracy, and may not be adequate for precise calculations"** | Same disclaimer | `docs/reports/2026-07-08-raw-data-uncertainty-metadata-survey.md` §1 (CRYSTAL-FACE-NASA/NP row) |
| CRYSTAL-FACE-UND | **Applanix POS** (`POS_Lat`/`POS_Lon`/`POS_Alt`) | Not documented in this dataset's raw archive | Not documented in this dataset's raw archive | `parsers/crystal_face_und.py:112`; see Applanix general spec below |
| ESCAPE | **Aventech AIMMS-20** (position) + **West Star Aviation RVSM altimetry** (altitude) | **±10 m** | **±60 ft (18.3 m)** | `data/raw/ESCAPE/spec-learjet-state/ESCAPE_SPEC_ReadMe_R0_10.2022.pdf` §2.0, confirmed earlier this session |
| ICE-L | Unnamed "IRS"/"GPS" (`LAT`/`LATC`/`GGLAT`; altitude `GGALT`/`ALT`/`PALT`) | Not documented | Not documented | `parsers/ice_l.py:240-243`; netCDF variable attributes + `README-info-105_004.txt` (checked this turn — bare "GPS Receivers" entry, no manufacturer) |
| IPHEX | **Applanix POS** — raw header confirms: "Aircraft latitude/longitude/altitude from the Applanix Position and Orientation System (POS)" | Not documented in this dataset's raw archive | Not documented in this dataset's raw archive | `data/raw/IPHEX/*.iphex` (checked this turn, raw header text) |
| ISDAC | **NovaTel GPS**, normal operation; **Litton INS dead-reckoning fallback** during GPS-outage periods | Not documented (normal operation); **up to ~8 km** (worse, ~10 km on some flights) **during GPS-outage/INS-fallback periods only** | GPS altitude channel (`ALT_G`) explicitly flagged "very noisy" by the raw readme; barometric alternative recommended instead | `data/raw/ISDAC/strapp-convair_bulk/ISDAC_BulkData_v3_README.rtf`, confirmed and corrected earlier this session |
| MACPEX | MMS-FlightPath (WB-57 platform, plausibly Litton LN-100G family — not independently named in this file) | Not documented | **±2.5 m (sea level) to ±35 m (20 km)**, altitude-dependent | `data/raw/MACPEX/MMS-FlightPath/*.ict`, per the existing uncertainty survey — Lat/Lon uncertainty explicitly absent despite altitude being given |
| MC3E | **Applanix POS** — raw header confirms: "Aircraft Latitude/Longitude/Altitude from the Applanix POS System" | Not documented in this dataset's raw archive | Not documented in this dataset's raw archive | `data/raw/MC3E/*.mc3e` (checked this turn, raw header text) |
| MIDCIX | **Litton 100-G INU**, primary; **C-MIGITS-II INU** fallback during 2 documented partial failures | Not documented in this dataset's raw archive for either system | Not documented in this dataset's raw archive for either system | `data/raw/MIDCIX/FP/*.WB57` (checked this turn) — explicit outage comments name both systems and the exact UT-second ranges affected |
| MPACE | **Applanix POS** (`POS_Lat`/`POS_Lon`/`POS_Alt`) | Not documented in this dataset's raw archive | Not documented in this dataset's raw archive | `parsers/mpace.py:25`; see Applanix general spec below |
| OLYMPEX | **Applanix POS** — raw header confirms: "Aircraft latitude/longitude/altitude from the Applanix Position and Orientation System (POS)" | Not documented in this dataset's raw archive | Not documented in this dataset's raw archive | `data/raw/OLYMPEX/*.olympex` (checked this turn, raw header text) |
| POSIDON | Unnamed GPS, primary (`G_LAT`/`G_LONG`/`G_ALT`, no "LN100g" qualifier in this campaign's files); **C-MIGITS INU** fallback during documented GPS/INU failures | Not documented for either system | Not documented for either system | `data/raw/POSIDON/MMS/*.ict` (checked this turn) — outage-period files state "GPS position data unavailable due to failure of CMIGITS INU" |

## External accuracy context for the newly-identified systems

These describe the general product/system as documented in manufacturer
literature or published technical papers — **not** independently
confirmed for this dataset's specific deployments, since none of the
raw archives above give a position-accuracy number for these systems.

- **Applanix POS (5 campaigns: CRYSTAL-FACE-UND, IPHEX, MC3E, MPACE,
  OLYMPEX)** — the POS AV product line's own specifications (from an
  earlier search this session) report **~5–15 cm** horizontal with
  carrier-phase differential GPS and post-processing, or **~1–5 m** in
  simpler real-time/non-differential configurations. Which configuration
  was actually used on these specific UND Citation deployments is not
  stated anywhere in the raw archive.
- **Litton LN-100G INS/GPS (ATTREX; plausibly the WB-57 MMS platform
  more broadly)** — manufacturer literature (Northrop Grumman, successor
  to Litton) states a **0.8 nautical-mile-per-hour free-inertial
  (GPS-denied) drift rate** — a high-end, ring-laser-gyro system,
  substantially better than older-generation aircraft INS. This is the
  GPS-*denied* dead-reckoning performance; under normal GPS-aided
  operation, accuracy would be expected to track standard GPS
  performance rather than this drift figure. **[Sources: Northrop
  Grumman LN-100G datasheet; GlobalSpec product page]**
- **C-MIGITS-II INU (fallback unit on MIDCIX and POSIDON)** — published
  specifications report **2.5 m horizontal, 3 m vertical** accuracy — a
  genuinely precise GPS-aided system in its own right, meaningfully
  different from ISDAC's situation, where the fallback was pure
  INS-only dead-reckoning with no independent GPS source (hence ISDAC's
  much larger ~8 km outage-period error). If the C-MIGITS-II's own GPS
  also lost lock, its unaided performance would be expected to degrade
  further (its documented gyro bias, 30°/hr, is substantially worse than
  the LN-100G's), but this scenario is not documented as having occurred
  in either MIDCIX's or POSIDON's raw archive. **[Source: published
  C-MIGITS-II technical specification, cited in airborne-testing
  literature]**

## Notable secondary finding (pipeline behavior, not accuracy)

POSIDON's parser (`parsers/posidon.py`) reads position exclusively from
`MMS-1HZ_G_LAT`/`G_LONG`/`G_ALT`. Of 5 POSIDON files sampled this turn, 4
— all during documented "GPS position data unavailable due to failure of
CMIGITS INU" periods — use a **different column set** (`LAT`/`LONG`/
`PALT`, non-GPS-labeled) instead. The current parser does not read this
fallback column set, meaning POSIDON's `Lat`/`Lon`/`Alt_m` are likely NaN
for at least some real time ranges where *a* position value (from
`LAT`/`LONG`/`PALT`) does exist in the raw file but isn't being
extracted. This is a coverage question, not an accuracy question, and is
noted here only because it surfaced during this investigation — no fix
attempted, out of scope for this report.

## Caveats

- "Not documented" means checked directly in the raw file/companion docs
  this turn (or, for the 6 campaigns established earlier this session,
  in a prior direct check) and genuinely absent — not that the original
  instrument team never characterized their system's accuracy, only that
  this archive doesn't carry that figure forward.
- The Applanix/LN-100G/C-MIGITS-II external figures above describe
  product-line specifications, not a deployment-specific calibration
  record for these campaigns. A reviewer should not cite these as this
  dataset's confirmed position accuracy — only as reasonable context for
  what these named systems are typically capable of.
- This report's per-campaign accuracy findings for non-position variables
  (temperature, pressure, water vapor) are already covered by
  `docs/reports/2026-08-28-instrument-inventory.md` and
  `docs/reports/2026-07-08-raw-data-uncertainty-metadata-survey.md` — not
  repeated here.
