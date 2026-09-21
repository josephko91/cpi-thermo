# Water-Vapor Instrument PI / Group Provenance — 2026-09-21

Per-campaign roster of the water-vapor instruments behind `qv`, `Sw`, and `Si`,
with the PI or research group responsible for each. Companion to
`latex/table-wv-instrument-pi-2026-09-21.tex`, and the water-vapor counterpart
to `docs/reports/2026-09-21-cpi-instrument-provenance.md`.

Written for the same reviewer request that prompted the CPI table: *"Different
groups have different methods for calibrating and operating these instruments,
so this could be relevant."*

This report is about **attribution**. For instrument *specifications* (accuracy,
response time, measurement principle) see
`latex/table-wv-instrument-specs-2026-08-31.tex` and
`docs/reports/2026-08-28-instrument-inventory.md`; for *which* instrument won at
each timestamp see `config.yaml`'s `h2o_ranking` and the manuscript's
`tab:h2o-ranking`.

## Method

Raw headers in this repo first, campaign archive documentation second, web
search only where neither resolves it. Most rows needed nothing beyond the raw
archive: ICARTT `.ict` files carry a PI name, affiliation, and
`PI_CONTACT_INFO` in their first lines, and the UND FFI1001-derived formats
(`.CIT`/`.iphex`/`.mc3e`/`.mpace`/`.olympex`) carry PI and institution on header
lines 2–3. Unresolved cells read "Unknown" rather than a plausible guess.

Scope is the full `h2o_ranking` roster (all 15 campaigns, MPACE included as
"none") **plus** water-vapor instruments present in `data/raw/` that the
pipeline does not use, marked "archive only" — these are listed because they
are part of each campaign's water-vapor measurement context even though they
contribute no dataset column.

## Ranked instruments (appear in `tab:h2o-ranking`)

| Campaign | Instrument | Rank | PI / responsible group | Source | Confidence |
|---|---|---|---|---|---|
| ARM | Chilled mirror | 1 | M. Poellot — University of North Dakota, Dept. of Atmospheric Sciences | `data/raw/ARM/readme.txt` (ARM stream `poellot-citation`); `data/raw/ARM/poellot-citation-t4-readme.txt` (UND Dept. of Atmospheric Sciences letterhead) | Group confirmed; the specific chilled-mirror unit is unresolved (ARM's raw archive here is binary with no text metadata) |
| AIRS-II | Chilled mirror | 1 | NSF/NCAR Research Aviation Facility; campaign PI J. Hallett (Desert Research Institute) | `parsers/airs_ii.py:11-16` (DPBC/DPTC dual sensors); [RAF AIRS-II instrument list](https://archive.eol.ucar.edu/raf/Projects/AIRS-II/inst.html) ("General Eastern, Model 1011B Dew Point Hygrometer", RAF-supplied) | Confirmed as RAF-supplied instrumentation; no individual instrument PI is named |
| ATTREX | DLH | 1 | G. S. Diskin — NASA Langley Research Center | `data/raw/ATTREX/DLH-H2O/*.ict` header lines 2–3 + `PI_CONTACT_INFO` | Confirmed |
| ATTREX | NOAA | 2 | T. D. Thornberry, A. W. Rollins — NOAA ESRL Chemical Sciences Division | `data/raw/ATTREX/NOAA-H2O/*.ict` header | Confirmed |
| ATTREX | UCATS | 3 | E. Hintsa, F. Moore, D. Nance, G. Dutton, B. Hall, J. Elkins — CIRES/University of Colorado and NOAA ESRL GMD | `data/raw/ATTREX/UCATS-H2O/*.ict` header | Confirmed |
| CRYSTAL-FACE-NASA | JLH | 1 | R. Herman — NASA JPL | `data/raw/CRYSTAL-FACE-NASA/JW/JW*.WB57` header lines 2–3 | Confirmed |
| CRYSTAL-FACE-NASA | HW | 2 | E. M. Weinstock, J. Vellovic, J. B. Smith, D. S. Sayres, J. G. Anderson — Harvard University | `data/raw/CRYSTAL-FACE-NASA/HW/HW*.WB57` header lines 2–3 | Confirmed |
| CRYSTAL-FACE-NASA | ALIAS | 3 | C. R. Webster, G. J. Flesch — NASA JPL | `data/raw/CRYSTAL-FACE-NASA/ALIAS/AL*.WB57` header lines 2–3 | Confirmed |
| CRYSTAL-FACE-UND | Lyman-alpha (unspec.) | 1 | M. Poellot — University of North Dakota | `data/raw/CRYSTAL-FACE-UND/ND_MIS/ND*__MIS.CIT` header lines 2–3 | Group confirmed; instrument identity explicitly unresolved in `parsers/crystal_face_und.py:8-9` ("potentially JLH?") |
| CRYSTAL-FACE-UND | Chilled mirror | 2 | M. Poellot — University of North Dakota | same header (ND_MET companion file) | Group confirmed; unit unresolved |
| ESCAPE | Chilled mirror | 1 | P. Lawson — SPEC Inc. (EdgeTech C-137 unit on the SPEC Learjet) | `data/raw/ESCAPE/spec-learjet-state/*/ESCAPE-Page0_Learjet_*.ict` header lines 2–3; unit identified in `ESCAPE_SPEC_ReadMe_R0_10.2022.pdf` | Confirmed |
| ICE-L | Chilled mirror | 1 | NSF/NCAR Research Aviation Facility (RAF T-Electric unit) | `parsers/ice_l.py:15-19`; NetCDF variable metadata ("Dew Point Temperature, T-Electric Top/Bottom") | Confirmed as RAF instrumentation; no individual PI named |
| ICE-L | MRTDL | 2 | Instrument design attributed to Maycomm (R. D. May); campaign operating group **Unknown** | `parsers/ice_l.py:21-30`; `docs/reports/2026-08-28-instrument-inventory.csv` (MRTDL row) | Manufacturer attribution web-confirmed previously; the "AC19-400" model number and the ICE-L operator are both unconfirmed |
| IPHEX | Chilled mirror | 1 | D. Delene — University of North Dakota | `data/raw/IPHEX/*.iphex` header lines 2–3; variable description line 22 ("Dewpoint Temperature from EG&G Probe") | Archive PI confirmed; the EG&G probe identification is the repo's existing moderate-confidence read |
| IPHEX | Ophir TDL | 2 | Ophir Corporation instrument, archived by UND (D. Delene); operating group **Unknown** | `data/raw/IPHEX/*.iphex` header line 5 ("Ophir") | Instrument origin confirmed in the raw header; no operator credited |
| ISDAC | Chilled mirror | 1 | J. W. Strapp (ARM stream `strapp-convair_bulk`); platform operated by NRC Canada, Convair-580 PI M. Wolde | `data/raw/ISDAC/strapp-convair_bulk/` (ARM PI-lastname stream convention, cf. `poellot-citation`, `heymsfield-pms`, `lawson-learjet` in `data/raw/ARM/readme.txt`); `data/raw/ISDAC/wolde-convair/`; `ISDAC_BulkData_v3_README.rtf` (IAR/NRC provenance); `Si` derived from the `EGGDew`/`ReHuI` columns | Stream-to-PI mapping follows ARM's documented naming convention rather than an explicit statement in the file itself |
| MACPEX | HWV | 1 | D. S. Sayres, J. B. Smith, M. R. Sargent, J. G. Anderson — Harvard University | `data/raw/MACPEX/HWV/*.ict` header | Confirmed |
| MACPEX | DLH | 2 | G. S. Diskin — NASA Langley Research Center | `data/raw/MACPEX/DLH/*.ict` header | Confirmed |
| MACPEX | JLH | 3 | R. L. Herman — NASA JPL | `data/raw/MACPEX/JLH/*.ict` header | Confirmed |
| MACPEX | FISH | 4 | Forschungszentrum Jülich; PI **Unknown** | declared in `config.yaml`'s MACPEX ranking; no FISH directory exists under `data/raw/MACPEX/` and the parser never loads it (always NaN) | Institution is the instrument's known home; no campaign PI traceable in this archive |
| MC3E | DLH (label — see note below) | 1 | M. Poellot — University of North Dakota | `data/raw/MC3E/*.mc3e` header lines 2–3; variable description lines 23–25 ("… from the Laser Hygrometer"); `parsers/mc3e.py:68-71` | Citation archive PI confirmed; the "DLH" label itself is **not** confirmed as a NASA Langley DLH deployment |
| MIDCIX | JLH | 1 | R. Herman and R. Troy — NASA JPL | `data/raw/MidCix/JW*.WB57` header lines 2–3 | Confirmed |
| MPACE | — | — | No water-vapor instrument flown | no humidity/dew-point/frost-point column in `data/raw/MPACE/*.mpace` (verified); documented in `config.yaml` | Confirmed |
| OLYMPEX | Frost point | 1 | D. Delene — University of North Dakota | `data/raw/OLYMPEX/*.olympex` header lines 2–3 | Archive PI confirmed; instrument model explicitly unresolved in `parsers/olympex.py:9` |
| POSIDON | DLH | 1 | G. S. Diskin — NASA Langley Research Center | `data/raw/POSIDON/DLH-H2O/*.ict` header | Confirmed |

## Archive-only water-vapor instruments (not used for qv/Si)

| Campaign | Instrument | PI / responsible group | Source |
|---|---|---|---|
| AIRS-II | SpectraSensors open-path TDL hygrometer (`TDL Humidity High/Low Value`) | NSF/NCAR Research Aviation Facility | `data/raw/AIRS-II/variable_descriptions.txt`; [RAF AIRS-II instrument list](https://archive.eol.ucar.edu/raf/Projects/AIRS-II/inst.html) ("SpectraSensors Open-Path TDL Hygrometer … (MRLH)") |
| ISDAC | LiCor frost/dew-point cross-check unit (`LicDew`/`LicFro`) | **Unknown** — manufacturer inferred from column naming only | `parsers/isdac.py:17`; instrument inventory CSV (LI-COR attribution not independently confirmed) |
| MACPEX | CLH (closed-path laser hygrometer, enhanced total water) | L. Avallone — LASP, University of Colorado Boulder | `data/raw/MACPEX/CLH/*.ict` header |
| MACPEX | ULH (UAS Laser Hygrometer) | R. L. Herman — NASA JPL | `data/raw/MACPEX/ULH/*.ict` header |
| MACPEX | CIMS-H2O (NOAA water vapor CIMS) | T. Thornberry, A. W. Rollins, R.-S. Gao, D. W. Fahey, L. A. Watts — NOAA ESRL Chemical Sciences Division | `data/raw/MACPEX/CIMS/*.ict` header |
| MC3E | EG&G dew-point probe (`DEWPT`) | M. Poellot — University of North Dakota | `data/raw/MC3E/*.mc3e` variable description line 22 |
| POSIDON | NOAA-H2O | T. D. Thornberry — NOAA ESRL Chemical Sciences Division | `data/raw/POSIDON/NOAA-H2O/*.ict` header |

## Notes

- **MC3E's "DLH" is a ranking key, not a verified instrument attribution.**
  Found while sourcing this table. `parsers/mc3e.py:68-71` derives `Si_DLH` from
  the UND Citation file's `FrostPoint` column, whose header text — "Frost Point
  Temperature from the Laser Hygrometer" — is byte-for-byte the same wording as
  IPHEX's, where the repo labels that same instrument the Ophir TDL (and where
  the header's organization line reads "Ophir"). MC3E's header gives no
  instrument manufacturer. There is no NASA Langley DLH `.ict` file anywhere
  under `data/raw/MC3E/`. The instrument inventory already flagged this
  ("MC3E … no raw MC3E .ict file inspected -- MC3E uses UND format, not
  ICARTT"). Consequence for attribution: the MC3E row is credited to the UND
  Citation PI, **not** to G. S. Diskin, and the table carries a footnote saying
  so. Renaming the `config.yaml` ranking key would change pipeline behavior and
  is deliberately out of scope here.
- **Four categories of responsible party**, which is the substance of the
  reviewer's point:
  1. *Agency laboratory groups* flying their own instrument — NASA Langley
     (DLH), NASA JPL (JLH, ULH, ALIAS), NOAA ESRL (NOAA-H2O, CIMS, UCATS with
     CIRES), Harvard (HW, HWV). These are the WB-57/Global Hawk campaigns, and
     they are also the rows with quantitative uncertainty figures in their raw
     headers.
  2. *Aircraft-facility instrumentation* — NSF/NCAR RAF's dew-point sensors on
     the C-130 (AIRS-II, ICE-L), NRC Canada's Convair-580 suite (ISDAC). The
     instrument belongs to the platform, not to a science PI.
  3. *University-operated suites* — UND's Citation (ARM, CRYSTAL-FACE-UND,
     IPHEX, MC3E, OLYMPEX) under M. Poellot or D. Delene depending on year;
     CU/LASP's CLH on MACPEX.
  4. *Manufacturer-operated* — SPEC Inc. on the ESCAPE Learjet; the Ophir
     instrument on IPHEX (operator uncredited).
  Note the same instrument can change hands across campaigns (JLH: Herman alone
  on CRYSTAL-FACE-NASA and MACPEX, Herman and Troy on MIDCIX), and one aircraft
  suite can change PI across campaigns (UND Citation: Poellot through MC3E,
  Delene for IPHEX and OLYMPEX).
- **Four "Unknown" cells**: the ICE-L MRTDL operator, the IPHEX Ophir TDL
  operator, MACPEX's FISH PI, and the ISDAC LiCor unit. In each case an
  organization is known or inferable but no responsible person is credited in
  any source checked; none is filled by inference.
- **Coverage relative to the ranking table**: every campaign/instrument pair in
  `tab:h2o-ranking` appears here exactly once, with the same spelling and the
  same order, plus seven archive-only rows that deliberately have no counterpart
  there.
