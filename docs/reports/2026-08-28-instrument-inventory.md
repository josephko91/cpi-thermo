# Instrument Inventory — 2026-08-28

Every distinct physical measurement instrument behind this dataset's
variables, across all 15 campaigns. Companion machine-readable table:
`docs/reports/2026-08-28-instrument-inventory.csv` (identical content, one
row per instrument).

## Method

Three sources, combined:

1. **Codebase survey** — `parsers/*.py` (comments/docstrings naming the
   physical instrument behind each `Si_X`/`qv_X`/`Wind_*`/`EDR_m23s1`
   column), `config.yaml`'s `standardized_columns` comments (per-column
   campaign lists), `parsers/README.md`, and `docs/decisions/*.md` /
   `docs/reports/*.md` for instrument names already documented from prior
   investigations.
2. **Raw file headers** — representative ICARTT (`.ict`), UND FFI1001-derived
   (`.CIT`/`.iphex`/`.mc3e`/`.mpace`/`.olympex`), and NetCDF (NCAR-RAF/Nimbus)
   files under `data/raw/<campaign>/`, read directly for PI name, instrument
   name, and (where present) sample rate.
3. **`docs/reports/2026-07-08-raw-data-uncertainty-metadata-survey.md`** —
   an existing, rigorous per-campaign survey of exactly which raw files
   carry a traceable numeric uncertainty value (written to prep this
   dataset's Technical Validation section for a Nature Scientific Data
   submission). This report is the authoritative accuracy source here and
   is cited directly rather than re-derived.
4. **Web search** — for a handful of instruments whose manufacturer isn't
   named anywhere in the repo (Ophir TDL, Maycomm AC19-400, FISH). Every
   externally-sourced fact is marked as such in the table; nothing is
   guessed or fabricated. Where an instrument's identity is genuinely
   unresolved in the repo itself (e.g. CRYSTAL-FACE-UND's "potentially
   JLH?" Lyman-alpha hygrometer, OLYMPEX's frost-point sensor), it stays
   marked "unresolved" rather than assigned a guessed manufacturer.

**32 distinct instruments** identified across the 15 campaigns.

## Key findings

- **Accuracy traceability tracks raw-file format, not instrument quality**
  (the uncertainty survey's headline finding, reused here): every
  water-vapor instrument shipped as an ICARTT `.ict` file states a
  quantitative uncertainty (DLH, NOAA-H2O, UCATS-H2O, MACPEX's JLH/HWV,
  CRYSTAL-FACE-NASA's HW); every UND Citation-format campaign
  (CRYSTAL-FACE-UND, IPHEX, MC3E, MPACE, OLYMPEX) and NetCDF/binary format
  (AIRS-II, ICE-L, ARM) has **no** uncertainty field in the file at all,
  regardless of which physical instrument is behind it.
- **MMS is the single most-reused instrument system**: identical
  ±0.3 hPa / ±0.3 K accuracy figures appear verbatim across ATTREX,
  MACPEX, and POSIDON's raw headers — same NASA Ames team/PI (T. Paul Bui)
  across three separate campaigns spanning 2011–2016.
- **ESCAPE has the single most complete instrument accuracy table** in
  this dataset — a companion PDF (not the `.ict` file itself, which just
  says "Please contact PI") gives per-variable figures for temperature,
  dew point, position, and altitude, each citing peer-reviewed instrument
  literature (e.g. Rosemount probes citing Lawson and Cooper 1990).
- **6 instrument manufacturers/labs are unresolved** even after this
  survey: the ARM chilled-mirror unit (binary archive has no text
  metadata at all), IPHEX's chilled-mirror probe (likely an EG&G-style
  probe based on the raw header, not confirmed), CRYSTAL-FACE-UND's
  chilled mirror and Lyman-alpha hygrometer, OLYMPEX's frost-point
  sensor, and ISDAC's on-board RH-ice sensor. All are marked "unresolved"
  rather than guessed.
- **`Si_FISH` and `Si_ALIAS` are effectively unused** — FISH is declared
  in `config.yaml` for MACPEX but never wired to a raw data source
  (always NaN in the output parquet); ALIAS has no documented accuracy
  anywhere in the repo despite being loaded for CRYSTAL-FACE-NASA.
- **The CPI (Cloud Particle Imager) itself is included** as an instrument,
  even though it feeds no thermodynamic column — it's the source of the
  `cpi_filename` join key behind every row in L1/L2, and is arguably the
  dataset's most consequential single instrument.

## Full inventory

| Instrument | Manufacturer / Lab | Measures | Column(s) | Accuracy | Response time / rate | Campaign(s) |
|---|---|---|---|---|---|---|
| MMS (Meteorological Measurement System) | NASA Ames Research Center (PI: T. Paul Bui) | Air temp, static pressure, 3-D wind, EDR, (MACPEX) position/altitude/TAS | Tair_C, P_hPa, Wind_U/V/W_ms, EDR_m23s1, Lat/Lon/Alt_m (MACPEX only) | P/T: ±0.3 hPa, ±0.3 K. Wind: ±1.0 m/s (1σ). MACPEX alt: ±2.5–35 m. MACPEX TAS: ±1 m/s. | 1 Hz (desampled from 20 Hz, MACPEX) | ATTREX, POSIDON, MACPEX, CRYSTAL-FACE-NASA (Tair_C/P_hPa only, no uncertainty) |
| Diode Laser Hygrometer (DLH) | NASA Langley (PI: Glenn S. Diskin) | Water vapor mixing ratio | Si_DLH, qv_dlh, H2O_DLH_ppmv | ATTREX/POSIDON: 5% or 1 ppmv. MACPEX: 10%. | 1 Hz | ATTREX, MACPEX, POSIDON, MC3E |
| NOAA-H2O | NOAA (ESRL/Chemical Sciences) | Water vapor mixing ratio | Si_NOAA, qv_noaa, H2O_NOAA_ppmv | ±(5%+0.23 ppm) [WV], ±(6%+0.35 ppm) [eTW] | 1 Hz | ATTREX |
| UCATS-H2O | NOAA | Water vapor mixing ratio | Si_UCATS, qv_ucats, H2O_UCATS_ppmv | 5%+1 ppm | ~1.5 s | ATTREX |
| JPL Laser Hygrometer (JLH) | NASA JPL (PI: Robert Herman; MIDCIX also Robert Troy) | Water vapor mixing ratio | Si_JLH, qv_jlh | MACPEX: 15%. CRYSTAL-FACE-NASA/MIDCIX: none in header. | 1 Hz (implied) | CRYSTAL-FACE-NASA, MACPEX, MIDCIX |
| Harvard Water Vapor (HW) | Harvard University (Anderson group) | Water vapor (Lyman-alpha) | Si_HW, qv_hw | ±5% | 5–10 s | CRYSTAL-FACE-NASA |
| Harvard Water Vapor (HWV) | Harvard University (PI: D.S. Sayres et al.) | Water vapor (Lyman-alpha + HHH) | Si_HWV, qv_hwv | Per-record ± columns; no single blanket % | 1 Hz | MACPEX |
| ALIAS | NASA JPL | Water vapor (one channel of a multi-gas spectrometer) | Si_ALIAS, qv_alias | Not found in repo | Not found in repo | CRYSTAL-FACE-NASA |
| FISH | **External:** Forschungszentrum Jülich, Germany | Water vapor (Lyman-alpha fluorescence) | Si_FISH (declared, never loaded — always NaN) | **External:** ~5–8% | **External:** ~1 Hz | MACPEX (declared, not implemented) |
| Ophir TDL | **Low-confidence:** Ophir Corporation, Littleton CO (inferred from raw header text "Ophir") | Water vapor / dew-frost point | Si_ophir_tdl, qv_ophir_tdl | Not found. Repo documents ~0.32 Si dry bias vs. chilled mirror on 21/32 flights. | 1 Hz (implied) | IPHEX |
| Maycomm AC19-400 | Maycomm Research Company / Maycomm Inc. (R.D. May) — **confirmed via web search** | Water vapor, dual optical path (~130 cm / ~10 cm) | Si_MRTDL, qv_mrtdl, MRTDLL_MC_ppmv | Not found in repo | 1 Hz | ICE-L |
| Lyman-alpha hygrometer (unspecified) | **Unresolved** — UND team, repo notes "potentially JLH?" | Water vapor mixing ratio | Si_LH_unspecified, qv_lh_unspecified | Not found | 1 Hz (implied) | CRYSTAL-FACE-UND |
| Frost-point sensor (unspecified) | **Unresolved** — UND team | Frost-point temp → Si | Si_frost_point, qv_frost_point | Not found | 1 Hz (implied) | OLYMPEX |
| Chilled-mirror hygrometer (ARM cryo unit) | **Unresolved** — ARM archive is binary, no text metadata | Dew/frost-point temp | Si_chilled_mirror, qv_chilled_mirror | ±0.3–0.5°C (general literature, **not** raw-traceable) | 1 Hz | ARM |
| DPBC/DPTC dual chilled-mirror sensors | NCAR Research Aviation Facility | Dew-point temp (2 collocated units: DPTC fast, DPBC slow) | Si_chilled_mirror, qv_chilled_mirror | Not found (NetCDF: qualitative DataQuality only) | 5 Hz | AIRS-II |
| Chilled-mirror hygrometer (IPHEX) | **Moderate confidence:** likely EG&G-style probe (raw header names "Dewpoint (EG&G Probe)" separately) | Dew-point temp | Si_chilled_mirror, qv_chilled_mirror | Not found (UND format has no uncertainty field) | 1 Hz | IPHEX |
| RAF T-Electric chilled-mirror hygrometer | NCAR Research Aviation Facility | Dew-point → RH → Si | Si_chilled_mirror (via RHUM) | Not found (NetCDF DataQuality only) | 1 Hz | ICE-L |
| Chilled-mirror hygrometer (CRYSTAL-FACE-UND) | **Unresolved** — UND team | Dew/frost-point temp | Si_chilled_mirror, qv_chilled_mirror | Not found | 1 Hz | CRYSTAL-FACE-UND |
| EdgeTech Chilled Mirror C-137 | EdgeTech | Dew-point temp | Si_chilled_mirror, qv_chilled_mirror | ±1°C | Not found | ESCAPE |
| Rosemount temperature probe (ARM) | Rosemount Aerospace | Air temperature | Tair_C | ±0.3–0.5°C (literature, not raw-traceable) | 1 Hz | ARM |
| Rosemount temperature probe (Model 102 & 510BH) | Rosemount Aerospace | Air temperature | Tair_C | ±0.5°C (cites Lawson and Cooper 1990) | Not found | ESCAPE |
| Applanix POS | Applanix (a Trimble company) | Lat/Lon/Alt | Lat, Lon, Alt_m | Not found | Not found | MPACE, CRYSTAL-FACE-UND |
| Aventech AIMMS-20 (Dual GPS) | Aventech Research Inc. | Lat/Lon | Lat, Lon | ±10 m | Not found | ESCAPE |
| West Star Aviation RVSM altimetry | West Star Aviation | Pressure altitude | Alt_m | ±60 ft (18.3 m) | Not found | ESCAPE |
| NCAR RAF/Nimbus temp & pressure sensors | NCAR Research Aviation Facility | Air temp (ATX), static pressure (PSXC) | Tair_C, P_hPa | Not found (NetCDF DataQuality only) | 1 Hz | AIRS-II, ICE-L |
| UND Citation onboard temp/pressure/dewpoint suite | University of North Dakota (PI: Mike Poellot / David Delene) | Air temp, static pressure | Tair_C, P_hPa | Not found (format has no uncertainty field, confirmed via format spec) | 1 Hz | CRYSTAL-FACE-UND, IPHEX, MC3E, MPACE, OLYMPEX |
| UND Citation wing-pitot EDR | University of North Dakota (PI: Mike Poellot / David Delene) | Eddy dissipation rate (eps^1/3) | EDR_m23s1 (unified) | Not found | 1 Hz | IPHEX, MC3E, MPACE, OLYMPEX, CRYSTAL-FACE-UND |
| ARM Turbulence_eps (T4 binary field 18) | University of North Dakota team (same house convention, older binary archive) | Eddy dissipation rate (eps^1/3) | EDR_m23s1 (unified) | Not found | 1 Hz | ARM |
| STRAPP Convair-580 bulk data system | National Research Council Canada / STRAPP team | Air temp, pressure, RH-ice → Si | Tair_C, P_hPa, Si | Not found (qualitative validation language only) | 1 s | ISDAC |
| STRAPP Convair-580 position system | National Research Council Canada / STRAPP team | Lat/Lon | Lat, Lon | ~8 km typical, worse on specific flights | 1 s | ISDAC |
| LiCor frost/dew-point cross-check unit | **Low-confidence:** LI-COR Biosciences (inferred from column-naming convention) | Frost/dew-point temp (cross-check only, not primary Si) | LicFro/LicDew (not exported to a standardized column) | Not found | Not found | ISDAC |
| SPEC Inc. Cloud Particle Imager (CPI) | Stratton Park Engineering Company (SPEC Inc.) | Cloud particle imagery (L1/L2 join key) | cpi_filename | Not applicable (imaging instrument) | Not found | All 12 campaigns with CPI imagery |

## Provenance key

- Plain text: fact traced directly to a file in this repo (raw data
  header, parser source, or an existing decision/report doc) — see the
  `source` column in the CSV companion for the exact citation.
- **"External"**: fact came from a web search this session, not from
  anything in the repo — currently only FISH's manufacturer/accuracy.
- **"Low-confidence"/"Moderate confidence"**: a plausible inference (from
  a raw-header name fragment or column-naming convention) that could not
  be independently confirmed.
- **"Unresolved"**: the repo's own parser comments already flag the
  instrument's identity as unknown or unconfirmed — not guessed here.
- **"Not found"**: genuinely absent from every source checked (repo and
  web) — not the same as "unresolved identity," just means no numeric
  value exists to report for that cell.

## Caveats

- This inventory covers only instruments whose output actually reaches a
  column in `combined_env_data*.parquet` — several campaigns' raw
  directories contain additional unused instrument subfolders (e.g.
  ESCAPE's `cfdc/`/`ccn/`/`nrc-aerosol/`, ICE-L's `dmt-caps/`/`cvi/`,
  MACPEX's `ULH/`/`CLH/`/`CIMS/`) that are out of scope, consistent with
  the uncertainty survey report's own stated scope.
- Accuracy figures are as stated in the original raw file/companion
  documentation, not independently verified by this pipeline.
- Response time is approximated by reported sample rate/interval where an
  instrument's own response-time constant isn't stated anywhere in the
  available sources — these are related but not identical concepts.
