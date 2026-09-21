# CPI Instrument & PI/Group Provenance — 2026-09-21

Per-campaign provenance of the cloud particle imagery underlying L1/L2:
which CPI variant flew, on which aircraft, and which group/PI operated,
processed, or archived the CPI data. Written in response to a reviewer-style
request: *"Different groups have different methods for calibrating and
operating these instruments, so this could be relevant."*

Companion LaTeX table: `latex/table-cpi-instrument-pi-2026-09-21.tex`.

## Method

Same rule as `docs/reports/2026-08-29-campaign-funding-aircraft-summary.md`:
raw-file headers in this repo first, campaign archives second, web search
only where neither resolves it — and anything unresolved stays marked
unresolved rather than guessed.

Two things are deliberately kept distinct:

- **Instrument hardware** — in all 12 campaigns this is a SPEC Inc.
  Cloud Particle Imager (2.3 µm nominal pixel resolution, ~15–2500 µm),
  in ATTREX's case as the CPI channel of the SPEC Hawkeye composite probe.
  This is the one uniform element.
- **Operating / archiving group** — this is *not* uniform, and is the part
  relevant to calibration and operating practice. Three distinct regimes
  appear: SPEC Inc. flying and archiving its own probe (the WB-57 and
  Global Hawk campaigns), UND flying the probe on its Citation with the
  data archived under the UND Citation PI, and third-party processing of
  someone else's probe data (NCAR for CRYSTAL-FACE-UND, University of
  Illinois for ISDAC).

## Table

| Campaign | CPI variant | Aircraft | CPI group / PI as credited | Source | Confidence |
|---|---|---|---|---|---|
| AIRS-II | CPI | NSF/NCAR C-130 | NSF/NCAR RAF (raw CPI archive); project scientist John Hallett (DRI), project manager Jørgen Jensen (NCAR) | [RAF mass-store log, AIRS-II Raw CPI](https://archive.eol.ucar.edu/raf/Catalog/taplog.cpi.109.html) | Organization confirmed; no CPI instrument PI named anywhere checked (the [C-130 AIRS-II instrument list](https://archive.eol.ucar.edu/raf/Projects/AIRS-II/inst.html) does not itemize a CPI) |
| ARM (SGP Cloud IOP, Mar 2000) | CPI | UND Citation II | University of North Dakota, Dept. of Atmospheric Sciences — Michael Poellot | `data/raw/ARM/readme.txt` ("poellot-citation  CPI and PMS data from the University of North Dakota Citation aircraft"); `data/raw/ARM/poellot-citation-t4-readme.txt` | Confirmed in repo's own raw archive |
| ATTREX | Hawkeye 3V-CPI | NASA Global Hawk | SPEC Inc. — R. Paul Lawson | [ESPO archive, ATTREX Global Hawk](https://espoarchive.nasa.gov/archive/browse/attrex/id1/GHawk) — listing reads "2DS; CDP; CPI  PI: Lawson", with a `Hawkeye-3VCPI` product directory | Confirmed |
| CRYSTAL-FACE (NASA) | CPI | NASA WB-57 | SPEC Inc. — Paul Lawson, Brad Baker, Qixu Mo | `data/raw/CRYSTAL-FACE-NASA/CP/CP*.WB57` header lines 2–4: PI names, "SPEC Inc, 3022 Sterling Cir…", "Cloud Particle Imager (CPI) measurements on NASA WB-57F" | Confirmed in repo's own raw archive |
| CRYSTAL-FACE (UND) | CPI | UND Citation | NCAR — Carl Schmitt, Andy Heymsfield, Aaron Bansemer | `data/raw/CRYSTAL-FACE-UND/CP/CP*.CIT` header lines 2–4: PI names, "NCAR, PO Box 3000, Boulder…", "Cloud Particle Imager (CPI) measurements from Citation" | Confirmed in repo's own raw archive. Note the split: same campaign, same probe type, *different* group from the WB-57 row |
| ICE-L | CPI | NSF/NCAR C-130 | NSF/NCAR EOL; campaign PIs Andrew Heymsfield (NCAR/MMM) and Jeff Stith (NCAR/EOL) | [EOL ICE-L project page](https://www.eol.ucar.edu/field_projects/ice-l) | **CPI operator unresolved.** EOL's [3V-CPI instrument page](https://www.eol.ucar.edu/instruments/three-view-cloud-particle-imager) names Paul Lawson (SPEC) as external contact and Sarah Woods as EOL lead, but its listed deployments (PREDICT, ICE-T, DC3, CSET, ARISTO-2016, IDEAS-4) all postdate ICE-L (2007), so that attribution is **not** carried over to this row |
| IPHEx | CPI | UND Citation II | University of North Dakota (Citation data archive PI David Delene); imaging-probe size spectra processed by NCAR under GPM GV | `data/raw/IPHEX/*.iphex` header line 2 ("David Delene") / line 3 ("University of North Dakota"); [GHRC GPM GV NCAR Particle Probes IPHEx](https://www.earthdata.nasa.gov/data/catalog/ghrc-daac-gpmparprbiphx-1) | **CPI-specific PI unresolved.** The header PI is for the Citation state/microphysics file this pipeline parses, not specifically for the CPI |
| ISDAC | CPI | NRC Canada Convair-580 | Aircraft/platform: NRC Canada (Convair PI Mengistu Wolde — cf. `data/raw/ISDAC/wolde-convair/`); CPI imagery produced for the ARM archive by Greg McFarquhar (University of Illinois) | [ISDAC Science Overview, DOE/SC-ARM-0705](https://www.osti.gov/servlets/purl/947999): "The CPI images will also be generated for the Archive (G. McFarquhar of Illinois)"; instrument table lists "SPEC Cloud Particle Imager" | Confirmed |
| MACPEX | CPI | NASA WB-57 | SPEC Inc. — R. Paul Lawson | [ESPO archive, MACPEX WB-57](https://espoarchive.nasa.gov/archive/browse/macpex) — "2DS; CPI; HVPS  PI: Lawson", with `CPI (PNG)` product directories | Confirmed |
| MC3E | CPI | UND Citation II | University of North Dakota (Citation data archive PI Mike Poellot); imaging-probe spectra processed by NCAR under GPM GV | `data/raw/MC3E/*.mc3e` header line 2 ("Mike Poellot") / line 3; [ARM MC3E campaign page](https://www.arm.gov/research/campaigns/sgp2011midlatcloud) | **CPI-specific PI unresolved**, same caveat as IPHEx |
| MidCiX | CPI | NASA WB-57 | SPEC Inc. — R. Paul Lawson | [ESPO archive, MidCiX WB-57](https://espoarchive.nasa.gov/archive/browse/midcix/WB57) — "CPI; SPP  PI: Lawson" | Confirmed |
| M-PACE | CPI | UND Citation II | University of North Dakota — Mike Poellot | `data/raw/MPACE/*.mpace` header line 2 ("Poellot, Mike"); ARM IOP stream [`poellot-citation`](https://iop.arm.gov/2004/nsa/mpace/poellot-citation) (linked from `parsers/README.md`) | Confirmed by naming convention consistent with the ARM 2000 row, where the same stream name is explicitly documented as holding CPI data |

## Notes

- **The three campaigns with no CPI imagery** (OLYMPEX, POSIDON, ESCAPE)
  are excluded, consistent with `CLAUDE.md` and the funding/aircraft
  table. POSIDON is worth one remark anyway: its WB-57 CPI *does* exist in
  the ESPO archive under PI Woods (SPEC), it simply isn't in
  `data/raw/cpi_embeddings_timestamps.csv`, so it contributes no L1/L2
  rows here.
- **Why the split matters for this dataset.** Rows attributed to SPEC Inc.
  (ATTREX, CRYSTAL-FACE-NASA, MACPEX, MidCiX) are the instrument
  manufacturer operating its own probe. UND Citation rows (ARM, IPHEx,
  MC3E, M-PACE, CRYSTAL-FACE-UND) are a probe integrated into a
  university aircraft's standard suite. CRYSTAL-FACE-UND and ISDAC add a
  third layer — a group other than either the manufacturer or the aircraft
  operator producing the archived imagery. Any habit-classification or
  size analysis pooled across campaigns is therefore pooling across at
  least three different operating/processing practices.
- **Unresolved cells are not filled by inference.** Four rows (AIRS-II,
  ICE-L, IPHEx, MC3E) name an organization where no individual CPI PI is
  credited in any source checked. The tempting inference for AIRS-II and
  ICE-L — that NCAR/EOL's CPI was operated with SPEC support — is
  plausible but unsupported for those specific years, so it's excluded.
- **Instrument reference.** The CPI itself is documented in Lawson et al.
  (2001) and in the ARM instrument handbook
  [DOE/SC-ARM-TR-240](https://www.arm.gov/publications/tech_reports/handbooks/doe-sc-arm-tr-240.pdf),
  which covers both the CPI and 3V-CPI.
