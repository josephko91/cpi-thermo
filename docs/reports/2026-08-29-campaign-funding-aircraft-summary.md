# Campaign Funding & Aircraft Platform Summary — 2026-08-29

Summary of the 12 field campaigns with CPI imagery in this dataset —
funding organization and aircraft platform. Companion LaTeX table:
`latex/table-campaign-funding-aircraft-2026-08-29.tex`.

## Method

Aircraft platform: `parsers/README.md` and
`docs/reports/2026-08-28-instrument-inventory.md` (both already
established in this repo/session). Funding organization: confirmed via
web search for every campaign — none asserted from the archive source
(NASA ESPO / NCAR EOL / ARM Data Center) alone, since hosting archive
doesn't necessarily equal funder. Sources cited per row below.

## Table

| Campaign | Funding organization | Aircraft platform | Source |
|---|---|---|---|
| ARM | DOE, Atmospheric Radiation Measurement (ARM) Program | UND Citation II | `parsers/README.md`; ARM is inherently a DOE program |
| AIRS-II | NASA (Glenn Research Center) + NRC Canada + Environment Canada (tri-agency) | NSF/NCAR C-130 | [NRC Canada](https://nrc.canada.ca/en/stories/national-research-council-canada-nasa-renew-agreement-icing-research); aircraft per COCPIT project README |
| ATTREX | NASA, Earth Venture Suborbital-1 (EVS-1) program | NASA Global Hawk (unmanned) | [NASA ATTREX overview](https://ntrs.nasa.gov/citations/20200000904) |
| CRYSTAL-FACE (NASA) | NASA, Radiation Sciences Program | NASA WB-57 | [NASA CASEI](https://impact.earthdata.nasa.gov/casei/campaign/CRYSTAL-FACE) |
| CRYSTAL-FACE (UND) | NASA, Radiation Sciences Program (same overall campaign; UND Citation was one of six contributing aircraft) | UND Citation | [NASA CASEI](https://impact.earthdata.nasa.gov/casei/campaign/CRYSTAL-FACE); `parsers/README.md` |
| ICE-L | NSF (NCAR-operated aircraft) | NSF/NCAR C-130 | COCPIT project README (aircraft); NSF funding inferred from NCAR/NSF C-130 operation, not independently confirmed by a dedicated funding-statement source |
| IPHEX | NASA, Global Precipitation Measurement (GPM) Ground Validation (co-led with Duke University, NOAA HMT partner) | UND Citation II | [NASA GPM IPHEx](http://pmm.nasa.gov/iphex) |
| ISDAC | DOE, Atmospheric Radiation Measurement (ARM) Program | NRC (Canada) Convair-580 | [DOE/ARM ISDAC](https://www.arm.gov/research/campaigns/aaf2008isdac) |
| MACPEX | NASA, Earth Science Research and Analysis Program (joint with NOAA ESRL, NCAR, university partners) | NASA WB-57 | [NASA Earthdata MACPEX](https://www.earthdata.nasa.gov/data/projects/macpex) |
| MC3E | NASA GPM Ground Validation + DOE ARM Program (joint campaign) | UND Citation II | [NASA GPM MC3E](https://pmm.nasa.gov/science/ground-validation/mid-latitude-continental-convective-clouds-experiment-mc3e) |
| MIDCIX | NASA, Radiation Sciences Program (DOE collaboration) | NASA WB-57 | [NASA NTRS](https://ntrs.nasa.gov/citations/20050180631) |
| MPACE | DOE, Atmospheric Radiation Measurement (ARM) Program | UND Citation II | [DOE/ARM MPACE](https://www.arm.gov/research/campaigns/nsa2004arcticcld) |

## Notes

- **ICE-L's funding** is the one cell not independently confirmed by a
  dedicated funding-statement source — NSF is inferred from the aircraft
  being NSF/NCAR-operated (a standard NSF Lower Atmosphere Observing
  Facility asset), not stated explicitly in any source checked. Flagged
  rather than asserted as fact.
- **Joint-funded campaigns** (MC3E, ISDAC, MPACE all touch DOE ARM;
  MIDCIX/AIRS-II have secondary partners) are listed with all confirmed
  funding parties, not simplified to a single agency, since that's what
  the sources actually describe.
- Aircraft designations use each campaign's specific tail/model where the
  source specifies it (e.g. "UND Citation II" vs. plain "UND Citation")
  — some sources are less precise than others; not normalized across rows
  since the underlying source precision differs.
