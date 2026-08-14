<!--
Report generated: 2026-07-27
Source data:  ssl-cpi-analysis/data/clean_combined_campaign_env_data.csv
              (COCPIT habit classification + geometry labels, 524,033 rows,
               6 campaigns; CRYSTAL_FACE_NASA subset = 47,219 labeled particles)
Habit column: `Classification` (single COCPIT class per particle)
Flight proxy: distinct calendar day of `date` timestamp
-->

# CRYSTAL-FACE-NASA — Research Flights & Per-Flight Habit Proportions

## Summary

- **7 research flights** carry classified CPI particle imagery, spanning
  **2002-07-11 to 2002-07-29**.
- **47,219 labeled particles** total across the 7 flights.
- **7 COCPIT habit classes** appear: `compact_irreg`, `agg` (aggregate),
  `planar_polycrystal`, `rimed`, `column`, `budding`, `bullet` (bullet
  rosette). No `sphere` or `fragment` particles in the NASA subset.
- Every flight is **dominated by compact irregular** (69.8–82.7%), with
  **aggregate** the consistent #2 (11.6–23.7%). All other habits are
  minor (each < 6% on every flight).

### Scope caveat

This count reflects flights with **habit-labeled CPI imagery**, not the
full campaign. The habit label file
(`ssl-cpi-analysis/data/clean_combined_campaign_env_data.csv`) covers only
6 of 15 campaigns. cpi-thermo's L0 env data for CRYSTAL-FACE-NASA spans
**20 distinct flight dates** (2002-05-09 – 2002-07-31; see
`docs/reports/2026-07-08-campaign-breakdown-descriptive-analysis.md`) — so
the environmental record covers many more flights than have classified
particle imagery here. "7 flights" = flights represented in the
habit-classified particle set only.

## Flights & particle counts

| Flight (date) | Labeled particles | % of NASA total |
|---|---:|---:|
| 2002-07-11 | 5,424 | 11.5% |
| 2002-07-16 | 11,177 | 23.7% |
| 2002-07-19 | 2,276 | 4.8% |
| 2002-07-21 | 3,404 | 7.2% |
| 2002-07-23 | 1,618 | 3.4% |
| 2002-07-28 | 11,349 | 24.0% |
| 2002-07-29 | 11,971 | 25.4% |
| **Total** | **47,219** | **100%** |

## Per-flight habit proportions (% of that flight's particles)

| Flight | n | compact_irreg | agg | planar_polycrystal | rimed | column | budding | bullet |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2002-07-11 | 5,424 | 69.8 | 23.7 | 3.1 | 1.8 | 1.4 | 0.1 | 0.0 |
| 2002-07-16 | 11,177 | 81.6 | 13.2 | 1.5 | 1.4 | 1.6 | 0.6 | 0.2 |
| 2002-07-19 | 2,276 | 81.3 | 11.6 | 0.3 | 5.7 | 0.7 | 0.4 | 0.0 |
| 2002-07-21 | 3,404 | 79.0 | 17.0 | 1.4 | 1.2 | 1.2 | 0.2 | 0.0 |
| 2002-07-23 | 1,618 | 82.7 | 12.9 | 1.5 | 1.0 | 1.2 | 0.4 | 0.3 |
| 2002-07-28 | 11,349 | 74.1 | 20.0 | 2.9 | 1.4 | 1.3 | 0.2 | 0.1 |
| 2002-07-29 | 11,971 | 74.7 | 19.5 | 3.0 | 1.2 | 1.3 | 0.3 | 0.0 |
| **All flights** | **47,219** | **76.5** | **17.8** | **2.3** | **1.6** | **1.3** | **0.3** | **0.1** |

## Raw counts (particles per habit per flight)

| Flight | agg | budding | bullet | column | compact_irreg | planar_polycrystal | rimed | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2002-07-11 | 1,287 | 8 | 2 | 78 | 3,784 | 168 | 97 | 5,424 |
| 2002-07-16 | 1,475 | 62 | 20 | 175 | 9,122 | 172 | 151 | 11,177 |
| 2002-07-19 | 265 | 10 | 0 | 15 | 1,850 | 7 | 129 | 2,276 |
| 2002-07-21 | 577 | 7 | 0 | 42 | 2,690 | 47 | 41 | 3,404 |
| 2002-07-23 | 209 | 6 | 5 | 19 | 1,338 | 25 | 16 | 1,618 |
| 2002-07-28 | 2,265 | 28 | 9 | 148 | 8,408 | 332 | 159 | 11,349 |
| 2002-07-29 | 2,335 | 33 | 5 | 158 | 8,941 | 354 | 145 | 11,971 |

## Campaign-wide habit proportions (all 7 flights pooled)

| Habit | % | count |
|---|---:|---:|
| compact_irreg | 76.5 | 36,133 |
| agg | 17.8 | 8,413 |
| planar_polycrystal | 2.3 | 1,105 |
| rimed | 1.6 | 738 |
| column | 1.3 | 635 |
| budding | 0.3 | 154 |
| bullet | 0.1 | 41 |

## Notes

- One flight = one distinct calendar day of the `date` timestamp.
  CRYSTAL-FACE (Florida, July 2002) flew single-day daytime missions, so
  day ≐ research flight; no day contains two flights and no flight spans
  midnight in this subset.
- `Classification` is a single hard COCPIT label per particle (the CSV also
  carries soft per-habit `[%]` columns — Aggregate/Column/… — not used
  here). Zero rows have a missing `Classification`.
- 2002-07-19 stands out: highest `rimed` fraction (5.7%, vs ≤1.8%
  elsewhere) and near-absent `planar_polycrystal` (0.3%).

## Reproduce

```bash
conda activate cpi-thermo
python - <<'PY'
import pandas as pd
df = pd.read_csv("data/clean_combined_campaign_env_data.csv",
                 usecols=["date","Classification","Campaign"])
c = df[df.Campaign=="CRYSTAL_FACE_NASA"].copy()
c["day"] = pd.to_datetime(c.date).dt.date
print("flights:", c.day.nunique(), "particles:", len(c))
print((pd.crosstab(c.day, c.Classification, normalize="index")*100).round(1))
PY
```
(run from `ssl-cpi-analysis/`)
