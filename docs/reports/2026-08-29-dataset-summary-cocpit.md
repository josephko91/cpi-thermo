# Dataset Summary — L1/L2 with COCPIT Particle Features — 2026-08-29

Companion to `docs/reports/2026-08-28-dataset-summary.md` (L0/L1/L2
overview) and `docs/reports/2026-08-29-cocpit-particle-feature-join.md`
(the join method, coverage investigation, and root-cause findings). This
report gives the full statistical picture of
`data/out/combined_env_data_L1_cocpit.parquet` /
`_L2_cocpit.parquet` — shape, every column's completeness, basic stats for
every joined COCPIT variable, and how the COCPIT-matched subset compares
to L1/L2 overall. Not committed to git (data outputs are gitignored); this
report documents them for reference.

## 1. Overview

| | L1_cocpit | L2_cocpit | (for reference) L1 | L2 |
|---|---:|---:|---:|---:|
| Rows | 2,997,447 | 1,828,818 | 2,997,447 | 1,828,818 |
| Columns | 78 | 78 | 47 | 47 |
| Campaigns | 12 | 11 | 12 | 11 |
| Rows with matched COCPIT features | 914,527 (30.5%) | 531,346 (29.0%) | — | — |

Row and campaign counts are identical to plain L1/L2 — the COCPIT join is
a left join, so no rows are added or dropped, only 31 columns appended
(NaN where unmatched). See the companion join report for why only ~30% of
rows have a match — it's a COCPIT-side processing-history gap (confirmed
via direct investigation on xcite), not a defect in this join.

## 2. Variables added by COCPIT (31 columns)

- **Size (3):** `particle_width_microns`, `particle_height_microns`
  (already in physical microns in COCPIT's raw data), `equiv_d_microns`
  (derived per-particle from `equiv_d` × that row's own
  `frame_width`/`frame_height`).
- **Geometric shape (12):** `circularity`, `solidity`, `complexity`,
  `phi`, `perim_area_ratio`, `roundness`, `filled_circular_area_ratio`,
  `convex_perim`, `hull_area`, `perim`, `cnt_area`, `extreme_points`.
- **Habit classification (10):** `classification` (the CNN's top-1 label,
  one of 7 classes observed in this data — see §5) plus 9 per-class
  probability columns: `agg`, `budding`, `bullet`, `column`,
  `compact_irreg`, `fragment`, `planar_polycrystal`, `rimed`, `sphere`.
- **Image-quality diagnostics (6):** `cutoff` (% of particle intersecting
  the frame border), `blur` (Laplacian variance — lower is blurrier),
  `contours`, `edges`, `std`, `contrast`.

## 3. Per-variable completeness, full column set

### L1_cocpit (2,997,447 rows, 12 campaigns)

| Variable | % | Variable | % | Variable | % |
|---|---:|---|---:|---|---:|
| cpi_filename | 100.0 | Timestamp | 100.0 | Campaign | 100.0 |
| source_file | 93.71 | Tair_C | 92.94 | P_hPa | 97.58 |
| Si | 61.05 | qv | 61.01 | Sw | 61.04 |
| Lat | 99.76 | Lon | 99.76 | Alt_m | 99.87 |
| Wind_U_ms | 55.95 | Wind_V_ms | 55.95 | Wind_W_ms | 62.26 |
| EDR_m23s1 | 68.05 | | | | |
| **particle_width_microns** | **30.51** | **particle_height_microns** | **30.51** | **equiv_d_microns** | **29.86** |
| **circularity** | **29.86** | **solidity** | **29.86** | **complexity** | **29.86** |
| **phi** | **29.86** | **perim_area_ratio** | **29.86** | **roundness** | **29.86** |
| **filled_circular_area_ratio** | **29.86** | **convex_perim** | **29.86** | **hull_area** | **29.86** |
| **perim** | **29.86** | **cnt_area** | **29.86** | **extreme_points** | **29.86** |
| **classification** | **29.78** | **agg** | **29.78** | **budding** | **29.78** |
| **bullet** | **29.78** | **column** | **29.78** | **compact_irreg** | **29.78** |
| **fragment** | **29.78** | **planar_polycrystal** | **29.78** | **rimed** | **29.78** |
| **sphere** | **29.78** | **cutoff** | **30.51** | **blur** | **29.86** |
| **contours** | **29.86** | **edges** | **29.78** | **std** | **29.78** |
| **contrast** | **29.86** | | | | |

(Per-instrument Si/qv fallback columns and raw `*_ppmv` intermediates
omitted from this table for brevity — unchanged from
`docs/reports/2026-08-28-dataset-summary.md` §3, all under 30%.)

### L2_cocpit (1,828,818 rows, 11 campaigns)

| Variable | % | Variable | % | Variable | % |
|---|---:|---|---:|---|---:|
| Tair_C | 100.0 | P_hPa | 100.0 | Si | 100.0 |
| qv | 100.0 | Sw | 100.0 | Lat | 100.0 |
| Lon | 100.0 | Alt_m | 100.0 | Wind_U_ms | 70.82 |
| Wind_V_ms | 70.82 | Wind_W_ms | 72.14 | EDR_m23s1 | 61.50 |
| **particle_width_microns** | **29.05** | **particle_height_microns** | **29.05** | **equiv_d_microns** | **28.00** |
| **circularity** | **28.00** | **solidity** | **28.00** | **complexity** | **28.00** |
| **phi** | **28.00** | **perim_area_ratio** | **28.00** | **roundness** | **28.00** |
| **filled_circular_area_ratio** | **28.00** | **convex_perim** | **28.00** | **hull_area** | **28.00** |
| **perim** | **28.00** | **cnt_area** | **28.00** | **extreme_points** | **28.00** |
| **classification** | **27.93** | **agg** | **27.93** | **budding** | **27.93** |
| **bullet** | **27.93** | **column** | **27.93** | **compact_irreg** | **27.93** |
| **fragment** | **27.93** | **planar_polycrystal** | **27.93** | **rimed** | **27.93** |
| **sphere** | **27.93** | **cutoff** | **29.05** | **blur** | **28.00** |
| **contours** | **28.00** | **edges** | **27.93** | **std** | **27.93** |
| **contrast** | **28.00** | | | | |

Within a matched row, geometric/habit/quality completeness (~27.9-29.9%)
runs about 0.5-0.7 percentage points below the two raw size columns
(`particle_width_microns`/`particle_height_microns`, ~29.0-30.5%) — a
small, consistent internal sparsity, **except for one campaign** (§8).

## 4. Basic stats, size/geometric/quality columns (matched rows only)

### L1 (n=914,527 matched)

| Variable | Mean | Std | Min | Max |
|---|---:|---:|---:|---:|
| particle_width_microns | 224.326 | 169.740 | 0.000 | 2385.141 |
| particle_height_microns | 224.252 | 169.889 | 0.000 | 2474.146 |
| equiv_d_microns | 188.644 | 116.285 | 1.256 | 1565.164 |
| circularity | 0.343 | 0.147 | 0.007 | 1.571 |
| solidity | 0.814 | 0.101 | 0.057 | 1.000 |
| complexity | 0.632 | 0.138 | -1.066 | 0.988 |
| phi | 0.780 | 0.172 | 0.009 | 1.000 |
| perim_area_ratio | 0.012 | 0.006 | 0.004 | 1.766 |
| roundness | 0.710 | 0.123 | 0.018 | 0.974 |
| filled_circular_area_ratio | 0.539 | 0.132 | 0.007 | 0.927 |
| convex_perim | 2526.253 | 371.013 | 24.325 | 3644.110 |
| hull_area | 450802.719 | 119664.846 | 19.000 | 887979.500 |
| perim | 3917.963 | 1160.794 | 23.828 | 20875.750 |
| cnt_area | 363471.804 | 95347.253 | 17.000 | 747783.500 |
| extreme_points | 309.199 | 47.594 | 11.247 | 496.771 |
| cutoff | 0.021 | 0.028 | 0.000 | 0.100 |
| blur | 31.861 | 13.368 | 4.225 | 592.600 |
| contours | 2.397 | 6.923 | 1.000 | 569.000 |
| edges | 44486.483 | 16086.478 | 92.000 | 295248.000 |
| std | 270.740 | 25.811 | 81.065 | 420.102 |
| contrast | 93.016 | 7.391 | 56.580 | 120.161 |

### L2 (n=531,346 matched)

| Variable | Mean | Std | Min | Max |
|---|---:|---:|---:|---:|
| particle_width_microns | 235.052 | 179.431 | 0.000 | 2378.497 |
| particle_height_microns | 234.989 | 179.731 | 0.000 | 2474.146 |
| equiv_d_microns | 199.170 | 123.760 | 1.353 | 1510.826 |
| circularity | 0.334 | 0.145 | 0.007 | 1.571 |
| solidity | 0.811 | 0.101 | 0.057 | 1.000 |
| complexity | 0.639 | 0.135 | -1.066 | 0.988 |
| phi | 0.774 | 0.173 | 0.009 | 1.000 |
| perim_area_ratio | 0.012 | 0.005 | 0.004 | 0.788 |
| roundness | 0.705 | 0.122 | 0.023 | 0.974 |
| filled_circular_area_ratio | 0.532 | 0.130 | 0.009 | 0.911 |
| convex_perim | 2545.106 | 342.523 | 39.174 | 3520.495 |
| hull_area | 454390.894 | 117383.534 | 76.000 | 887979.500 |
| perim | 3990.560 | 1172.990 | 37.657 | 20713.786 |
| cnt_area | 364685.921 | 92176.381 | 53.000 | 747783.500 |
| extreme_points | 311.525 | 46.390 | 11.247 | 496.771 |
| cutoff | 0.021 | 0.028 | 0.000 | 0.100 |
| blur | 32.022 | 13.325 | 4.515 | 592.600 |
| contours | 2.353 | 6.395 | 1.000 | 447.000 |
| edges | 43897.016 | 15680.544 | 92.000 | 238954.000 |
| std | 269.499 | 25.341 | 81.065 | 420.102 |
| contrast | 93.217 | 7.604 | 56.580 | 120.161 |

L1 and L2 stats are nearly identical throughout — L2's core-variable
completeness filter doesn't meaningfully shift the particle-geometry
distributions, since it filters on env variables, not on anything
COCPIT-derived. `perim_area_ratio`'s L1 max (1.766) exceeds its L2 max
(0.788) — a small number of geometrically extreme outliers exist only in
env-incomplete rows.

## 5. Habit classification distribution (matched rows)

| Class | L1 count | L1 % | L2 count | L2 % |
|---|---:|---:|---:|---:|
| compact_irreg | 484,475 | 53.0% | 263,964 | 49.7% |
| agg (aggregate) | 174,341 | 19.1% | 107,107 | 20.2% |
| rimed | 77,513 | 8.5% | 50,553 | 9.5% |
| planar_polycrystal | 68,844 | 7.5% | 41,388 | 7.8% |
| column | 32,337 | 3.5% | 19,049 | 3.6% |
| budding | 31,619 | 3.5% | 18,443 | 3.5% |
| bullet | 23,556 | 2.6% | 10,313 | 1.9% |

`compact_irreg` dominates in both tiers (~50-53%), consistent with it
being the catch-all "small, arbitrary-shaped particle" class per COCPIT's
own README class definitions. No `fragment` or `sphere` rows appear in
either tier at this filter — consistent with COCPIT's `run_model.py`
*intending* to exclude those two classes (see the join report's note on
that filter being a no-op bug in COCPIT's own code — the absence here may
instead simply reflect that few/no particles in the L1/L2-matched subset
were classified into those categories, not that the exclusion code
worked; not independently verified which explanation is correct).

## 6. Per-campaign match rate and mean particle size (L1)

| Campaign | L1 rows | Matched | % Matched | Mean width (µm) | Mean equiv_d (µm) |
|---|---:|---:|---:|---:|---:|
| AIRS-II | 92,168 | 27,303 | 29.62% | 253.1 | 202.7 |
| ARM | 230,029 | 19,778 | 8.60% | 267.5 | 219.3 |
| ATTREX | 122,050 | 19,616 | 16.07% | 102.9 | N/A (see §8) |
| CRYSTAL-FACE-NASA | 78,151 | 61,858 | 79.15% | 130.9 | 115.4 |
| CRYSTAL-FACE-UND | 1,608,674 | 393,751 | 24.48% | 218.5 | 179.5 |
| ICE-L | 46,203 | 36,994 | 80.07% | 317.3 | 261.5 |
| IPHEX | 38,697 | 16,081 | 41.56% | 467.2 | 363.3 |
| ISDAC | 400,805 | 67,413 | 16.82% | 225.2 | 190.0 |
| MACPEX | 80,240 | 2,226 | 2.77% | 94.8 | 89.1 |
| MC3E | 173,766 | 151,439 | 87.15% | 260.5 | 220.8 |
| MIDCIX | 90,667 | 88,785 | 97.92% | 172.5 | 150.7 |
| MPACE | 35,997 | 29,283 | 81.35% | 253.1 | 197.5 |

IPHEX has the largest mean particle size (467 µm) of any campaign;
MACPEX and ATTREX the smallest (~95-103 µm) — plausible given IPHEX's
mixed-phase precipitation focus (Appalachians) versus ATTREX's
upper-troposphere/lower-stratosphere cirrus focus (smaller ice crystals
expected at those altitudes/temperatures).

## 7. Complete-record counts (L2): embeddings + core thermo + size (+ geometric)

L2 already guarantees a matched CPI embedding (100% — every L2 row was
checked against the SSL embedding archive,
`data/raw/SSL-Model-v3/cpi3m_campaign_cls_head_features_compressed.parquet`,
and every one matches) and complete core thermodynamics (100%, by
construction — `Tair_C, P_hPa, Si, qv, Lat, Lon, Alt_m`). So for L2, the
"how many records have everything" question reduces entirely to whether
COCPIT size/geometric features are also present:

| Requirement | Rows | % of L2 |
|---|---:|---:|
| Embeddings + core thermo + **size** | 531,346 | 29.05% |
| Embeddings + core thermo + size + **geometric shape** | 512,145 | 28.00% |

The 19,192-row gap between the two is entirely ATTREX (§8 caveat below):
it has particle size but zero geometric shape descriptors in COCPIT's
data, so it contributes fully to the size-only count and zero to the
size+geometric count.

Per-campaign, embeddings + core thermo + size (no geometric requirement):

| Campaign | Complete | L2 total |
|---|---:|---:|
| AIRS-II | 27,303 | 92,168 |
| ARM | 12,980 | 64,706 |
| ATTREX | 19,192 | 120,595 |
| CRYSTAL-FACE-NASA | 16,127 | 20,441 |
| CRYSTAL-FACE-UND | 201,302 | 848,940 |
| ICE-L | 36,993 | 46,202 |
| IPHEX | 11,884 | 28,189 |
| ISDAC | 67,113 | 399,668 |
| MACPEX | 1,337 | 51,747 |
| MC3E | 118,750 | 137,272 |
| MIDCIX | 18,365 | 18,890 |

Per-campaign, embeddings + core thermo + size + geometric shape (ATTREX
drops to 0, all others unchanged from the size-only table):

| Campaign | Complete | L2 total |
|---|---:|---:|
| AIRS-II | 27,303 | 92,168 |
| ARM | 12,971 | 64,706 |
| ATTREX | 0 | 120,595 |
| CRYSTAL-FACE-NASA | 16,127 | 20,441 |
| CRYSTAL-FACE-UND | 201,302 | 848,940 |
| ICE-L | 36,993 | 46,202 |
| IPHEX | 11,884 | 28,189 |
| ISDAC | 67,113 | 399,668 |
| MACPEX | 1,337 | 51,747 |
| MC3E | 118,750 | 137,272 |
| MIDCIX | 18,365 | 18,890 |

Note ARM differs by 9 rows between the two tables (12,980 vs 12,971) —
a handful of ARM particles have size but not all geometric columns
populated, distinct from ATTREX's all-or-nothing gap.

## 8. Caveats

- **ATTREX has size but no geometric shape descriptors at all.** Every
  other campaign's matched rows are 100% complete for `circularity` (and
  the other 11 geometric columns); ATTREX's matched rows are **0%**
  complete for all of them (`equiv_d_microns`, `circularity`, `solidity`,
  etc. all NaN), while `particle_width_microns`/`particle_height_microns`
  are still populated. This means ATTREX's COCPIT v1.4.0 CSV is missing
  the shape-descriptor columns entirely — not a join-logic issue (the
  columns are just absent/NaN in the source data for this campaign) but
  worth knowing before using geometric-shape features across all
  campaigns uniformly. Not yet root-caused against COCPIT's own
  processing history (`docs/reports/2026-07-08-raw-data-uncertainty-metadata-survey.md`-style
  investigation would need to inspect ATTREX's raw v1.4.0 CSV columns
  directly).
- **COCPIT-matched rows are not systematically better-instrumented for
  env data.** Comparing L1's COCPIT-matched subset (n=914,527) against
  its unmatched subset (n=2,082,920): Tair_C completeness is actually
  *lower* among matched rows (82.8% vs 97.4%), P_hPa similarly (92.3% vs
  99.9%), while Si/qv/position are close either way. Having a COCPIT
  feature match is roughly independent of — if anything mildly
  anti-correlated with — having complete environmental data, so joining
  COCPIT features doesn't meaningfully bias which env-complete rows you'd
  keep for a combined analysis.
- Size/geometric feature completeness gaps and the underlying ~30%
  overall match rate are a COCPIT processing-history artifact, not
  something this join can fix — see
  `docs/reports/2026-08-29-cocpit-particle-feature-join.md` for the full
  investigation (traced to COCPIT's own upstream source, confirmed on
  xcite and independently reproduced locally).
- `particle_width_microns`/`particle_height_microns` include some exact
  0.0 values (see §4 min column) — likely degenerate detections, not
  filtered here (same caveat as the join report).

## Reproduce

```bash
conda activate cpi-thermo
python scripts/join_cocpit_features.py
```

then load `data/out/combined_env_data_L1_cocpit.parquet` /
`_L2_cocpit.parquet` directly with pandas for further analysis. Both
files are gitignored (regenerable, not portable — depend on the external
COCPIT path).
