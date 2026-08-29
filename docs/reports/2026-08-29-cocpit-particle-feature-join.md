# COCPIT Particle Size/Geometric Feature Join — 2026-08-29

Joins per-particle size (microns) and geometric-shape/habit-classification
features from the external COCPIT vgg16 derived-feature database into this
pipeline's L1/L2 tiers. Built by the new `scripts/join_cocpit_features.py`.
Outputs: `data/out/combined_env_data_L1_cocpit.parquet`,
`data/out/combined_env_data_L2_cocpit.parquet` (new files — the canonical
`combined_env_data_L1.parquet`/`_L2.parquet` are untouched, since this join
depends on an external, non-portable path that only exists on this
machine, consistent with how the other COCPIT-reading scripts are already
treated per CLAUDE.md).

## Method

- **Source**: `/Users/josephko/research/cocpit/final_databases/vgg16/v1.4.0/`.
  Of 5 versions on disk, v1.4.0 is the only one with all 15 campaigns
  present (v1.2.0 has 12, v1.3.0 has 5 with no shape descriptors at all,
  v1.5.0 is empty, v3.1.0 has 11). v1.4.0 is also the canonical version
  per `docs/decisions/2026-07-24-derived-feature-version-equiv-d-shift.md`:
  v3.1.0 is a wholly disjoint particle extraction (zero shared filenames
  with v1.2.0/v1.4.0) with an unresolved ~7x scale discontinuity in
  absolute-scale features. **Confirmed this session**: v3.1.0's filenames
  are also missing the millisecond field entirely
  (`2011_0523_183354_0.png`, 4 fields) versus v1.4.0/this pipeline's
  5-field format (`..._163505_942_29.png`) — a direct string join against
  `cpi_filename` would silently fail for v3.1.0 even setting the scale
  issue aside.
- **Join key**: `cpi_filename` (this pipeline) against COCPIT's `filename`
  column, both the same underlying CPI-archive naming convention. Format
  match confirmed exact for every campaign checked.
- **Join type**: per-campaign **left** join (campaign names mapped via
  the existing `parsers/cpi_timestamps.py::CPI_TO_ENV_CAMPAIGN`) — L1/L2
  rows are always kept, with NaN COCPIT columns where unmatched, so match
  rate is measurable rather than rows silently disappearing.
- **Features joined**: size (`particle_width_microns`,
  `particle_height_microns`, `equiv_d_microns` — the first two are
  already in physical microns in the raw COCPIT data; `equiv_d_microns`
  is derived per-particle from `equiv_d` × each row's own
  `frame_width`/`frame_height`, reusing
  `compare_derived_feature_versions.py::_add_equiv_d_microns`), geometric
  shape (`circularity`, `solidity`, `complexity`, `phi`,
  `perim_area_ratio`, `roundness`, `filled_circular_area_ratio`,
  `convex_perim`, `hull_area`, `perim`, `cnt_area`, `extreme_points`),
  habit (`classification` + 9 per-class probability columns), and
  image-quality diagnostics (`cutoff`, `blur`, `contours`, `edges`,
  `std`, `contrast`).
- Row counts are unchanged by the join (2,997,447 at L1, 1,828,818 at
  L2, matching `docs/reports/2026-08-28-dataset-summary.md`) — confirms
  no duplicate-filename row multiplication occurred during the merge.

## Summary statistics (matched rows only)

| Column | Tier | n | Mean | Std | Min | Max |
|---|---|---:|---:|---:|---:|---:|
| particle_width_microns | L1 | 914,527 | 224.3 | 169.7 | 0.0 | 2385.1 |
| particle_height_microns | L1 | 914,527 | 224.3 | 169.9 | 0.0 | 2474.1 |
| equiv_d_microns | L1 | 894,892 | 188.6 | 116.3 | 1.3 | 1565.2 |
| circularity | L1 | 894,892 | 0.343 | 0.147 | 0.007 | 1.571 |
| solidity | L1 | 894,892 | 0.814 | 0.101 | 0.057 | 1.000 |
| complexity | L1 | 894,892 | 0.632 | 0.138 | -1.066 | 0.988 |
| roundness | L1 | 894,892 | 0.710 | 0.123 | 0.018 | 0.974 |
| particle_width_microns | L2 | 531,346 | 235.1 | 179.4 | 0.0 | 2378.5 |
| particle_height_microns | L2 | 531,346 | 235.0 | 179.7 | 0.0 | 2474.1 |
| equiv_d_microns | L2 | 512,145 | 199.2 | 123.8 | 1.4 | 1510.8 |
| circularity | L2 | 512,145 | 0.334 | 0.145 | 0.007 | 1.571 |
| solidity | L2 | 512,145 | 0.811 | 0.101 | 0.057 | 1.000 |
| complexity | L2 | 512,145 | 0.639 | 0.135 | -1.066 | 0.988 |
| roundness | L2 | 512,145 | 0.705 | 0.122 | 0.023 | 0.974 |

Particle sizes (hundreds of microns, mean ~224 µm) are physically
plausible for CPI-imaged ice crystals. **Caveat carried over from
existing reports** — these are COCPIT's own sizing convention, not an
absolute microphysical calibration: `docs/reports/2026-07-24-cpi-size-distribution-verification.md`
found a ~3.4–5.2x gap between COCPIT sizes and an independent SPEC PSD
product, attributed to sizing-definition differences and a ~36.7 µm
detection floor, not a pipeline defect in this join.

## Coverage: does every available L1/L2 record have matching features?

**No — overall 30.51% of L1 rows (914,527 / 2,997,447) and 29.05% of L2
rows (531,346 / 1,828,818) have a matched COCPIT feature row.** Full
per-campaign breakdown:

| Campaign | L1 rows | L1 matched | L1 % | L2 rows | L2 matched | L2 % |
|---|---:|---:|---:|---:|---:|---:|
| AIRS-II | 92,168 | 27,303 | 29.6% | 92,168 | 27,303 | 29.6% |
| ARM | 230,029 | 19,778 | 8.6% | 64,706 | 12,980 | 20.1% |
| ATTREX | 122,050 | 19,616 | 16.1% | 120,595 | 19,192 | 15.9% |
| CRYSTAL-FACE-NASA | 78,151 | 61,858 | 79.2% | 20,441 | 16,127 | 78.9% |
| CRYSTAL-FACE-UND | 1,608,674 | 393,751 | 24.5% | 848,940 | 201,302 | 23.7% |
| ESCAPE | 0 | 0 | — | 0 | 0 | — |
| ICE-L | 46,203 | 36,994 | 80.1% | 46,202 | 36,993 | 80.1% |
| IPHEX | 38,697 | 16,081 | 41.6% | 28,189 | 11,884 | 42.2% |
| ISDAC | 400,805 | 67,413 | 16.8% | 399,668 | 67,113 | 16.8% |
| MACPEX | 80,240 | 2,226 | 2.8% | 51,747 | 1,337 | 2.6% |
| MC3E | 173,766 | 151,439 | 87.2% | 137,272 | 118,750 | 86.5% |
| MIDCIX | 90,667 | 88,785 | 97.9% | 18,890 | 18,365 | 97.2% |
| MPACE | 35,997 | 29,283 | 81.4% | 0 | 0 | — |
| OLYMPEX | 0 | 0 | — | 0 | 0 | — |
| POSIDON | 0 | 0 | — | 0 | 0 | — |

Full CSVs: `logs/join_cocpit_features/latest/match_coverage_by_campaign.csv`
(includes the reverse direction — COCPIT rows with no L1/L2 match),
`feature_completeness.csv` (per-column completeness within matched rows —
uniformly ~29-30%, i.e. once a row is matched essentially every feature
column is populated; habit columns run ~0.7 percentage points lower than
size columns, a minor internal sparsity), `summary_stats.csv`.

## Problems and missing data

1. **ESCAPE, OLYMPEX, POSIDON: 0% coverage, but not a new problem.**
   These three campaigns already have 0 rows at L1/L2 — `data/raw/cpi_embeddings_timestamps.csv`
   (this pipeline's CPI image archive) has no imagery for them at all, a
   pre-existing, documented limitation (CLAUDE.md's "Campaigns" section)
   unrelated to COCPIT. COCPIT itself *does* have particle data for all
   three (7,106 / 93,741 / 22,841 rows respectively) — that data simply
   has nothing in this pipeline to join against.

2. **MACPEX: 2.6-2.8% coverage — a real, COCPIT-side data gap.** MACPEX
   has 80,240 CPI images at L1, but COCPIT's v1.4.0 MACPEX CSV contains
   only 2,226 particle rows total, *all* of which matched. COCPIT simply
   never processed the vast majority of MACPEX's raw CPI imagery into
   derived features — this is not a join-key or campaign-mapping bug (0
   COCPIT rows went unmatched), it's a gap in COCPIT's own processing
   coverage for this campaign specifically.

3. **Low coverage on ARM (8.6%), ATTREX (16.1%), ISDAC (16.8%): mixed
   causes, not fully resolved.** Each of these has a non-trivial number
   of COCPIT rows unmatched to any L1 row too (ARM: 4,181 of 23,959;
   ATTREX: 21 of 19,637; ISDAC: 5,981 of 73,394) — meaning the gap runs
   in both directions and isn't simply "COCPIT has fewer particles than
   this pipeline's CPI archive." Two known, already-documented factors
   likely contribute: (a) the pipeline's own ~6.34% CPI-image/env
   timestamp mismatch (CLAUDE.md's "Known issues," dominated by ISDAC and
   ARM specifically) removes some images from L1 entirely before this
   join even runs, and (b) COCPIT's own processing may not cover every
   flight date this pipeline's CPI archive does. Distinguishing these two
   effects precisely would need a direct COCPIT-filename vs.
   full-CPI-archive-filename set comparison, independent of L1 — **not
   done in this pass**, flagged as a follow-up rather than guessed at.

4. **CRYSTAL-FACE-UND (24.5%) and ARM/ATTREX/ISDAC above are collectively
   most of the reason overall coverage is ~30%, not because of any join
   defect** — MIDCIX (97.9%), MC3E (87.2%), MPACE (81.4%), ICE-L (80.1%),
   and CRYSTAL-FACE-NASA (79.2%) all show COCPIT features are genuinely
   available and correctly joinable for the majority of a campaign's
   images when COCPIT actually processed that campaign thoroughly — the
   low-coverage campaigns are specific COCPIT-processing-coverage gaps,
   not a systemic join failure.

5. **A stale, unrelated artifact found and explicitly rejected.**
   `v1.4.0/merged_env/` and `v1.4.0/environment/` subdirectories contain
   per-particle CSVs that already combine geometric AND environmental
   columns (e.g. `merged_env/CRYSTAL_FACE_NASA.csv`, same filename
   convention). These are dated 2022–2023 (predate this repo), and every
   sampled row's environmental columns (`Latitude`, `Pressure`,
   `Temperature`, `Ice Water Content`, ...) are the `-999.99` sentinel —
   i.e. missing. This is a broken artifact from an unrelated prior
   project, not this pipeline's env data, and was **not used** as a
   shortcut for this join.

6. **Some `particle_width_microns`/`particle_height_microns` values are
   0.0** (see summary-stats min column) — likely degenerate/near-zero
   detections at COCPIT's own processing floor; not investigated further
   in this pass, worth a follow-up filter if these rows matter for a
   downstream size-distribution analysis.

## Follow-up: why does `final_databases` undercount raw images? (2026-08-29, xcite)

Item 3 above ("mixed causes, not fully resolved") was investigated further
via two runs on xcite (the HPC cluster where COCPIT's actual pipeline and
raw image directories live) plus a read of COCPIT's own source
(`github.com/vprzybylo/cocpit`). Summary — **the mismatch is a
processing-history artifact, not a quality filter or a bug we can fix by
rerunning something**:

- **Confirmed on xcite, not just inferred from L1**: comparing raw
  `single_imgs_v1.4.0/` image filenames directly against
  `final_databases/vgg16/v1.4.0/<campaign>.csv` (bypassing this pipeline's
  L1 entirely) reproduces the same ~2.8%–98% per-campaign range found
  above, and **100% of every COCPIT CSV row has a matching raw image on
  disk in every campaign** — COCPIT's features are a strict *subset* of
  the raw images, never a mismatched/different set. This rules out a
  filename-format or campaign-mapping bug as the cause.
- **Independently reproduced on this machine**, via
  `scripts/verify_cpi_cocpit_overlap_local.py` — same method, but using
  `data/raw/cpi_embeddings_timestamps.csv` (this pipeline's own CPI image
  manifest) in place of xcite's raw image directories, since those don't
  exist locally. Row counts match xcite's raw-image counts almost exactly
  campaign for campaign, confirming the manifest is a reliable stand-in:

  | Campaign | Embeddings filenames | COCPIT rows | Overlap | Embeddings-only | COCPIT-only | % Embeddings matched | % COCPIT matched |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | AIRS_II | 92,201 | 27,303 | 27,303 | 64,898 | 0 | 29.61% | 100.0% |
  | ARM | 295,703 | 23,959 | 23,959 | 271,744 | 0 | 8.10% | 100.0% |
  | ATTREX | 129,128 | 19,637 | 19,637 | 109,491 | 0 | 15.21% | 100.0% |
  | CRYSTAL_FACE_NASA | 78,152 | 61,858 | 61,858 | 16,294 | 0 | 79.15% | 100.0% |
  | CRYSTAL_FACE_UND | 1,617,826 | 396,138 | 396,138 | 1,221,688 | 0 | 24.49% | 100.0% |
  | ICE_L | 46,236 | 37,020 | 37,019 | 9,217 | 1 | 80.07% | 100.0% |
  | IPHEX | 40,692 | 16,900 | 16,900 | 23,792 | 0 | 41.53% | 100.0% |
  | ISDAC | 505,812 | 73,394 | 73,394 | 432,418 | 0 | 14.51% | 100.0% |
  | MACPEX | 80,240 | 2,226 | 2,226 | 78,014 | 0 | 2.77% | 100.0% |
  | MC3E | 187,558 | 160,630 | 160,630 | 26,928 | 0 | 85.64% | 100.0% |
  | MIDCIX | 90,761 | 88,857 | 88,857 | 1,904 | 0 | 97.90% | 100.0% |
  | MPACE | 36,042 | 29,316 | 29,316 | 6,726 | 0 | 81.34% | 100.0% |
  | **Overall** | **3,200,351** | **937,238** | **937,237** | **2,263,114** | **1** | **29.29%** | **100.0%** |

  The single `ICE_L` discrepancy (37,020 vs 37,019 COCPIT rows matched)
  is consistent with one duplicate filename in the COCPIT CSV — not
  investigated further, negligible at this scale. Full run:
  `logs/verify_cpi_cocpit_overlap_local/latest/overlap_report.csv`.
- **Traced the actual cause in COCPIT's source**: `cocpit/__main__.py`
  runs 4 stages per campaign — crop-every-particle-into-`single_imgs_{TAG}/`
  (`_preprocess_sheets`), then CNN-classify-and-**overwrite the same CSV**
  (`_ice_classification`), then append geometry, then add dates. The
  historical script *at the `v1.4.0` git tag itself* has every campaign
  except `ICE_L` manually commented out of its campaign list — i.e. this
  is a hand-edited working script, re-targeted to one or a few campaigns
  per run, not an atomic all-campaigns batch job. Since
  `_preprocess_sheets` never deletes old crops (each run just adds more
  files to `single_imgs_{TAG}/`), but `_ice_classification` only reflects
  whichever run's classification pass last completed for that campaign,
  a campaign whose sheets kept growing after its last classification run
  ends up with far more images on disk than CSV rows — exactly the
  observed pattern, and consistent with why coverage varies so
  unpredictably campaign to campaign (each one has a different
  reprocessing history).
- Two candidate mechanisms considered and ruled out from the code itself:
  a fragment/sphere exclusion in `run_model.py` (`df[(df["classification"]
  != "fragment") & ...]` — the filtered result is never assigned back to
  `df`, so it's a real bug but a no-op, not the cause) and silent
  image-load skipping in `data_loaders.py`'s `TestDataSet` (no try/except
  around `Image.open()` — a bad file would crash the loader, not vanish
  quietly).
- **v3.1.0 follow-up (also run on xcite)**: checking `single_imgs_v1.4.0/`
  against `final_databases/vgg16/v3.1.0/` gave 0% overlap for 8 of 11
  campaigns (MC3E, ARM, IPHEX, AIRS_II, ICE_L, CRYSTAL_FACE_NASA,
  CRYSTAL_FACE_UND, ESCAPE) and 100%-COCPIT-matched for 3 (MACPEX,
  ATTREX, ISDAC). The 0%s are a filename-format break, not zero coverage:
  v3.1.0 uses a 4-field filename for those 8 campaigns
  (`2011_0523_183354_0.png`, no millisecond field) versus
  `single_imgs_v1.4.0`'s 5-field format
  (`..._163505_942_29.png`) — this is a real difference, confirmed by the
  0% overlap itself, but where the 4-field name comes from is unresolved
  (filenames here are `<sheet filename>_<contour index>.png`; the sheet
  filename originates outside this codebase, from SPEC Inc.'s `cpiview`
  tool per COCPIT's README, and the contour index depends on
  `cv2.findContours()`'s detection order, so neither is a stable,
  content-based ID guaranteed to survive reprocessing). MACPEX and ATTREX
  came back with row counts identical to v1.4.0, suggesting those two
  were simply carried over unchanged rather than reprocessed for v3.1.0;
  ISDAC kept the old filename format but a different row count, so it
  was genuinely reprocessed.
- **Unresolved**: no `single_imgs_v3.1.0/`-style directory exists
  anywhere under `cpi_data/campaigns/` on xcite, so v3.1.0's raw crops
  can't be directly inspected to confirm the filename-format hypothesis
  against the actual images that produced them. Checked and ruled out
  DVC as the explanation (the repo's `.dvc/config` remote is scoped only
  to the CNN's `training_datasets`, and zero `.dvc` pointer files exist
  in the repo for any `cpi_data`/`final_databases` path). The project's
  own base directory has demonstrably moved at least once (`v1.4.0`-era
  config template used `/data/data/cpi_data/...`; the live path today is
  `/home/vanessa/hulk/cocpit/cpi_data/...`), so the most likely
  explanation is that v3.1.0's raw crops were deleted after processing to
  reclaim disk space (standard practice — keep the small CSV of derived
  features, discard the much larger image crops) and/or were produced
  under the old, since-migrated path and never carried forward. Not
  confirmed; asking the data owner directly would resolve this faster
  than further code archaeology.
- **Bottom line for this pipeline**: the ~30% overall coverage figure
  above is real and not something a rerun of `join_cocpit_features.py`
  can improve — it reflects COCPIT's own incomplete/inconsistent
  processing history across campaigns and versions, not a defect in this
  join. v1.4.0 remains the right version to use (still the only one with
  all 15 campaigns and a filename convention confirmed to match
  `cpi_filename` for every campaign checked).

## Reproduce

```bash
conda activate cpi-thermo
python scripts/join_cocpit_features.py
```

Requires the external COCPIT database at
`/Users/josephko/research/cocpit/final_databases/vgg16/` — not portable
to another machine without that path (same caveat as
`derive_particle_size_microns.py`, `compare_derived_feature_versions.py`,
and the other COCPIT-reading scripts already in this repo).
