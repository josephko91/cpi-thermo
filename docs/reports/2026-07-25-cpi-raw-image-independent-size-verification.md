# Independent CPI size verification from raw particle-image archive

Follow-up to `docs/reports/2026-07-24-cpi-size-distribution-verification.md`,
which compared COCPIT's derived `particle_width_microns`/
`particle_height_microns`/`equiv_d_microns` (CRYSTAL-FACE-NASA) against SPEC
Inc's own bulk, volume-normalized ambient PSD product and found a
consistent 3.4x-5.2x mean-size gap. That comparison had a structural
limitation: the SPEC PSD is itself a *derived* product (someone else's
sizing algorithm's output), not raw imagery, so it couldn't determine
whether the gap was a COCPIT-pipeline problem or an artifact of comparing
two different derived products. This follow-up adds a third, more primary
reference point: particle sizes measured **directly from raw, individual
CPI particle-image crops**, run through neither SPEC's nor COCPIT's own
sizing algorithm.

**Headline result: this revises the prior report's framing.** COCPIT's
derived sizes track the independent raw-image measurement closely (within
14-22%, consistently, across all 8 flight dates) -- not the 3.4x-5.2x gap
that made COCPIT's numbers look potentially broken. The 3-4x-scale gap is
real, but it's a property of comparing against SPEC's bulk PSD specifically,
not evidence that COCPIT's pipeline is inflating sizes.

## Data source and access

Same ESPO archive as the prior report (`espoarchive.nasa.gov`, anonymous
HTTPS, no login), a second CPI product for CRYSTAL-FACE-NASA/WB57: gallery
1637, "CPI cloud particle images" (PI: Paul Lawson/SPEC Inc -- the same PI
as the PSD product), one PDF per minute-of-flight, for the same 8 matched
flight dates used previously. Confirmed by hand: individual page timestamps
match SPEC's own PSD record times exactly, and each page's imagery is
genuine per-particle content (habit-labeled: sph/col/sir/bir/...), not a
plot or summary graphic.

Downloaded all 8 matching dates (384 PDF files, 402MB total -- well under
the 5GB budget), respecting the archive's `robots.txt` `Crawl-delay: 10`.
Cached at `data/raw/CRYSTAL-FACE-NASA/CPI_raw_images_verification/<date>/`
(gitignored, like all of `data/`).

## Extraction method (and a real structural surprise)

The original plan assumed each individual particle crop was a separate
embedded image (PDF XObject) that could be pulled out directly. That
assumption was **wrong**, discovered immediately on the first real
extraction attempt: every embedded image count matched the page count
exactly (e.g. 13 pages, 13 embedded images) -- each PDF **page** embeds
exactly **one** composite raster covering the whole montage layout
(particle crops + background texture + text labels baked into a single
bitmap), not one image per particle.

This required a different segmentation approach: Otsu-threshold the whole
page, connected-component label it, and classify each component. A second
surprise during calibration: individual text glyphs (timestamp digits,
3-letter habit codes) turned out to be near-square at this resolution
(aspect ratio ~1.0-1.3) -- indistinguishable from small particles by aspect
ratio alone, invalidating the "text is elongated" assumption from the
original plan. The actual working discriminator turned out to be **area**:
the per-page component-size distribution is extremely bimodal (median ~5
px^2 -- single/few-pixel compression noise; genuine particle crops
300-37,000 px^2), with almost no components in between. Final filters,
calibrated against a visual spot-check (`--debug` mode, annotated
bounding-box overlays on real pages, manually inspected):

- exclude the page's top 20px (the fixed single-line date/title header,
  present identically on every page)
- reject components under 150 px^2 (noise/text-glyph fragments)
- reject components with bounding-box aspect ratio over 3.0 (merged
  multi-character text runs)

Visual inspection of the debug overlays (2 full pages, ~95 candidate
regions) showed clean, accurate particle boxing with no obvious
text-glyph false positives -- see the module docstring in
`scripts/verify_cpi_raw_image_sizes.py` for the full calibration writeup.
A caught-and-fixed bug along the way: an off-by-one in the Otsu threshold
comparison (`arr < t` vs the correct `arr <= t`) that would have produced
empty particle masks -- verified with a synthetic test before trusting any
real output. A second bug -- the gallery listing pages are paginated at 15
items/page, silently truncating every date with more than 15 files -- was
caught by comparing downloaded file counts against the archive's own
listed totals and fixed before the real download run.

**Known limitation, disclosed not eliminated:** the 150 px^2 / 3.0
aspect-ratio filters can't perfectly separate small text labels from small
particles. Any residual contamination biases the independent measurement
toward *smaller* sizes -- which, if anything, works against (not for) the
close agreement with COCPIT found below, so it doesn't undermine the
headline result.

## Result

147,253 particles independently measured across 384 files, 8 flight dates
(`logs/verify_cpi_raw_image_sizes/latest/raw_image_particle_measurements.csv`).
Aggregate stats: mean bounding-box width 109.1 microns (median 94.3),
mean equivalent diameter 109.8 microns; aspect ratios mostly 1.0-1.3
(compact, round-ish shapes -- consistent with real particle silhouettes,
not stray text fragments); only 0.26% of accepted components touch the
crop border (negligible edge-truncation risk).

| Date | SPEC PSD | COCPIT width | COCPIT equiv_d | Raw-image bbox width | Raw-image equiv_d | raw/PSD ratio | COCPIT/raw ratio |
|---|---|---|---|---|---|---|---|
| 2002-07-09 | 33.0 | 120.0 | 109.7 | 98.5 | 98.1 | 2.99x | 1.22x |
| 2002-07-11 | 30.4 | 158.6 | 138.7 | 137.3 | 143.5 | 4.52x | 1.16x |
| 2002-07-16 | 28.8 | 126.9 | 113.5 | 106.6 | 106.5 | 3.70x | 1.19x |
| 2002-07-19 | 29.0 | 113.3 | 99.6 | 97.2 | 96.1 | 3.35x | 1.17x |
| 2002-07-21 | 29.8 | 120.3 | 107.0 | 105.7 | 106.0 | 3.55x | 1.14x |
| 2002-07-23 | 33.0 | 113.1 | 100.6 | 96.5 | 96.3 | 2.93x | 1.17x |
| 2002-07-28 | 29.6 | 133.0 | 116.1 | 109.4 | 111.0 | 3.70x | 1.22x |
| 2002-07-29 | 27.8 | 132.8 | 115.9 | 111.2 | 111.1 | 4.00x | 1.19x |

Two patterns, both tight and consistent across all 8 independently-sampled
dates (not scattered or coincidental):

1. **COCPIT vs independent raw-image measurement: 1.14x-1.22x** -- COCPIT's
   derived `particle_width`/`equiv_d` run only 14-22% higher than sizes
   measured completely independently from raw imagery, with neither
   COCPIT's nor SPEC's sizing code involved. This is a small, explainable
   gap (COCPIT's minAreaRect-based width vs. this analysis's simpler
   thresholded bounding box, plus COCPIT's `remove_text()` floor removing a
   slightly different small-particle population than this analysis's area
   filter) -- not evidence of a broken pipeline.
2. **Independent raw-image measurement vs SPEC PSD: 2.93x-4.52x** -- nearly
   identical in magnitude to the COCPIT-vs-SPEC gap found in the prior
   report. Since this measurement uses neither COCPIT's nor SPEC's
   algorithm, the fact that it *also* lands 3-4.5x above SPEC's PSD means
   the gap is not something specific to COCPIT's pipeline -- it's a
   property of SPEC's bulk PSD product itself (see below).

See `figs/verify_cpi_raw_image_sizes/latest/01_aggregate_three_way_comparison.png`
(pooled density comparison -- COCPIT and the independent raw-image
measurement visually overlap closely across the whole distribution shape;
SPEC's PSD is sharply concentrated near zero, a clear outlier) and
`02_per_date_three_way_mean_size.png` (per-date bars showing the same
pattern holds on every individual date, not just in aggregate).

## Why SPEC's bulk PSD sits so much lower

Only **4.9%** of the independently-measured raw particles fall below the
36.7-micron floor implicated in the prior report's `remove_text()`
explanation -- compared to **76.4%** of SPEC's own PSD's number-weight
sitting below that same floor (prior report's finding). That's a striking
asymmetry: whatever is producing SPEC's PSD's enormous population of very
small particles, it is not something visible in this archived,
gallery-rendered particle-image product -- not because of COCPIT's
`remove_text()` filtering (this analysis never touches COCPIT's code or
images), but because these PDF-archived crops themselves don't show nearly
as many tiny particles as SPEC's own bulk concentration numbers imply.
Plausible explanations (not resolved here): SPEC's PSD algorithm may
operate directly on higher-fidelity/lower-compression native sensor frames
than what ended up in this public-facing PDF gallery product, or bins many
sub-resolution/near-noise-floor detections into its concentration counts
in a way that a from-scratch, conservative image-segmentation approach
naturally excludes.

## Revised verdict on COCPIT's derived size columns

The prior report's verdict ("not broken, but biased toward larger sizes,
magnitude ~3.4x-5.2x vs. verified PSD, mechanism partially unexplained")
should be **updated**: COCPIT's `particle_width_microns`/
`particle_height_microns`/`equiv_d_microns` agree closely (within ~15-22%)
with an independent, from-scratch measurement off the same campaign's raw
particle imagery. The large gap against SPEC's bulk PSD specifically
appears to be a property of that PSD product -- not a defect in COCPIT's
pipeline. Anyone using COCPIT's derived sizes for absolute microphysical
work should still be aware they may under-sample the very smallest
particles (any pixel-based method from this era's compressed/archived CPI
imagery seems to, based on this analysis's own small-particle detection
rate), but the core sizing itself is corroborated, not contradicted, by
this independent check.

## Reproducing this analysis

```bash
python scripts/verify_cpi_raw_image_sizes.py --download   # ~70 min, ~402MB
python scripts/verify_cpi_raw_image_sizes.py --debug --dates 20020709  # spot-check
python scripts/verify_cpi_raw_image_sizes.py               # full run
```

Outputs to `logs/verify_cpi_raw_image_sizes/<timestamp>/` (CSVs,
`summary_report.md`, `run_config.json`) and
`figs/verify_cpi_raw_image_sizes/<timestamp>/` (2 PNGs + debug crops), both
with `latest` symlinks.
