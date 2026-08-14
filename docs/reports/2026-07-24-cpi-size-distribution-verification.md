# CPI derived size verification against an archived, PI-produced size distribution

> **Update 2026-07-25:** see `docs/reports/2026-07-25-cpi-raw-image-independent-size-verification.md`.
> A follow-up independent measurement directly from raw CPI particle-image
> crops (not SPEC's PSD algorithm, not COCPIT's own pipeline) found COCPIT's
> derived sizes agree closely (within 14-22%) with that independent
> measurement. The 3.4x-5.2x gap documented below turned out to be a
> property of SPEC's bulk PSD product specifically, not evidence that
> COCPIT's pipeline inflates sizes. Treat this report's mechanism-hunting
> (the `remove_text()` floor discussion) as still useful background, but
> its overall framing ("COCPIT sizes are biased large, mechanism partially
> unexplained") is superseded by the newer report's conclusion.

**Question:** are COCPIT's derived `particle_width_microns` / `particle_height_microns`
/ `equiv_d_microns` (see `scripts/compare_derived_feature_versions.py` and
`scripts/derive_particle_size_microns.py`) physically reasonable, checked
against an independent, verified size-distribution product for the same
campaign?

**Answer, short version:** yes, in the sense that matters most -- no
units/scale bug. All three COCPIT size metrics and the archive's verified
particle size distribution (PSD) agree in order of magnitude, and the ratio
between them is tight and consistent (3.4x-5.2x, same direction) across 8
independently-sampled flight dates. But they are **not** a 1:1 match, and
part of that gap has a specific, identified, and partially-quantified cause
inside COCPIT's own pipeline (see "What explains the gap" below) -- this is
a genuine, still-partially-open finding, not a clean confirmation.

## What was checked, and why this campaign

Every campaign parser in this repo documents its raw-data source
(`parsers/<campaign>.py` docstrings). Looking for a size-distribution
product independent of COCPIT's own derived-feature archive -- something
the original instrument PI team produced and archived, that could serve as
ground truth for "does this many-microns number make physical sense" --
several campaign archives were tried and ruled out before finding one that
was both machine-accessible and login-free:

| Archive | Result |
|---|---|
| ARM (`iop.arm.gov`, `www.arm.gov`) | Both are now React single-page apps with no server-rendered content; the actual data archive (`archive.arm.gov/armlogin/login.jsp`) requires an account. |
| NCAR EOL (`data.eol.ucar.edu`, ICE-L/ESCAPE) | Also a React SPA (`create-react-app` bundle); dataset listing/file pages return only a header with no login-free way found to enumerate or fetch files via plain HTTP. |
| NASA Earthdata / GHRC (IPHEX DOI) | DOI resolves to an Earthdata catalog page; typical GHRC/ASDC downloads require an Earthdata Login account. |
| **NASA ESPO archive** (`espoarchive.nasa.gov`, CRYSTAL-FACE/ATTREX/MidCiX/MACPEX/POSIDON) | **Server-rendered Drupal 7 site, browsable and downloadable with plain anonymous HTTPS -- no login, confirmed by directly downloading files.** |

CRYSTAL-FACE-NASA (WB-57 aircraft) was the campaign used because its ESPO
archive directory (`archive/browse/crystalf/WB57/CP`) hosts exactly the
right product: files named `CP<YYYYMMDD>.WB57`, described as "CPI particle
habit & size distribution vs t," PI **Paul Lawson / SPEC Inc** -- the actual
company that designed, built, and operates the Cloud Particle Imager
instrument. This is about as authoritative a size-distribution product as
exists for CPI data: not a re-derivation from raw images by a third party
(like COCPIT), but the instrument team's own processing of their own raw
imagery. 8 flight dates are archived (2002-07-09, 11, 16, 19, 21, 23, 28,
29) and all 8 have exact-match coverage in COCPIT's `CRYSTAL_FACE_NASA`
derived-feature CSV (`v1.4.0`, which carries a `date` column).

## The verified PSD file format, and how it was validated

`CP<YYYYMMDD>.WB57` files are NASA Ames "irregular" format (FFI 2110): each
10-second measurement record has a *variable* number of particle-size bins
(number density in #/liter/micron vs. bin lower-edge in microns), plus 42
auxiliary fields (total concentration, surface area, ice water content,
and habit-resolved counts/surface-area/mass for 5 shape classes). The
parser (`scripts/verify_cpi_size_distribution.py::parse_cpi_wb57()`) reads
this record-by-record.

**Self-consistency validation:** for every record, `sum(concentration_i *
bin_width_i)` over that record's bins was compared against the record's own
independently-reported "total particle concentration" auxiliary field.
Across all 8 files and ~900 total 10-second records, the mean relative
error was **0.017%**, max **0.283%** (`logs/verify_cpi_size_distribution/latest/parser_closure_check.csv`)
-- strong evidence the bin-edge, bin-width, and concentration parsing is
correct, independent of trusting the file's (occasionally mislabeled,
e.g. "#/m^3" where the numbers are actually #/L) header units.

## Comparison result

Aggregated across all 8 matching flight dates (`figs/verify_cpi_size_distribution/latest/01_aggregate_comparison.png`):

| Date | SPEC verified PSD mean (µm) | COCPIT particle_width (µm) | COCPIT particle_height (µm) | COCPIT equiv_d (µm) | width/PSD | equiv_d/PSD |
|---|---|---|---|---|---|---|
| 2002-07-09 | 33.0 | 120.0 | 119.2 | 109.7 | 3.64x | 3.33x |
| 2002-07-11 | 30.4 | 158.6 | 159.5 | 138.7 | 5.22x | 4.57x |
| 2002-07-16 | 28.8 | 126.9 | 126.9 | 113.5 | 4.40x | 3.94x |
| 2002-07-19 | 29.0 | 113.3 | 115.4 | 99.6 | 3.91x | 3.44x |
| 2002-07-21 | 29.8 | 120.3 | 121.5 | 107.0 | 4.03x | 3.59x |
| 2002-07-23 | 33.0 | 113.1 | 112.3 | 100.6 | 3.43x | 3.05x |
| 2002-07-28 | 29.6 | 133.0 | 133.9 | 116.1 | 4.49x | 3.92x |
| 2002-07-29 | 27.8 | 132.8 | 133.5 | 115.9 | 4.77x | 4.16x |

**Average ratio: COCPIT particle_width is 4.24x the verified PSD mean;
COCPIT equiv_d is 3.75x.** Both ratios are tight (range ~3.4x-5.2x, not
scattered by orders of magnitude) and hold the same direction on every one
of the 8 independently-sampled dates -- this is the signature expected from
two products measuring genuinely the same physical population differently,
not the signature of a random bug (which would produce inconsistent or
wildly-off ratios date to date).

## What explains the gap: a confirmed size floor plus an unconfirmed residual

Two products should **not** be expected to match 1:1 here even with a
"correct" pipeline: SPEC's PSD is a true ambient number *concentration*
(per liter of sampled air, volume-normalized) while COCPIT's per-particle
columns are raw *counts of saved images* with no sample-volume weighting;
and each uses a different geometric size definition (SPEC/Lawson's own CPI
sizing algorithm vs. COCPIT's `cv2.minAreaRect()`-based bounding-box
width/height and `cv2.contourArea()`-based equivalent diameter). Some gap
was expected going in.

But the size of the gap has a specific, checkable partial explanation.
COCPIT's `parsers`-adjacent COCPIT extraction code
(`process_sheets.py::remove_text()`, in the external COCPIT repo, not this
one) masks any contour smaller than 200 native-pixel² as presumed sheet
text/timestamp noise **before** particles are cropped and saved -- so no
particle image is ever produced for a true ice crystal whose native-pixel
silhouette falls below that area. At the CPI probe's fixed 2.3
microns/pixel resolution, a 200 px² contour has an equivalent circular
diameter of **36.7 microns** -- matching, closely, the ~37-48 micron
1st-5th-percentile floor actually observed in COCPIT's own
`particle_width_microns`/`particle_height_microns` distributions.

Quantifying this against the *verified* PSD: **76.4%** of the verified
PSD's total number-weight lies below that 36.7 micron floor (most of what
the CPI instrument actually detected that day was smaller than what
COCPIT's pipeline could ever image). Excluding that population from the
verified PSD raises its weighted mean from 28.8 to 49.7 microns -- a 1.72x
shift, in the same direction as, but smaller than, the observed 3.4x-5.2x
gap.

**So `remove_text()`'s size floor explains part of the gap (~1.7x of the
~3.7-4.2x average), not all of it.** The residual is consistent with (but
not confirmed as) COCPIT's `particle_width`/`particle_height` measuring a
*maximum bounding-box dimension* -- for elongated or aggregate ice crystals
(the dominant habit in cirrus, CRYSTAL-FACE's target cloud regime), bounding-box
length is well known in the cloud-physics literature to run substantially
larger than an area-equivalent or instrument-native size metric. Additional
automated image-quality filtering further downstream in the COCPIT pipeline
(not traced in this investigation) may also contribute. **This residual
factor is flagged as an open item**, not resolved here.

## Verdict on the derived microns columns

`particle_width_microns`, `particle_height_microns`, and `equiv_d_microns`
are **not units-broken** -- their absolute scale is physically plausible
(tens to hundreds of microns, correct for cirrus ice crystals) and their
relationship to a verified, independently-produced PSD from the same
campaign is stable and explicable, not arbitrary. They should **not** be
treated as a direct stand-in for true ambient particle size distributions,
however: COCPIT's own `remove_text()` filtering step measurably excludes
the majority (by number) of the smallest particles the CPI instrument
actually recorded, and the residual size inflation beyond that is not yet
fully explained. Anyone using these columns for microphysical
size-distribution work (not just relative/comparative use, e.g. within-COCPIT
version comparisons) should treat them as **biased toward larger sizes**
relative to the true ambient PSD, with the bias's magnitude established
here (~3.4x-5.2x mean-size inflation for CRYSTAL-FACE-NASA) but not yet
fully mechanistically decomposed.

## Reproducing this analysis

```bash
python scripts/verify_cpi_size_distribution.py
```

Reads the 8 `CP<YYYYMMDD>.WB57` files already downloaded to
`data/raw/CRYSTAL-FACE-NASA/CPI_SPEC_verification/` (gitignored, like all
of `data/`; re-download from
https://espoarchive.nasa.gov/archive/browse/crystalf/WB57/CP if missing --
anonymous HTTPS, no login) and COCPIT's `CRYSTAL_FACE_NASA` v1.4.0 CSV.
Outputs to `logs/verify_cpi_size_distribution/<timestamp>/` (CSVs,
`summary_report.md`, `run_config.json`) and
`figs/verify_cpi_size_distribution/<timestamp>/` (3 PNGs), both with
`latest` symlinks.
