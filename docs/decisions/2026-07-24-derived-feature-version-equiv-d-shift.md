# Why v3.1.0 geometric features (equiv_d, extreme_points, perim_area_ratio, ...) differ sharply from v1.2.0/v1.4.0

Follow-up to `scripts/compare_derived_feature_versions.py`, which found large
Kolmogorov-Smirnov shifts in pixel-scale geometric features (`equiv_d`,
`extreme_points`, `perim`, `hull_area`, `convex_perim`, `cnt_area`) between
the COCPIT vgg16 derived-feature CSVs' v1.2.0/v1.4.0 and v3.1.0 for nearly
every campaign (see `logs/compare_derived_feature_versions/latest/summary_report.md`).
For ARM, mean `equiv_d` drops from ~650 px (v1.2.0/v1.4.0) to ~70 px (v3.1.0)
-- roughly a 9x difference, too large to be an algorithm-precision change.

## Investigation

Cloned `github.com/vprzybylo/cocpit` and compared the geometry-calculation
code across git tags `v1.2.0`, `v1.4.0`, and the closest match to "v3.1.0"
(tag `3.1.0`, no `v` prefix -- the repo's tag naming is inconsistent; `v3.0.0`
and `v3.2.0` bracket it and share the same code for the files in question).

**Ruled out: a geometry-calculation code change.** `find_contours()`,
`morph_contours()`, and the equiv_d/perim/area formulas
(`cocpit/pic.py` in v1.4.0 vs `cocpit/geometry.py`/`geometry_runner.py` in
3.1.0, introduced by commit `b4c63a5` "create read the docs using sphinx",
2022-04-29, first present in v3.0.0/3.1.0/v3.2.0) are byte-identical in
threshold value (50), contour-finding flags, and the order area/perim/hull
are computed relative to `morph_contours()`. The v1.4.0 -> v3.1.0 refactor
split one file into two and added type hints/docstrings, but did not change
the pixel math.

**Ruled out: a resize/crop-size change in image extraction.**
`cocpit/process_sheets.py`'s `extract_contours()` is also byte-identical
between v1.4.0 and 3.1.0: both crop each particle's bounding rect from the
raw sheet, then `cv2.resize(cropped, (1000, 1000), interpolation=cv2.INTER_AREA)`
before saving. Spot-checked local PNGs in
`/Users/josephko/research/cocpit/cpi_data/ARM/{single_imgs_v1.2.0,single_imgs_v1.4.0,single_imgs}`
-- all confirmed 1000x1000, no resolution difference across the versioned
image directories that do exist locally.

**Found: the two CSVs were generated from different, non-overlapping raw
image extractions.**
- `v1.2.0/ARM.csv` and `v1.4.0/ARM.csv` share 21,490 filenames (of ~24-25k
  each) -- they're two runs of the pipeline against essentially the same
  extracted-particle image set.
- `v3.1.0/ARM.csv` shares **zero** filenames with either v1.2.0 or v1.4.0
  (checked directly: `set(v1.2.0.filename) & set(v3.1.0.filename)` and
  `set(v1.4.0.filename) & set(v3.1.0.filename)` are both empty).
- The filename *convention* itself differs. v1.2.0/v1.4.0 use
  `<date>_<time>_<millisecond>_<particle_index>.png` (e.g.
  `2000_0313_211521_883_0.png`). v3.1.0 uses
  `<date>_<time>_<particle_index>.png` -- **no millisecond field**
  (e.g. `2000_0309_214544_5.png`). This matches the naming pattern of a
  different raw-sheet source: `cpi_data/ARM/sheets/2000_0309_214544_644.png`
  is a 792x1024 whole (uncropped, multi-particle) raw sheet image using a
  filename pattern (`date_time_millisecond.png`, no particle index) that,
  once run through `extract_contours()`'s per-particle loop
  (`self.file_out = self.file[:-4] + "_" + str(i) + ".png"`), would produce
  child filenames matching v3.1.0's pattern *if the parent sheet name lacked
  the millisecond field* -- i.e. v3.1.0's particles were extracted from a
  sheet-image archive with a different/older naming convention than the one
  behind v1.2.0/v1.4.0.
- v3.1.0's date coverage is also wider: 2000-03-06 to 2000-03-19, vs
  v1.4.0's 2000-03-09 to 2000-03-19 -- v3.1.0 includes 3 additional early
  flight days not present in v1.4.0's CSV at all.

**Units clarification (important):** `equiv_d`/`perim`/`hull_area`/`cnt_area`/
`extreme_points`/`convex_perim` are computed in `cocpit/pic.py`'s
`get_attributes()` (v1.4.0) / `cocpit/geometry_runner.py` (3.1.0) by loading
the **already-saved 1000x1000 PNG** from disk (`image.resize_stretch()` is
called nowhere in this path -- it's commented out) and running
`find_contours()`/`calculate_area()`/etc directly on it. So these features
are in pixels of the **resized 1000x1000 canvas**, not native CPI sensor
pixels. The native 2.3 microns/pixel constant (`process_sheets.py`'s
`particle_dimensions()`) is only ever applied to `particle_width`/
`particle_height`, measured **before** the resize on the true crop
(`self.width`/`self.height`, saved as the `frame width`/`frame height`
columns). Since `cv2.resize(cropped, (1000, 1000))` is a **non-uniform**
stretch (aspect ratio not preserved -- both dimensions forced to exactly
1000), the microns-per-resized-pixel scale is not a fixed constant; it's
different per particle and per axis:

```
scale_x = frame_width_native_px  * 2.3 / 1000   # microns per resized-x-pixel
scale_y = frame_height_native_px * 2.3 / 1000   # microns per resized-y-pixel
```

For an area-derived length like `equiv_d`, the appropriate approximate
conversion is the geometric mean `sqrt(scale_x * scale_y)` (area scales as
`scale_x * scale_y`, and equiv_d is `sqrt(area)`-like).

**Tested this conversion directly on ARM and it does NOT reconcile the
v1.4.0/v3.1.0 gap.** `frame width`/`frame height` (the native pre-resize
crop size -- i.e. the scale factor itself) are nearly identical between
versions (mean ~148px/148px for v1.4.0 vs ~145px/146px for v3.1.0) -- so a
resize-stretch-scale artifact from *differently-sized source crops* is
**ruled out**, contradicting the original hypothesis in this document's
first draft. Applying the per-particle micron conversion to both versions:

| Version | equiv_d (px, canvas space) | equiv_d converted to microns |
|---|---|---|
| v1.4.0 | mean 670 px | mean ~227 microns |
| v3.1.0 | mean 96 px | mean ~33 microns |

A ~7x gap remains even in physical units. This means the discrepancy is not
a units/scale artifact at all -- the segmented contour genuinely occupies a
much smaller fraction of the resized canvas in v3.1.0 than in v1.4.0, for
crops of matched native size. That points to a difference in **what the
contour-finding/thresholding actually segmented** (different source pixel
content for nominally-matched crop boxes, different effective threshold
behavior, or the v3.1.0 batch's particles genuinely being smaller within
their bounding boxes) rather than a pure resize/units issue. This was not
fully resolved -- pinning it down further needs the raw sheet archives and
extraction run logs for the v3.1.0 batch, which aren't in this git history
(extraction is triggered externally, not committed).

Circularity/solidity/roundness/phi (ratios of two same-scale measurements)
are far less affected than absolute measures like `equiv_d`/`perim`/
`hull_area`/`cnt_area` (ARM circularity/solidity KS ~0.13-0.25 vs
equiv_d/perim/hull_area KS ~0.96-0.99) -- consistent with *some* scale-like
effect still being part of the story, but the frame-width test above shows
it can't be explained by crop-size/resize-stretch alone.

## Recommendation: which version to use

**Use v1.4.0** for any analysis involving the absolute-scale shape
descriptors (`equiv_d`, `perim`, `hull_area`, `convex_perim`, `cnt_area`,
`extreme_points`, `blur`):
- It shares 84-90%+ of its particles with v1.2.0 (cross-validated, stable
  pipeline output), while v3.1.0 is an entirely disjoint extraction with an
  unexplained scale shift in exactly the pixel-absolute features.
- v1.4.0 has the fullest, most consistent column schema of the three
  non-v1.3.0 versions (36 cols, includes both the raw-pixel geometry and the
  `[microns]`-scale particle_width/height), avoiding v1.3.0's much smaller
  17-column schema (no shape descriptors at all) and v3.1.0's currently
  unexplained scale discontinuity.
- It's also what the repo's own dashboard code (`cocpit/dash_app/processing_scripts/process.py`)
  reads as the production-merged dataset (`final_databases/vgg16/v1.4.0/merged_env/`),
  suggesting the pipeline authors themselves treated v1.4.0, not v3.1.0, as
  the canonical/finalized geometric-feature build.

**Do not mix v1.4.0 and v3.1.0 equiv_d/perim/hull_area/etc. in the same
analysis** without first confirming (e.g. by pulling a handful of the same
physical particles from both extraction batches, if that's ever possible, or
finding the extraction logs/config for the v3.1.0 run) whether a rescaling
correction is needed. Ratio-based shape descriptors (circularity, solidity,
roundness, phi, complexity) are comparatively safe to use from either
version if v3.1.0's wider date/campaign coverage is otherwise wanted, since
those are far less sensitive to the apparent scale artifact.

If the wider flight-date coverage in v3.1.0 (3 extra ARM flight days, and
likely similar for other campaigns -- not individually checked here) is
important, consider re-deriving `particle_width`/`particle_height`-based
size metrics from v3.1.0 (those use a fixed micron/px constant on the
pre-resize bounding rect and are not implicated in this shift) rather than
trusting its raw-pixel equiv_d/perim family directly.

## Converting equiv_d (and perim/hull_area/cnt_area) to physical units

Yes, possible, per-particle, using the `frame width [pixels]`/
`frame height [pixels]` (v1.4.0: `frame width`/`frame height`) columns as
the scale reference -- these record the true pre-resize crop box, and the
CPI probe's native resolution is a fixed 2.3 microns/pixel
(`process_sheets.py::particle_dimensions()`):

```python
scale_x = df["frame width"]  * 2.3 / 1000   # microns per resized-canvas x-pixel
scale_y = df["frame height"] * 2.3 / 1000   # microns per resized-canvas y-pixel

# length-like features measured along a single known axis: use scale_x or scale_y directly
# area-like / equiv_d-like features: use the geometric mean (area scales as scale_x * scale_y)
df["equiv_d_microns"] = df["equiv_d"] * np.sqrt(scale_x * scale_y)
df["perim_microns"]   = df["perim"]   * np.sqrt(scale_x * scale_y)  # approximate: perim isn't
                                                                      # purely area-scaled, but
                                                                      # this is the best available
                                                                      # single-factor approximation
```

This conversion is legitimate and worth applying **within a single version**
(e.g. to get real micron-scale equiv_d out of v1.4.0). It does **not**,
however, explain or fix the v1.4.0-vs-v3.1.0 discrepancy -- as shown above,
both versions' `frame width`/`frame height` (hence `scale_x`/`scale_y`) are
already nearly identical, so converting both to microns preserves essentially
the same ~7x gap. Don't treat a same-version micron conversion as license to
then mix v1.4.0 and v3.1.0 equiv_d values together -- the unexplained
per-version discrepancy is upstream of pixels vs. microns.
