# Data Harmonization — Source Material — 2026-08-28

Source material for a Nature Scientific Data paper's "Data harmonization"
methods subsection — standardization logic and final dataset construction.
Companion report: `docs/reports/2026-08-28-dataset-construction-and-qc.md`
(data sources, pre-processing, QC).

## 1. Standardized schema

Every campaign parser's `extract_*_standard()` function returns the same
core column set, regardless of the campaign's native raw format:

```
Timestamp, Tair_C, P_hPa, Si, Si_chilled_mirror, Si_<instrument>, qv,
qv_chilled_mirror, qv_<instrument>, Sw, Lat, Lon, Alt_m, Campaign,
source_file
```

plus, for the 14 of 15 campaigns with turbulence data, `Wind_U_ms`,
`Wind_V_ms`, `Wind_W_ms`, and unified `EDR_m23s1`. Conventions applied
uniformly across all 15 campaigns regardless of source format:

- Temperature in **Celsius** (`Tair_C`), pressure in **hPa**, altitude in
  **meters** — chosen for direct cross-campaign comparability, since raw
  formats mix Celsius/Kelvin, hPa/Pa/mb, and meters/feet.
- All timestamps normalized to **UTC** (`tz_localize`/`tz_convert`),
  resolving campaigns whose raw archive uses local time (e.g. the MC3E CDT
  mislabel found during pre-processing, see the construction/QC report).
  §2 of that report has other fixes categorized similarly.
- `Si` (ice supersaturation, dimensionless, >0 = supersaturated) and `Sw`
  (liquid-water supersaturation) computed from a common thermodynamic
  basis (`parsers/utils.py`, Murphy & Koop 2005 saturation vapor
  pressure), not each instrument's own onboard Si/RH calculation —
  ensures every campaign's Si is comparable on the same physical formula
  regardless of which raw quantity (frost point, RH-ice, ppmv mixing
  ratio) the source instrument reported.
- `qv` (water vapor mass mixing ratio, g/kg) similarly computed from a
  common `qv_from_e_P` / `qv_from_ppmv` basis.
- A shared missing-value convention: all fill values replaced with NaN
  during `load_*()`, before any downstream calculation — never after.

A campaign whose parser doesn't produce a given optional column (e.g. a
turbulence field) simply contributes NaN there — no registry or schema
migration is needed, and `main.py`'s final `pd.concat(all_dfs)` is a plain
outer union of columns across campaigns.

## 2. Multi-instrument harmonization

**Best-available Si/qv selection.** Many campaigns flew more than one
water-vapor instrument simultaneously. Rather than picking one arbitrarily
or averaging across instruments with different physics, each campaign has
an explicit, ordered fallback priority (`config.yaml`'s `h2o_ranking`),
and the parser fills `Si`/`qv` from the first available (non-NaN) source
in that order, row by row:

| Campaign | h2o_ranking (priority order) |
|---|---|
| ARM | chilled-mirror |
| AIRS-II | chilled-mirror |
| ATTREX | DLH → NOAA → UCATS |
| CRYSTAL-FACE-NASA | JLH → HW → ALIAS |
| CRYSTAL-FACE-UND | LH-unspecified → chilled-mirror |
| ESCAPE | chilled-mirror |
| ICE-L | chilled-mirror → MRTDL |
| IPHEX | chilled-mirror → ophir-tdl |
| ISDAC | chilled-mirror (on-board RH-ice sensor) |
| MACPEX | HWV → DLH → JLH → FISH |
| MC3E | DLH |
| MIDCIX | JLH |
| MPACE | *(none — no water-vapor instrument flown; Si/qv NaN by design)* |
| OLYMPEX | frost-point |
| POSIDON | DLH |

Every raw per-instrument reading is *also* preserved in its own column
(`Si_JLH`, `qv_dlh`, etc. — not overwritten by the ranking), so the
"which instrument actually produced this campaign's `Si`" provenance is
never lost, and a downstream analysis that wants one specific instrument's
reading rather than the campaign's best-available blend can still get it.

**EDR unification — a worked example of physical-unit harmonization.**
Eddy dissipation rate (turbulence intensity, eps^(1/3)) arrived from three
genuinely different raw conventions: NASA Ames MMS files report log10(eps)
in kW/kg (ATTREX, POSIDON); the UND Citation ASCII pipeline reports
eps^(1/3) directly but in cm^(2/3)·s⁻¹ (IPHEX, MC3E, MPACE, OLYMPEX,
CRYSTAL-FACE-UND); ARM's older binary archive uses the same UND house
cm-convention despite a different file format. Each was converted to a
single ICAO/WMO-standard `EDR_m23s1` column in m^(2/3)·s⁻¹
(`parsers/utils.py::edr_from_mms_log10kWkg`, `edr_from_und_cm23s1`),
verified by confirming all three converted families land in the same
physical range on a shared histogram rather than three disjoint
sub-ranges (`docs/decisions/2026-07-13-edr-unification.md`,
`docs/reports/2026-07-13-turbulence-scope-reduction-diagnostics.md`).
This process also caught two real bugs (an undocumented MMS fill-flag
cluster, and ARM's raw values being in the wrong assumed length unit) —
documented in the same decision doc as a worked example of why
verifying a physical-range overlap, not just applying a stated conversion
factor, matters for harmonizing across instrument families.

## 3. Cross-instrument and cross-campaign temporal alignment

Every merge in this pipeline — combining multiple instruments within one
campaign, and joining CPI particle images to environmental data — is an
**exact-second join, never a merge-tolerance join.** Each instrument's own
timestamp is floored (not rounded) to the nearest whole second
(`parsers/utils.py::round_timestamp_to_second`), then combined via
`pd.merge(..., on="Timestamp", how="outer")`. A second with no reading
from a given instrument is NaN for that instrument's columns — never a
nearest-neighbor value borrowed from a different second.

This is a deliberate, documented design choice
(`docs/decisions/2026-07-07-exact-second-merge-rewrite.md`, GitHub issue
#12), replacing an earlier `pd.merge_asof(tolerance=...)` architecture
that was found to be silently fabricating time resolution the raw data
didn't actually have — i.e. borrowing a stale reading from up to a
tolerance window away and presenting it as if it were simultaneous. The
rewrite grew the L0 row count substantially (documented in
`docs/dataset-changelog.md`), since previously-merged rows that had only
existed because of the tolerance window disappeared once the tolerance
was removed. Floor (not round) is used specifically because some raw
sources sample at a fixed 0.5-second offset; banker's-rounding would
collide two physically distinct adjacent samples into one duplicate
timestamp, while flooring preserves the original spacing with no
collisions.

## 4. Final dataset construction: L0 → L1 → L2

Three tiers, built by `main.py` (L0) and `scripts/build_data_tiers.py`
(L1, L2):

**L0** (`data/out/combined_env_data.parquet`): every whole second where
*any* instrument in a campaign reported *anything* — the union of every
instrument's timestamps within that campaign, produced directly by each
parser's internal outer merges, then `pd.concat`-ed across all 15
campaigns. **4,572,581 rows, 46 columns, 15 campaigns.**

**L1** (`data/out/combined_env_data_L1.parquet`): one row per CPI
(Cloud Particle Imager) particle image, produced by an inner join of each
campaign's CPI image timestamps (`parsers/cpi_timestamps.py`, the
canonical loader for `data/raw/cpi_embeddings_timestamps.csv`) against
that same campaign's L0 rows, on the exact floored second. This join is
done **per campaign, not globally** — a CPI image from one campaign can
never spuriously match an L0 row from a different campaign that happens
to share the same wall-clock second. Because some campaigns' L0 is
genuinely multi-Hz (e.g. ARM's native 4 Hz stream), L0 is deduplicated to
one row per (campaign, floored second) — keeping the first actually-
observed sample, never an average, since a mean across sub-second samples
would synthesize a value that existed at no real instant — **for this
join only**; L0 itself and its native sub-second resolution are
untouched. Multiple CPI images sharing one second each get their own L1
row, with that second's env data duplicated across them (a `cpi_filename`
column keeps every row traceable to its source image). An image with no
matching L0 second is dropped — no instrument reported anything at that
time, so there's nothing to join it to. **2,997,447 rows, 47 columns
(adds `cpi_filename`), 12 campaigns** (OLYMPEX, POSIDON, ESCAPE
contribute 0 rows — no CPI imagery in this pipeline's inputs for those
three).

**L2** (`data/out/combined_env_data_L2.parquet`): L1 filtered to rows
where every one of 7 "core" variables is present —
`Tair_C, P_hPa, Si, qv, Lat, Lon, Alt_m` (`CORE_COLS` in
`scripts/build_data_tiers.py`; `Sw` is excluded since it's derived
entirely from `Si`+`Tair_C` and adds no independent completeness signal).
**1,828,818 rows, 47 columns, 11 campaigns** (MPACE additionally drops out
at L2 — it has CPI imagery and reaches L1, but flew no water-vapor
instrument, so `Si`/`qv` are NaN on every row and it fails the
core-completeness filter entirely). Turbulence and per-instrument fallback
columns are deliberately **not** part of the L2 gate — they ride through
ungated at every tier, since gating on them would drop the many campaigns
with strong core-thermodynamic coverage but partial turbulence coverage.

Full per-tier column inventory, per-variable completeness, and
per-campaign row-count breakdown: `docs/reports/2026-08-28-dataset-summary.md`.

## 5. Reproducibility

The entire construction pipeline is deterministic and re-runnable from
the raw archives with no manual steps:

```bash
conda activate cpi-thermo
python main.py                          # L0
python scripts/build_data_tiers.py      # L1, L2
python scripts/qa_checks.py             # 9 QC checks
python scripts/diagnose_cpi_fusion.py   # CPI/env timestamp match diagnostic
```

This was independently re-verified during this session
(`docs/reports/2026-08-28-dataset-validation.md`): rebuilding L0/L1/L2 and
rerunning all 9 QC checks from the current code reproduced the exact same
row counts, QC flag counts, and CPI/env fusion percentages as the
prior known-good build — confirming the pipeline has no hidden
non-determinism (e.g. dict/set ordering, floating-point accumulation
order, or file-glob ordering dependence) between runs.
