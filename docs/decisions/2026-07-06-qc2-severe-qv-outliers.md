# 2026-07-06 — QC2 severe qv_exceeds_saturation outside the in-cloud set

## Fix applied in commit: (pending — AIRS-II in-cloud classification)

## What we found

QC2's severe `qv_exceeds_saturation` flags (`qv > 1.20 x qv_sat_liq`) include
4 campaigns outside the `IN_CLOUD_CAMPAIGNS` set documented in
`docs/decisions/2026-07-05-qc2-severity-tiers.md`, with no prior
investigation: AIRS-II (399), MACPEX (77), ATTREX (13), POSIDON (1).
Investigated each; only AIRS-II turned out to be a genuine categorization gap.

**AIRS-II (399 rows, all 12 flights) — miscategorized, not investigated.**
AIRS-II (Alliance Icing Research Study II) uses the same chilled-mirror
dew-point instrument (`DPXC`/`RHUM`) as ARM, IPHEX, CRYSTAL-FACE-UND, and
CRYSTAL-FACE-NASA, all of which are already in `IN_CLOUD_CAMPAIGNS`. AIRS-II's
entire purpose is sampling in-cloud icing conditions, making it a textbook
case for the same expected chilled-mirror-in-cloud behavior documented in
`2026-07-05-qc2-severity-tiers.md` — it was simply missing from the set.

**MACPEX (77 rows) — cross-instrument `Si`/`qv` "best" inconsistency, tiny
volume, not worth a structural fix.** `load_macpex()` resolves `Si` (best)
via a per-row NaN fallback across `Si_HWV -> Si_DLH -> Si_JLH` after each
instrument's own value has been bounds-clipped (`SI_BOUNDS`), while
`extract_macpex_standard()` resolves `qv` (best) via an *independent*,
identically-ordered fallback across `qv_hwv -> qv_dlh -> qv_jlh` computed
from the same raw ppmv columns but with no equivalent bounds-clipping. When
one instrument's `Si` gets clipped to NaN on a given row but its `qv` does
not, `Si` (best) falls through to a different instrument than `qv` (best) for
that same row, producing thermodynamically inconsistent pairs (e.g. one
sample row: `Si_best = -0.155` from one instrument while `qv_best = 0.238
g/kg` from another, an inconsistent pairing). Confirmed by direct inspection
of the raw merged rows: values alternate between two populations from one
row to the next during a ~1-minute window on 2011-04-03. This is a structural
characteristic of resolving `Si` and `qv` independently across a multi-
instrument ranking rather than a simple bug, and affects only 77 of 279,073
rows (0.03%) — a joint-consistency fix would require reworking both
resolution functions together, disproportionate risk for the volume
affected. Documented and accepted, not fixed.

**ATTREX (13 rows) — measurement noise at the extreme low end of the
sensor range, not a data error.** All 13 rows are at -85 to -87°C (tropical
tropopause layer, ATTREX's specific target region) with `qv` of only
0.003-0.005 g/kg. At these extremely low absolute humidities, a fixed,
instrument-precision-level absolute discrepancy (a few times 1e-4 g/kg
between `qv` and `qv_sat_liq`) produces a large *relative* ratio (1.20-1.35x)
even though the absolute magnitude is far below any realistic measurement
precision. Not actionable.

**POSIDON (1 row) — trivial, below any reasonable action threshold**, ratio
1.20 (exactly at the mild/severe boundary). Consistent with how QC1's 6-row
ESCAPE residual is treated elsewhere in this project.

## Fix details

`scripts/qa_checks.py`: added `"AIRS-II"` to `IN_CLOUD_CAMPAIGNS`. This does
not change the severe/mild ratio thresholds or the raw flag counts (severity
is determined purely by `qv_ratio`, independent of campaign) — it corrects
the `in_cloud_campaign` label so downstream analysts see AIRS-II's 399 severe
rows correctly marked as expected chilled-mirror-in-cloud behavior, exactly
as IPHEX/MC3E/OLYMPEX/CRYSTAL-FACE-UND/CRYSTAL-FACE-NASA/ICE-L/ARM already
are, rather than appearing as an unexplained anomaly.

MACPEX/ATTREX/POSIDON: no code changes.

## How to detect

`logs/qaqc/latest/02_consistency_flags.csv`, filter
`check_type == "qv_exceeds_saturation" and severity == "severe"`, then check
`in_cloud_campaign` and cross-reference the campaign's water-vapor instrument
type against `docs/decisions/2026-07-05-qc2-severity-tiers.md`'s documented
in-cloud population before assuming an anomaly.

## What remains

None actionable. MACPEX's Si/qv cross-instrument inconsistency is a known,
accepted structural characteristic at negligible volume (0.03% of that
campaign's rows) — worth keeping in mind if MACPEX's multi-instrument
resolution logic is ever reworked for other reasons, but not worth a
dedicated fix on its own.
