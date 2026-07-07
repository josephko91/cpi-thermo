# 2026-07-06 — QC3 residual stuck-sensor diagnostic (GitHub issue #10)

## Question

Issue #10 (split from #5) asks whether the 309 QC3 stuck-sensor runs left
after the 2026-07-06 ARM/AIRS-II altitude fixes (`docs/decisions/2026-07-06-qc3-stuck-sensor.md`)
are really a merge-tolerance/reference-cadence artifact of each campaign's
multi-instrument `merge_asof` architecture, as that doc concluded from manual
inspection, or something that deserves its own fix. This adds a targeted,
reproducible diagnostic rather than relying on the earlier manual read.

**Scope**: the 4 campaigns/variable sets the decision doc names as
"remaining" — CRYSTAL-FACE-NASA (Tair_C, P_hPa), ESCAPE (Tair_C, P_hPa, qv,
Alt_m), ISDAC (Alt_m, P_hPa), MIDCIX (P_hPa). This is 303 runs, not 309 —
the QC3 report's 309 total also includes 3 AIRS-II Alt_m and 3
CRYSTAL-FACE-UND Alt_m runs that the stuck-sensor doc doesn't attribute to
the merge-tolerance mechanism (AIRS-II's are separately noted as "short
GGALT-native repeats"); those 6 are out of scope here.

## Method

`scripts/diagnose_qc3_merge_granularity.py` re-detects the same stuck runs
directly from `combined_env_data.parquet` (same run-length-encoding logic
as QC3, `>=30` bit-exact-identical samples) and, for each run, tests two
signatures that distinguish "reference source repeating its last value
while a faster stream keeps ticking" from "genuine multi-minute sensor
freeze":

1. **Timestamp cadence during the run** — `implied_dt_s` (run duration /
   (run_length − 1)) vs `file_median_dt_s` (the file's own native sample
   spacing outside stuck runs). A ratio near 1 means the Timestamp column
   kept advancing normally while only the flagged variable froze — the
   signature of a sparser reference being nearest-matched onto a denser
   timeline. A ratio far above 1 would mean the whole file went sparse
   during the run, pointing to a genuine dropout instead (as was the case
   for the AIRS-II and ARM bugs already fixed).
2. **Run-duration clustering** — if a periodic reference source with
   update interval R drives the stuck runs, run durations should cluster
   in a narrow band near R rather than spreading arbitrarily.

## Results

`logs/qc3_merge_granularity/latest/qc3_merge_granularity_summary.csv`:

| Campaign | variable | n_runs | median ratio | % native-cadence-consistent | median duration (s) | modal duration (s) |
|---|---|---|---|---|---|---|
| CRYSTAL-FACE-NASA | P_hPa | 22 | 1.28 | 90.9% | 40.5 | 37 |
| CRYSTAL-FACE-NASA | Tair_C | 74 | 1.28 | 90.5% | 43.0 | 37 |
| ESCAPE | Alt_m | 11 | 1.00 | 100% | 37.0 | 30 |
| ESCAPE | P_hPa | 11 | 1.00 | 100% | 37.0 | 30 |
| ESCAPE | Tair_C | 15 | 1.00 | 100% | 36.0 | 31 |
| ESCAPE | qv | 2 | 1.00 | 100% | 51.0 | 51 |
| ISDAC | Alt_m | 52 | 1.00 | 100% | 33.0 | 29 |
| ISDAC | P_hPa | 1 | 1.00 | 100% | 29.0 | 29 |
| MIDCIX | P_hPa | 115 | 1.28 | 95.7% | 48.0 | 41 |

Full per-run detail in `qc3_merge_granularity_runs.csv`; figures in
`figs/qc3_merge_granularity/latest/`.

**Signature 1 holds everywhere.** 90.5–100% of runs in every
campaign/variable group have `ratio_implied_to_file_dt` within
[0.7, 1.3] of 1 (`qc3_merge_granularity_ratio_hist.png`) — the Timestamp
column keeps advancing at essentially the file's normal cadence throughout
the "stuck" run. There is no sign of the file itself going sparse, which is
what a genuine sensor dropout (as found for AIRS-II `ALT` and ARM
`GPS_Alt_m`) would look like.

**Signature 2 holds too.** Run durations cluster tightly just above the
30-sample detection floor in every group (`qc3_merge_granularity_duration_hist.png`):
medians 29–48 s, IQRs mostly within a ~10–20 s band, and even the max
across all 303 runs is 159 s (2.65 min) — vs. the genuine bugs already
fixed, which ran up to ~4 min (ARM) and ~68 min (AIRS-II). This is
consistent with a periodic reference source update interval (tens of
seconds) rather than an open-ended freeze.

## Conclusion

Both diagnostic signatures support the `2026-07-06-qc3-stuck-sensor.md`
conclusion: these 303 runs are a merge-tolerance/low-precision
characteristic of matching a higher-frequency stream against a sparser
reference via nearest-neighbor `merge_asof`, not a sensor-freeze bug like
the ARM/AIRS-II cases already fixed. No change to the parquet or parsers
from this diagnostic. Per issue #10's next step, a proper fix
(interpolating the sparser reference sources instead of nearest-matching)
remains a bigger architectural change across `crystal_face_nasa.py`,
`escape.py`, `isdac.py`, and `midcix.py`, worth doing only if one of those
parsers is being reworked for another reason.

## How to reproduce

```bash
python scripts/diagnose_qc3_merge_granularity.py
```

Reads `data/out/combined_env_data.parquet`; writes to
`logs/qc3_merge_granularity/<timestamp>/` and
`figs/qc3_merge_granularity/<timestamp>/` (with `latest` symlinks), per
`scripts/log_paths.py` convention.
