# 2026-07-06 — QC3 stuck-sensor investigation

## Fix applied in commit: 60a92b5 (AIRS-II and ARM altitude fixes)

## What we found

QC3 (`>=30` consecutive bit-exact identical values) flagged 1,567 runs across
7 campaigns with zero prior investigation. Two campaigns had a genuine,
large-magnitude root cause; the rest are small, borderline runs with
physically plausible values.

**AIRS-II Alt_m (6 runs, up to 4,413 consecutive samples / ~68 minutes) —
inertial altitude dropout.** `parsers/airs_ii.py` read `ALT` (IRS
Baro-Inertial Altitude) preferentially via `ds.get("ALT", ds.get("GGALT",
...))`, but `ds.get()`'s fallback only triggers when a key is entirely
absent — `ALT` is present in every file, just sometimes invalid. `ALT` reads
exactly `0.0` for extended stretches in several files (up to 4,413 of 33,444
samples in RF09) — an inertial-system alignment/dropout artifact, not a real
reading of sea level for over an hour of flight. GPS-derived `GGALT` is far
more reliable across the whole campaign (at most 7 zero samples in any
single file, checked all 15).

**ARM Alt_m (1,215 runs, up to 1,027 consecutive samples / ~4 minutes) —
a second, distinct GPS altitude failure mode beyond the one fixed for QC2.**
The 2026-07-06 QC2 fix (`docs/decisions/2026-07-06-qc2-t-altitude-inconsistent.md`)
only rejected `GPS_Alt_m` once it disagreed from `Pressure_Altitude_m` by
>1,000 m. But a frozen GPS receiver can hold the exact same whole-meter
altitude for minutes while the aircraft keeps climbing/descending on
`Pressure_Altitude_m`, and can still be within the 1,000 m tolerance at the
*start* of the freeze — confirmed directly in `citation.0312001649`:
`GPS_Alt_m` frozen at exactly `8926` m for 727 samples (~3 minutes) while
`Pressure_Altitude_m` smoothly descended from 8,828 m to 7,782 m; only the
back half of that run had diverged past 1,000 m by the time the tolerance
check would have caught it. This pattern recurred across nearly every ARM
flight file (~70,000 rows total).

**Remaining 309 runs (CRYSTAL-FACE-NASA Tair_C/P_hPa, ESCAPE
Tair_C/P_hPa/qv/Alt_m, ISDAC Alt_m/P_hPa, MIDCIX P_hPa) — not fixed.**
Inspected sample values: all are physically plausible atmospheric readings
(no obvious fill/sentinel patterns), and run lengths are mostly just above
the 30-sample detection threshold (medians 32-40 samples). CRYSTAL-FACE-NASA
in particular ties back to its multi-instrument `merge_asof` architecture
(documented in `2026-07-06-qc7-duplicate-timestamps.md` and
`2026-07-05-cpi-fusion-gap-fixes.md`): when a higher-frequency stream (e.g.
JLH) is matched against a sparser reference (MMS) via nearest-neighbor
`merge_asof`, several consecutive samples can legitimately share the same
matched value between reference updates. This is an inherent characteristic
of the merge-tolerance approach, not a bug — a fix would require
interpolating the sparser source, a bigger architectural change
disproportionate to the ~300 affected runs. Documented and accepted.

## Fix details

**`parsers/airs_ii.py`**: added a per-sample validity check on `ALT` (exact
`0.0` treated as invalid) falling back to `GGALT`, instead of the
key-presence-only `ds.get()` fallback.

**`parsers/arm.py`**: extended the Gap 2 altitude-selection logic with a
second, independent invalidation condition — `GPS_Alt_m` repeating a
bit-exact value for `>=30` consecutive samples (matching QC3's own
threshold) is now also rejected in favor of `Pressure_Altitude_m`,
regardless of whether it happens to still agree with `Pressure_Altitude_m`
at the time.

Before → after:
- QC3 total: 1,567 runs (7 campaigns) → 309 runs (4 campaigns)
- AIRS-II Alt_m stuck runs: 6 → 3 (residual are short GGALT-native repeats)
- ARM Alt_m stuck runs: 1,215 → 0

## How to detect

For a GPS/inertial altitude source with a redundant pressure-altitude
channel: check for both (a) large absolute disagreement between the two
sources and (b) extended bit-exact repeats in the GPS/inertial source while
the pressure-derived source keeps changing — a stuck reading that happens to
still agree at first will only show up via (b), not (a).

## What remains

None for AIRS-II/ARM. The 309 residual runs across CRYSTAL-FACE-NASA,
ESCAPE, ISDAC, and MIDCIX are accepted as a merge-tolerance/low-precision
characteristic, not tracked as a new open issue given the small volume and
plausible values.
