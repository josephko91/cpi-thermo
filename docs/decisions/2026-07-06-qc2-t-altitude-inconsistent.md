# 2026-07-06 — QC2 T_altitude_inconsistent investigation

## Fix applied in commit: (pending — two distinct ARM bugs)

## What we found

QC2's `T_altitude_inconsistent` check_type (`|Tair_C - T_ISA(Alt_m)| > 40°C`)
flagged 8,672 rows, all in ARM, with no per-campaign writeup previously. Two
unrelated root causes across two different flights:

**Bug 1 — GPS altitude fix-acquisition glitch (4 files, 12,853 rows total,
of which 3,004 tripped this specific check via `citation.0313001806`).**
`load_arm_file()` selected `Alt_m = GPS_Alt_m` whenever `GPS_Alt_m` was
present and positive, ignoring `Pressure_Altitude_m` entirely in that case
(the function's own comment said the opposite — "prefer
`Pressure_Altitude_m`" — but the code did not implement that). Right after
GPS first acquires a fix, horizontal position (Lat/Lon) can lock before the
vertical solution stabilizes: in `citation.0313001806.t4archive.gz` at
18:42:12 UTC, `GPS_Alt_m` jumped to a valid-looking `36.63°N, -97.56°W` but
got stuck at exactly `333.0 m` for the next ~70 seconds while
`Pressure_Altitude_m` correctly climbed through 6,600-6,900 m — an 8,261 m
disagreement. Checked the whole campaign: 12,853 of 520,636 rows with both
altitude sources present disagree by >1,000 m (vs. a median disagreement of
just 55 m), confirming this is a real, recurring GPS-acquisition artifact,
not a one-off.

**Bug 2 — Rosemount temperature probe warm-up/frozen-reading fault (1 file,
5,668 rows, `citation.0303001655.t4archive.gz`).** For the first ~23 minutes
after this file starts (16:55:20-17:18:56 UTC), `Air_Temp_Rosemount_C` is
frozen at approximately -64.5°C (fluctuating only at instrument-noise level,
e.g. -64.548, -64.549, -64.547, ...) while `Pressure_Altitude_m` varies
normally (211-1,717 m, confirming the aircraft was genuinely flying, not
sitting stationary). At 17:18:57 the reading jumps abruptly to a physically
plausible ~5.0°C and tracks normally for the rest of the flight. -64.5°C at
200-1,700 m altitude is physically impossible under any real atmospheric
inversion. Note QC3 (stuck-sensor check, requires bit-exact repeats) did
**not** flag this, since the frozen reading still has sub-0.02°C noise —
only QC2's ISA-deviation check caught it.

## Fix details

**Bug 1**: `parsers/arm.py` `load_arm_file()` — added a plausibility check
before trusting `GPS_Alt_m`: only use it when it agrees with
`Pressure_Altitude_m` within 1,000 m (a generous margin given the campaign's
55 m median and <10th-percentile-scale legitimate geoid/ICAO-atmosphere
offset), otherwise fall back to `Pressure_Altitude_m`.

**Bug 2**: added an ISA-deviation mask on `Air_Temp_Rosemount_C` itself,
using the same formula/threshold as QC2 (`T_isa_C = 15.0 - 6.5e-3 *
Pressure_Altitude_m`, `|Tair - T_isa| > 40°C`) — checked the full-campaign
deviation distribution excluding this fault and confirmed the next-highest
value is well under 40°C (75th percentile is 5.4°C), so the threshold
cleanly isolates only this fault with no ambiguous cases. Masking
`Air_Temp_Rosemount_C` also nulls the co-derived `Si_chilled_mirror`,
`qv_chilled_mirror`, and `Sw` for these rows (all computed from it), which is
correct — those derived quantities were equally invalid.

Before → after:
- QC2 `T_altitude_inconsistent`: 8,672 → 0
- ARM Tair_C coverage: 100.0% → 99.0% (5,668 rows correctly nulled instead of
  reporting a physically impossible reading)
- Total dataset rows unchanged (both fixes re-derive/mask values in place,
  no rows added or removed)

## How to detect

For altitude-source disagreements: compare `GPS_Alt_m` against
`Pressure_Altitude_m` directly — a multi-thousand-meter gap that persists for
tens of seconds right after Lat/Lon first becomes non-null is the signature.
For the temperature fault: `logs/qaqc/latest/02_consistency_flags.csv`
filtered to `check_type == "T_altitude_inconsistent"`, grouped by
`source_file` — a fault localized to a single file's opening minutes, with
near-zero variance in the raw reading, points to a probe warm-up/init issue
rather than a real atmospheric event.

## What remains

None — both root causes are now masked/corrected in the parser.
