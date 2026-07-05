# 2026-07-05 — ARM cryo hygrometer masking

## Fix applied in commit: 0df6b02

## Background: T4 binary encoding

All ARM `.t4archive.gz` records encode physical values as:

```
physical = raw_int32 / 1000.0 - 100.0
```

Three distinct failure modes arise from this encoding:

---

## Issue A — GPS fill values

When the GPS receiver has not acquired a fix, hardware outputs `raw = 100000` for all
three GPS channels, which decodes to `Lat = Lon = Alt = 0.0`.

Detection: `|GPS_Lat_deg| < 0.01 AND |GPS_Lon_deg| < 0.01` simultaneously.
Fix: null all three GPS columns for those rows.

---

## Issue B — Cryo below detection limit

The Cryo-Electric Mirror (CEM) outputs `raw < 0` when the mirror is cooled below its
lower measurement limit (~-80°C frost point). This decodes to physically impossible
temperatures (-100 to -150°C), which drives `qv_from_e_P` to essentially zero, creating
a false "bone-dry" cluster.

Detection: `Dew_Point_Cryo_C < -80.0` or `Frost_Point_Cryo_C < -80.0`
Fix: null both cryo channels below `CRYO_FLOOR = -80.0°C`.

Effect on qv distribution: ARM had 46% of qv ≈ 0 g/kg (all from this cause). After fix, 63.6%
of qv is NaN — this is correct: the cryo was below range throughout the dry upper troposphere.
The 63.6% NaN rate is real data sparsity, not a parser bug.

---

## Issue C — Cryo cloud flooding (Frost_Point > Tair)

When the aircraft flies through cloud or precipitation, liquid droplets coat the cryo mirror
and it equilibrates to the liquid dew point, which under in-cloud conditions can be very close
to Tair. However, instrument thermal lag and droplet flooding sometimes causes the cryo to
report Frost_Point > Tair for tens of seconds — thermodynamically impossible.

Observed in `citation.0317001715`: Frost_Point_Cryo = 20–25°C while Tair = 5°C and
Dew_Point_EGG = 3.9°C (Egg hygrometer reported correctly).

Detection threshold: `CRYO_TAIR_MARGIN = 1.0°C` above Tair.
WHY NOT STRICT (> Tair exactly): In-cloud conditions near 100% RH produce Frost_Point ≈ Tair.
CEM and Rosemount temperature sensors each carry ±0.3–0.5°C precision. Strict threshold (`> Tair`)
incorrectly nulled 79,196 legitimate in-cloud near-saturation readings. The 1°C buffer correctly
passes readings within instrument precision while catching the +15–20°C flooding excursions.

Applied to: `Dew_Point_EGG_C`, `Dew_Point_Cryo_C`, `Frost_Point_Cryo_C`.

---

## Altitude fallback

ARM GPS Alt can be unavailable (GPS fill) or unreliable early in flight. Added:

```python
df["Alt_m"] = np.where(
    df["GPS_Alt_m"].notna() & (df["GPS_Alt_m"] > 0),
    df["GPS_Alt_m"],
    df["Pressure_Altitude_m"],
)
```

---

## What remains

63.6% ARM qv NaN is correct. The ARM EGG dew-point instrument (`Dew_Point_EGG_C`) is not
currently used as a fallback for above-freezing conditions when cryo is NaN. This could
partially fill the dry UT gap at low altitudes where the EGG is reliable.
