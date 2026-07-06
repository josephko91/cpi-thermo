# 2026-07-05 — ESCAPE temperature sensor failure (2022-06-10)

## Fix applied in commit: 6503c22

## What we found

The 2022-06-10 ESCAPE flight file reports Tair_C in the range -7 to +12°C at altitudes of
12,000–17,000 m. The ISA standard temperature at those altitudes is approximately -56.5°C.
This is a physically impossible excursion of ~70°C — clearly a sensor failure, not an
atmospheric anomaly (even a sudden stratospheric warming event peaks around -30°C at 10 km).

The root cause was a faulty or disconnected temperature sensor on that specific flight day.
The bad Tair values then propagated into qv calculations via `es_liq_hPa(T)`, producing
wildly super-saturated qv flags (the saturation vapor pressure at +12°C is ~14 hPa vs
~0.01 hPa at the actual ambient temperature).

## Detection threshold

Threshold: `Tair_C > -20.0°C at Alt_m > 10,000 m`

Why -20°C (not something closer to the ISA value of -56.5°C):
- Deliberately conservative to avoid masking legitimate anomalies (polar vortex disruptions,
  dynamic tropopause folds, etc.)
- The +12°C readings are 32°C above the threshold — clearly captured
- Even extreme SSW events don't bring 10+ km temperatures above ~-30°C
- The threshold also catches any future similar failures without being too aggressive

## Fix details

Added after the physical bounds block in `parsers/escape.py`:

```python
if temp_col and alt_col:
    sensor_failure = (
        df[temp_col].notna() & df[alt_col].notna()
        & (df[temp_col] > -20.0) & (df[alt_col] > 10_000.0)
    )
    if sensor_failure.any():
        df.loc[sensor_failure, temp_col] = np.nan
        if dew_col:
            df.loc[sensor_failure, dew_col] = np.nan
```

Nulled 5,930 rows on the 2022-06-10 flight.

## What remains

After this fix, ESCAPE still has 1,674 severe `qv_exceeds_saturation` flags in QC2.
Additionally 1,104 rows flagged in QC1 with P_hPa < 50 hPa — these appear to come from
a stuck or erroneous pressure altitude reading driving the ICAO formula to produce
stratospheric pressures. Separate investigation needed.
