# 2026-07-05 — Altitude unit bugs in ATTREX and ICE-L

## Fix applied in commit: 6aef37a

## What we found

**ATTREX**: Median Alt_m was ~163,924 m (should be ~16,392 m — a ×10 inflation).
Root cause: `parsers/attrex.py` read the MMS G_ALT ICARTT binary column but never applied
its per-column scale factor of 0.1. All other temperature/pressure columns used a global
0.01 factor that happened to be correct for them. G_ALT was silently wrong.

**ICE-L**: Median Alt_m was ~16,237 m (should be ~4,987 m — a ×3.3 inflation).
Root cause: `parsers/ice_l.py` used `_pick_var(ds, [..., "PALTF"])` and PALTF was the
only altitude variable present in several files. PALTF has `units="feet"` in its netCDF
attributes. The parser never checked units and never converted.

## How to detect

QC1 physical range check flagged 584k rows for Alt_m > 25,000 m; dropped to 1,683 after fix.
Also caught by scatter plots of Alt_m vs Tair_C — ATTREX points appeared at >100 km altitude.

## Fix details

ATTREX (`parsers/attrex.py`):
- Detect MMS G_ALT column by name pattern: `"MMS" in col.upper() and "G_ALT" in col.upper()`
- Apply `× 0.1` scale; mask `< -500` or `> 25000` as NaN
- Also masked Lat/Lon fill values that survived the `× 0.00001` scaling

ICE-L (`parsers/ice_l.py`):
- Reordered `_pick_var` priority to `["GGALT", "ALT", "PALT", "PALTF"]` — prefers m-native vars
- Added explicit unit check: `ds[alt_name].attrs.get("units", "").lower() in ("feet", "ft", "foot")`
- Convert feet → meters with `× 0.3048` if matched

## What remains

ESCAPE still has 1,104 rows where P_hPa < 50 hPa — possibly the Palt was stuck at low
altitude and the ICAO formula then yields stratospheric pressure. Separate from this fix.
