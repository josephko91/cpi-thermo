# 2026-07-05 — QC2 qv_exceeds_saturation severity tiers

## Fix applied in commit: 54020cf

## The problem

QC2 flagged 196k rows for `qv > 1.05 × qv_sat_liq`. Investigation revealed two distinct
populations:

1. **In-cloud instrument physics** (~60k rows): Chilled-mirror hygrometers equilibrate to
   the liquid dew point in cloud. Near 100% RH, qv approaches qv_sat_liq closely. The 1.05×
   threshold catches legitimate in-cloud readings where instrument precision (±5%) pushes
   the measurement just above the saturation line. These are not actionable data problems.

2. **Genuine data errors** (~21k rows): Sensor malfunctions (cryo flooding, stuck sensors,
   pressure inconsistencies) produce qv values that are 20–120% above saturation. These
   need investigation or masking.

## Solution: two-tier severity

```
MILD    = 1.05 × qv_sat_liq  to  1.20 × qv_sat_liq   → expected in-cloud instrument behavior
SEVERE  = > 1.20 × qv_sat_liq                          → likely data errors, investigate
```

The 1.20 threshold was chosen empirically: 20% supersaturation is beyond what any realistic
instrument uncertainty or in-cloud effect can produce at non-explosive-convection altitudes.

## In-cloud campaigns

Campaigns expected to have elevated MILD flag rates (deep cloud penetration):
`{ARM, IPHEX, MC3E, OLYMPEX, ICE-L, CRYSTAL-FACE-UND, CRYSTAL-FACE-NASA}`

The severity column plus `in_cloud_campaign` boolean in the QC2 CSV output allows filtering
by actionability without changing the detection threshold.

## QC2 flag counts after split (QA run 2026-07-05)

| Severity | Count  |
|----------|--------|
| Mild     | ~60k   |
| Severe   | ~21k   |
| Total    | ~92k   |

Remaining severe flags by campaign:
- IPHEX: 11,931 (4.1%) — possible rain contamination on chilled mirror
- MC3E: 4,727 (2.9%) — convective campaign, some genuine data issues
- ESCAPE: 1,674 — ICAO pressure inconsistency from Palt
- OLYMPEX: 1,341 (0.6%) — marine precipitation, LWC flag needed

## What remains

IPHEX and OLYMPEX severe flags are the next investigation target. Both campaigns involve
cloud/precipitation environments where rain contamination of chilled mirrors is known to
produce intermittent super-saturation spikes. Check if a Liquid Water Content (LWC) flag
is available in the raw data to identify cloud-flagged legs.
