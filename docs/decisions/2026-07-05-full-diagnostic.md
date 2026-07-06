# Decision: Full dataset diagnostic — 2026-07-05

**Commit at time of analysis:** e143084  
**Script:** `scripts/full_diagnostic.py`  
**Figures:** `figs/all-campaigns_20260705/`

---

## Dataset snapshot

| Metric | Value |
|--------|-------|
| Total rows | 3,644,847 |
| Columns | 41 |
| Campaigns | 14 |
| Parquet size | ~214 MB |

---

## Per-variable summary

| Variable | Non-null | % avail | Min | P5 | Median | P95 | Max | Notes |
|----------|----------|---------|-----|----|--------|-----|-----|-------|
| Tair_C | 3,627,125 | 99.5% | −88.8 | −81.6 | −24.8 | +12.5 | +41.7 | Very complete; P_hPa < 0 outliers in ESCAPE |
| P_hPa | 3,639,476 | 99.9% | −1000 | 86.6 | 446 | 939 | 1040 | −1000 fill values from ESCAPE (1,104 rows stuck Palt) |
| Alt_m | 2,887,633 | 79.2% | −97 | 551 | 5,150 | 17,500 | 24,100 | Missing for CRYSTAL-FACE-NASA, CRYSTAL-FACE-UND, MACPEX, MIDCIX (no altitude column in raw) |
| Si | 2,856,601 | 78.4% | −1.0 | −0.891 | −0.202 | 0.213 | 2.0 | Clipped at −1 and 2 (parser hard limits); ARM 36.4% only |
| Sw | 2,855,702 | 78.3% | −1.0 | −0.93 | −0.434 | −0.019 | 5.03 | Near-identical coverage to Si; 5.03 max is IPHEX rain contamination |
| qv | 2,858,516 | 78.4% | ~0 | 0.002 | 0.374 | 7.87 | 279 | 279 g/kg max is physically impossible — IPHEX |

---

## Per-campaign availability

| Campaign | N rows | % of total | Tair_C | P_hPa | Alt_m | Si | qv | Notes |
|----------|--------|-----------|--------|-------|-------|----|----|-------|
| ARM | 567,760 | 15.6% | 100% | 100% | 100% | 36.4% | 36.4% | 63.6% qv NaN — real upper-troposphere dryness |
| AIRS-II | 312,792 | 8.6% | 100% | 100% | 100% | 100% | 100% | Cleanest campaign |
| ATTREX | 581,370 | 16.0% | 99.7% | 99.7% | 99.4% | 86.3% | 86.4% | Largest campaign; altitude fix applied (×0.1 scale) |
| CRYSTAL-FACE-NASA | 154,723 | 4.2% | 100% | 100% | 0% | 100% | 100% | No altitude in raw data |
| CRYSTAL-FACE-UND | 185,420 | 5.1% | 98.7% | 100% | 0% | 71.7% | 71.4% | No altitude in raw data |
| ESCAPE | 67,380 | 1.8% | 91.2% | 100% | 100% | 90.9% | 90.9% | 1,104 rows P_hPa < 50 (stuck Palt); temp sensor failure masked |
| ICE-L | 210,561 | 5.8% | 100% | 100% | 99.7% | 99.7% | 99.8% | Altitude foot→m fix applied |
| IPHEX | 287,600 | 7.9% | 99.6% | 100% | 99.4% | 68.4% | 68.8% | 1,391 rows Si > 1.05 (rain contamination); qv max 279 g/kg |
| ISDAC | 357,071 | 9.8% | 99.9% | 99.4% | 100% | 100% | 99.3% | Clean |
| MACPEX | 279,073 | 7.7% | 100% | 100% | 0% | 69.9% | 69.9% | No altitude in raw data |
| MC3E | 165,431 | 4.5% | 98.6% | 100% | 100% | 88.2% | 88.2% | Clean |
| MIDCIX | 75,927 | 2.1% | 100% | 100% | 0% | 100% | 100% | No altitude in raw data |
| OLYMPEX | 209,321 | 5.7% | 100% | 100% | 100% | 58.6% | 58.6% | 128 rows Si > 1.05; marine precip |
| POSIDON | 190,418 | 5.2% | 97.9% | 99.3% | 70.4% | 96.0% | 98.0% | Some altitude gaps; otherwise clean |

---

## Known issues confirmed

| Issue | Count | Status |
|-------|-------|--------|
| ESCAPE P_hPa < 50 hPa | 1,104 rows | **Open** — stuck/erroneous Palt driving ICAO formula |
| IPHEX Si > 1.05 (severe) | 1,391 rows | **Open** — rain contamination on chilled mirror |
| OLYMPEX Si > 1.05 (severe) | 128 rows | **Open** — marine precipitation |
| ARM qv NaN | 360,869 / 567,760 (63.6%) | **Accepted** — real upper-troposphere dryness |
| qv max 279 g/kg (IPHEX) | ~few rows | **Open** — physically impossible, needs capping |

---

## Figures generated

| File | Contents |
|------|----------|
| `Tair_C_per_campaign.png` | Temperature histograms per campaign |
| `P_hPa_per_campaign.png` | Pressure histograms per campaign |
| `Alt_m_per_campaign.png` | Altitude histograms per campaign |
| `Si_per_campaign.png` | Ice supersaturation histograms per campaign |
| `Sw_per_campaign.png` | Liquid supersaturation histograms per campaign |
| `qv_per_campaign.png` | Water vapor mixing ratio histograms per campaign |
| `availability_heatmap.png` | % non-null heatmap: campaign × variable |
| `temporal_coverage.png` | Timeline bar chart of each campaign's date range |
| `alt_vs_tair_all.png` | Altitude vs temperature scatter, all campaigns colored |
| `Si_vs_Tair_per_campaign.png` | Si vs Tair scatter faceted by campaign |

---

## Next priority items

1. Fix ESCAPE P_hPa < 50 residual (stuck Palt → ICAO formula giving stratospheric pressure)
2. Investigate IPHEX qv = 279 g/kg outliers — cap or mask
3. Add LWC flag to IPHEX/OLYMPEX to separate in-cloud Si > 1 from sensor errors
4. Recover altitude for CRYSTAL-FACE-NASA, CRYSTAL-FACE-UND, MACPEX, MIDCIX if available in raw
