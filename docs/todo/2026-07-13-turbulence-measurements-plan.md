<!--
Plan drafted: 2026-07-13 (local)
Status: not yet implemented
Related: docs/reports/2026-07-13-turbulence-measurements-survey.md (the
         research survey this plan is based on)
-->

# Incorporate 3-D Wind Vectors, EDR, and Related Turbulence Measurements

## Context

A prior research survey (`docs/reports/2026-07-13-turbulence-measurements-survey.md`)
found that essentially every one of the 15 campaigns in this pipeline flew some form
of gust-probe/turbulence instrumentation, but none of it reaches the combined
dataset today — every `extract_*_standard()` function silently drops it. In most
cases the data isn't even hard to get: it's already sitting in an in-memory
DataFrame inside the existing loader, just never selected into the final
standardized output. This plan turns that survey into an ordered set of parser
changes that add wind vector, EDR, aircraft attitude, angle-of-attack/sideslip,
and true-airspeed columns to `data/out/combined_env_data.parquet`, without
disturbing the existing thermodynamic columns or the exact-second-merge
architecture the rest of the pipeline depends on.

Three decisions were confirmed with the user up front and are treated as fixed
constraints below, not open questions: (1) TKE is not a raw variable anywhere in
this data — this plan ingests raw wind components and leaves TKE/spectral
derivation to a future analysis script; (2) ARM's 4 Hz and ISDAC's unused 5 Hz
source both get floored to 1 Hz like every other parser, no new high-rate tier;
(3) new turbulence columns are NOT added to `build_data_tiers.py`'s `CORE_COLS`
— they ride through L1/L2 as optional, ungated columns.

## Architecture facts this plan relies on (verified, not assumed)

- `main.py:306` combines all campaigns via plain `pd.concat(all_dfs, ignore_index=True)`
  — an outer union of columns. Campaigns do **not** need identical schemas; a
  campaign whose extractor doesn't emit a turbulence column simply gets NaN
  there. No registry or schema file needs updating for this to work.
- `scripts/build_data_tiers.py`'s L1 build (`build_l1`) does a full-row merge
  that carries every L0 column through automatically — zero code change needed
  there. Only `CORE_COLS` (line ~63) gates L2 completeness, and per the decision
  above, new columns are deliberately left out of it.
- There is no code-enforced "standard schema" anywhere — each
  `extract_*_standard()` independently builds a `pd.DataFrame({...})` dict
  literal. `config.yaml`'s `output.standardized_columns` block (lines 176-214)
  is documentation only (confirmed via repo-wide grep), kept in sync by
  convention, not by any parser.
- `parsers/utils.py` has thermodynamic helpers, `round_timestamp_to_second`
  (floors to whole seconds — never `.dt.round()`, to avoid banker's-rounding
  collisions), `clean_column_name`, `parse_columns_with_units`,
  `extract_takeoff_date`, `COMMON_NA_VALUES` — but nothing for wind, angles, or
  unit conversion (e.g. knots→m/s). This plan adds a new section there.

## The three instrument families (drives the naming design)

- **Family A — NASA Ames MMS** (`ATTREX`, `POSIDON`, `MACPEX`, and
  CRYSTAL-FACE-NASA's unused `MG` dir): `U`/`V`/`W` (wind, m/s), `TEDR` (EDR,
  **log10 kW/kg**), `REYN`, `ROLL`/`PITCH`/`HDG`.
- **Family B — UND Citation pipeline** (`CRYSTAL-FACE-UND`, `MC3E`, `MPACE`,
  `OLYMPEX`, `IPHEX`): `Wind_Z`/`Wind_M`/`Wind_D`, `TURB` (EDR, **linear
  cm^(2/3)·s⁻¹** — incompatible with Family A's unit), `POS_Pitch`/`POS_Roll`/
  `POS_Head`, `POSZ_Acc`, `Alpha`/`Beta` (AoA/sideslip — only this family has
  them explicitly), `VERT_VEL` (aircraft's own vertical motion, not wind),
  `TAS`/`IAS`/`MachNo_N`.
- **Family C — NCAR/NRC RAF-Nimbus** (`AIRS-II`, `ICE-L`, both NetCDF): `WI`/
  `WIC` (wind, CF `standard_name=upward_air_velocity`), `PITCH`/`ROLL`/`THDG`,
  `ACINS`, `ATTACK`/`AKRD`, `SSLIP`, `TASX` family.
- **Outliers**: ARM's binary `.t4archive.gz` (`Vertical_Wind_m_s`,
  `Turbulence_eps` — units unconfirmed, do not assume it matches Family A/B),
  ISDAC's two sources (1 Hz `strapp-convair_bulk` currently used, 5 Hz
  `wolde-convair` unused with proprietary `TDREDDY` coefficient), ESCAPE's two
  aircraft (Learjet currently used; NRC Convair unused, different flight
  track — excluded from this plan), MIDCIX (TAS only, nothing else exists).

## Standard column naming

**Physically-unified columns** (safe to merge across families once
unit-converted — same physical quantity, well-defined conversion):

| Column | Units | Notes |
|---|---|---|
| `Wind_U_ms`, `Wind_V_ms`, `Wind_W_ms` | m/s | Vertical-only sources (`Wind_Z`, `VWND_P`, `Vertical_Wind_m_s`, `WIC`) populate `Wind_W_ms` alone |
| `WindSpeed_ms`, `WindDir_deg` | m/s, deg | Prefer GPS-corrected variant where both exist (e.g. Family C's `WSC`/`WDC` over `WS`/`WD`) |
| `Roll_deg`, `Pitch_deg`, `Heading_deg` | deg | |
| `AngleOfAttack_deg`, `Sideslip_deg` | deg | Only Family B (`Alpha`/`Beta`) and Family C (`ATTACK`/`SSLIP`) have these |
| `TAS_ms` | m/s | **Verify each file's documented units before converting — do not assume m/s.** MPACE's TAS/IAS may be in knots per header; check at implementation time |
| `VertVel_ms` | m/s | Aircraft's own vertical velocity (`VERT_VEL`, `WP3`) — kept separate from `Wind_W_ms` deliberately; conflating platform motion with atmospheric wind would corrupt updraft/downdraft interpretation |
| `Accel_Vert_ms2` | m/s² | `POSZ_Acc`, `ACINS` |

Rationale for unifying these (unlike EDR below): once unit-converted, m/s is
m/s and degrees are degrees regardless of which gust probe measured them —
`Campaign` already tells the reader the provenance, same as how `Tair_C`/
`P_hPa` are already unified across differently-sensored campaigns. This is a
different situation from `Si_<instrument>`, which exists because *competing*
hygrometers can coexist in the *same file* with a defined `h2o_ranking`
preference order — there's no such same-file competition for wind/attitude
here (MPACE's nose/wing dual channel is the one exception — see below).

**EDR — do NOT unify.** Keep separate, explicitly-tagged columns:
`EDR_mms_log10kWkg` (Family A), `EDR_und_cm23s1` (Family B), `EDR_arm` (ARM,
units unconfirmed — do not merge into `EDR_und_cm23s1` even though same
institution produced both, without a verified conversion), `EDR_isdac_tdreddy`
(Phase 3, ISDAC's proprietary TAMDAR coefficient — explicitly not labeled EDR
to avoid implying equivalence). These are different published quantities from
different processing pipelines, not just different units of one quantity —
fabricating an equivalence between them is exactly the kind of manufactured
precision `docs/decisions/2026-07-07-exact-second-merge-rewrite.md` already
warns against elsewhere in this codebase. Record this as its own decision doc
(`docs/decisions/<date>-turbulence-schema.md`) so a future contributor doesn't
"clean up" the three EDR columns into one.

**MPACE's dual nose/wing channels**: use the wing (primary) channel for the
unified columns above; keep the nose channel as MPACE-local `_nose`-suffixed
extras (`WindW_nose_ms`, `EDR_und_cm23s1_nose`). Don't design a repo-wide
nose/wing schema for one campaign.

## New helpers in `parsers/utils.py`

Add a `# Wind / Attitude / Airspeed Utilities` section:
- `knots_to_ms(knots) -> np.ndarray` (`knots * 0.514444`) — apply only where a
  file's own header confirms knots (MPACE is the lead to check, not a given).
- No angle-conversion helper needed — every attitude/AoA/sideslip field found
  is already in degrees.
- No EDR conversion helper — per the no-unification decision.
- Recommended small refactor: factor the repeated
  `round_timestamp_to_second(...)` → `dropna` → `drop_duplicates(keep="first")`
  idiom (already appears verbatim in `crystal_face_und.py` and `midcix.py`)
  into a `first_per_second(df, ts_col="Timestamp")` helper, since ARM (this
  plan) and ISDAC's 5 Hz source (Phase 3) add two more call sites.

## ARM's 4 Hz → 1 Hz flooring

`parsers/arm.py::load_arm_file` builds `Timestamp` via direct `tz_localize`
(lines 205-207) with no call to `round_timestamp_to_second` anywhere in the
file today. Add, immediately after Timestamp construction:
```python
df["Timestamp"] = round_timestamp_to_second(df["Timestamp"])
df = df.dropna(subset=["Timestamp"]).drop_duplicates(subset=["Timestamp"], keep="first")
```
This is the same two-line idiom already used in `crystal_face_und.py` and
`midcix.py` for exactly this situation (denser-than-1Hz raw source onto a 1 Hz
grid) — not a new pattern. Take the first real sample per second rather than
averaging: a mean across 4 samples would synthesize a value that never existed
at any single instant, which is worse for a genuinely fluctuating turbulence
quantity than picking one actually-observed instant. Apply this to the whole
`load_arm_file` output (not just new turbulence columns), and check during
implementation whether ARM currently produces duplicate-timestamp rows
downstream — if so, this fix changes ARM's existing row count, which is worth
flagging as a side effect, not just an addition.

## Phased rollout

**Phase 1 — pure-additive, `extract_*_standard()` only (lowest risk, ship first):**
The turbulence columns already exist in each function's intermediate
DataFrame and are dropped only at the final dict-literal construction. One
self-contained edit each:
- `parsers/attrex.py::extract_attrex_standard` (~587-608) — `MMS_TAS/U/V/W/TEDR/REYN/ROLL/HDG/PITCH`
- `parsers/posidon.py::extract_posidon_standard` (~314-346) — `MMS-1HZ_*` (identical instrument to ATTREX)
- `parsers/macpex.py::extract_macpex_standard` (~746-765) — `MMS-Met_U/V/W` + `MMS-FlightPath_TAS` only (no EDR/attitude archived for this campaign)
- `parsers/iphex.py::extract_iphex_standard` (~399-418) — full 14-field Family-B set, single file already open for Tair/P
- `parsers/mc3e.py::extract_mc3e_standard` (~155-169) — richest single-file set (17 fields); verified exact lines above
- `parsers/mpace.py::extract_mpace_standard` (~218-232) — 16 fields incl. dual nose/wing EDR
- `parsers/olympex.py::extract_olympex_standard` (~190-204) — 12 fields
- `parsers/isdac.py::extract_isdac_standard` (~346-362) — `ALPHA/BETA/PITCH/ROLL/driftA/HDG/TRK/MWSpd/MWDir/VWND_P` from the already-read 1 Hz `strapp-convair_bulk` source
- `parsers/arm.py::extract_arm_standard` (~333-347) — bundled with the 1 Hz-flooring fix above; verified exact lines above

**Phase 2 — needs an upstream loader/merge-list fix, not just the extractor:**
- `parsers/crystal_face_und.py::load_crystal_face_und_file` — extend
  `nav_cols_to_merge` (line 126-128, currently `Timestamp,POS_Lat,POS_Lon,POS_Alt`
  — add `POS_Pitch,POS_Roll,POS_Trk,POS_Head,TAS_n`) and `met_cols_to_merge`
  (line 169-173, currently `Timestamp`+temp+pressure — add
  `Wind_Z_Nose,Wind_M_Nose,Wind_D_Nose,TURB`); both verified exact against
  current source. Then extend `extract_crystal_face_und_standard` (~238-297).
- `parsers/crystal_face_nasa.py` — 4 slice points to restore dropped columns:
  `load_mm_met_file` (line 372, add back `U,V,W`), `load_np_file` (line 199,
  add back `Pitch,Roll,TAS,gSpd,trkA,T_Head`), `load_nm_met_file` (line 483,
  add back `WindSpd,WindDir,Mach`), shared `load_mms_file` (line 133, add back
  `TAS` — this function is imported directly by `midcix.py` too). Then extend
  `extract_crystal_face_nasa_standard` (~901-919).
- `parsers/midcix.py` — after fixing the shared `load_mms_file` above, also
  add `"TAS"` to MIDCIX's own re-slice of `nav_df` (~line 106); MIDCIX has no
  wind/EDR/attitude data anywhere in its raw archive, so TAS is all this
  campaign gains.
- `parsers/escape.py` — `VaV`/`Hdg`/`TAS` are never isolated by name today
  (unlike other parsers, `load_escape_file` just returns the whole raw `df`).
  Add `_choose_column(df, [...])` lookups (the helper already exists at
  lines 129-150 for temp/pressure/etc. — same pattern) for these three, then
  extend `extract_escape_standard` (~437-451). ESCAPE's second raw source
  (NRC Convair, a different aircraft's flight track) stays excluded.

**Phase 3 — genuinely new read paths (highest effort, do last):**
- `parsers/ice_l.py` — extend the existing `_pick_var(ds, candidates)`
  fallback helper (lines 94-98) with candidate lists for `WI`/`WIC`, `WS`/
  `WSC`/`WD`/`WDC`, `PITCH`/`ROLL`/`THDG`, `ACINS`, `ATTACK`/`AKRD`, `SSLIP`,
  `TASX` family. Body-rate/body-accel fields (`BPITCHR`/`BROLLR`/`BYAWR`,
  `BLATA`/`BLONGA`/`BNORMA`) and the independent `IWD`/`IWS` estimate are
  lower priority within this phase.
- `parsers/airs_ii.py` — same variables, brand-new read path (currently never
  dereferences any of them). Recommend introducing the same `_pick_var`
  helper pattern here (currently uses inline `for pvar in (...)` / nested
  `ds.get(ds.get(...))` chains) for consistency with `ice_l.py`, since several
  new candidate-list lookups would make the inline style noisy.
- ISDAC's `wolde-convair/*.cdf` 5 Hz source — new loader, floored to 1 Hz via
  the same `first_per_second` approach as ARM. Lower priority than the two
  NetCDF campaigns since ISDAC already gets wind/attitude/AoA from its Phase 1
  1 Hz source; this phase only adds `TDREDDY` and angular-rate/3-axis-accel.

**Excluded from this plan entirely (flag as follow-up, do not implement now):**
ESCAPE's NRC-Convair source (different aircraft/track than what's ingested);
CRYSTAL-FACE-NASA's `MG`/`FP`/`FT`/`PT` subdirectories (never read at all
today, would need whole new loader functions — the campaign already gets
Family-A wind/attitude via Phase 2); TKE and any derived spectral turbulence
metric (belongs in a future `scripts/analyze_turbulence.py`, not the ingest
pipeline).

## `config.yaml` documentation sync

Update `output.standardized_columns` (lines 176-214) alongside each phase,
not batched to the end — add the new column names with one-line comments in
the existing `# <what> — <units> (<campaigns>)` style. Phase 1 adds the bulk
of the list; Phase 3 adds `EDR_isdac_tdreddy`.

## Verification

1. **Rebuild per phase**: run `python main.py`, then check
   `df.groupby("Campaign")[new_cols].count()` — new columns should be
   non-null only for the campaigns just changed, and existing columns
   (`Tair_C`, `Si`, `Lat`, etc.) must be unchanged for every campaign,
   including ones this phase didn't touch.
2. **Tier derivation**: run `python scripts/build_data_tiers.py` — confirm L1
   automatically carries the new L0 columns (verify this actually happens
   rather than trusting the merge-carries-everything claim), and confirm L2
   row counts are unaffected (new columns are absent from `CORE_COLS` by
   design).
3. **QC**: run `python scripts/qa_checks.py` unmodified first — every
   consumer of column lists there already guards with `if var in df.columns`,
   so it should pass with no changes needed. Then add plausible
   `HARD_BOUNDS` entries (`scripts/qa_checks.py`) for boundable columns only:
   `Roll_deg`/`Pitch_deg` (~±90 or ±180, check sign convention per family
   first), `AngleOfAttack_deg`/`Sideslip_deg` (~±30), `TAS_ms` (~0-300),
   `Heading_deg` (0-360). Do **not** bound the EDR columns or
   `Wind_*_ms`/`WindSpeed_ms` — turbulence/wind speed are legitimately
   unbounded by storm severity; a hard bound would false-flag real severe
   events.
4. **New diagnostic script**: add `scripts/diagnose_turbulence_coverage.py`
   (mirroring `scripts/diagnose_cpi_fusion.py`'s structure — argparse CLI,
   per-campaign coverage table, `logs/diagnose_turbulence_coverage/<ts>/` +
   `figs/.../<ts>/` via `scripts/log_paths.py`, same as every other diagnostic
   entry point) that reports, per campaign: non-null fraction for every new
   column, and per-EDR-family value-range histograms — this is the concrete
   check that the "don't unify EDR" boundary was actually respected (Family A
   should show small negative-to-single-digit log10 values; Family B a very
   different linear range; a mixup would be obvious in the histogram but
   invisible in isolation). Also print a level-flight `Roll_deg`/`Pitch_deg`
   sanity check per campaign (should hover near 0, not show a constant
   offset, which would indicate a sign/convention mismatch).
5. **Existing per-campaign tests**: extend `scripts/tests/test_attrex.py`,
   `test_iphex.py`, `test_macpex.py`, `test_posidon.py`, `test_ice_l.py`,
   `test_airs_ii.py` with new-column assertions as each campaign's phase
   lands, rather than one new turbulence-specific test file — keeps the test
   co-located with the extractor it covers, consistent with current layout.

## Critical files

- `parsers/utils.py` — new wind/attitude/airspeed helper section
- `parsers/arm.py`, `parsers/mc3e.py`, `parsers/mpace.py`, `parsers/olympex.py`,
  `parsers/iphex.py`, `parsers/attrex.py`, `parsers/posidon.py`,
  `parsers/macpex.py`, `parsers/isdac.py` — Phase 1
- `parsers/crystal_face_und.py`, `parsers/crystal_face_nasa.py`,
  `parsers/midcix.py`, `parsers/escape.py` — Phase 2
- `parsers/ice_l.py`, `parsers/airs_ii.py` — Phase 3
- `config.yaml` (`output.standardized_columns`, lines 176-214)
- `scripts/qa_checks.py` (`HARD_BOUNDS`)
- `scripts/diagnose_turbulence_coverage.py` (new)
