#!/usr/bin/env python3
"""
QC3 residual stuck-sensor / merge_asof-granularity diagnostic
================================================================
Tests the hypothesis in docs/decisions/2026-07-06-qc3-stuck-sensor.md and
GitHub issue #10: that the 309 QC3 stuck-sensor runs remaining in
CRYSTAL-FACE-NASA, ESCAPE, ISDAC, and MIDCIX are a merge-tolerance /
low-precision artifact of each campaign's multi-instrument merge_asof
architecture (a sparser reference source repeating its last-known value
across several samples of a higher-frequency stream), not a genuine
multi-minute sensor freeze.

Two signatures distinguish the two explanations for a run of >=30
bit-exact-identical values in some variable:

  1. Timestamp cadence during the run. If the underlying Timestamp column
     keeps advancing at the same rate as the rest of the file while the
     variable value is frozen, that's the signature of a fast stream being
     matched against a slower reference (merge_asof artifact). If the
     Timestamps themselves go sparse/flat too, that points to a genuine
     instrument dropout instead.
  2. Run-duration clustering. A periodic reference source updating every
     R seconds should produce run durations that cluster near R (or its
     multiples), not durations spread uniformly at random.

Outputs (logs/qc3_merge_granularity/<timestamp>/ and
figs/qc3_merge_granularity/<timestamp>/, with `latest` symlinks):
  qc3_merge_granularity_runs.csv       - one row per residual stuck run, with
                                          run_duration_s, implied_dt_s (avg
                                          spacing between samples inside the
                                          run), file_median_dt_s (native file
                                          cadence outside stuck runs), and
                                          their ratio
  qc3_merge_granularity_summary.csv    - per campaign/variable aggregate:
                                          median ratio, % runs consistent
                                          with native cadence, modal run
                                          duration
  qc3_merge_granularity_ratio_hist.png - histogram of implied_dt/file_dt
                                          ratio, one panel per campaign
  qc3_merge_granularity_duration_hist.png - histogram of run durations,
                                          one panel per campaign/variable

Usage:
    python scripts/diagnose_qc3_merge_granularity.py
    python scripts/diagnose_qc3_merge_granularity.py --env data/out/combined_env_data.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.log_paths import timestamp as _run_timestamp, update_latest  # noqa: E402

# ---------------------------------------------------------------------------
# Scope: the 309 residual runs from docs/decisions/2026-07-06-qc3-stuck-sensor.md
# ---------------------------------------------------------------------------
RESIDUAL_VARS: dict[str, list[str]] = {
    "CRYSTAL-FACE-NASA": ["Tair_C", "P_hPa"],
    "ESCAPE":            ["Tair_C", "P_hPa", "qv", "Alt_m"],
    "ISDAC":             ["Alt_m", "P_hPa"],
    "MIDCIX":            ["P_hPa"],
}
MIN_RUN = 30  # matches QC3's own threshold

COLORS = {
    "CRYSTAL-FACE-NASA": "#ff7f0e",
    "ESCAPE":            "#ff9896",
    "ISDAC":             "#e377c2",
    "MIDCIX":            "#d62728",
}

STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor":   "#f8f8f8",
    "axes.grid":        True,
    "grid.color":       "white",
    "grid.linewidth":   0.8,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "font.size":        9,
}

# Native-cadence-consistent band: implied_dt/file_dt within this range means
# the Timestamp column advanced through the run at roughly the file's normal
# sample rate (merge_asof/reference-cadence signature).
RATIO_LO, RATIO_HI = 0.7, 1.3


def _parse_args() -> argparse.Namespace:
    ts = _run_timestamp()
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", type=Path,
                   default=ROOT / "data" / "out" / "combined_env_data.parquet",
                   help="Path to combined parquet")
    p.add_argument("--out", type=Path,
                   default=ROOT / "logs" / "qc3_merge_granularity" / ts,
                   help="Directory for CSV output")
    p.add_argument("--figs", type=Path,
                   default=ROOT / "figs" / "qc3_merge_granularity" / ts,
                   help="Directory for figure output")
    return p.parse_args()


def find_stuck_runs(sf_grp: pd.DataFrame, var: str) -> list[dict]:
    """Run-length-encode `var` within one source_file's rows (sorted by Timestamp).

    Mirrors qa_checks.py's check_03_stuck_sensor run-detection exactly so
    results line up with the QC3 report, restricted to the single variable
    of interest here.
    """
    vals = pd.to_numeric(sf_grp[var], errors="coerce").values
    runs = []
    if np.isnan(vals).all():
        return runs

    run_start, run_len = 0, 1
    for j in range(1, len(vals)):
        v_prev, v_curr = vals[j - 1], vals[j]
        same = (np.isnan(v_prev) and np.isnan(v_curr)) or (
            not np.isnan(v_prev) and not np.isnan(v_curr)
            and abs(v_curr - v_prev) < 1e-6
        )
        if same:
            run_len += 1
        else:
            if run_len >= MIN_RUN and not np.isnan(vals[run_start]):
                runs.append((run_start, j - 1, run_len))
            run_start, run_len = j, 1
    if run_len >= MIN_RUN and not np.isnan(vals[run_start]):
        runs.append((run_start, len(vals) - 1, run_len))
    return runs


def main() -> None:
    args = _parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.figs.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.env} ...")
    df = pd.read_parquet(args.env)
    print(f"  {len(df):,} records\n")

    rows = []
    for camp, variables in RESIDUAL_VARS.items():
        sub_c = df[df["Campaign"] == camp]
        if sub_c.empty:
            continue
        for sf, sf_grp in sub_c.groupby("source_file"):
            sf_grp = sf_grp.sort_values("Timestamp").reset_index(drop=True)
            ts = pd.to_datetime(sf_grp["Timestamp"], utc=True, errors="coerce")
            dt_all = ts.diff().dt.total_seconds()

            # Native file cadence: median sample spacing, excluding non-positive
            # or absurdly large (>2h intra-file gap) diffs that would skew it.
            dt_valid = dt_all[(dt_all > 0) & (dt_all < 7200)]
            file_median_dt = dt_valid.median() if len(dt_valid) else np.nan

            for var in variables:
                if var not in sf_grp.columns:
                    continue
                for run_start, run_end, run_len in find_stuck_runs(sf_grp, var):
                    ts_start, ts_end = ts.iloc[run_start], ts.iloc[run_end]
                    duration_s = (ts_end - ts_start).total_seconds()
                    implied_dt = duration_s / (run_len - 1) if run_len > 1 else np.nan
                    ratio = (implied_dt / file_median_dt
                             if file_median_dt and file_median_dt > 0 else np.nan)
                    rows.append({
                        "Campaign": camp,
                        "source_file": sf,
                        "variable": var,
                        "run_start": ts_start,
                        "run_end": ts_end,
                        "run_length": run_len,
                        "run_duration_s": round(duration_s, 3),
                        "implied_dt_s": round(implied_dt, 4) if not np.isnan(implied_dt) else np.nan,
                        "file_median_dt_s": round(file_median_dt, 4) if not np.isnan(file_median_dt) else np.nan,
                        "ratio_implied_to_file_dt": round(ratio, 3) if not np.isnan(ratio) else np.nan,
                        "native_cadence_consistent": bool(
                            not np.isnan(ratio) and RATIO_LO <= ratio <= RATIO_HI
                        ),
                    })

    runs_df = pd.DataFrame(rows)
    runs_path = args.out / "qc3_merge_granularity_runs.csv"
    runs_df.to_csv(runs_path, index=False)
    print(f"Saved {runs_path}  ({len(runs_df):,} runs)")

    # ── Per campaign/variable summary ───────────────────────────────────────
    summary_rows = []
    for (camp, var), grp in runs_df.groupby(["Campaign", "variable"]):
        valid_ratio = grp["ratio_implied_to_file_dt"].dropna()
        pct_native = (grp["native_cadence_consistent"].mean() * 100
                      if len(grp) else np.nan)
        # Modal run duration, rounded to nearest second, as a proxy for a
        # periodic reference-update interval.
        modal_duration = (grp["run_duration_s"].round(0).mode().iloc[0]
                          if len(grp) else np.nan)
        summary_rows.append({
            "Campaign": camp,
            "variable": var,
            "n_runs": len(grp),
            "median_ratio_implied_to_file_dt": round(valid_ratio.median(), 3) if len(valid_ratio) else np.nan,
            "pct_native_cadence_consistent": round(pct_native, 1),
            "median_run_duration_s": round(grp["run_duration_s"].median(), 1),
            "modal_run_duration_s": modal_duration,
            "median_run_length": round(grp["run_length"].median(), 1),
        })
    summary_df = pd.DataFrame(summary_rows).sort_values(["Campaign", "variable"])
    summary_path = args.out / "qc3_merge_granularity_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved {summary_path}")
    print("\n" + summary_df.to_string(index=False))

    # ── Figure 1: ratio histogram, one panel per campaign ───────────────────
    campaigns = [c for c in RESIDUAL_VARS if c in runs_df["Campaign"].unique()]
    if campaigns:
        with plt.rc_context(STYLE):
            fig, axes = plt.subplots(1, len(campaigns), figsize=(4 * len(campaigns), 3.5),
                                      constrained_layout=True, sharey=False)
            if len(campaigns) == 1:
                axes = [axes]
            for ax, camp in zip(axes, campaigns):
                vals = runs_df.loc[runs_df["Campaign"] == camp, "ratio_implied_to_file_dt"].dropna()
                ax.hist(vals, bins=20, color=COLORS.get(camp, "gray"), alpha=0.8, edgecolor="white")
                ax.axvspan(RATIO_LO, RATIO_HI, color="green", alpha=0.12,
                           label="native-cadence-consistent")
                ax.axvline(1.0, color="k", linestyle="--", linewidth=1)
                ax.set_title(camp, fontsize=9, fontweight="bold", color=COLORS.get(camp, "k"))
                ax.set_xlabel("implied_dt / file_median_dt")
                ax.set_ylabel("n runs")
                ax.legend(fontsize=6, loc="upper right")
            fig.suptitle(
                "QC3 residual runs — Timestamp cadence during the stuck run\n"
                "(ratio ≈ 1 ⇒ Timestamp kept advancing normally while the value froze: merge-tolerance signature)",
                fontsize=9, fontweight="bold")
            out_p = args.figs / "qc3_merge_granularity_ratio_hist.png"
            fig.savefig(out_p, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {out_p}")

    # ── Figure 2: run-duration histogram, one panel per campaign+variable ──
    combos = sorted(runs_df.groupby(["Campaign", "variable"]).groups.keys())
    if combos:
        n = len(combos)
        ncols = 3
        nrows = int(np.ceil(n / ncols))
        with plt.rc_context(STYLE):
            fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                                      constrained_layout=True)
            axes = np.atleast_1d(axes).ravel()
            for ax, (camp, var) in zip(axes, combos):
                vals = runs_df.loc[
                    (runs_df["Campaign"] == camp) & (runs_df["variable"] == var),
                    "run_duration_s",
                ]
                ax.hist(vals, bins=15, color=COLORS.get(camp, "gray"), alpha=0.8, edgecolor="white")
                ax.set_title(f"{camp} — {var} (n={len(vals)})", fontsize=8, fontweight="bold")
                ax.set_xlabel("run duration (s)", fontsize=7)
                ax.set_ylabel("n runs", fontsize=7)
                ax.tick_params(labelsize=7)
            for ax in axes[len(combos):]:
                ax.axis("off")
            fig.suptitle("QC3 residual runs — run-duration clustering by campaign/variable",
                         fontsize=10, fontweight="bold")
            out_p = args.figs / "qc3_merge_granularity_duration_hist.png"
            fig.savefig(out_p, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {out_p}")

    update_latest(args.out.parent, args.out)
    update_latest(args.figs.parent, args.figs)
    print(f"\nLatest run: {args.out.parent / 'latest'} -> {args.out.name}")


if __name__ == "__main__":
    main()
