"""
Full diagnostic of combined_env_data.parquet.
Console-only: per-variable summary stats, per-campaign availability table,
row counts, and known-issue quick checks. No figures are written here — the
distribution/scatter/availability plots that used to live in this script now
live in scripts/plot_all_campaigns.py (see plots 01/02/07/08/10/11/12), which
is the single home for all-campaigns figures under figs/all-campaigns/.
"""

import sys
from pathlib import Path
from datetime import date

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PARQUET = ROOT / "data" / "out" / "combined_env_data.parquet"

VARIABLES = ["Tair_C", "P_hPa", "Alt_m", "Si", "Sw", "qv"]

CAMPAIGN_ORDER = [
    "ARM", "AIRS-II", "ATTREX", "CRYSTAL-FACE-NASA", "CRYSTAL-FACE-UND",
    "ESCAPE", "ICE-L", "IPHEX", "ISDAC", "MACPEX", "MC3E",
    "MIDCIX", "MPACE", "OLYMPEX", "POSIDON",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pct(n, total):
    return f"{100*n/total:.1f}%" if total else "N/A"


def availability_table(df):
    campaigns = sorted(df["Campaign"].unique(), key=lambda c: CAMPAIGN_ORDER.index(c) if c in CAMPAIGN_ORDER else 99)
    rows = []
    for c in campaigns:
        sub = df[df["Campaign"] == c]
        n   = len(sub)
        row = {"Campaign": c, "N_rows": n}
        for v in VARIABLES:
            if v in sub.columns:
                nn  = sub[v].notna().sum()
                row[v] = f"{nn:,} ({pct(nn,n)})"
            else:
                row[v] = "—"
        rows.append(row)
    return pd.DataFrame(rows)


def variable_stats(df):
    rows = []
    for v in VARIABLES:
        if v not in df.columns:
            continue
        s   = df[v].dropna()
        n   = len(s)
        tot = len(df)
        row = {
            "Variable":  v,
            "Non-null":  n,
            "Pct_avail": pct(n, tot),
            "Min":       f"{s.min():.3g}" if n else "—",
            "P5":        f"{s.quantile(.05):.3g}" if n else "—",
            "Median":    f"{s.median():.3g}" if n else "—",
            "Mean":      f"{s.mean():.3g}" if n else "—",
            "P95":       f"{s.quantile(.95):.3g}" if n else "—",
            "Max":       f"{s.max():.3g}" if n else "—",
            "Std":       f"{s.std():.3g}" if n else "—",
        }
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"\n{'='*70}")
    print(f"  CPI-THERMO Full Diagnostic  —  {date.today()}")
    print(f"{'='*70}\n")

    print(f"Loading {PARQUET} …")
    df = pd.read_parquet(PARQUET)
    print(f"  {len(df):,} rows × {len(df.columns)} columns\n")

    # ---- 1. overall shape & dtypes ----------------------------------------
    print("── Columns in parquet ──────────────────────────────────────────────")
    for col in df.columns:
        nn  = df[col].notna().sum()
        pct_v = pct(nn, len(df))
        print(f"  {col:<30} dtype={str(df[col].dtype):<12} non-null={nn:>9,}  ({pct_v})")

    # ---- 2. per-variable stats ---------------------------------------------
    print("\n── Per-variable summary stats ──────────────────────────────────────")
    stats = variable_stats(df)
    print(stats.to_string(index=False))

    # ---- 3. per-campaign availability --------------------------------------
    print("\n── Per-campaign data availability ──────────────────────────────────")
    avail = availability_table(df)
    print(avail.to_string(index=False))

    # ---- 4. row counts by campaign -----------------------------------------
    print("\n── Row counts by campaign ──────────────────────────────────────────")
    counts = df["Campaign"].value_counts().reindex(CAMPAIGN_ORDER).dropna()
    total  = len(df)
    for camp, n in counts.items():
        bar = "█" * int(40 * n / total)
        print(f"  {camp:<22} {n:>8,} ({pct(n,total):>5})  {bar}")
    print(f"  {'TOTAL':<22} {total:>8,}")

    # ---- 5. known-issue quick check ----------------------------------------
    print("\n── Known-issue quick checks ────────────────────────────────────────")
    esc = df[df["Campaign"] == "ESCAPE"]
    esc_low = (esc["P_hPa"] < 50).sum()
    print(f"  ESCAPE P_hPa < 50 hPa  : {esc_low:,} rows")

    iph = df[df["Campaign"] == "IPHEX"]
    if "Si" in df.columns:
        iph_sat = (iph["Si"] > 1.05).sum()
        print(f"  IPHEX Si > 1.05        : {iph_sat:,} rows")

    olym = df[df["Campaign"] == "OLYMPEX"]
    olym_sat = (olym["Si"] > 1.05).sum() if "Si" in df.columns else 0
    print(f"  OLYMPEX Si > 1.05      : {olym_sat:,} rows")

    arm_nan = df[df["Campaign"] == "ARM"]["qv"].isna().sum()
    arm_tot = len(df[df["Campaign"] == "ARM"])
    print(f"  ARM qv NaN             : {arm_nan:,} / {arm_tot:,} ({pct(arm_nan,arm_tot)})")

    print(f"\n{'='*70}")
    print("  Done. (Figures live in scripts/plot_all_campaigns.py --> figs/all-campaigns/)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
