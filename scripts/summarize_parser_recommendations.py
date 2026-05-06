#!/usr/bin/env python3
"""Summarize and prioritize parser recommendations from diagnostics CSV output.

Reads campaign_parser_recommendations.csv and campaign_variable_detail.csv,
then prints a ranked parser-fix backlog to help guide extractor updates.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError


GAP_WEIGHT = {
    "unavailable_in_raw": 0.6,
    "dropped_by_extractor": 1.0,
    "mostly_invalid_or_missing_post_extraction": 0.9,
    "partially_missing_post_extraction": 0.5,
    "covered": 0.0,
}

CONFIDENCE_WEIGHT = {
    "high": 1.0,
    "medium": 0.7,
    "low": 0.4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize parser recommendation CSVs")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("logs/diagnostics"),
        help="Directory containing diagnostics CSV outputs",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=25,
        help="How many prioritized rows to print",
    )
    parser.add_argument(
        "--write-csv",
        action="store_true",
        help="Also write ranked_parser_backlog.csv into --input-dir",
    )
    return parser.parse_args()


def load_inputs(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rec_path = input_dir / "campaign_parser_recommendations.csv"
    detail_path = input_dir / "campaign_variable_detail.csv"

    if not rec_path.exists() or not detail_path.exists():
        missing = [str(p) for p in [rec_path, detail_path] if not p.exists()]
        raise FileNotFoundError(
            "Missing diagnostics input files. Run diagnose_campaign_missingness first. "
            f"Missing: {', '.join(missing)}"
        )

    try:
        rec_df = pd.read_csv(rec_path)
    except EmptyDataError:
        rec_df = pd.DataFrame(
            columns=[
                "campaign",
                "variable",
                "gap_class",
                "suggestion_confidence",
                "suggested_raw_column",
                "raw_match_score",
                "recommended_action",
                "parser_file",
                "parser_function",
                "risk_hint",
            ]
        )

    try:
        detail_df = pd.read_csv(detail_path)
    except EmptyDataError:
        detail_df = pd.DataFrame(
            columns=[
                "campaign",
                "variable",
                "std_null_fraction_cleaned",
                "n_raw_rows",
                "raw_match_score",
            ]
        )

    return rec_df, detail_df


def build_ranked_backlog(rec_df: pd.DataFrame, detail_df: pd.DataFrame) -> pd.DataFrame:
    merged = rec_df.merge(
        detail_df[
            [
                "campaign",
                "variable",
                "std_null_fraction_cleaned",
                "n_raw_rows",
                "raw_match_score",
            ]
        ],
        on=["campaign", "variable"],
        how="left",
        suffixes=("", "_detail"),
    )

    merged["gap_weight"] = merged["gap_class"].map(GAP_WEIGHT).fillna(0.3)
    merged["confidence_weight"] = merged["suggestion_confidence"].map(CONFIDENCE_WEIGHT).fillna(0.5)
    merged["null_weight"] = merged["std_null_fraction_cleaned"].fillna(1.0)
    merged["row_weight"] = merged["n_raw_rows"].fillna(0).clip(lower=0)

    merged["impact_score"] = (
        merged["gap_weight"]
        * merged["confidence_weight"]
        * merged["null_weight"]
        * (merged["row_weight"] + 1).pow(0.25)
    )

    ranked = merged.sort_values("impact_score", ascending=False).reset_index(drop=True)
    cols = [
        "campaign",
        "variable",
        "gap_class",
        "suggestion_confidence",
        "suggested_raw_column",
        "raw_match_score",
        "std_null_fraction_cleaned",
        "n_raw_rows",
        "impact_score",
        "recommended_action",
        "parser_file",
        "parser_function",
        "risk_hint",
    ]
    present_cols = [c for c in cols if c in ranked.columns]
    return ranked[present_cols]


def main() -> None:
    args = parse_args()
    rec_df, detail_df = load_inputs(args.input_dir)
    ranked = build_ranked_backlog(rec_df, detail_df)

    if ranked.empty:
        print("No recommendation rows available.")
        return

    print("Top prioritized parser updates:")
    print(ranked.head(args.top).to_string(index=False))

    if args.write_csv:
        out_path = args.input_dir / "ranked_parser_backlog.csv"
        ranked.to_csv(out_path, index=False)
        print(f"\nWrote ranked backlog: {out_path}")


if __name__ == "__main__":
    main()
