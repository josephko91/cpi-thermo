#!/usr/bin/env python3
"""
Main script for processing field campaign environmental and positional data.

This script orchestrates the loading and processing of data from multiple
atmospheric field campaigns, extracting standardized environmental (Si, temperature)
and positional (lat, lon, altitude) measurements.

Usage:
    python main.py --campaigns ARM MC3E --output combined_env_data.parquet
    python main.py --all --output combined_env_data.parquet
    python main.py --config config.yaml
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import yaml
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from parsers import CAMPAIGN_LOADERS
from parsers.arm import extract_arm_standard
from parsers.crystal_face_nasa import extract_crystal_face_nasa_standard
from parsers.crystal_face_und import extract_crystal_face_und_standard
from parsers.mc3e import extract_mc3e_standard
from parsers.midcix import extract_midcix_standard
from parsers.olympex import extract_olympex_standard
from parsers.airs_ii import extract_airs_ii_standard
from parsers.attrex import extract_attrex_standard
from parsers.iphex import extract_iphex_standard
from parsers.isdac import extract_isdac_standard
from parsers.posidon import extract_posidon_standard
from parsers.escape import extract_escape_standard
from parsers.ice_l import extract_ice_l_standard
from parsers.macpex import extract_macpex_standard


# =============================================================================
# Campaign Configuration
# =============================================================================

# Default data directories (update these paths for your system)
DEFAULT_CAMPAIGN_CONFIG: Dict[str, Dict] = {
    "ARM": {
        "path": "./data/raw/ARM",
        "pattern": "*.t4archive.gz",
        "loader": "load_arm",
        "extractor": extract_arm_standard,
    },
    "CRYSTAL-FACE-NASA": {
        "path": "./data/raw/CRYSTAL-FACE-NASA/JW",
        "pattern": "*.WB57",
        "loader": "load_crystal_face_nasa",
        "extractor": extract_crystal_face_nasa_standard,
    },
    "CRYSTAL-FACE-UND": {
        "path": "./data/raw/CRYSTAL-FACE-UND",
        "pattern": "*MIS.CIT",
        "loader": "load_crystal_face_und",
        "extractor": extract_crystal_face_und_standard,
    },
    "MC3E": {
        "path": "./data/raw/MC3E",
        "pattern": "*",
        "loader": "load_mc3e",
        "extractor": extract_mc3e_standard,
    },
    "MIDCIX": {
        "path": "./data/raw/MidCix",
        "pattern": "*",
        "loader": "load_midcix",
        "extractor": extract_midcix_standard,
    },
    "OLYMPEX": {
        "path": "./data/raw/OLYMPEX",
        "pattern": "*",
        "loader": "load_olympex",
        "extractor": extract_olympex_standard,
    },
    "AIRS-II": {
        "path": "./data/raw/AIRS-II",
        "pattern": "*.nc",
        "loader": "load_airs_ii",
        "extractor": extract_airs_ii_standard,
    },
    "ATTREX": {
        "path": "./data/raw/ATTREX",
        "pattern": "*.ict",
        "loader": "load_attrex",
        "extractor": extract_attrex_standard,
    },
    "IPHEX": {
        "path": "./data/raw/IPHEX",
        "pattern": "*.iphex",
        "loader": "load_iphex",
        "extractor": extract_iphex_standard,
    },
    "ISDAC": {
        "path": "./data/raw/ISDAC/strapp-convair_bulk/CommaDelimited",
        "pattern": "*.txt",
        "loader": "load_isdac",
        "extractor": extract_isdac_standard,
    },
    "POSIDON": {
        "path": "./data/raw/POSIDON",
        "pattern": "*.ict",
        "loader": "load_posidon",
        "extractor": extract_posidon_standard,
    },
    "ESCAPE": {
        "path": "./data/raw/ESCAPE/spec-learjet-state",
        "pattern": "ESCAPE-Page0_Learjet_*.ict",
        "loader": "load_escape",
        "extractor": extract_escape_standard,
    },
    "ICE-L": {
        "path": "./data/raw/ICE-L",
        "pattern": "*.PNI.nc",
        "loader": "load_ice_l",
        "extractor": extract_ice_l_standard,
    },
    "MACPEX": {
        "path": "./data/raw/MACPEX",
        "pattern": "*.ict",
        "loader": "load_macpex",
        "extractor": extract_macpex_standard,
    },
}


# =============================================================================
# Processing Functions
# =============================================================================

def load_campaign_config(config_path: Path) -> Dict:
    """Load campaign configuration from YAML file.

    The YAML schema stores campaigns under a 'campaigns' key and does not
    include extractor callables. This function flattens to the same dict
    structure as DEFAULT_CAMPAIGN_CONFIG and injects extractors so that
    parser-specific logic is preserved when running with --config.
    """
    with open(config_path, 'r') as f:
        raw = yaml.safe_load(f)
    # Support both flat {"ARM": {...}} and nested {"campaigns": {"ARM": {...}}}
    campaigns_yaml = raw.get("campaigns", raw)
    result: Dict = {}
    for name, yaml_cfg in campaigns_yaml.items():
        default = DEFAULT_CAMPAIGN_CONFIG.get(name, {})
        # YAML keys (path, pattern, h2o_ranking) override defaults; extractor is
        # injected from DEFAULT_CAMPAIGN_CONFIG since it cannot be serialised to YAML.
        result[name] = {**default, **yaml_cfg}
    return result


def process_campaign(
    campaign_name: str,
    config: Optional[Dict] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Process a single campaign and return standardized DataFrame.
    
    Parameters
    ----------
    campaign_name : str
        Name of the campaign (e.g., "ARM", "MC3E").
    config : dict, optional
        Campaign configuration. If None, uses DEFAULT_CAMPAIGN_CONFIG.
    verbose : bool
        Whether to print progress messages.
        
    Returns
    -------
    pd.DataFrame
        Standardized environmental and positional data.
    """
    if config is None:
        config = DEFAULT_CAMPAIGN_CONFIG.get(campaign_name)
    
    if config is None:
        raise ValueError(f"Unknown campaign: {campaign_name}")
    
    data_dir = Path(config["path"])
    pattern = config.get("pattern", "*")
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    if verbose:
        logging.info(f"Processing {campaign_name}...")
        logging.info(f"  Data path: {data_dir}")
        logging.info(f"  Pattern: {pattern}")
    
    # Get loader function
    loader = CAMPAIGN_LOADERS.get(campaign_name)
    if loader is None:
        raise ValueError(f"No loader found for campaign: {campaign_name}")

    # Pass campaign-specific extra kwargs from config (e.g. h2o_ranking for ATTREX)
    extra_kwargs = {}
    for key in ("h2o_ranking",):
        if key in config:
            extra_kwargs[key] = config[key]

    # Load raw data
    df_raw = loader(data_dir, pattern=pattern, **extra_kwargs)
    
    if verbose:
        logging.info(f"  Loaded {len(df_raw):,} records")
    
    # Extract standardized columns
    extractor = config.get("extractor")
    if extractor:
        df_std = extractor(df_raw)
    else:
        # Basic extraction if no custom extractor
        df_std = pd.DataFrame({
            "Timestamp": df_raw.get("Timestamp", pd.NaT),
            "Tair_C": df_raw.get("Air_Temp", df_raw.get("T_C", pd.NA)),
            "Si": df_raw.get("Si", pd.NA),
            "Lat": pd.NA,
            "Lon": pd.NA,
            "Alt_m": pd.NA,
            "Campaign": campaign_name,
            "source_file": df_raw.get("source_file", ""),
        })

    # Uniform Si floor/ceiling: Si < -1 is physically impossible; Si > 2 is
    # almost certainly an instrument artefact across all campaigns.
    for col in [c for c in df_std.columns if c == "Si" or c.startswith("Si_")]:
        df_std[col] = pd.to_numeric(df_std[col], errors="coerce").clip(-1.0, 2.0)

    # qv < 0 is physically impossible; clip negative values to NaN.
    for col in [c for c in df_std.columns if c == "qv" or c.startswith("qv_")]:
        s = pd.to_numeric(df_std[col], errors="coerce")
        df_std[col] = s.where(s >= 0, np.nan)

    # Ensure Campaign column
    df_std["Campaign"] = campaign_name
    
    if verbose:
        n_valid = df_std[["Timestamp", "Tair_C", "Si"]].notna().all(axis=1).sum()
        logging.info(f"  Valid records (with Timestamp, Tair_C, Si): {n_valid:,}")
    
    return df_std


def process_all_campaigns(
    campaigns: Optional[List[str]] = None,
    config: Optional[Dict] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Process multiple campaigns and return combined DataFrame.
    
    Parameters
    ----------
    campaigns : list of str, optional
        List of campaign names to process. If None, processes all campaigns.
    config : dict, optional
        Campaign configurations. If None, uses DEFAULT_CAMPAIGN_CONFIG.
    verbose : bool
        Whether to print progress messages.
        
    Returns
    -------
    pd.DataFrame
        Combined standardized data from all campaigns.
    """
    if config is None:
        config = DEFAULT_CAMPAIGN_CONFIG
    
    if campaigns is None:
        campaigns = list(config.keys())
    
    all_dfs = []
    
    for campaign in campaigns:
        try:
            campaign_config = config.get(campaign, DEFAULT_CAMPAIGN_CONFIG.get(campaign))
            df = process_campaign(campaign, campaign_config, verbose)
            all_dfs.append(df)
        except Exception as e:
            logging.error(f"Error processing {campaign}: {e}")
            continue
    
    if not all_dfs:
        raise RuntimeError("No campaigns were successfully processed")
    
    combined = pd.concat(all_dfs, ignore_index=True)
    
    if verbose:
        logging.info(f"\nCombined dataset: {len(combined):,} total records")
        logging.info(f"Campaigns: {combined['Campaign'].unique().tolist()}")
    
    return combined


def save_output(
    df: pd.DataFrame,
    output_path: Path,
    verbose: bool = True,
) -> None:
    """Save DataFrame to file (supports parquet and CSV)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.suffix == ".parquet":
        df.to_parquet(output_path, index=False)
    elif output_path.suffix == ".csv":
        df.to_csv(output_path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {output_path.suffix}")
    
    if verbose:
        logging.info(f"Saved to: {output_path}")

# =============================================================================
# Diagnostics
# =============================================================================

# Ordered list of all known Si instrument columns (excludes 'Si' itself)
_SI_INSTRUMENT_COLS = [
    "Si_chilled_mirror",
    "Si_DLH",
    "Si_JLH",
    "Si_HWV",
    "Si_HW",
    "Si_NOAA",
    "Si_UCATS",
    "Si_MRTDL",
    "Si_ophir_tdl",
    "Si_LH_unspecified",
    "Si_frost_point",
    "Si_ALIAS",
    "Si_FISH",
]

# Short display labels for the terminal table
_SI_SHORT = {
    "Si":                 "Si",
    "Si_chilled_mirror":  "CM",
    "Si_DLH":             "DLH",
    "Si_JLH":             "JLH",
    "Si_HWV":             "HWV",
    "Si_HW":              "HW",
    "Si_NOAA":            "NOAA",
    "Si_UCATS":           "UCATS",
    "Si_MRTDL":           "MRTDL",
    "Si_ophir_tdl":       "TDL",
    "Si_LH_unspecified":  "LHU",
    "Si_frost_point":     "FP",
    "Si_ALIAS":           "ALIAS",
    "Si_FISH":            "FISH",
}

# Ordered list of all known qv instrument columns (excludes 'qv' itself)
_QV_INSTRUMENT_COLS = [
    "qv_chilled_mirror",
    "qv_dlh",
    "qv_jlh",
    "qv_hwv",
    "qv_hw",
    "qv_noaa",
    "qv_ucats",
    "qv_mrtdl",
    "qv_ophir_tdl",
    "qv_lh_unspecified",
    "qv_frost_point",
]

_QV_SHORT = {
    "qv":                  "qv",
    "qv_chilled_mirror":   "CM",
    "qv_dlh":              "DLH",
    "qv_jlh":              "JLH",
    "qv_hwv":              "HWV",
    "qv_hw":               "HW",
    "qv_noaa":             "NOAA",
    "qv_ucats":            "UCATS",
    "qv_mrtdl":            "MRTDL",
    "qv_ophir_tdl":        "TDL",
    "qv_lh_unspecified":   "LHU",
    "qv_frost_point":      "FP",
}


def generate_diagnostics(df: pd.DataFrame, diag_dir: Path) -> None:
    """Write diagnostic CSVs to diag_dir."""
    diag_dir = Path(diag_dir)
    diag_dir.mkdir(parents=True, exist_ok=True)

    campaigns = sorted(df["Campaign"].unique())

    # 1. Campaign summary
    rows = []
    for camp in campaigns:
        sub = df[df["Campaign"] == camp]
        n = len(sub)
        ts = sub["Timestamp"].dropna()
        rows.append({
            "Campaign":        camp,
            "n_records":       n,
            "date_start":      ts.min().date() if len(ts) else None,
            "date_end":        ts.max().date() if len(ts) else None,
            "n_flight_days":   sub["Timestamp"].dt.date.nunique() if len(ts) else 0,
            "tair_mean":       sub["Tair_C"].mean(),
            "tair_std":        sub["Tair_C"].std(),
            "tair_pct_valid":  sub["Tair_C"].notna().mean() * 100,
            "si_mean":         sub["Si"].mean(),
            "si_std":          sub["Si"].std(),
            "si_pct_valid":    sub["Si"].notna().mean() * 100,
            "si_pct_iss":      (sub["Si"] > 0).sum() / n * 100 if n else None,
        })
    summary_df = pd.DataFrame(rows)
    summary_path = diag_dir / "campaign_summary.csv"
    summary_df.round(4).to_csv(summary_path, index=False)
    logging.info(f"  Diagnostics: {summary_path}")

    # 2. Si instrument coverage — pct_valid per campaign × Si_* column
    si_cols = ["Si"] + [c for c in _SI_INSTRUMENT_COLS if c in df.columns]
    cov_rows = []
    for camp in campaigns:
        sub = df[df["Campaign"] == camp]
        n = len(sub)
        row = {"Campaign": camp, "n_records": n}
        for col in si_cols:
            if col in sub.columns:
                row[col] = sub[col].notna().sum() / n * 100 if n else 0.0
            else:
                row[col] = np.nan
        cov_rows.append(row)
    cov_df = pd.DataFrame(cov_rows)
    cov_path = diag_dir / "si_instrument_coverage.csv"
    cov_df.round(2).to_csv(cov_path, index=False)
    logging.info(f"  Diagnostics: {cov_path}")

    # 3. qv instrument coverage — pct_valid per campaign × qv_* column
    qv_cols = ["qv"] + [c for c in _QV_INSTRUMENT_COLS if c in df.columns]
    qv_cov_rows = []
    for camp in campaigns:
        sub = df[df["Campaign"] == camp]
        n = len(sub)
        row = {"Campaign": camp, "n_records": n}
        for col in qv_cols:
            if col in sub.columns:
                row[col] = sub[col].notna().sum() / n * 100 if n else 0.0
            else:
                row[col] = np.nan
        qv_cov_rows.append(row)
    qv_cov_df = pd.DataFrame(qv_cov_rows)
    qv_cov_path = diag_dir / "qv_instrument_coverage.csv"
    qv_cov_df.round(2).to_csv(qv_cov_path, index=False)
    logging.info(f"  Diagnostics: {qv_cov_path}")


# =============================================================================
# Summary Statistics
# =============================================================================

def print_summary(df: pd.DataFrame, output_path: Optional[Path] = None,
                  diag_dir: Optional[Path] = None,
                  figs_dir: Optional[Path] = None) -> None:
    """Print a comprehensive summary to stdout and the log."""
    campaigns = sorted(df["Campaign"].unique())
    n_total = len(df)
    si_cols_present = ["Si"] + [c for c in _SI_INSTRUMENT_COLS if c in df.columns]

    lines: List[str] = []
    W = 72

    lines.append("=" * W)
    lines.append(" PIPELINE SUMMARY")
    lines.append("=" * W)
    lines.append(f"  Campaigns processed : {len(campaigns)}")
    lines.append(f"  Total records       : {n_total:,}")
    lines.append("")

    # --- Per-campaign table ---
    lines.append(f"  {'Campaign':<24} {'Records':>9}  {'Date range':<24}  {'Si valid':>8}  {'Tair valid':>10}")
    lines.append("  " + "-" * (W - 2))
    for camp in campaigns:
        sub = df[df["Campaign"] == camp]
        n = len(sub)
        ts = sub["Timestamp"].dropna()
        if len(ts):
            date_range = f"{ts.min().date()} → {ts.max().date()}"
        else:
            date_range = "—"
        si_pct   = sub["Si"].notna().mean() * 100 if "Si" in sub.columns else 0.0
        tair_pct = sub["Tair_C"].notna().mean() * 100
        lines.append(
            f"  {camp:<24} {n:>9,}  {date_range:<24}  {si_pct:>7.1f}%  {tair_pct:>9.1f}%"
        )
    lines.append("")

    # --- Si instrument coverage table ---
    # Only show columns that have at least 1 non-NaN value anywhere
    active_si = [c for c in si_cols_present
                 if c in df.columns and df[c].notna().any()]

    if len(active_si) > 1:
        headers = [_SI_SHORT.get(c, c) for c in active_si]
        col_w = max(len(h) for h in headers) + 2

        lines.append("  Si instrument coverage (% of campaign records with valid data):")
        hdr = f"  {'Campaign':<24}" + "".join(f"{h:>{col_w}}" for h in headers)
        lines.append(hdr)
        lines.append("  " + "-" * (W - 2))
        for camp in campaigns:
            sub = df[df["Campaign"] == camp]
            n = len(sub)
            row = f"  {camp:<24}"
            for col in active_si:
                if col in sub.columns and n > 0:
                    pct = sub[col].notna().sum() / n * 100
                    cell = f"{pct:.0f}%" if pct > 0 else "—"
                else:
                    cell = "—"
                row += f"{cell:>{col_w}}"
            lines.append(row)
        lines.append("")

    # --- Overall Si stats ---
    lines.append(f"  {'Metric':<28} {'Mean':>8}  {'Std':>7}  {'Median':>8}  {'% valid':>8}")
    lines.append("  " + "-" * (W - 2))
    for col in active_si:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if len(s) == 0:
            continue
        lines.append(
            f"  {col:<28} {s.mean():>8.4f}  {s.std():>7.4f}  {np.median(s):>8.4f}  {len(s)/n_total*100:>7.1f}%"
        )
    lines.append("")

    # --- qv instrument coverage table ---
    qv_cols_present = ["qv"] + [c for c in _QV_INSTRUMENT_COLS if c in df.columns]
    active_qv = [c for c in qv_cols_present if c in df.columns and df[c].notna().any()]

    if len(active_qv) >= 1:
        qv_headers = [_QV_SHORT.get(c, c) for c in active_qv]
        qv_col_w = max(len(h) for h in qv_headers) + 2

        lines.append("  qv instrument coverage (% of campaign records with valid data):")
        hdr = f"  {'Campaign':<24}" + "".join(f"{h:>{qv_col_w}}" for h in qv_headers)
        lines.append(hdr)
        lines.append("  " + "-" * (W - 2))
        for camp in campaigns:
            sub = df[df["Campaign"] == camp]
            n = len(sub)
            row = f"  {camp:<24}"
            for col in active_qv:
                if col in sub.columns and n > 0:
                    pct = sub[col].notna().sum() / n * 100
                    cell = f"{pct:.0f}%" if pct > 0 else "—"
                else:
                    cell = "—"
                row += f"{cell:>{qv_col_w}}"
            lines.append(row)
        lines.append("")

    # --- Overall qv stats ---
    if active_qv:
        lines.append(f"  {'qv Metric':<28} {'Mean':>8}  {'Std':>7}  {'Median':>8}  {'% valid':>8}")
        lines.append("  " + "-" * (W - 2))
        for col in active_qv:
            if col not in df.columns:
                continue
            s = df[col].dropna()
            if len(s) == 0:
                continue
            lines.append(
                f"  {col:<28} {s.mean():>8.4f}  {s.std():>7.4f}  {np.median(s):>8.4f}  {len(s)/n_total*100:>7.1f}%"
            )
        lines.append("")

    # --- Output paths ---
    if output_path:
        lines.append(f"  Output   : {output_path}")
    if diag_dir:
        lines.append(f"  Diag dir : {diag_dir}/")
    if figs_dir:
        lines.append(f"  Figs dir : {figs_dir}/")
    lines.append("=" * W)

    text = "\n".join(lines)
    print(text)
    logging.info("\n" + text)

# =============================================================================
# CLI Interface
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    root = Path(__file__).parent
    parser = argparse.ArgumentParser(
        description="Process field campaign environmental data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process specific campaigns
    python main.py --campaigns ARM MC3E --output data/out/combined.parquet

    # Process all campaigns
    python main.py --all --output data/out/combined.parquet

    # Use custom config file
    python main.py --config config.yaml --output data/out/combined.parquet

    # Skip figures
    python main.py --all --no-plots
        """,
    )

    parser.add_argument(
        "--campaigns",
        nargs="+",
        choices=list(DEFAULT_CAMPAIGN_CONFIG.keys()),
        help="Campaign(s) to process",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all available campaigns",
    )

    parser.add_argument(
        "--config",
        type=Path,
        help="Path to YAML configuration file",
    )

    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=root / "data" / "out" / "combined_env_data.parquet",
        help="Output parquet/csv path",
    )

    parser.add_argument(
        "--log-file",
        type=Path,
        default=root / "logs" / "output.log",
        help="Path to log file",
    )

    parser.add_argument(
        "--diag-dir",
        type=Path,
        default=root / "logs" / "diagnostics",
        help="Directory for diagnostic CSV outputs",
    )

    parser.add_argument(
        "--figs-dir",
        type=Path,
        default=root / "figs" / "all-campaigns",
        help="Directory for figure outputs",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process data without saving output or figures",
    )

    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip figure generation",
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress messages",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    verbose = not args.quiet

    # Ensure log directory exists before configuring file handler
    args.log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(args.log_file),
            logging.StreamHandler(),
        ],
    )

    try:
        # Determine campaigns to process
        if args.all:
            campaigns = None
        elif args.campaigns:
            campaigns = args.campaigns
        else:
            logging.error("Specify --campaigns or --all")
            sys.exit(1)

        # Load custom config if provided
        config = None
        if args.config:
            config = load_campaign_config(args.config)

        # Process campaigns
        df = process_all_campaigns(campaigns, config, verbose)

        # Save parquet / CSV
        if not args.dry_run:
            save_output(df, args.output, verbose)

        # Diagnostics CSVs
        if not args.dry_run:
            generate_diagnostics(df, args.diag_dir)

        # Figures — run plot_all_campaigns.py as subprocess so it can find the saved parquet
        if not args.dry_run and not args.no_plots:
            plot_script = Path(__file__).parent / "scripts" / "plot_all_campaigns.py"
            if plot_script.exists():
                python = sys.executable
                env_path = str(args.output)
                out_dir  = str(args.figs_dir)
                logging.info(f"Generating figures → {out_dir}")
                result = subprocess.run(
                    [python, str(plot_script), "--env", env_path, "--out", out_dir],
                    capture_output=True, text=True,
                )
                if result.stdout:
                    logging.info(result.stdout.strip())
                if result.returncode != 0:
                    logging.warning(f"plot_all_campaigns.py exited with code {result.returncode}")
                    if result.stderr:
                        logging.warning(result.stderr.strip())
            else:
                logging.warning(f"Plot script not found: {plot_script}")

        # Summary to terminal
        if verbose:
            print_summary(
                df,
                output_path=args.output if not args.dry_run else None,
                diag_dir=args.diag_dir if not args.dry_run else None,
                figs_dir=args.figs_dir if not args.dry_run and not args.no_plots else None,
            )

        return df

    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
