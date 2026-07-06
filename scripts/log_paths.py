"""
Shared helpers for a consistent, non-overwriting logs/ and figs/ layout.

Every diagnostic entry point (main.py, qa_checks.py, diagnose_cpi_fusion.py,
diagnose_campaign_missingness.py) writes into its own

    <logs-or-figs>/<script-name>/<YYYYMMDD_HHMMSS>/

directory so re-running a diagnostic never clobbers a previous run, and
refreshes a `<logs-or-figs>/<script-name>/latest` symlink so downstream
tools/humans can find the most recent run without knowing its timestamp.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def timestamp() -> str:
    """A sortable, filesystem-safe timestamp shared across a single run's outputs."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def new_run_dir(base: Path, ts: str) -> Path:
    """Create base/<ts>/, point base/latest at it, and return the run dir."""
    run_dir = base / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    update_latest(base, run_dir)
    return run_dir


def update_latest(base: Path, run_dir: Path) -> None:
    """Point base/latest at run_dir (relative symlink), replacing any prior link."""
    base.mkdir(parents=True, exist_ok=True)
    latest = base / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(run_dir.name)
