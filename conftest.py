"""Pytest hooks for optional single-file test output logging."""

from __future__ import annotations

from pathlib import Path


LOG_PATH: Path | None = None


def _write(text: str) -> None:
    if LOG_PATH is None:
        return
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n\n")


def pytest_sessionstart(session):
    global LOG_PATH
    log_file = session.config.getoption("test_log_file")
    LOG_PATH = Path(log_file) if log_file else None

    if LOG_PATH is not None:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        if LOG_PATH.exists():
            LOG_PATH.unlink()
        LOG_PATH.write_text("Pytest combined output log\n" + "=" * 60 + "\n\n", encoding="utf-8")


def pytest_addoption(parser):
    """Add a CLI option for optional single-file test logging.

    Usage: `pytest --test-log-file path/to/pytest_output.txt`.
    """
    parser.addoption(
        "--test-log-file",
        action="store",
        default=None,
        dest="test_log_file",
        help="Write captured output/errors and tracebacks to one file; disabled by default",
    )


def pytest_runtest_logreport(report):
    if LOG_PATH is None:
        return

    blocks = [f"=== {report.nodeid} [{report.when}] {report.outcome.upper()} ==="]
    for title, content in getattr(report, "sections", []):
        if title.startswith("Captured ") and content:
            blocks.append(f"{title}:\n{content}")

    if report.failed:
        longrepr = getattr(report, "longreprtext", "")
        if longrepr:
            blocks.append(f"Traceback:\n{longrepr}")

    blocks.append(f"Duration: {getattr(report, 'duration', 0.0):.3f}s")
    message = "\n".join(blocks)
    _write(message)


def pytest_sessionfinish(session, exitstatus):
    if LOG_PATH is None:
        return
    _write(f"=== SESSION END exitstatus={exitstatus} ===")