"""Pytest hooks for writing test run logs to files."""

from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
TEST_DIR = ROOT_DIR / "scripts" / "tests"
TEST_LOG_DIR = ROOT_DIR / "logs"
LOG_PATH = TEST_LOG_DIR / "pytest_test_results.log"


def _write(text: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n\n")


def _per_test_log_path(nodeid: str) -> Path | None:
    test_path = nodeid.split("::", 1)[0]
    path = Path(test_path)
    if path.suffix != ".py":
        return None
    return TEST_LOG_DIR / f"{path.stem}.log"


def _write_per_test(nodeid: str, text: str) -> None:
    path = _per_test_log_path(nodeid)
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(
            f"Pytest log for {path.stem}\n{'=' * 60}\n\n",
            encoding="utf-8",
        )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n\n")


def pytest_sessionstart(session):
    if LOG_PATH.exists():
        LOG_PATH.unlink()
    TEST_LOG_DIR.mkdir(parents=True, exist_ok=True)
    for path in TEST_LOG_DIR.glob("test_*.log"):
        path.unlink()
    for test_file in sorted(TEST_DIR.glob("test_*.py")):
        log_path = TEST_LOG_DIR / f"{test_file.stem}.log"
        log_path.write_text(
            f"Pytest log for {test_file.name}\n{'=' * 60}\n\n",
            encoding="utf-8",
        )


def pytest_runtest_logreport(report):
    if report.when != "call":
        return

    blocks = [f"=== {report.nodeid} {report.outcome.upper()} ==="]
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
    _write_per_test(report.nodeid, message)