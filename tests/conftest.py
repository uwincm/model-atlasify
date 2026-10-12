from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from wrf_fixture import hourly, write_wrfout  # noqa: E402

T0 = datetime(2026, 10, 6, 0, tzinfo=timezone.utc)


@pytest.fixture
def wrf_run(tmp_path) -> Path:
    """A 3-hour, single-domain Lambert-conformal run, one time per file."""
    run = tmp_path / "run01"
    write_wrfout(run, hourly(T0, 4))
    (run / "namelist.input").write_text("&time_control\n/\n")  # non-WRF-output file
    return run


def _network_available() -> bool:
    import urllib.request

    try:
        urllib.request.urlopen("https://schemas.stacspec.org/v1.1.0/item-spec/json-schema/item.json", timeout=5)
        return True
    except Exception:  # noqa: BLE001
        return False


@pytest.fixture(scope="session")
def network() -> None:
    if not _network_available():
        pytest.skip("STAC schemas cannot be fetched (offline)")
