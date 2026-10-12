from datetime import timedelta

import pytest

from conftest import T0
from wrf_fixture import hourly, write_wrfout
from modelatlasify.core.errors import DiscoveryError
from modelatlasify.models.wrf import WRFAdapter


def test_discovers_files_and_times(wrf_run):
    adapter = WRFAdapter()
    (ds,) = adapter.discover(wrf_run)
    assert ds.model == "WRF"
    assert ds.run_id == "run01"
    assert ds.dataset_id == "d01"
    assert len(ds.files) == 4
    assert ds.valid_times == hourly(T0, 4)
    assert ds.reference_time == T0
    assert all(t.tzinfo is not None for t in ds.valid_times)
    assert adapter.skipped_files == []  # namelist.input is simply not a candidate


def test_metadata_is_extracted_not_invented(wrf_run):
    (ds,) = WRFAdapter().discover(wrf_run)
    m = ds.metadata
    assert m["wrf:version"] == "4.5.1"
    assert m["wrf:map_proj"] == 1
    assert m["wrf:dx"] == 12000.0
    assert m["wrf:truelat1"] == 30.0
    assert m["wrf:simulation_start_date"] == "2026-10-06_00:00:00"
    assert m["wrf:mp_physics"] == 8
    assert m["wrf:output_interval_minutes"] == 60.0
    assert "wrf:ra_lw_physics" not in m  # absent from the file -> absent from metadata


def test_multiple_times_per_file_and_domains(tmp_path):
    run = tmp_path / "run"
    write_wrfout(run, hourly(T0, 6), times_per_file=3)
    write_wrfout(run, hourly(T0, 3), domain=2, nx=20, ny=16, dx=4000.0)
    datasets = WRFAdapter().discover(run)
    assert [d.dataset_id for d in datasets] == ["d01", "d02"]
    d01, d02 = datasets
    assert len(d01.files) == 2 and len(d01.valid_times) == 6
    assert d01.valid_times[3] == T0 + timedelta(hours=3)
    assert d02.grid.width == 20 and abs(d02.grid.transform.a - 4000.0) < 1e-6


def test_corrupt_and_inconsistent_files_are_skipped(tmp_path):
    run = tmp_path / "run"
    times = hourly(T0, 4)
    write_wrfout(run, times, corrupt_xtime=(times[2],))
    (run / f"wrfout_d01_{(T0 + timedelta(hours=4)):%Y-%m-%d_%H:%M:%S}").write_bytes(b"\0" * 1024)
    adapter = WRFAdapter()
    (ds,) = adapter.discover(run)
    assert ds.valid_times == [times[0], times[1], times[3]]
    reasons = {p.name: r for p, r in adapter.skipped_files}
    assert len(reasons) == 2
    assert any("XTIME" in r for r in reasons.values())
    # Unusable files still define the output cadence.
    assert WRFAdapter.nominal_output_interval(ds) == timedelta(hours=1)


def test_no_wrf_files(tmp_path):
    (tmp_path / "readme.txt").write_text("hi")
    with pytest.raises(DiscoveryError, match="No WRF output files"):
        WRFAdapter().discover(tmp_path)


def test_missing_directory(tmp_path):
    with pytest.raises(DiscoveryError, match="does not exist"):
        WRFAdapter().discover(tmp_path / "nope")


def test_only_unreadable_files(tmp_path):
    (tmp_path / "wrfout_d01_2026-10-06_00:00:00").write_bytes(b"garbage")
    with pytest.raises(DiscoveryError, match="none could be used"):
        WRFAdapter().discover(tmp_path)


def test_late_started_nest_accumulates_from_its_start(tmp_path):
    run = tmp_path / "run"
    write_wrfout(run, hourly(T0, 3))
    nest_start = T0 + timedelta(hours=1)
    write_wrfout(run, hourly(nest_start, 2), domain=2, sim_start=T0, start_date=nest_start)
    d01, d02 = WRFAdapter().discover(run)
    assert d01.metadata["wrf:accumulation_start"] == "2026-10-06T00:00:00Z"
    assert d02.metadata["wrf:accumulation_start"] == "2026-10-06T01:00:00Z"
    assert d02.reference_time == T0  # initialization time is still the simulation start
