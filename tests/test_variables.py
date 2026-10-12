from datetime import timedelta

import numpy as np
import pytest

from conftest import T0
from wrf_fixture import hourly, write_wrfout
from modelatlasify.core.errors import FieldQualityError, VariableUnavailableError
from modelatlasify.core.variables import resolve_names
from modelatlasify.models.wrf import WRFAdapter
from modelatlasify.variables.precipitation import interval_precipitation
from modelatlasify.variables.wind import destagger, wind_speed


# ------------------------------------------------------------------ wind
def test_wind_speed_formula():
    np.testing.assert_allclose(wind_speed(np.array([3.0, -6.0, 0.0]), np.array([4.0, 8.0, 0.0])),
                               [5.0, 10.0, 0.0])


def test_destagger_averages_faces():
    u_stag = np.array([[0.0, 2.0, 6.0], [1.0, 1.0, 1.0]])  # (ny=2, nx_stag=3)
    np.testing.assert_allclose(destagger(u_stag, axis=1), [[1.0, 4.0], [1.0, 1.0]])
    v_stag = np.array([[0.0, 0.0], [2.0, 4.0], [4.0, 8.0]])  # (ny_stag=3, nx=2)
    np.testing.assert_allclose(destagger(v_stag, axis=0), [[1.0, 2.0], [3.0, 6.0]])


def test_adapter_wind_from_u10_v10(wrf_run):
    adapter = WRFAdapter()
    (ds,) = adapter.discover(wrf_run)
    f = adapter.compute(ds, "surface_wind_speed", 1)
    assert f.data.shape == ds.grid.shape
    nx = ds.grid.width
    np.testing.assert_allclose(f.data, np.broadcast_to(5.0 * (1 + np.arange(nx) / nx), f.data.shape),
                               rtol=1e-6)
    assert f.provenance["source_variables"] == ["U10", "V10"]


def test_adapter_wind_falls_back_to_destaggered_lowest_level(tmp_path):
    write_wrfout(tmp_path / "r", hourly(T0, 2), include=("U", "V", "RAINC", "RAINNC"), nx=6, ny=5)
    adapter = WRFAdapter()
    (ds,) = adapter.discover(tmp_path / "r")
    f = adapter.compute(ds, "surface_wind_speed", 0)
    # U_stag = i (0..6) -> U = i + 0.5 ; V_stag = j (0..5) -> V = j + 0.5 (south-up)
    u = np.arange(6) + 0.5
    v = (np.arange(5) + 0.5)[::-1]  # output is north-up
    np.testing.assert_allclose(f.data, np.hypot(u[None, :], v[:, None]))
    assert "NOT a 10 m wind" in f.provenance["method"]


def test_missing_wind_variables_give_clear_error(tmp_path):
    write_wrfout(tmp_path / "r", hourly(T0, 2), include=("RAINC", "RAINNC"))
    adapter = WRFAdapter()
    (ds,) = adapter.discover(tmp_path / "r")
    with pytest.raises(VariableUnavailableError, match="neither U10/V10 nor U/V"):
        adapter.compute(ds, "surface_wind_speed", 0)


# --------------------------------------------------------- precipitation
def test_accumulated_precipitation(wrf_run):
    adapter = WRFAdapter()
    (ds,) = adapter.discover(wrf_run)
    f = adapter.compute(ds, "precipitation", 2)  # 2 h after start
    nx = ds.grid.width
    expected = 0.5 * 2 + 1.0 * 2 * (1 + np.arange(nx) / nx)  # RAINC + RAINNC
    np.testing.assert_allclose(f.data, np.broadcast_to(expected, f.data.shape), rtol=1e-6)
    assert f.start_time == T0 and f.end_time == T0 + timedelta(hours=2)
    assert f.provenance["source_variables"] == ["RAINC", "RAINNC"]


def test_interval_precipitation(wrf_run):
    adapter = WRFAdapter()
    (ds,) = adapter.discover(wrf_run)
    assert adapter.compute(ds, "precipitation_interval", 0) is None  # no previous output
    f = adapter.compute(ds, "precipitation_interval", 3)
    nx = ds.grid.width
    expected = 0.5 + 1.0 * (1 + np.arange(nx) / nx)  # one hour of RAINC + RAINNC
    np.testing.assert_allclose(f.data, np.broadcast_to(expected, f.data.shape), rtol=1e-5)
    assert f.start_time == T0 + timedelta(hours=2)
    assert f.end_time == T0 + timedelta(hours=3)


def test_interval_precipitation_skipped_across_gap(tmp_path):
    times = hourly(T0, 5)
    write_wrfout(tmp_path / "r", times)
    (tmp_path / "r" / f"wrfout_d01_{times[2]:%Y-%m-%d_%H:%M:%S}").unlink()  # missing output
    adapter = WRFAdapter()
    (ds,) = adapter.discover(tmp_path / "r")
    assert ds.valid_times == [times[0], times[1], times[3], times[4]]
    assert adapter.compute(ds, "precipitation_interval", 2) is None  # 01 -> 03 spans a gap
    assert adapter.compute(ds, "precipitation_interval", 3) is not None


def test_bucketed_precipitation_is_reconstructed(tmp_path):
    write_wrfout(tmp_path / "r", hourly(T0, 4), bucket_mm=1.0,
                 include=("U10", "V10", "RAINC", "RAINNC", "I_RAINC", "I_RAINNC"))
    adapter = WRFAdapter()
    (ds,) = adapter.discover(tmp_path / "r")
    f = adapter.compute(ds, "precipitation", 3)
    nx = ds.grid.width
    expected = 0.5 * 3 + 3 * (1 + np.arange(nx) / nx)
    np.testing.assert_allclose(f.data, np.broadcast_to(expected, f.data.shape), rtol=1e-5)
    assert "I_RAINNC" in f.provenance["source_variables"]


def test_rain_units_are_checked(tmp_path):
    write_wrfout(tmp_path / "r", hourly(T0, 2), rain_units="kg m-2 s-1")
    adapter = WRFAdapter()
    (ds,) = adapter.discover(tmp_path / "r")
    with pytest.raises(VariableUnavailableError, match="expected 'mm'"):
        adapter.compute(ds, "precipitation", 0)


def test_missing_rain_variable(tmp_path):
    write_wrfout(tmp_path / "r", hourly(T0, 2), include=("U10", "V10", "RAINNC"))
    adapter = WRFAdapter()
    (ds,) = adapter.discover(tmp_path / "r")
    with pytest.raises(VariableUnavailableError, match="missing RAINC"):
        adapter.compute(ds, "precipitation", 0)


def test_decreasing_accumulation_is_rejected(tmp_path):
    import netCDF4

    times = hourly(T0, 3)
    paths = write_wrfout(tmp_path / "r", times)
    with netCDF4.Dataset(paths[2], "a") as nc:  # zero-fill the last time, like a damaged file
        nc["RAINNC"][0] = 0.0
        nc["RAINC"][0] = 0.0
    adapter = WRFAdapter()
    (ds,) = adapter.discover(tmp_path / "r")
    with pytest.raises(FieldQualityError, match="lower than at"):
        adapter.compute(ds, "precipitation", 2)
    with pytest.raises(FieldQualityError):
        adapter.compute(ds, "precipitation_interval", 2)


def test_interval_helper_clips_roundoff_and_masks_real_decreases():
    out, n_bad = interval_precipitation(np.array([1.0, 0.995, 0.5]), np.array([0.5, 1.0, 1.0]))
    np.testing.assert_allclose(out[:2], [0.5, 0.0])
    assert np.isnan(out[2]) and n_bad == 1


def test_variable_aliases():
    assert resolve_names(["wind"]) == ["surface_wind_speed"]
    assert resolve_names(["precipitation"]) == ["precipitation", "precipitation_interval"]
    with pytest.raises(KeyError, match="Unknown variable"):
        resolve_names(["temperature"])
