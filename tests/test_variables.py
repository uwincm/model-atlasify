from datetime import timedelta

import numpy as np
import pytest

from conftest import T0
from wrf_fixture import SPRAY_FIELDS, hourly, write_uwin_nml, write_wrfout
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


def test_interval_precipitation_from_prec_acc(tmp_path):
    write_wrfout(tmp_path / "r", hourly(T0, 4), prec_acc_dt=60.0,
                 include=("U10", "V10"))  # no RAINC/RAINNC: PREC_ACC alone suffices
    adapter = WRFAdapter()
    (ds,) = adapter.discover(tmp_path / "r")
    adapter.prepare(ds, ["precipitation_interval"])  # no accumulation check needed
    assert adapter.compute(ds, "precipitation_interval", 0) is None
    f = adapter.compute(ds, "precipitation_interval", 2)
    nx = ds.grid.width
    expected = 0.5 + 1.0 * (1 + np.arange(nx) / nx)
    np.testing.assert_allclose(f.data, np.broadcast_to(expected, f.data.shape), rtol=1e-5)
    assert f.start_time == T0 + timedelta(hours=1)
    assert f.end_time == T0 + timedelta(hours=2)
    assert f.provenance["source_variables"] == ["PREC_ACC_C", "PREC_ACC_NC"]


def test_prec_acc_is_scaled_to_hourly(tmp_path):
    times = hourly(T0, 5, step_hours=0.25)
    write_wrfout(tmp_path / "r", times, prec_acc_dt=15.0)
    adapter = WRFAdapter()
    (ds,) = adapter.discover(tmp_path / "r")
    f = adapter.compute(ds, "precipitation_interval", 3)
    nx = ds.grid.width
    expected = 0.5 + 1.0 * (1 + np.arange(nx) / nx)  # 15 min of rain * 4
    np.testing.assert_allclose(f.data, np.broadcast_to(expected, f.data.shape), rtol=1e-5)
    assert f.start_time == times[3] - timedelta(minutes=15)
    assert f.provenance["hourly_equivalent_scale"] == 4.0


def test_prec_acc_off_cycle_times_are_skipped(tmp_path):
    times = hourly(T0, 5, step_hours=0.5)
    write_wrfout(tmp_path / "r", times, prec_acc_dt=60.0)
    adapter = WRFAdapter()
    (ds,) = adapter.discover(tmp_path / "r")
    assert adapter.compute(ds, "precipitation_interval", 1) is None  # 00:30, partial window
    assert adapter.compute(ds, "precipitation_interval", 2) is not None  # 01:00


def test_prec_acc_disabled_falls_back_to_differences(tmp_path):
    write_wrfout(tmp_path / "r", hourly(T0, 3), prec_acc_dt=0.0)
    adapter = WRFAdapter()
    (ds,) = adapter.discover(tmp_path / "r")
    f = adapter.compute(ds, "precipitation_interval", 2)
    assert f.provenance["source_variables"] == ["RAINC", "RAINNC"]


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


def test_prepare_precomputes_accumulation_checks(tmp_path):
    import netCDF4

    times = hourly(T0, 4)
    paths = write_wrfout(tmp_path / "r", times)
    with netCDF4.Dataset(paths[2], "a") as nc:
        nc["RAINNC"][0] = 0.0
        nc["RAINC"][0] = 0.0
    adapter = WRFAdapter()
    (ds,) = adapter.discover(tmp_path / "r")
    adapter.prepare(ds, ["precipitation"])
    problems = ds.adapter_state.accumulation_problems
    assert [p is None for p in problems] == [True, True, False, True]  # t=3 compared with t=1
    fresh = WRFAdapter()  # like a worker process: relies on the precomputed checks
    with pytest.raises(FieldQualityError, match="lower than at"):
        fresh.compute(ds, "precipitation", 2)
    assert fresh.compute(ds, "precipitation", 3) is not None
    with pytest.raises(FieldQualityError, match="previous output time is unusable"):
        fresh.compute(ds, "precipitation_interval", 3)


def test_interval_helper_clips_roundoff_and_masks_real_decreases():
    out, n_bad = interval_precipitation(np.array([1.0, 0.995, 0.5]), np.array([0.5, 1.0, 1.0]))
    np.testing.assert_allclose(out[:2], [0.5, 0.0])
    assert np.isnan(out[2]) and n_bad == 1


def test_variable_aliases():
    assert resolve_names(["wind"]) == ["surface_wind_speed", "u10", "v10"]
    assert resolve_names(["precipitation"]) == ["precipitation", "precipitation_interval"]
    assert resolve_names(["spray"]) == ["spray_mass_flux", "spray_latent_heat_flux",
                                        "spray_sensible_heat_flux"]
    assert resolve_names(["mspr", "hlspr", "hsspr"]) == resolve_names(["spray"])
    with pytest.raises(KeyError, match="Unknown variable"):
        resolve_names(["humidity"])


SPRAY_INCLUDE = ("U10", "V10", "U", "V", "RAINC", "RAINNC", "OLR", "T2", *SPRAY_FIELDS)


@pytest.mark.parametrize("variable,scale", [("spray_mass_flux", 1e-5),
                                            ("spray_latent_heat_flux", 2.0),
                                            ("spray_sensible_heat_flux", 0.1)])
def test_spray_fields_when_spray_is_on(tmp_path, variable, scale):
    run = tmp_path / "r"
    write_wrfout(run, hourly(T0, 3), include=SPRAY_INCLUDE, levels=True)
    write_uwin_nml(run, spray=True)
    adapter = WRFAdapter()
    (ds,) = adapter.discover(run)
    assert adapter.unavailable_variables(ds) == {}
    assert variable in adapter.dataset_variables(ds)
    f = adapter.compute(ds, variable, 2)
    nx = ds.grid.width
    np.testing.assert_allclose(
        f.data, np.broadcast_to(scale * 2.0 * (1 + np.arange(nx) / nx), f.data.shape), rtol=1e-5)


@pytest.mark.parametrize("spray_nml", [False, None])  # .F., or no uwin.nml at all
def test_spray_fields_unavailable_when_spray_is_off(tmp_path, spray_nml):
    run = tmp_path / "r"
    write_wrfout(run, hourly(T0, 2), include=SPRAY_INCLUDE, levels=True)  # WRF writes them anyway (zeros)
    if spray_nml is not None:
        write_uwin_nml(run, spray=spray_nml)
    adapter = WRFAdapter()
    (ds,) = adapter.discover(run)
    unavailable = adapter.unavailable_variables(ds)
    assert set(unavailable) == {"spray_mass_flux", "spray_latent_heat_flux",
                                "spray_sensible_heat_flux"}
    assert all("sea spray was not on" in r for r in unavailable.values())
    assert "surface_wind_speed" in adapter.dataset_variables(ds)
    assert "spray_mass_flux" not in adapter.dataset_variables(ds)
    with pytest.raises(VariableUnavailableError, match="sea spray was not on"):
        adapter.compute(ds, "spray_mass_flux", 1)


def test_spray_units_are_checked(tmp_path):
    run = tmp_path / "r"
    write_wrfout(run, hourly(T0, 2), include=SPRAY_INCLUDE, spray_units={"MSPR": "g m-2 s-1"})
    write_uwin_nml(run, spray=True)
    adapter = WRFAdapter()
    (ds,) = adapter.discover(run)
    with pytest.raises(VariableUnavailableError, match="expected 'kg m-2 s-1'"):
        adapter.compute(ds, "spray_mass_flux", 1)


def test_spray_mass_flux_uses_log_spaced_levels():
    from modelatlasify.core.variables import get_variable

    levels = get_variable("spray_mass_flux").levels
    assert levels[0] == 1e-7 and levels[-1] == 1e-2
    ratios = np.array(levels[1:]) / np.array(levels[:-1])
    np.testing.assert_allclose(ratios, np.tile([2.0, 2.5, 2.0], 5))


# ------------------------------------------------------------------- OLR
def test_olr(tmp_path):
    run = tmp_path / "r"
    write_wrfout(run, hourly(T0, 2), include=("RAINC", "RAINNC", "OLR"))
    adapter = WRFAdapter()
    (ds,) = adapter.discover(run)
    assert {"olr", "olr_precipitation"} <= set(adapter.dataset_variables(ds))
    f = adapter.compute(ds, "olr", 1)
    nx = ds.grid.width
    np.testing.assert_allclose(
        f.data, np.broadcast_to(100.0 + 150.0 * np.arange(nx) / nx, f.data.shape), rtol=1e-6)
    assert f.provenance["source_variables"] == ["OLR"]


def test_olr_unavailable_without_olr(wrf_run):
    adapter = WRFAdapter()
    (ds,) = adapter.discover(wrf_run)
    unavailable = adapter.unavailable_variables(ds)
    assert "no OLR" in unavailable["olr"] and "no OLR" in unavailable["olr_precipitation"]
    assert not {"olr", "olr_precipitation"} & set(adapter.dataset_variables(ds))


def test_olr_units_are_checked(tmp_path):
    run = tmp_path / "r"
    write_wrfout(run, hourly(T0, 2), include=("OLR",), olr_units="erg cm-2 s-1")
    adapter = WRFAdapter()
    (ds,) = adapter.discover(run)
    with pytest.raises(VariableUnavailableError, match="expected 'W m-2'"):
        adapter.compute(ds, "olr", 0)


def test_olr_alias_and_overlay():
    from modelatlasify.core.variables import get_variable

    assert resolve_names(["olr"]) == ["olr", "olr_precipitation"]
    assert "olr_precipitation" not in resolve_names(["precipitation"])
    olr_map = get_variable("olr_precipitation")
    assert (olr_map.draws, olr_map.overlay) == ("olr", "precipitation_interval")
    assert not get_variable("olr").maps


def test_olr_zeros_are_missing_and_mostly_zero_is_rejected(tmp_path):
    import netCDF4

    run = tmp_path / "r"
    paths = write_wrfout(run, hourly(T0, 2), include=("OLR",))
    with netCDF4.Dataset(paths[0], "a") as nc:  # initialization: radiation not run yet
        nc["OLR"][0] = 0.0
    with netCDF4.Dataset(paths[1], "a") as nc:  # a moving nest's new edge columns
        nc["OLR"][0, :, -1] = 0.0
    adapter = WRFAdapter()
    (ds,) = adapter.discover(run)
    with pytest.raises(FieldQualityError, match="exactly 0"):
        adapter.compute(ds, "olr", 0)
    f = adapter.compute(ds, "olr", 1)
    assert np.isnan(f.data[:, -1]).all() and np.isfinite(f.data[:, :-1]).all()
    assert f.provenance["n_zero_set_missing"] == ds.grid.height


# ------------------------------------------------------- u10 / v10 vectors
@pytest.mark.parametrize("map_proj", [1, 2, 3])  # Lambert, polar stereographic, Mercator
def test_grid_rotation_matches_meridian_convergence(tmp_path, map_proj):
    from pyproj import Proj

    from modelatlasify.variables.wind import grid_rotation

    write_wrfout(tmp_path / "r", hourly(T0, 1), map_proj=map_proj)
    (ds,) = WRFAdapter().discover(tmp_path / "r")
    g = ds.grid
    convergence = np.radians(Proj(g.crs).get_factors(g.lon, g.lat).meridian_convergence)
    np.testing.assert_allclose(grid_rotation(g), -convergence, atol=1e-4)


@pytest.mark.parametrize("map_proj", [1, 3])
def test_u10_v10_are_earth_relative(tmp_path, map_proj):
    from modelatlasify.variables.wind import grid_rotation

    write_wrfout(tmp_path / "r", hourly(T0, 2), map_proj=map_proj)
    adapter = WRFAdapter()
    (ds,) = adapter.discover(tmp_path / "r")
    assert {"u10", "v10"} <= set(adapter.dataset_variables(ds))
    u = adapter.compute(ds, "u10", 1).data
    v = adapter.compute(ds, "v10", 1).data
    nx = ds.grid.width
    ug = np.broadcast_to(3.0 * (1 + np.arange(nx) / nx), u.shape)  # fixture U10, V10
    vg = np.broadcast_to(4.0 * (1 + np.arange(nx) / nx), u.shape)
    np.testing.assert_allclose(np.hypot(u, v), np.hypot(ug, vg), rtol=1e-6)  # speed kept
    theta = grid_rotation(ds.grid)
    np.testing.assert_allclose(u, ug * np.cos(theta) - vg * np.sin(theta), rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(v, ug * np.sin(theta) + vg * np.cos(theta), rtol=1e-6, atol=1e-9)
    if map_proj == 3:  # Mercator: grid-relative is already earth-relative
        np.testing.assert_allclose(u, ug, rtol=1e-6)
    else:
        assert np.abs(theta).max() > 0.01  # the Lambert test actually rotates


def test_u10_v10_unavailable_without_u10(tmp_path):
    write_wrfout(tmp_path / "r", hourly(T0, 1), include=("U", "V", "RAINC", "RAINNC"))
    adapter = WRFAdapter()
    (ds,) = adapter.discover(tmp_path / "r")
    unavailable = adapter.unavailable_variables(ds)
    assert "no U10/V10" in unavailable["u10"] and "no U10/V10" in unavailable["v10"]
    assert "surface_wind_speed" in adapter.dataset_variables(ds)  # falls back to U/V


# ------------------------------------------------------------------- t2m
def test_t2m_in_celsius(tmp_path):
    write_wrfout(tmp_path / "r", hourly(T0, 2), include=("U10", "V10", "T2"))
    adapter = WRFAdapter()
    (ds,) = adapter.discover(tmp_path / "r")
    assert "t2m" in adapter.dataset_variables(ds)
    f = adapter.compute(ds, "t2m", 1)
    nx = ds.grid.width
    np.testing.assert_allclose(
        f.data, np.broadcast_to(-10.0 + 50.0 * np.arange(nx) / nx, f.data.shape), atol=1e-4)
    assert f.provenance["source_variables"] == ["T2"]


def test_t2m_unavailable_without_t2(wrf_run):
    adapter = WRFAdapter()
    (ds,) = adapter.discover(wrf_run)
    assert "no T2" in adapter.unavailable_variables(ds)["t2m"]


def test_t2m_zero_filled_is_rejected(tmp_path):
    import netCDF4

    paths = write_wrfout(tmp_path / "r", hourly(T0, 1), include=("T2",))
    with netCDF4.Dataset(paths[0], "a") as nc:
        nc["T2"][0, :5, :] = 0.0
    adapter = WRFAdapter()
    (ds,) = adapter.discover(tmp_path / "r")
    with pytest.raises(FieldQualityError, match="0 K"):
        adapter.compute(ds, "t2m", 0)


def test_shifted_levels_fit_the_data_in_steps():
    from modelatlasify.core.variables import shifted_levels
    from modelatlasify.variables.temperature import T2M_LEVELS

    def shift(data):
        out = shifted_levels(T2M_LEVELS, 5.0, np.asarray(data, dtype=float))
        assert np.allclose(np.diff(out), 1.5) and len(out) == len(T2M_LEVELS)  # same scale
        return out[0], out[-1]

    assert shift(np.linspace(-5.0, 30.0, 100)) == (-10.0, 35.0)  # fits already: unchanged
    # Fits at -30..15, -35..10 and -40..5: the smallest shift wins.
    assert shift(np.linspace(-30.0, -20.0, 100)) == (-30.0, 15.0)
    # 90 values at 30-40 C and 10 at -10 C cannot all fit: -5..40 and 0..45
    # leave out only the 10 (fewer than -10..35, which leaves out most of the
    # warm ones); -5..40 is the smaller shift.
    assert shift(np.r_[np.linspace(30.0, 40.0, 90), np.full(10, -10.0)]) == (-5.0, 40.0)
    assert shift(np.r_[np.linspace(36.0, 44.0, 50), np.nan]) == (0.0, 45.0)  # NaN ignored
    assert shift(np.full(4, np.nan)) == (-10.0, 35.0)  # nothing to fit


def test_t2m_maps_shift_the_colour_scale_to_fit_d01_at_the_start(tmp_path, monkeypatch, capsys):
    import netCDF4

    from modelatlasify.cli import main
    from modelatlasify.plotting import maps

    run = tmp_path / "cold"
    include = ("U10", "V10", "T2")
    d01 = write_wrfout(run, hourly(T0, 3), include=include)
    write_wrfout(run, hourly(T0, 3), include=include, domain=2, nx=20, ny=16, dx=4000.0)
    with netCDF4.Dataset(d01[0], "a") as nc:  # first time zero-filled: fails the checks
        nc["T2"][0] = 0.0
    with netCDF4.Dataset(d01[1], "a") as nc:  # so the next one sets the scale: -28 to -20 C
        nx = nc["T2"].shape[-1]
        nc["T2"][0] = 245.15 + 8.0 * np.arange(nx) / nx
    drawn = []
    real = maps._draw_map
    monkeypatch.setattr(maps, "_draw_map", lambda path, view, layers, variable, *a, **k: (
        drawn.append((path.name, variable.levels)), real(path, view, layers, variable, *a, **k)
    )[1])
    assert main([str(run), "-o", str(tmp_path / "atlas"), "--no-validate",
                 "--variables", "t2m"]) == 0
    assert (f"cold': -30 to 15 degC (fits d01 at {T0 + timedelta(hours=1):%Y-%m-%d %H:%M} UTC)"
            in capsys.readouterr().err)
    t2m = [lv for name, lv in drawn if name.startswith("t2m") or ".t2m." in name]
    # Per-item maps (d01 at its two good times, d02 at all three) and composites
    # (over d01: its two good times).
    assert len(t2m) == 2 + 3 + 2
    assert all(lv[0] == -30.0 and lv[-1] == 15.0 for lv in t2m)
    from modelatlasify.core.variables import get_variable

    assert get_variable("t2m").levels[0] == -10.0  # the definition itself is unchanged


def test_t2m_sub_area_maps_fit_their_own_area(tmp_path, monkeypatch, capsys):
    from modelatlasify.cli import main
    from modelatlasify.plotting import maps

    run = tmp_path / "run"
    write_wrfout(run, hourly(T0, 1), include=("U10", "V10", "T2"))  # T2: -10 C (W) to 40 C (E)
    drawn = []
    real = maps._draw_map
    monkeypatch.setattr(maps, "_draw_map", lambda path, view, layers, variable, *a, **k: (
        drawn.append((path.name, variable.levels[0], variable.levels[-1])),
        real(path, view, layers, variable, *a, **k))[1])
    # d01 is centred on 40N, 100W, about 5.6 degrees of longitude wide: "east"
    # holds its eastern quarter, about 29 to 39 C.
    assert main([str(run), "-o", str(tmp_path / "atlas"), "--no-validate", "--variables", "t2m",
                 "--plot-sub-areas=east,-98.5,-96,35,45,away,10,20,40,50"]) == 0
    assert "-10 to 35, area 'east' -5 to 40 degC" in capsys.readouterr().err
    ranges = {name.split(".")[1] if name.startswith("uwincm.") else "item": (lo, hi)
              for name, lo, hi in drawn}
    # The whole domain (composite and per-item map) keeps -10..35 (as good as
    # -5..40, and the smaller shift); "east" shifts to fit its warm points.
    assert ranges == {"item": (-10.0, 35.0), "full": (-10.0, 35.0), "east": (-5.0, 40.0)}


def test_t2m_levels_and_colours():
    import matplotlib.colors as mcolors

    from modelatlasify.core.variables import get_variable
    from modelatlasify.plotting.maps import _norm_and_cmap

    var = get_variable("t2m")
    assert var.levels[0] == -10.0 and var.levels[-1] == 35.0
    assert np.allclose(np.diff(var.levels), 1.5) and len(var.levels) == 31
    assert var.vectors == ("u10", "v10")
    assert resolve_names(["temperature"]) == ["t2m"]
    norm, cmap = _norm_and_cmap(var, np.array([0.0]))
    first, last = (np.array(c) / 255 for c in ((109, 227, 255), (163, 112, 255)))
    rgba = lambda x: np.array(cmap(norm(np.array([x])))[0][:3])  # noqa: E731
    assert np.allclose(rgba(-9.0), first) and np.allclose(rgba(34.0), last)
    assert np.allclose(rgba(-30.0), first) and np.allclose(rgba(45.0), last)  # extends
    assert np.allclose(rgba(0.1), np.array((0, 0, 128)) / 255)  # -1 to 0.5 C: navy
    assert np.allclose(rgba(5.5), np.array((255, 255, 255)) / 255)  # 5 to 6.5 C: white
    assert isinstance(norm, mcolors.BoundaryNorm)
