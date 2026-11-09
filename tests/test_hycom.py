"""HYCOM ocean model: a/b reading, discovery, fields, and alongside WRF."""

import json
from datetime import timedelta

import numpy as np
import pytest

from conftest import T0, run_out
from hycom_fixture import write_archive, write_grid, write_hycom
from wrf_fixture import hourly, write_wrfout
from modelatlasify.cli import main
from modelatlasify.core.errors import NoModelOutputError
from modelatlasify.models.hycom import HYCOMAdapter
from modelatlasify.models.hycom import hycom_io

NX, NY, LAND = 30, 20, 2


def _uwin(directory, ocean=True):
    (directory / "uwin.nml").write_text(
        "&MAIN\n  atmosphere = .T.\n  ocean = %s\n/\n&TIME\n"
        "  startDate = 2026, 10, 06, 00, 00, 00\n/\n" % (".T." if ocean else ".F."))


def _dataset(directory):
    adapter = HYCOMAdapter()
    (ds,) = adapter.discover(directory)
    return adapter, ds


def test_ab_reading(tmp_path):
    a = write_archive(tmp_path, T0 + timedelta(hours=1), nx=NX, ny=NY)
    h = hycom_io.read_header(a.with_suffix(".b"))
    assert (h.idm, h.jdm, h.yrflag) == (NX, NY, 3)
    assert h.index("temp", 1) == 7 and h.index("temp", 2) == 12 and h.index("u_btrop") == 2
    temp = hycom_io.read_record(a, h, h.index("temp", 1))
    assert temp.shape == (NY, NX) and np.isnan(temp[:, :LAND]).all()
    np.testing.assert_allclose(temp[0, LAND:], 20 + 10 * np.arange(LAND, NX) / NX, rtol=1e-6)
    assert hycom_io.model_day_to_datetime(h.records[0].model_day, 3) - (
        T0 + timedelta(hours=1)) < timedelta(minutes=1)
    assert hycom_io.filename_time("archv.2026_279_01.a") == T0 + timedelta(hours=1)


def test_discovery(tmp_path):
    write_hycom(tmp_path, hourly(T0 + timedelta(hours=1), 3), nx=NX, ny=NY)
    _uwin(tmp_path)
    adapter, ds = _dataset(tmp_path)
    assert (ds.model, ds.dataset_id) == ("HYCOM", "HYCOM")
    assert ds.valid_times == hourly(T0 + timedelta(hours=1), 3)
    assert ds.reference_time == T0  # from uwin.nml startDate
    assert (ds.grid.width, ds.grid.height) == (NX, NY)
    assert ds.metadata["hycom:georeference"] == "regular grid in Mercator"
    assert ds.grid.max_georeference_error_m < 1.0
    assert set(adapter.dataset_variables(ds)) == {
        "sst", "surface_salinity", "surface_currents", "surface_u", "surface_v"}


def test_archs_preferred_over_archv(tmp_path):
    write_grid(tmp_path, NX, NY)
    t = T0 + timedelta(hours=1)
    write_archive(tmp_path, t, "archv", nx=NX, ny=NY)
    write_archive(tmp_path, t, "archs", nx=NX, ny=NY)
    write_archive(tmp_path, t + timedelta(hours=1), "archv", nx=NX, ny=NY)  # archv only
    _, ds = _dataset(tmp_path)
    assert [p.name for p in ds.time_sources] == ["archs.2026_279_01.a", "archv.2026_279_02.a"]


def test_ocean_off_in_uwin_nml_means_no_hycom(tmp_path):
    write_hycom(tmp_path, hourly(T0, 1), nx=NX, ny=NY)
    _uwin(tmp_path, ocean=False)
    with pytest.raises(NoModelOutputError, match="ocean = .F."):
        HYCOMAdapter().discover(tmp_path)


def test_inconsistent_model_day_is_skipped(tmp_path):
    write_grid(tmp_path, NX, NY)
    write_archive(tmp_path, T0, nx=NX, ny=NY)
    bad = write_archive(tmp_path, T0 + timedelta(hours=1), nx=NX, ny=NY,
                        day_error=timedelta(hours=3))
    adapter, ds = _dataset(tmp_path)
    assert ds.valid_times == [T0]
    assert adapter.skipped_files[0][0] == bad and "inconsistent" in adapter.skipped_files[0][1]


def test_sst_and_salinity_north_up_with_land_missing(tmp_path):
    write_hycom(tmp_path, [T0], nx=NX, ny=NY)
    adapter, ds = _dataset(tmp_path)
    sst = adapter.compute(ds, "sst", 0).data
    sss = adapter.compute(ds, "surface_salinity", 0).data
    assert np.isnan(sst[:, :LAND]).all() and np.isfinite(sst[:, LAND:]).all()
    np.testing.assert_allclose(sst[5, LAND:], 20 + 10 * np.arange(LAND, NX) / NX, rtol=1e-6)
    # North-up: row 0 is the northernmost model row (j = NY - 1).
    np.testing.assert_allclose(sss[0, LAND], 30 + 5 * (NY - 1) / NY, rtol=1e-6)
    np.testing.assert_allclose(sss[-1, LAND], 30.0, rtol=1e-6)


def test_currents_are_total_and_at_cell_centres(tmp_path):
    write_hycom(tmp_path, [T0], nx=NX, ny=NY)
    adapter, ds = _dataset(tmp_path)
    u = adapter.compute(ds, "surface_u", 0).data[::-1]  # back to row 0 = south
    v = adapter.compute(ds, "surface_v", 0).data[::-1]
    speed = adapter.compute(ds, "surface_currents", 0).data[::-1]
    i = np.arange(NX)
    interior = slice(LAND + 1, NX - 1)
    # Baroclinic face values averaged to the centre, plus the barotropic part.
    np.testing.assert_allclose(u[3, interior], (0.1 * (i + 0.5) + 0.05)[interior], rtol=1e-6)
    np.testing.assert_allclose(v[:-1, 10], 0.02 * (np.arange(NY - 1) + 0.5) - 0.01, atol=1e-7)
    # The coastal cell's land-side face carries no flow (0), not a copy of the open face.
    np.testing.assert_allclose(u[3, LAND], 0.5 * (0.1 * (LAND + 1) + 0.05), rtol=1e-6)
    assert np.isnan(u[:, :LAND]).all()  # land
    assert np.isnan(u[:, -1]).all() and np.isnan(v[-1, LAND:]).all()  # no east/north face
    np.testing.assert_allclose(speed, np.hypot(u, v), equal_nan=True)


def test_rotated_grid_has_no_currents(tmp_path):
    write_hycom(tmp_path, [T0], nx=NX, ny=NY, pang=0.1)
    adapter, ds = _dataset(tmp_path)
    unavailable = adapter.unavailable_variables(ds)
    assert set(unavailable) == {"surface_u", "surface_v", "surface_currents"}
    assert "rotated" in ds.warnings[0]
    assert adapter.compute(ds, "sst", 0) is not None


def _record_maps(monkeypatch):
    from modelatlasify.plotting import maps

    saved = []
    real = maps._draw_map
    monkeypatch.setattr(maps, "_draw_map", lambda path, *a, **k: (
        saved.append((str(path), bool(k.get("vectors")),
                      getattr(k.get("vector_style"), "units", None))), real(path, *a, **k))[1])
    return saved


def test_coupled_run_with_wrf_and_ocean(tmp_path, monkeypatch):
    run = tmp_path / "coupled"
    write_wrfout(run, hourly(T0, 3))
    write_hycom(run, hourly(T0 + timedelta(hours=1), 2), nx=NX, ny=NY)
    _uwin(run)
    saved = _record_maps(monkeypatch)
    out = tmp_path / "atlas"
    assert main([str(run), "-o", str(out), "--no-validate", "--variables", "wind,ocean"]) == 0
    run_dir = run_out(out, run)
    item = run_dir / "items" / "HYCOM-20261006T010000Z"
    assert {p.name for p in item.iterdir() if p.suffix in (".tif", ".png")} == {
        "sst.tif", "sst.png", "surface_salinity.tif", "surface_salinity.png",
        "surface_currents.tif", "surface_currents.png", "surface_u.tif", "surface_v.tif"}
    assets = json.loads((item / f"{item.name}.json").read_text())["assets"]
    assert {"sst", "surface_salinity", "surface_u", "surface_v"} <= set(assets)
    for name in ("sst", "surface_salinity", "surface_currents"):
        ocean_maps = [(v, u) for p, v, u in saved if f"/{name}" in p or f".{name}." in p]
        assert ocean_maps and all(v and u == "m/s" for v, u in ocean_maps)  # current vectors
    wind_maps = [(v, u) for p, v, u in saved if "surface_wind_speed" in p]
    assert wind_maps and all(v and u is None for v, u in wind_maps)  # wind style (default)
    images = run_dir / "model-images" / "full"
    assert len(list((images / "sst").glob("*.png"))) == 2
    collection = json.loads((run_dir / "collection.json").read_text())
    assert {d["model"] for d in collection["modelatlasify:domains"]} == {"WRF", "HYCOM"}


def test_current_vector_style():
    from modelatlasify.core.variables import get_variable
    from modelatlasify.variables.ocean import CURRENT_VECTORS
    from modelatlasify.variables.wind import WIND_VECTORS

    assert CURRENT_VECTORS.spacing_km == WIND_VECTORS.spacing_km / 2  # twice as dense
    assert CURRENT_VECTORS.min_across == 2 * WIND_VECTORS.min_across
    assert CURRENT_VECTORS.max_speed == 2.0 and CURRENT_VECTORS.units_per_m_s == 1.0
    for v in ("sst", "surface_salinity", "surface_currents"):
        var = get_variable(v)
        assert var.vectors == ("surface_u", "surface_v") and var.vector_style is CURRENT_VECTORS


def test_items_without_start_time_declare_no_forecast_extension(tmp_path):
    from modelatlasify.core.catalog import FORECAST_EXTENSION

    run = tmp_path / "ocean_only"
    write_hycom(run, [T0], nx=NX, ny=NY)  # no uwin.nml: start time unknown
    out = tmp_path / "atlas"
    assert main([str(run), "-o", str(out), "--no-validate", "--no-plots"]) == 0
    (item,) = run_out(out, run).glob("items/HYCOM-*/HYCOM-*.json")
    data = json.loads(item.read_text())
    assert FORECAST_EXTENSION not in data["stac_extensions"]
    assert "forecast:reference_datetime" not in data["properties"]
