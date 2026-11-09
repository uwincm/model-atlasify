"""DWM wave model: discovery, georeferencing, fields, and alongside WRF."""

import json
from datetime import timedelta

import numpy as np
import pytest
import rasterio
from pyproj import Transformer

from conftest import T0, run_out
from dwm_fixture import mercator_lonlat, write_dwm
from wrf_fixture import PROJECTIONS, grid_lonlat, hourly, true_crs, write_wrfout
from modelatlasify.cli import main
from modelatlasify.core.errors import NoModelOutputError
from modelatlasify.models.dwm import DWMAdapter
from modelatlasify.models.dwm.adapter import read_namelist

NX, NY, DX = 40, 24, 50000.0


def test_discovery_times_grid_and_start(tmp_path):
    write_dwm(tmp_path, hourly(T0 + timedelta(hours=1), 3), start=T0)
    (ds,) = DWMAdapter().discover(tmp_path)
    assert ds.model == "DWM" and ds.dataset_id == "DWM"
    assert ds.valid_times == hourly(T0 + timedelta(hours=1), 3)  # from the file names
    assert ds.reference_time == T0  # from namelists/main.nml
    g = ds.grid
    assert (g.width, g.height) == (NX, NY)
    assert g.transform.a == pytest.approx(DX) and g.transform.e == pytest.approx(-DX)
    # Every cell centre of the fitted grid lands on the lon/lat in the file.
    lon, lat, _, _ = mercator_lonlat(NX, NY, DX, 180.0, 5.0)
    rows, cols = np.mgrid[0:NY, 0:NX]
    t = g.transform
    x, y = t.c + (cols + 0.5) * t.a, t.f + (rows + 0.5) * t.e
    glon, glat = Transformer.from_crs(g.crs, g.crs.geodetic_crs, always_xy=True).transform(x, y)
    np.testing.assert_allclose(np.asarray(glon) % 360.0, lon[::-1], atol=1e-6)
    np.testing.assert_allclose(glat, lat[::-1], atol=1e-6)
    assert g.max_georeference_error_m < 5.0
    assert ds.metadata["dwm:starttimestr"] == f"{T0:%Y-%m-%d %H:%M:%S}"


def test_fields_are_north_up_with_land_missing(tmp_path):
    write_dwm(tmp_path, hourly(T0, 2))
    adapter = DWMAdapter()
    (ds,) = adapter.discover(tmp_path)
    swh = adapter.compute(ds, "significant_wave_height", 1).data
    mwp = adapter.compute(ds, "mean_wave_period", 0).data
    i = np.arange(NX)
    np.testing.assert_allclose(swh[0, 5:], (1 + i / NX + 1.0)[5:], rtol=1e-6)
    np.testing.assert_allclose(mwp[:, 10], 5 + np.arange(NY)[::-1] / NY, rtol=1e-6)  # north-up
    assert np.isnan(swh[-5:, :5]).all() and not np.isnan(swh[:-5]).any()  # land: SW corner


def test_no_namelist_no_reference_time_and_files_without_nc_suffix(tmp_path):
    write_dwm(tmp_path, hourly(T0, 2), namelist=False, suffix="")
    (ds,) = DWMAdapter().discover(tmp_path)
    assert ds.reference_time is None and len(ds.valid_times) == 2


def test_no_dwm_output_is_a_specific_error(tmp_path):
    with pytest.raises(NoModelOutputError):
        DWMAdapter().discover(tmp_path)


def test_read_namelist(tmp_path):
    p = tmp_path / "main.nml"
    p.write_text("&DOMAIN\n isGlobal = .false. ! x\n mm = 12\n fmin = 0.0313\n"
                 " startTimeStr = '2023-03-01 00:00:00'\n/\n")
    assert read_namelist(p) == {"isGlobal": False, "mm": 12, "fmin": 0.0313,
                                "startTimeStr": "2023-03-01 00:00:00"}


def test_waves_only_run(tmp_path):
    run = tmp_path / "waverun"
    write_dwm(run, hourly(T0, 2))
    out = tmp_path / "atlas"
    assert main([str(run), "-o", str(out), "--no-validate"]) == 0
    run_dir = run_out(out, run)
    items = sorted((run_dir / "items").iterdir())
    assert [p.name for p in items] == ["DWM-20261006T000000Z", "DWM-20261006T010000Z"]
    assert {p.name for p in items[0].glob("*.tif")} == {
        "significant_wave_height.tif", "mean_wave_period.tif", "dominant_wave_period.tif"}
    with rasterio.open(items[0] / "significant_wave_height.tif") as src:
        assert "Mercator" in src.crs.to_wkt() and np.isnan(src.read(1)[-1, 0])
    full = run_dir / "model-images" / "full"
    assert sorted(p.name for p in full.iterdir() if p.is_dir()) == [
        "dominant_wave_period", "mean_wave_period", "significant_wave_height"]
    assert sorted(p.name for p in full.glob("*.html")) == [
        "dominant_wave_period.html", "mean_wave_period.html", "significant_wave_height.html"]
    assert len(list(full.rglob("*.png"))) == 6
    collection = json.loads((run_dir / "collection.json").read_text())
    assert collection["title"] == "DWM run waverun"


def test_coupled_run_with_wrf_and_waves(tmp_path, capsys):
    from modelatlasify.core.validation import validate_catalog

    run = tmp_path / "coupled"
    write_wrfout(run, hourly(T0, 2))
    write_dwm(run, hourly(T0, 2), cen_lon=-100.0, cen_lat=40.0)
    out = tmp_path / "atlas"
    assert main([str(run), "-o", str(out), "--no-validate", "--variables", "wind,waves",
                 "--plot-sub-areas=box,-101,-99,39,41"]) == 0
    assert "Found WRF + DWM run" in capsys.readouterr().err
    run_dir = run_out(out, run)
    assert {p.name.split("-")[0] for p in (run_dir / "items").iterdir()} == {"d01", "DWM"}
    assert not list(run_dir.glob("items/DWM-*/surface_wind_speed.tif"))  # each model its own
    for area in ("full", "box"):
        names = {p.name for p in (run_dir / "model-images" / area).iterdir() if p.is_dir()}
        assert names == {"surface_wind_speed", "significant_wave_height", "mean_wave_period",
                         "dominant_wave_period"}
    collection = json.loads((run_dir / "collection.json").read_text())
    assert collection["title"] == "WRF + DWM run coupled"
    assert {d["model"] for d in collection["modelatlasify:domains"]} == {"WRF", "DWM"}
    assert "wrf:version" in collection and "dwm:starttimestr" in collection
    assert validate_catalog(out / "catalog.json", schemas=False).ok

    # --model wrf ignores the waves; asking a model for what it cannot make is an error.
    out2 = tmp_path / "atlas2"
    assert main([str(run), "-o", str(out2), "--no-validate", "--no-plots",
                 "--model", "wrf"]) == 0
    assert not list(out2.rglob("DWM-*"))
    assert main([str(run), "-o", str(tmp_path / "a3"), "--no-plots", "--model", "wrf",
                 "--variables", "swh"]) == 1
    assert "No model found in the input can produce: significant_wave_height" in (
        capsys.readouterr().err)


def test_composite_outlines_the_wave_domain_dashed(tmp_path, monkeypatch):
    from modelatlasify.plotting import maps

    run = tmp_path / "w"
    write_dwm(run, hourly(T0, 1))
    seen = []
    real = maps._draw_map

    def spy(*args, **kwargs):
        seen.append(len(kwargs.get("dashed_outlines", ())))
        return real(*args, **kwargs)

    monkeypatch.setattr(maps, "_draw_map", spy)
    assert main([str(run), "-o", str(tmp_path / "atlas"), "--no-validate",
                 "--variables", "swh"]) == 0
    assert seen == [0, 1]  # per-item map: no edge; composite: the wave domain, dashed



# ------------------------------------------- grids that are not Mercator
LCC_DX = 50000.0


def lambert_lonlat(nx: int = NX, ny: int = NY):
    """South-up lon/lat of a Lambert conformal grid in the WRF fixture's
    projection: far from Mercator or lon/lat over 2000 x 1200 km at 40N."""
    lon, lat, _, _ = grid_lonlat(PROJECTIONS[1], nx, ny, LCC_DX, LCC_DX)
    return lon, lat


def test_lambert_waves_take_the_wrf_projection_of_the_run(tmp_path):
    run = tmp_path / "coupled"
    write_wrfout(run, hourly(T0, 2))  # Lambert conformal WRF, same projection
    write_dwm(run, hourly(T0, 2), lonlat=lambert_lonlat())
    adapter = DWMAdapter()
    (ds,) = adapter.discover(run)
    g = ds.grid
    assert g.crs.equals(true_crs(PROJECTIONS[1]), ignore_axis_order=True)
    assert g.transform.a == pytest.approx(LCC_DX) and g.max_georeference_error_m < 5.0
    assert g.uses_lonlat_view and g.lonlat_view_buffer_deg == 1.0
    assert ds.metadata["dwm:georeference"] == (
        "regular grid in the Lambert Conformal projection of WRF d01")
    assert ds.warnings == [] and ds.adapter_state["regrid"] is None
    swh = adapter.compute(ds, "significant_wave_height", 1).data
    np.testing.assert_allclose(swh[0, 5:], (1 + np.arange(NX) / NX + 1.0)[5:], rtol=1e-6)


def test_lambert_waves_without_wrf_are_resampled_to_lonlat(tmp_path):
    from modelatlasify.geospatial.footprint import grid_footprint

    lon, lat = lambert_lonlat()
    write_dwm(tmp_path, hourly(T0, 2), lonlat=(lon, lat))
    adapter = DWMAdapter()
    (ds,) = adapter.discover(tmp_path)
    g = ds.grid
    assert g.crs.is_geographic and g.max_georeference_error_m is None
    assert g.lonlat_view_buffer_deg == 1.0 and g.outline_lonlat is not None
    assert len(ds.warnings) == 1 and "resampled" in ds.warnings[0]
    assert "no WRF output here" in ds.warnings[0]

    swh = adapter.compute(ds, "significant_wave_height", 1).data
    mwp = adapter.compute(ds, "mean_wave_period", 1).data
    assert swh.shape == g.shape
    # Nearest model cell, never interpolated: swh = 2 + i/NX, mwp = 5 + j/NY, i, j integers.
    ok = np.isfinite(swh)
    assert ok.mean() > 0.5
    i = (swh[ok] - 2.0) * NX
    np.testing.assert_allclose(i, np.round(i), atol=1e-3)
    j = (mwp[np.isfinite(mwp)] - 5.0) * NY
    np.testing.assert_allclose(j, np.round(j), atol=1e-3)
    # The regular cell holding model cell (i=20, j=12)'s centre has its value.
    t = g.transform
    col = int((lon[12, 20] - t.c) // t.a)
    row = int((lat[12, 20] - t.f) // t.e)
    assert swh[row, col] == pytest.approx(2.0 + 20 / NX, abs=1e-5)
    # Outside the model's curved edge (a corner of the bounding box) is missing.
    assert np.isnan(swh[0, 0]) or np.isnan(swh[-1, 0])

    # The STAC footprint follows the model's edge, not the lon/lat box.
    fp = grid_footprint(g)
    assert fp.bbox[0] == pytest.approx(lon.min(), abs=1.0)
    assert fp.bbox[3] == pytest.approx(lat.max(), abs=1.0)


def test_regular_lonlat_waves_fit_exactly(tmp_path):
    jj, ii = np.mgrid[0:NY, 0:NX]
    lon, lat = -120.0 + 0.5 * ii, 10.0 + 0.5 * jj
    write_dwm(tmp_path, hourly(T0, 1), lonlat=(lon, lat))
    (ds,) = DWMAdapter().discover(tmp_path)
    g = ds.grid
    assert g.crs.is_geographic and g.max_georeference_error_m < 1e-6
    assert g.transform.a == pytest.approx(0.5) and g.transform.c == pytest.approx(-120.25)
    assert ds.warnings == [] and ds.adapter_state["regrid"] is None
    assert g.uses_lonlat_view and g.lonlat_view_buffer_deg is None  # its own projection


def test_lonlat_box_view_is_the_domain_plus_a_buffer(tmp_path):
    from modelatlasify.plotting.maps import full_view, lonlat_box

    lon, lat = lambert_lonlat()
    write_wrfout(tmp_path, hourly(T0, 1))
    write_dwm(tmp_path, hourly(T0, 1), lonlat=(lon, lat))
    (ds,) = DWMAdapter().discover(tmp_path)
    box = lonlat_box(ds.grid)
    # The outer cell edges lie half a cell (25 km) outside the centres; then 1 degree.
    assert box.west == pytest.approx(lon.min() - 1.0, abs=0.4)
    assert box.east == pytest.approx(lon.max() + 1.0, abs=0.4)
    assert box.south == pytest.approx(lat.min() - 1.0, abs=0.4)
    assert box.north == pytest.approx(lat.max() + 1.0, abs=0.4)
    view = full_view(ds.grid)
    assert view.crs is None  # PlateCarree, not the Lambert projection
    assert view.extent == pytest.approx((box.west - box.center_lon, box.east - box.center_lon,
                                         box.south, box.north))


def test_cli_lambert_waves_without_wrf(tmp_path, capsys, monkeypatch):
    from modelatlasify.core.validation import validate_catalog
    from modelatlasify.plotting import maps

    run = tmp_path / "w"
    write_dwm(run, hourly(T0, 2), lonlat=lambert_lonlat())
    calls = []
    real = maps._draw_map

    def spy(*args, **kwargs):
        calls.append((len(kwargs.get("outlines", ())), len(kwargs.get("dashed_outlines", ()))))
        return real(*args, **kwargs)

    monkeypatch.setattr(maps, "_draw_map", spy)
    out = tmp_path / "atlas"
    assert main([str(run), "-o", str(out), "--no-validate", "--variables", "swh"]) == 0
    err = capsys.readouterr().err
    assert "resampled to a lon/lat grid." in err
    assert "WARNING: DWM of run '" in err and "resampled (each cell takes the value" in err
    # The wave edge is outlined solid (like WRF domains), on item maps and composites.
    assert calls and all(c == (1, 0) for c in calls)
    run_dir = run_out(out, run)
    with rasterio.open(next(run_dir.glob("items/*/significant_wave_height.tif"))) as src:
        assert src.crs.is_geographic
    collection = json.loads((run_dir / "collection.json").read_text())
    (dom,) = collection["modelatlasify:domains"]
    assert "resolution_deg" in dom and dom["georeference_max_error_m"] is None
    assert validate_catalog(out / "catalog.json", schemas=False).ok


def _record_maps(monkeypatch):
    """Record (path, vector grid widths) for every map drawn."""
    from modelatlasify.plotting import maps

    saved = []
    real_draw = maps._draw_map
    monkeypatch.setattr(maps, "_draw_map", lambda path, *a, **k: (
        saved.append((str(path), [g.width for _, _, g in (k.get("vectors") or [])])),
        real_draw(path, *a, **k))[1])
    return saved


def test_wave_maps_get_wrf_wind_vectors(tmp_path, monkeypatch, capsys):
    run = tmp_path / "coupled"
    write_wrfout(run, hourly(T0, 2))  # d01, 40 wide
    write_wrfout(run, hourly(T0, 2), domain=2, nx=20, ny=16, dx=4000.0)
    write_dwm(run, hourly(T0, 2), cen_lon=-100.0, cen_lat=40.0)
    saved = _record_maps(monkeypatch)
    out = tmp_path / "atlas"
    assert main([str(run), "-o", str(out), "--no-validate", "--variables", "swh"]) == 0
    assert "Also making u10" in capsys.readouterr().err
    run_dir = run_out(out, run)
    # WRF is kept only to supply the wind: its u10/v10 GeoTIFFs, no WRF maps.
    assert {p.name for p in run_dir.glob("items/d01-*/*")} >= {"u10.tif", "v10.tif"}
    assert not list(run_dir.glob("items/d0*/*.png"))
    wave_maps = [(p, w) for p, w in saved if "significant_wave_height" in p]
    assert len(wave_maps) == 2 + 2  # 2 per-item DWM maps + 2 composites
    assert all(w == [40, 20] for _, w in wave_maps)  # WRF d01 and d02 winds, coarse first


def test_wave_maps_without_wrf_have_no_vectors(tmp_path, monkeypatch, capsys):
    run = tmp_path / "coupled"
    write_wrfout(run, hourly(T0, 2))
    write_dwm(run, hourly(T0, 2), cen_lon=-100.0, cen_lat=40.0)
    saved = _record_maps(monkeypatch)
    out = tmp_path / "atlas"
    assert main([str(run), "-o", str(out), "--no-validate", "--model", "dwm"]) == 0
    assert "Also making u10" not in capsys.readouterr().err
    assert saved and all(w == [] for _, w in saved)
    assert not list(run_out(out, run).glob("items/d01-*"))


def test_wave_maps_without_matching_wrf_times_have_no_vectors(tmp_path, monkeypatch):
    run = tmp_path / "coupled"
    write_wrfout(run, hourly(T0, 2))  # 00, 01
    write_dwm(run, hourly(T0 + timedelta(minutes=30), 2), start=T0,
              cen_lon=-100.0, cen_lat=40.0)  # 00:30, 01:30
    saved = _record_maps(monkeypatch)
    assert main([str(run), "-o", str(tmp_path / "atlas"), "--no-validate",
                 "--variables", "swh"]) == 0
    wave_maps = [w for p, w in saved if "significant_wave_height" in p]
    assert wave_maps and all(w == [] for w in wave_maps)


def test_swh_lowest_band_is_white():
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    from modelatlasify.core.variables import get_variable
    from modelatlasify.plotting.maps import _norm_and_cmap

    var = get_variable("significant_wave_height")
    norm, cmap = _norm_and_cmap(var, np.array([0.0]))
    x = np.array([0.2, 0.7, 2.2, 13.0, 20.0])
    new = cmap(norm(x))[:, :3]
    old = plt.get_cmap("turbo")(mcolors.BoundaryNorm(var.levels, 256, extend="max")(x))[:, :3]
    assert np.allclose(new[0], 1.0)  # 0-0.5 m is white
    np.testing.assert_allclose(new[1:], old[1:])  # every other band is turbo as before
