"""DWM wave model: discovery, georeferencing, fields, and alongside WRF."""

import json
from datetime import timedelta

import numpy as np
import pytest
import rasterio
from pyproj import Transformer

from conftest import T0, run_out
from dwm_fixture import mercator_lonlat, write_dwm
from wrf_fixture import hourly, write_wrfout
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
