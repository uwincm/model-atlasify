"""Pressure-level products: RH, geopotential height and wind at 700/500 hPa."""

import json

import numpy as np
import pytest

from conftest import T0, run_out
from wrf_fixture import (LEVEL_P_HPA, LEVEL_T_K, LEVEL_Z_STAG_M, hourly, write_wrfout)
from modelatlasify.cli import main
from modelatlasify.core.processing import _with_dependencies
from modelatlasify.core.validation import validate_catalog
from modelatlasify.core.variables import get_variable, resolve_names
from modelatlasify.models.wrf import WRFAdapter
from modelatlasify.variables.upper_air import interp_to_pressure, relative_humidity
from modelatlasify.variables.wind import earth_relative

LEVEL_PRODUCTS = {f"{q}_{lev}" for lev in (700, 500)
                  for q in ("rh", "geopotential_height", "u", "v")}


def _expected_height(level_hpa, p_lo, p_hi, z_lo, z_hi):
    w = (np.log(p_lo) - np.log(level_hpa)) / (np.log(p_lo) - np.log(p_hi))
    return z_lo + w * (z_hi - z_lo)


def test_names_and_aliases():
    assert set(resolve_names(["upper_air"])) == LEVEL_PRODUCTS
    assert resolve_names(["500hpa"]) == ["rh_500", "geopotential_height_500", "u_500", "v_500"]
    rh, z = get_variable("rh_700"), get_variable("geopotential_height_700")
    assert (rh.overlay, rh.vectors) == ("geopotential_height_700", ("u_700", "v_700"))
    assert rh.composite and not z.composite  # height: per-item maps only
    assert z.overlay == z.name and z.contour is not None  # shaded plus its own contours
    assert not get_variable("u_700").maps and not get_variable("v_500").maps


def test_dependencies_put_winds_before_height():
    # rh_700 needs the height (its contours), which in turn needs the winds
    # (vectors on the height maps): winds must come first.
    order = _with_dependencies(["rh_700"], lambda _: None)
    assert order.index("u_700") < order.index("geopotential_height_700") < order.index("rh_700")
    assert _with_dependencies(["geopotential_height_500"], lambda _: None) == [
        "u_500", "v_500", "geopotential_height_500"]


def test_interpolation_in_log_pressure():
    p = np.array([900.0, 600.0, 400.0])[:, None, None] * 100.0 * np.ones((3, 1, 2))
    p[0, 0, 0] = 650e2  # 700 hPa below ground here
    z = np.array([1000.0, 4000.0, 7000.0])[:, None, None] * np.ones((3, 1, 2))
    (z700,) = interp_to_pressure([z], p, 700e2)
    assert np.isnan(z700[0, 0])
    assert z700[0, 1] == pytest.approx(_expected_height(700, 900, 600, 1000, 4000))
    (z300,) = interp_to_pressure([z], p, 300e2)  # above the top level
    assert np.isnan(z300).all()


def test_relative_humidity_saturation_and_limits():
    t = np.array([300.0, 273.15, 250.0])
    p = np.full(3, 8.5e4)
    es = 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))
    qvs = 0.622 * es / (p - 0.378 * es)
    np.testing.assert_allclose(relative_humidity(qvs, p, t), 100.0)
    np.testing.assert_allclose(relative_humidity(0.25 * qvs, p, t), 25.0)
    assert relative_humidity(2 * qvs, p, t).max() == 100.0  # supersaturation capped


def test_wrf_pressure_level_fields(tmp_path):
    run = tmp_path / "r"
    write_wrfout(run, hourly(T0, 1), levels=True)
    adapter = WRFAdapter()
    (ds,) = adapter.discover(run)
    assert LEVEL_PRODUCTS <= set(adapter.dataset_variables(ds))
    nx, ny = ds.grid.width, ds.grid.height
    below = np.broadcast_to(np.arange(nx) < 2, (ny, nx))  # 700 hPa below ground
    z_mass = [0.5 * (a + b) for a, b in zip(LEVEL_Z_STAG_M, LEVEL_Z_STAG_M[1:])]
    i = np.broadcast_to(np.arange(nx), (ny, nx))

    z700 = adapter.compute(ds, "geopotential_height_700", 0)
    assert np.isnan(z700.data[below]).all()
    np.testing.assert_allclose(z700.data[~below], _expected_height(
        700, LEVEL_P_HPA[0], LEVEL_P_HPA[1], z_mass[0], z_mass[1]) + 10.0 * i[~below],
        rtol=1e-5)
    z500 = adapter.compute(ds, "geopotential_height_500", 0)
    np.testing.assert_allclose(z500.data, _expected_height(
        500, LEVEL_P_HPA[1], LEVEL_P_HPA[2], z_mass[1], z_mass[2]) + 10.0 * i, rtol=1e-5)
    assert z500.provenance["pressure_hpa"] == 500

    rh = adapter.compute(ds, "rh_500", 0)
    assert LEVEL_T_K == 280.0  # RH from T = 280 K and the fixture's mixing ratio:
    np.testing.assert_allclose(rh.data, 30.0 + 40.0 * i / nx, rtol=1e-3)

    # U/V (staggered i, j at every level): destaggered, then rotated to east/north.
    u, v = (adapter.compute(ds, n, 0).data for n in ("u_500", "v_500"))
    gu = np.broadcast_to(np.arange(nx) + 0.5, (ny, nx))
    gv = np.broadcast_to((np.arange(ny) + 0.5)[:, None], (ny, nx))[::-1]  # north-up
    eu, ev = earth_relative(gu, gv, ds.grid)
    np.testing.assert_allclose(u, eu, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(v, ev, rtol=1e-5, atol=1e-5)
    assert not np.allclose(u, gu)  # Lambert grid: the rotation is not zero


def test_unavailable_without_3d_fields(wrf_run):
    adapter = WRFAdapter()
    (ds,) = adapter.discover(wrf_run)
    unavailable = adapter.unavailable_variables(ds)
    assert set(unavailable) >= LEVEL_PRODUCTS
    assert "no P/PB/PH/PHB/T/QVAPOR" in unavailable["rh_700"]


def test_pressure_level_maps(tmp_path, monkeypatch):
    from modelatlasify.plotting import maps

    run = tmp_path / "run01"
    write_wrfout(run, hourly(T0, 2), levels=True)
    write_wrfout(run, hourly(T0, 2), domain=2, nx=20, ny=16, dx=4000.0, levels=True)
    drawn, contoured = [], []
    real_draw, real_contours = maps._draw_map, maps._draw_contours
    monkeypatch.setattr(maps, "_draw_map", lambda path, *a, **k: (
        drawn.append((str(path), bool(k.get("vectors")))), real_draw(path, *a, **k))[1])
    monkeypatch.setattr(maps, "_draw_contours", lambda ax, view, layers, style: (
        contoured.append(len(layers)), real_contours(ax, view, layers, style)))
    out = tmp_path / "atlas"
    assert main([str(run), "-o", str(out), "--variables", "upper_air"]) == 0

    item = sorted(run_out(out, run).glob("items/d02-*"))[-1]
    files = {p.name for p in item.iterdir() if p.suffix in (".tif", ".png")}
    assert files == {f"{n}.tif" for n in LEVEL_PRODUCTS} | {
        f"{q}_{lev}.png" for lev in (700, 500) for q in ("rh", "geopotential_height")}
    assets = json.loads((item / f"{item.name}.json").read_text())["assets"]
    assert assets["u_700"]["roles"] == ["data"] and "u_700_map" not in assets

    pngs = [(p, v) for p, v in drawn if p.endswith(".png")]
    # per item: d01, d02 x 2 times x 4 maps; composites: rh_700, rh_500 x 2 times
    assert len(pngs) == 2 * 2 * 4 + 2 * 2
    assert all(v for _, v in pngs)  # wind vectors on every map
    assert len(contoured) == len(pngs)  # height contours on every map
    assert 2 in contoured  # composites contour both domains

    images = run_out(out, run) / "model-images" / "full"
    assert sorted(p.name for p in images.iterdir() if p.is_dir()) == ["rh_500", "rh_700"]
    assert validate_catalog(out / "catalog.json", schemas=False).ok


def test_smoothing_keeps_missing_cells_and_constants():
    from modelatlasify.plotting.maps import _smooth121

    a = np.full((6, 7), 5.0)
    a[2, 3] = np.nan
    s = _smooth121(a, 5)
    assert np.isnan(s[2, 3]) and np.allclose(s[np.isfinite(s)], 5.0)
    b = np.zeros((5, 5))
    b[2, 2] = 16.0
    sb = _smooth121(b, 1)
    assert sb[2, 2] == pytest.approx(16.0 * 0.25) and sb.sum() == pytest.approx(16.0)
