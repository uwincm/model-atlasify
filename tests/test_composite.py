import pytest

from conftest import T0
from wrf_fixture import hourly, write_wrfout
from modelatlasify.cli import main
from modelatlasify.core.validation import validate_catalog


def test_composite_maps_with_nest(tmp_path):
    run = tmp_path / "run01"
    write_wrfout(run, hourly(T0, 3))
    write_wrfout(run, hourly(T0, 3), domain=2, nx=20, ny=16, dx=4000.0)
    out = tmp_path / "atlas"
    assert main([str(run), "-o", str(out), "--no-validate"]) == 0

    images = out / "run01" / "model-images" / "full"
    pngs = {p.relative_to(images).as_posix() for p in images.rglob("*.png")}
    assert "precipitation/uwincm.full.precipitation.20261006T010000Z.png" in pngs
    d01_tifs = list(out.glob("run01/items/d01-*/*.tif"))
    assert len(pngs) == len(d01_tifs) == 3 + 3 + 2
    for p in pngs:
        var, name = p.split("/")
        assert name.startswith(f"uwincm.full.{var}.")
    assert validate_catalog(out / "catalog.json", schemas=False).ok


def test_mercator_nest_is_placed_without_resampling(tmp_path):
    from modelatlasify.models.wrf.adapter import WRFAdapter
    from modelatlasify.plotting.maps import _extent_in

    write_wrfout(tmp_path, hourly(T0, 1), map_proj=3)
    write_wrfout(tmp_path, hourly(T0, 1), map_proj=3, domain=2, nx=20, ny=16, dx=4000.0)
    d01, d02 = WRFAdapter().discover(tmp_path)
    extent = _extent_in(d02.grid, d01.grid.crs)
    assert extent is not None
    assert extent[1] - extent[0] == pytest.approx(20 * 4000.0, rel=1e-6)
