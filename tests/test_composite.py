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


def test_sub_area_maps(tmp_path, capsys):
    run = tmp_path / "run01"
    write_wrfout(run, hourly(T0, 2))  # d01 centred on 40N, 100W
    write_wrfout(run, hourly(T0, 2), domain=2, nx=20, ny=16, dx=4000.0)
    out = tmp_path / "atlas"
    assert main([str(run), "-o", str(out), "--no-validate", "--variables", "wind",
                 "--plot-sub-areas=zoom,-101,-99,39.5,40.5,away,10,20,40,50"]) == 0
    images = out / "run01" / "model-images"
    zoom = sorted(p.name for p in images.glob("zoom/surface_wind_speed/*.png"))
    assert zoom == ["uwincm.zoom.surface_wind_speed.20261006T000000Z.png",
                    "uwincm.zoom.surface_wind_speed.20261006T010000Z.png"]
    assert len(list(images.glob("full/*/*.png"))) == 2
    assert not (images / "away").exists()
    assert "sub-area 'away' is entirely outside" in capsys.readouterr().err


def test_crop_window_and_overlap():
    from modelatlasify.core.areas import SubArea
    from modelatlasify.models.wrf.adapter import WRFAdapter
    from modelatlasify.plotting.maps import (_corners_in_view, _crop_window, subarea_view,
                                             view_overlaps)
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        write_wrfout(Path(d), hourly(T0, 1))
        (ds,) = WRFAdapter().discover(Path(d))
    grid = ds.grid  # 40x30 cells of 12 km around 40N, 100W
    view = subarea_view(SubArea("z", -101.0, -99.0, 39.5, 40.5))
    r0, r1, c0, c1 = _crop_window(*_corners_in_view(grid, view), view)
    assert 0 < r1 - r0 < grid.height and 0 < c1 - c0 < grid.width
    assert not view_overlaps(subarea_view(SubArea("far", 10.0, 20.0, 40.0, 50.0)), grid)
    # A box much smaller than one cell still finds that cell.
    tiny = subarea_view(SubArea("t", -100.01, -100.0, 40.0, 40.01))
    assert view_overlaps(tiny, grid)
