import pytest

from conftest import T0, run_out
from wrf_fixture import hourly, write_wrfout
from modelatlasify.cli import main
from modelatlasify.core.validation import validate_catalog


def test_composite_maps_with_nest(tmp_path):
    run = tmp_path / "run01"
    write_wrfout(run, hourly(T0, 3))
    write_wrfout(run, hourly(T0, 3), domain=2, nx=20, ny=16, dx=4000.0)
    out = tmp_path / "atlas"
    assert main([str(run), "-o", str(out), "--no-validate"]) == 0

    images = run_out(out, run) / "model-images" / "full"
    pngs = {p.relative_to(images).as_posix() for p in images.rglob("*.png")}
    assert "precipitation/uwincm.full.precipitation.20261006T010000Z.png" in pngs
    d01_tifs = [p for p in run_out(out, run).glob("items/d01-*/*.tif")
                if p.stem not in ("u10", "v10")]  # GeoTIFF only, no maps
    assert len(pngs) == len(d01_tifs) == 3 + 3 + 2
    for p in pngs:
        var, name = p.split("/")
        assert name.startswith(f"uwincm.full.{var}.")
    for var_dir in (d for d in images.iterdir() if d.is_dir()):
        frames = _animation_frames(images / f"{var_dir.name}.html")
        assert [f["src"] for f in frames] == sorted(
            f"{var_dir.name}/{p.name}" for p in var_dir.glob("*.png"))
        assert all((images / f["src"]).is_file() for f in frames)
    frames = _animation_frames(images / "precipitation.html")
    assert [f["label"] for f in frames] == ["2026-10-06 00:00 UTC", "2026-10-06 01:00 UTC",
                                            "2026-10-06 02:00 UTC"]
    assert validate_catalog(out / "catalog.json", schemas=False).ok


def _animation_frames(page):
    import json
    import re

    m = re.search(r"const FRAMES = (.*?);\n", page.read_text(encoding="utf-8"), flags=re.DOTALL)
    return json.loads(m[1].replace("<\\/", "</"))


def test_animation_page(tmp_path):
    import modelatlasify.variables  # noqa: F401 - registers the long names
    from modelatlasify.plotting.animation import write_animation

    var_dir = tmp_path / "run" / "model-images" / "full" / "surface_wind_speed"
    var_dir.mkdir(parents=True)
    assert write_animation(var_dir) is None  # no frames
    (var_dir / "uwincm.full.surface_wind_speed.20261006T010000Z.png").write_bytes(b"")
    page = write_animation(var_dir)
    assert page == var_dir.parent / "surface_wind_speed.html"
    (var_dir / "uwincm.full.surface_wind_speed.20261006T000000Z.png").write_bytes(b"")
    assert write_animation(var_dir) == page  # rewritten with the new frame
    assert [f["label"] for f in _animation_frames(page)] == ["2026-10-06 00:00 UTC",
                                                             "2026-10-06 01:00 UTC"]
    assert "<title>Surface wind speed — full — run</title>" in page.read_text(encoding="utf-8")


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
    images = run_out(out, run) / "model-images"
    zoom = sorted(p.name for p in images.glob("zoom/surface_wind_speed/*.png"))
    assert zoom == ["uwincm.zoom.surface_wind_speed.20261006T000000Z.png",
                    "uwincm.zoom.surface_wind_speed.20261006T010000Z.png"]
    assert len(list(images.glob("full/*/*.png"))) == 2
    assert (images / "zoom" / "surface_wind_speed.html").is_file()
    assert not (images / "away").exists()
    assert not (images / "away.html").exists()
    assert "sub-area 'away' is entirely outside" in capsys.readouterr().err

    summary = images.parent / "model-images.html"
    text = summary.read_text(encoding="utf-8")
    assert text.index('<th scope="col">full</th>') < text.index('<th scope="col">zoom</th>')
    assert ('<a href="model-images/zoom/surface_wind_speed.html"' in text
            and 'src="model-images/zoom/surface_wind_speed/'
                'uwincm.zoom.surface_wind_speed.20261006T010000Z.png"' in text)  # latest frame
    for ref in _summary_links(text):
        assert (images.parent / ref).is_file(), ref


def _summary_links(text):
    import re

    return re.findall(r'(?:href|src)="([^"]+)"', text)


def test_summary_table(tmp_path):
    import modelatlasify.variables  # noqa: F401 - registers the variable order and names
    from modelatlasify.core.variables import all_variables
    from modelatlasify.plotting.animation import write_summary

    images = tmp_path / "run" / "model-images"
    assert write_summary(images) is None  # no maps yet
    for area, var in [("gulf", "precipitation"), ("full", "surface_wind_speed"),
                      ("full", "precipitation")]:
        (images / area / var).mkdir(parents=True)
        for h in ("00", "01"):
            (images / area / var / f"uwincm.{area}.{var}.20261006T{h}0000Z.png").write_bytes(b"")
    page = write_summary(images)
    assert page == tmp_path / "run" / "model-images.html"
    text = page.read_text(encoding="utf-8")
    rows = text.split("<tr>")[1:]
    assert [r.count("<th") for r in rows] == [3, 1, 1]  # corner + 2 areas; row headings
    assert "full" in rows[0].split("gulf")[0]  # full is the first column
    # Rows follow the registered variable order.
    order = list(all_variables())
    expected = sorted(["precipitation", "surface_wind_speed"], key=order.index)
    assert [r.split('class="code">')[1].split("<")[0] for r in rows[1:]] == expected
    by_var = dict(zip(expected, rows[1:]))
    assert 'class="none"' in by_var["surface_wind_speed"]  # no gulf wind map
    assert 'class="none"' not in by_var["precipitation"]
    assert 'width="300"' in by_var["precipitation"]
    assert "20261006T010000Z.png" in by_var["precipitation"]
    assert "20261006T000000Z.png" not in text  # only the latest frame is shown


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


def test_no_full_plots_keeps_sub_areas_and_item_maps(tmp_path):
    run = tmp_path / "run01"
    write_wrfout(run, hourly(T0, 2))
    out = tmp_path / "atlas"
    assert main([str(run), "-o", str(out), "--no-validate", "--variables", "wind",
                 "--no-full-plots", "--plot-sub-areas=zoom,-101,-99,39.5,40.5"]) == 0
    images = run_out(out, run) / "model-images"
    assert not (images / "full").exists()
    assert len(list(images.glob("zoom/surface_wind_speed/*.png"))) == 2
    assert len(list(run_out(out, run).glob("items/*/surface_wind_speed.png"))) == 2


def test_olr_precipitation_maps(tmp_path, capsys):
    import json

    import rasterio

    include = ("U10", "V10", "RAINC", "RAINNC", "OLR")
    run = tmp_path / "run01"
    write_wrfout(run, hourly(T0, 3), include=include)
    write_wrfout(run, hourly(T0, 3), domain=2, nx=20, ny=16, dx=4000.0, include=include)
    out = tmp_path / "atlas"
    assert main([str(run), "-o", str(out), "--variables", "olr", "--no-validate"]) == 0
    assert "Also making precipitation_interval" in capsys.readouterr().err

    item = sorted(run_out(out, run).glob("items/d01-*"))[-1]
    assert {p.name for p in item.iterdir() if p.suffix in (".tif", ".png")} == {
        "olr.tif", "olr_precipitation.png",
        "precipitation_interval.tif", "precipitation_interval.png",
        "u10.tif", "v10.tif"}  # for the vectors on the precipitation_interval maps
    with rasterio.open(item / "olr.tif") as src:  # the full range: no 240 W m-2 cutoff
        assert src.read(1).max() > 240.0
    assets = json.loads((item / f"{item.name}.json").read_text())["assets"]
    assert assets["olr"]["roles"] == ["data"]
    assert "visual" in assets["olr_precipitation_map"]["roles"]
    assert "olr_map" not in assets and "olr_precipitation" not in assets

    images = run_out(out, run) / "model-images" / "full"
    olr = sorted(p.name for p in (images / "olr_precipitation").glob("*.png"))
    assert len(olr) == 3  # also at the first time, without precipitation on top
    assert (images / "olr_precipitation.html").is_file()
    assert not (images / "olr").exists()
    summary = (run_out(out, run) / "model-images.html").read_text(encoding="utf-8")
    assert "OLR with interval precipitation" in summary
    assert validate_catalog(out / "catalog.json", schemas=False).ok


def test_wind_vectors_on_every_domain_and_composites(tmp_path, monkeypatch):
    from modelatlasify.plotting import maps

    run = tmp_path / "run01"
    write_wrfout(run, hourly(T0, 2))
    write_wrfout(run, hourly(T0, 2), domain=2, nx=20, ny=16, dx=4000.0)
    calls = []
    real = maps._draw_vectors
    monkeypatch.setattr(maps, "_draw_vectors", lambda fig, ax, view, vectors, *a: (
        calls.append((ax.figure, [g.width for _, _, g in vectors])),
        real(fig, ax, view, vectors, *a)))
    saved = []
    real_draw = maps._draw_map
    monkeypatch.setattr(maps, "_draw_map", lambda path, *a, **k: (
        saved.append((str(path), bool(k.get("vectors")))), real_draw(path, *a, **k))[1])
    out = tmp_path / "atlas"
    assert main([str(run), "-o", str(out), "--variables", "wind", "--no-validate"]) == 0
    items = [(p, v) for p, v in saved if "/items/" in p]
    assert {p.split("/items/")[1][:3] for p, _ in items} == {"d01", "d02"}
    assert all(v for _, v in items)  # every domain's per-item maps have vectors
    composites = [p for p, v in saved if "/model-images/" in p]
    assert composites and all(v for p, v in saved if "/model-images/" in p)
    assert any(widths == [40, 20] for _, widths in calls)  # composites use the nest's winds


def test_vector_lengths_spacing_and_cap():
    import matplotlib.pyplot as plt
    import numpy as np
    from pyproj import CRS
    from rasterio.transform import from_origin

    from modelatlasify.core.model import GridDefinition
    from modelatlasify.plotting.maps import _draw_vectors, full_view

    crs = CRS.from_proj4("+proj=merc +lat_ts=0 +lon_0=0 +R=6370000 +units=m")
    dx = 25e3
    # 6000 km wide: 24 arrows across at 250 km, so the 20-across rule does not apply.
    grid = GridDefinition(crs=crs, transform=from_origin(0.0, 2e6, dx, dx), width=240, height=80)
    view = full_view(grid)
    u = np.zeros((80, 240))
    u[:, :120] = 10.0 / (3600 / 1852)  # 10 kt westerly in the west half
    u[:, 120:] = 100.0 / (3600 / 1852)  # 100 kt (capped at 70) in the east half
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1], projection=view.projection)
    ax.set_extent(view.extent, crs=view.projection)
    _draw_vectors(fig, ax, view, [(u, np.zeros_like(u), grid)])
    (q,) = [c for c in ax.collections if hasattr(c, "U")]
    x, y = q.X, q.Y
    lat = np.degrees(2 * np.arctan(np.exp(y / 6370e3)) - np.pi / 2)
    k = 1 / np.cos(np.radians(lat))  # Mercator scale factor: map metres per ground metre
    length_km = np.asarray(q.U) / k / 1e3
    assert np.allclose(np.asarray(q.V), 0, atol=1e-6)  # due east
    np.testing.assert_allclose(length_km[x < 3e6], 100.0, rtol=0.01)  # 100 km per 10 kt
    np.testing.assert_allclose(length_km[x > 3e6], 700.0, rtol=0.01)  # capped at 70 kt
    spacing_km = np.diff(np.unique(np.round(x)))[0] / 1e3 / k[0]
    assert 240 < spacing_km < 260  # every ~250 km on the ground (centre of the map)
    assert len(np.unique(np.round(x))) == 24  # 6000 km / 250 km
    plt.close(fig)


def test_small_maps_get_twenty_vectors_across():
    import matplotlib.pyplot as plt
    import numpy as np
    from pyproj import CRS
    from rasterio.transform import from_origin

    from modelatlasify.core.model import GridDefinition
    from modelatlasify.plotting.maps import _draw_vectors, full_view

    crs = CRS.from_proj4("+proj=merc +lat_ts=0 +lon_0=0 +R=6370000 +units=m")
    # 1000 km wide, 600 km tall: 250 km spacing would give only 4 across.
    grid = GridDefinition(crs=crs, transform=from_origin(0.0, 6e5, 5e3, 5e3), width=200,
                          height=120)
    view = full_view(grid)
    u = np.full((120, 200), 5.0)
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1], projection=view.projection)
    ax.set_extent(view.extent, crs=view.projection)
    _draw_vectors(fig, ax, view, [(u, np.zeros_like(u), grid)])
    (q,) = [c for c in ax.collections if hasattr(c, "U")]
    assert len(np.unique(np.round(q.X))) == 20  # 50 km apart
    assert len(np.unique(np.round(q.Y))) == 12  # the same spacing vertically
    length = np.asarray(q.U) / (1 / np.cos(np.radians(2.7)))  # ~ground metres near 2.7 N
    # 50 km spacing instead of 250 km: arrows are 1/5 as long per knot (2 km per knot).
    np.testing.assert_allclose(length, 5.0 * 3600 / 1852 * 10e3 / 5, rtol=0.01)
    plt.close(fig)


def test_t2m_maps_with_wind_vectors(tmp_path, monkeypatch, capsys):
    import json

    from modelatlasify.plotting import maps

    include = ("U10", "V10", "RAINC", "RAINNC", "T2")
    run = tmp_path / "run01"
    write_wrfout(run, hourly(T0, 2), include=include)
    write_wrfout(run, hourly(T0, 2), domain=2, nx=20, ny=16, dx=4000.0, include=include)
    saved = []
    real_draw = maps._draw_map
    monkeypatch.setattr(maps, "_draw_map", lambda path, *a, **k: (
        saved.append((str(path), bool(k.get("vectors")))), real_draw(path, *a, **k))[1])
    out = tmp_path / "atlas"
    assert main([str(run), "-o", str(out), "--variables", "t2m", "--no-validate"]) == 0
    assert "Also making u10" in capsys.readouterr().err

    item = sorted(run_out(out, run).glob("items/d02-*"))[-1]
    assert {p.name for p in item.iterdir() if p.suffix in (".tif", ".png")} == {
        "t2m.tif", "t2m.png", "u10.tif", "v10.tif"}
    assets = json.loads((item / f"{item.name}.json").read_text())["assets"]
    assert assets["t2m"]["roles"] == ["data"] and "t2m_map" in assets
    t2m_maps = [(p, v) for p, v in saved if p.endswith(".png") and "t2m" in p]
    assert len(t2m_maps) == 2 * 2 + 2  # per-item d01, d02 at 2 times + 2 composites
    assert all(v for _, v in t2m_maps)  # every t2m map has wind vectors
    images = run_out(out, run) / "model-images" / "full"
    assert len(list((images / "t2m").glob("*.png"))) == 2
    assert "2 m temperature" in (run_out(out, run) / "model-images.html").read_text()


@pytest.mark.parametrize("variable", ["precipitation", "precipitation_interval"])
def test_precipitation_maps_have_wind_vectors(tmp_path, monkeypatch, variable):
    from modelatlasify.plotting import maps

    run = tmp_path / "run01"
    write_wrfout(run, hourly(T0, 2))
    saved = []
    real_draw = maps._draw_map
    monkeypatch.setattr(maps, "_draw_map", lambda path, *a, **k: (
        saved.append((str(path), bool(k.get("vectors")))), real_draw(path, *a, **k))[1])
    assert main([str(run), "-o", str(tmp_path / "atlas"), "--variables", variable,
                 "--no-validate"]) == 0
    mine = [v for p, v in saved if f"{variable}." in p.rsplit("/", 1)[-1]]
    assert mine and all(mine)


def test_olr_maps_have_no_wind_vectors():
    from modelatlasify.core.variables import get_variable

    assert get_variable("olr_precipitation").vectors is None  # top-of-atmosphere field
    for v in ("precipitation", "precipitation_interval", "spray_mass_flux",
              "spray_latent_heat_flux", "spray_sensible_heat_flux", "significant_wave_height",
              "mean_wave_period", "dominant_wave_period", "surface_wind_speed", "t2m"):
        assert get_variable(v).vectors == ("u10", "v10")


def _curly_setup(u, v, width_m):
    """A Mercator grid (R = 6370 km) from the equator, its full view, and the
    curly vectors drawn on it with the ocean current style."""
    import matplotlib.pyplot as plt
    from pyproj import CRS
    from rasterio.transform import from_origin

    from modelatlasify.core.model import GridDefinition
    from modelatlasify.plotting.maps import _draw_vectors, full_view
    from modelatlasify.variables.ocean import CURRENT_VECTORS

    ny, nx = u.shape
    crs = CRS.from_proj4("+proj=merc +lat_ts=0 +lon_0=0 +R=6370000 +units=m")
    dx = width_m / nx
    grid = GridDefinition(crs=crs, transform=from_origin(0.0, ny * dx, dx, dx), width=nx,
                          height=ny)
    view = full_view(grid)
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1], projection=view.projection)
    ax.set_extent(view.extent, crs=view.projection)
    _draw_vectors(fig, ax, view, [(u, v, grid)], CURRENT_VECTORS)
    from matplotlib.collections import LineCollection, PolyCollection
    lines = [c for c in ax.collections if isinstance(c, LineCollection)]
    heads = [c for c in ax.collections if isinstance(c, PolyCollection)]
    return fig, lines, heads


def test_curly_vectors_in_uniform_flow_are_straight_and_scaled():
    import matplotlib.pyplot as plt
    import numpy as np

    u = np.zeros((40, 240))
    u[:, :120] = 0.2  # m/s, east, in the west half
    u[:, 120:] = 3.0  # above the 2 m/s cap in the east half
    fig, (lines,), (heads,) = _curly_setup(u, np.zeros_like(u), 6.0e6)  # 6000 km: 125 km steps
    segs = lines.get_segments()
    assert len(segs) == len(heads.get_paths()) > 0
    for s in segs:
        assert np.allclose(s[:, 1], s[0, 1])  # due east, straight
        assert np.all(np.diff(s[:, 0]) > 0)  # pointing downstream (east)
    lat = np.degrees(2 * np.arctan(np.exp(np.array([s[0, 1] for s in segs]) / 6370e3)) - np.pi / 2)
    ground_km = np.array([s[-1, 0] - s[0, 0] for s in segs]) * np.cos(np.radians(lat)) / 1e3
    mid = np.array([s[len(s) // 2, 0] for s in segs])
    west = mid < 3.0e6 - 200e3  # away from the speed jump
    east = (mid > 3.0e6 + 800e3) & (mid < 6.0e6 - 800e3)  # ...and from the grid's east edge
    np.testing.assert_allclose(ground_km[west], 0.2 * 375, rtol=0.02)  # 375 km per m/s
    np.testing.assert_allclose(ground_km[east], 2.0 * 375, rtol=0.02)  # capped at 2 m/s
    # Centred on the seed (the middle point): equal halves upstream and downstream.
    seed = mid
    up = np.array([s[len(s) // 2, 0] - s[0, 0] for s in segs])
    down = np.array([s[-1, 0] - s[len(s) // 2, 0] for s in segs])
    np.testing.assert_allclose(up[west], down[west], rtol=1e-6)
    # Seeds every 125 km on the ground (at the map's centre latitude).
    row = np.isclose([s[0, 1] for s in segs], segs[0][0, 1])
    spacing = np.diff(np.sort(seed[row & west]))  # (truncated edge curves lose points)
    lat_c = np.degrees(2 * np.arctan(np.exp(0.5e6 / 6370e3)) - np.pi / 2)
    np.testing.assert_allclose(spacing * np.cos(np.radians(lat_c)), 125e3, rtol=1e-3)
    plt.close(fig)


def test_curly_vectors_curve_with_the_flow_and_stop_at_land():
    import matplotlib.pyplot as plt
    import numpy as np

    n, width = 120, 1.2e6  # 10 km cells; 40 curves across: 30 km apart, 90 km per m/s
    y, x = np.mgrid[n - 1:-1:-1, 0:n] + 0.5 - n / 2  # north-up cell centres, origin centred
    radius = np.hypot(x, y)
    u, v = -y / radius, x / radius  # 1 m/s everywhere, counter-clockwise around the centre
    u[:, :10] = v[:, :10] = np.nan  # land strip on the west edge
    fig, (lines,), _ = _curly_setup(u, v, width)
    segs = lines.get_segments()
    centre = np.array([width / 2, width / 2])
    mid_r = np.array([np.hypot(*(s[len(s) // 2] - centre)) for s in segs])
    chord = np.array([np.hypot(*(s[-1] - s[0])) for s in segs])
    arc = np.array([np.hypot(*np.diff(s, axis=0).T).sum() for s in segs])
    near = (mid_r > 20e3) & (mid_r < 60e3)  # 90 km curves on 20-60 km circles
    assert near.any() and np.all(chord[near] < 0.9 * arc[near])  # clearly curved (straight: 1)
    # Counter-clockwise: the end lies to the left of the start direction.
    for s in np.array(segs, dtype=object)[near]:
        d0, d1 = s[1] - s[0], s[-1] - s[0]
        assert d0[0] * d1[1] - d0[1] * d1[0] > 0
    xmin = min(s[:, 0].min() for s in segs)
    assert xmin >= 10 / n * width - 1.0  # no curve enters the land strip
    plt.close(fig)


def test_wind_vectors_stay_straight():
    from modelatlasify.variables.ocean import CURRENT_VECTORS
    from modelatlasify.variables.wind import WIND_VECTORS

    assert CURRENT_VECTORS.curly and not WIND_VECTORS.curly
    assert CURRENT_VECTORS.km_per_unit == 3 * CURRENT_VECTORS.spacing_km  # x3 the arrow scale
