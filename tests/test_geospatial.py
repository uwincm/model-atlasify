import numpy as np
import pytest
import rasterio
from pyproj import Transformer

from conftest import T0
from wrf_fixture import PROJECTIONS, grid_lonlat, hourly, write_wrfout
from modelatlasify.core.errors import GeoreferenceError
from modelatlasify.core.variables import get_variable
from modelatlasify.geospatial.footprint import grid_footprint
from modelatlasify.geospatial.geotiff import write_geotiff
from modelatlasify.models.wrf import WRFAdapter
from modelatlasify.models.wrf.projection import wrf_crs, wrf_grid
import modelatlasify.variables  # noqa: F401


@pytest.mark.parametrize("map_proj", [1, 2, 3])
def test_projection_recovers_known_grid(map_proj):
    nx, ny, dx = 50, 40, 15000.0
    p = PROJECTIONS[map_proj]
    lon, lat, x0, y0 = grid_lonlat(p, nx, ny, dx, dx)
    attrs = dict(p, DX=dx, DY=dx)
    g = wrf_grid(attrs, lat.astype("f4"), lon.astype("f4"))
    t = g.transform
    assert (g.width, g.height) == (nx, ny)
    assert t.a == pytest.approx(dx) and t.e == pytest.approx(-dx)
    # North-west corner = SW mass point - half a cell in x, + (ny - 0.5) cells in y.
    assert t.c == pytest.approx(x0 - dx / 2, abs=1.0)
    assert t.f == pytest.approx(y0 + (ny - 0.5) * dx, abs=1.0)
    assert g.max_georeference_error_m < 5.0
    assert g.crs.ellipsoid.semi_major_metre == pytest.approx(6370000.0)
    # Geolocation is flipped to north-up along with the data.
    assert g.lat[0, 0] > g.lat[-1, 0]


def test_inconsistent_latlon_is_rejected():
    p = PROJECTIONS[1]
    lon, lat, _, _ = grid_lonlat(p, 30, 20, 12000.0, 12000.0)
    attrs = dict(p, DX=12000.0, DY=12000.0)
    attrs["TRUELAT1"] = 45.0  # metadata no longer matches the coordinates
    with pytest.raises(GeoreferenceError, match="not regular"):
        wrf_grid(attrs, lat, lon)


def test_unsupported_projection():
    with pytest.raises(GeoreferenceError, match="MAP_PROJ=6"):
        wrf_crs({"MAP_PROJ": 6})


def test_geotiff_georeferencing_and_metadata(wrf_run, tmp_path):
    adapter = WRFAdapter()
    (ds,) = adapter.discover(wrf_run)
    fld = adapter.compute(ds, "precipitation", 2)
    var = get_variable("precipitation")
    info = write_geotiff(tmp_path / "p.tif", fld, ds.grid, var)

    with rasterio.open(tmp_path / "p.tif") as src:
        assert src.crs == ds.grid.crs
        assert src.transform.almost_equals(ds.grid.transform)
        assert (src.width, src.height) == (ds.grid.width, ds.grid.height)
        assert src.dtypes[0] == "float32"
        assert np.isnan(src.nodata)
        assert src.units[0] == "mm"
        assert src.tags(1)["standard_name"] == "precipitation_amount"
        assert src.tags(1)["valid_time"] == "2026-10-06T02:00:00Z"
        assert src.tags(1)["max_error"] == "0.01"
        data = src.read(1)
        np.testing.assert_allclose(data, fld.data.astype("f4"), rtol=0,
                                   atol=var.max_error * 1.00001)  # + float32 rounding

        # The centre of pixel (row, col) must land on WRF's XLAT/XLONG for that
        # mass point (row 0 of the GeoTIFF is the northernmost WRF row).
        to_ll = Transformer.from_crs(src.crs, src.crs.geodetic_crs, always_xy=True)
        for row, col in [(0, 0), (5, 17), (src.height - 1, src.width - 1)]:
            x, y = src.xy(row, col)
            lon, lat = to_ll.transform(x, y)
            assert lon == pytest.approx(ds.grid.lon[row, col], abs=1e-4)
            assert lat == pytest.approx(ds.grid.lat[row, col], abs=1e-4)
    assert info.statistics["maximum"] == pytest.approx(float(np.nanmax(fld.data)), rel=1e-6)


@pytest.mark.parametrize("max_error", [None, 0.1, 1.0])
def test_geotiff_keeps_values_within_max_error(tmp_path, wrf_run, max_error):
    import dataclasses

    from modelatlasify.core.model import FieldResult

    (ds,) = WRFAdapter().discover(wrf_run)
    rng = np.random.default_rng(0)
    # Noisy (compresses badly) and never negative, like precipitation: LERC
    # stores values relative to each block's minimum, so zeros stay exactly
    # zero when nothing in the block is below zero.
    data = 10.0 * np.abs(rng.standard_normal(ds.grid.shape))
    data[:3, :] = np.nan  # missing stays missing
    data[3:6, :] = 0.0  # exact zeros stay zero
    fld = FieldResult("t2m", data, T0)
    var = dataclasses.replace(get_variable("t2m"), max_error=max_error)
    write_geotiff(tmp_path / "t.tif", fld, ds.grid, var)
    with rasterio.open(tmp_path / "t.tif") as src:
        out = src.read(1).astype(np.float64)
        assert src.compression.name == ("deflate" if max_error is None else "lerc_zstd")
        assert ("max_error" in src.tags(1)) == (max_error is not None)
    assert np.array_equal(np.isnan(out), np.isnan(data))
    assert (out[3:6] == 0.0).all()
    err = np.nanmax(np.abs(out - data.astype("f4")))
    assert err == 0.0 if max_error is None else err <= max_error * 1.00001  # + float32 rounding


def test_every_geotiff_variable_has_a_deliberate_max_error():
    from modelatlasify.core.variables import all_variables

    for v in all_variables().values():
        if v.draws is not None:  # map-only product: no GeoTIFF of its own
            continue
        if v.name == "spray_mass_flux":  # log scale over 1e-7..1e-2: stays lossless
            assert v.max_error is None
            continue
        assert v.max_error is not None and v.max_error > 0, v.name
        if v.levels:  # well below the colour-level spacing ...
            assert v.max_error <= 0.2 * min(np.diff(v.levels)), v.name
        if v.transparent_below:  # ... and any transparency threshold
            assert v.max_error <= 0.1 * v.transparent_below, v.name


def test_footprint_simple(wrf_run):
    (ds,) = WRFAdapter().discover(wrf_run)
    fp = grid_footprint(ds.grid)
    w, s, e, n = fp.bbox
    assert not fp.crosses_antimeridian and fp.geometry["type"] == "Polygon"
    assert w < ds.grid.lon.min() and e > ds.grid.lon.max()  # outer edges, beyond mass points
    assert s < ds.grid.lat.min() and n > ds.grid.lat.max()


def test_footprint_across_antimeridian(tmp_path):
    # A Mercator grid centred on 156W that spans > 180 degrees of longitude.
    write_wrfout(tmp_path / "r", hourly(T0, 1), map_proj=3, nx=165, ny=70, dx=120000.0)
    (ds,) = WRFAdapter().discover(tmp_path / "r")
    fp = grid_footprint(ds.grid)
    assert fp.crosses_antimeridian
    assert fp.geometry["type"] == "MultiPolygon"
    w, _, e, _ = fp.bbox
    assert w > e  # GeoJSON convention for antimeridian-crossing boxes
    assert 100 < w < 120 and -75 < e < -55
