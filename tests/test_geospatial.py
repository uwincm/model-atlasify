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
        data = src.read(1)
        np.testing.assert_allclose(data, fld.data.astype("f4"))

        # The centre of pixel (row, col) must land on WRF's XLAT/XLONG for that
        # mass point (row 0 of the GeoTIFF is the northernmost WRF row).
        to_ll = Transformer.from_crs(src.crs, src.crs.geodetic_crs, always_xy=True)
        for row, col in [(0, 0), (5, 17), (src.height - 1, src.width - 1)]:
            x, y = src.xy(row, col)
            lon, lat = to_ll.transform(x, y)
            assert lon == pytest.approx(ds.grid.lon[row, col], abs=1e-4)
            assert lat == pytest.approx(ds.grid.lat[row, col], abs=1e-4)
    assert info.statistics["maximum"] == pytest.approx(float(np.nanmax(fld.data)), rel=1e-6)


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
