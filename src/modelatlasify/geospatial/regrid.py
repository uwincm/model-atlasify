"""Nearest-cell resampling of a curvilinear lon/lat grid onto a regular lon/lat grid.

For model output whose grid is not regular in any known projection (only its
cell-centre lon/lat are written). Each regular cell takes the value of the
model cell it falls in, as found by inverting the model's lon/lat over a
triangulation of its own cells: values are never interpolated, and points
outside the model's outer edge stay missing.
"""

from __future__ import annotations

import matplotlib.tri as mtri
import numpy as np
from affine import Affine
from pyproj import CRS

from modelatlasify.core.errors import GeoreferenceError
from modelatlasify.core.model import GridDefinition


def lonlat_crs(radius_m: float) -> CRS:
    """Geographic lon/lat CRS on a sphere of the given radius."""
    return CRS.from_proj4(f"+proj=longlat +R={radius_m:.0f} +no_defs +type=crs")


def _cell_corners(a: np.ndarray) -> np.ndarray:
    """(ny+1, nx+1) cell-corner values of cell-centre values ``a`` (ny, nx),
    extrapolating linearly by half a cell at the edges."""
    p = np.pad(a, 1, mode="edge")
    p[0, 1:-1], p[-1, 1:-1] = 2 * a[0] - a[1], 2 * a[-1] - a[-2]
    p[1:-1, 0], p[1:-1, -1] = 2 * a[:, 0] - a[:, 1], 2 * a[:, -1] - a[:, -2]
    p[0, 0], p[0, -1] = 2 * p[1, 1] - p[2, 2], 2 * p[1, -2] - p[2, -3]
    p[-1, 0], p[-1, -1] = 2 * p[-2, 1] - p[-3, 2], 2 * p[-2, -2] - p[-3, -3]
    return 0.25 * (p[:-1, :-1] + p[1:, :-1] + p[:-1, 1:] + p[1:, 1:])


def outer_edge(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Closed ring (lon, lat) along the outer cell edges of a curvilinear grid
    (lon continuous, i.e. not wrapped)."""
    clon, clat = _cell_corners(lon), _cell_corners(lat)

    def ring(c):
        return np.concatenate([c[0, :-1], c[:-1, -1], c[-1, :0:-1], c[:0:-1, 0], c[:1, 0]])

    return ring(clon), ring(clat)


def resample_to_lonlat(lon: np.ndarray, lat: np.ndarray, radius_m: float, description: str = ""
                       ) -> tuple[GridDefinition, np.ndarray]:
    """A regular, north-up lon/lat grid covering the curvilinear grid ``lon``/``lat``
    (2-D cell centres, any row order), at about the model's spacing, and the
    index map: for each regular cell, the flat index into the model arrays (as
    stored) of the model cell it falls in, or -1 outside the model.

    The grid's ``outline_lonlat`` is the model's true outer edge.
    """
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    if lon.ndim != 2 or lon.shape != lat.shape or min(lon.shape) < 2:
        raise GeoreferenceError("lon/lat must be 2-D arrays of identical shape, at least 2x2.")
    if not (np.all(np.isfinite(lon)) and np.all(np.isfinite(lat))):
        raise GeoreferenceError("lon/lat contain missing values; cannot resample.")
    ny, nx = lon.shape
    # Longitudes relative to the domain centre, continuous across the antimeridian.
    lon0 = float(lon[ny // 2, nx // 2])
    rel = (lon - lon0 + 180.0) % 360.0 - 180.0

    dlon = float(np.median(np.abs(np.diff(rel, axis=1))))
    dlat = float(np.median(np.abs(np.diff(lat, axis=0))))
    if not (dlon > 0 and dlat > 0):
        raise GeoreferenceError("lon/lat do not vary along the grid's rows and columns.")
    west, east = float(rel.min()), float(rel.max())
    south, north = float(lat.min()), float(lat.max())
    width = int(round((east - west) / dlon)) + 1
    height = int(round((north - south) / dlat)) + 1
    tlon = west + dlon * np.arange(width)  # cell centres, relative to lon0
    tlat = north - dlat * np.arange(height)  # north-up
    glon, glat = np.meshgrid(tlon, tlat)

    # Invert the model grid: fractional (col, row) at each regular cell centre,
    # linear over two triangles per model cell; masked outside the model.
    idx = np.arange(ny * nx).reshape(ny, nx)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()
    tri = mtri.Triangulation(rel.ravel(), lat.ravel(),
                             np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)]))
    jj, ii = np.mgrid[0:ny, 0:nx]
    col = mtri.LinearTriInterpolator(tri, ii.ravel().astype(np.float64))(glon, glat)
    row = mtri.LinearTriInterpolator(tri, jj.ravel().astype(np.float64))(glon, glat)
    outside = np.ma.getmaskarray(col) | np.ma.getmaskarray(row)
    col = np.clip(np.rint(np.ma.filled(col, 0.0)), 0, nx - 1).astype(np.int64)
    row = np.clip(np.rint(np.ma.filled(row, 0.0)), 0, ny - 1).astype(np.int64)
    index = np.where(outside, -1, row * nx + col).astype(np.int32)

    edge_lon, edge_lat = outer_edge(rel, lat)
    grid = GridDefinition(
        crs=lonlat_crs(radius_m),
        transform=Affine(dlon, 0.0, lon0 + west - dlon / 2.0, 0.0, -dlat, north + dlat / 2.0),
        width=width, height=height,
        lat=glat, lon=(glon + lon0 + 180.0) % 360.0 - 180.0,
        description=description or (
            f"{width} x {height} regular lon/lat grid, {dlon:.4g} x {dlat:.4g} deg, resampled "
            f"(nearest model cell) from a {nx} x {ny} curvilinear grid"),
        max_georeference_error_m=None,
        outline_lonlat=(edge_lon + lon0, edge_lat),
    )
    return grid, index
