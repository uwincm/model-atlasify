"""Recover a regular projected grid from 2-D longitude/latitude arrays.

For models that write only cell-centre lon/lat (no projection metadata), the
grid is identified by trying candidate CRSs: in the right one, x depends only
on the column and y only on the row, each with a constant step. The worst
residual from that regular grid is kept, and a candidate is rejected if it
exceeds a fraction of a cell, so a GeoTIFF is never written in the wrong place.
"""

from __future__ import annotations

import numpy as np
from affine import Affine
from pyproj import CRS, Transformer

from modelatlasify.core.errors import GeoreferenceError
from modelatlasify.core.model import GridDefinition

#: Reject a fit whose worst cell centre is off by more than this fraction of a cell.
MAX_RESIDUAL_FRACTION = 0.1


def fit_regular_grid(crs: CRS, lon: np.ndarray, lat: np.ndarray, description: str = ""
                     ) -> GridDefinition:
    """North-up grid in ``crs`` whose cell centres are ``lon``/``lat``.

    ``lon``/``lat`` are 2-D (rows, cols); row order may be south-up or
    north-up (the result is always north-up). Raises GeoreferenceError if the
    points are not a regular grid in ``crs``.
    """
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    if lon.ndim != 2 or lon.shape != lat.shape:
        raise GeoreferenceError("lon/lat must be 2-D arrays of identical shape.")
    ny, nx = lon.shape
    if nx < 2 or ny < 2:
        raise GeoreferenceError(f"A {nx}x{ny} grid is too small to georeference.")
    x, y = Transformer.from_crs(crs.geodetic_crs, crs, always_xy=True).transform(lon, lat)
    x, y = np.asarray(x), np.asarray(y)
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise GeoreferenceError(f"Some lon/lat points cannot be projected with {crs.srs}.")

    jj, ii = np.mgrid[0:ny, 0:nx]
    dx = float(np.mean(np.diff(x, axis=1)))
    dy = float(np.mean(np.diff(y, axis=0)))
    if dx <= 0 or dy == 0:
        raise GeoreferenceError("Grid columns do not increase eastward in the candidate CRS.")
    x0 = float(np.mean(x - ii * dx))
    y0 = float(np.mean(y - jj * dy))
    residual = float(np.max(np.hypot(x - (x0 + ii * dx), y - (y0 + jj * dy))))
    if residual > MAX_RESIDUAL_FRACTION * min(abs(dx), abs(dy)):
        raise GeoreferenceError(
            f"lon/lat are not a regular grid in {crs.srs}: off by up to {residual:.0f} m "
            f"(cells {dx:.0f} x {abs(dy):.0f} m)."
        )

    south_up = dy > 0
    dy = abs(dy)
    west = x0 - dx / 2.0
    north = (y0 + (ny - 1) * dy if south_up else y0) + dy / 2.0
    flip = (lambda a: a[::-1].copy()) if south_up else (lambda a: a.copy())
    return GridDefinition(
        crs=crs, transform=Affine(dx, 0.0, west, 0.0, -dy, north), width=nx, height=ny,
        lat=flip(lat), lon=flip((lon + 180.0) % 360.0 - 180.0),
        description=description or f"{nx} x {ny} regular grid, {dx:.0f} m x {dy:.0f} m",
        max_georeference_error_m=residual,
    )
