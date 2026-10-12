"""WRF map projection -> CRS + affine transform.

Approach
--------
1. Build the CRS from the WRF global attributes (MAP_PROJ, TRUELAT1/2,
   STAND_LON, ...). WRF does all of its map math on a sphere of radius
   6 370 000 m, so the CRS uses that sphere (``+R=6370000``), not WGS84. The
   latitudes and longitudes WRF writes (XLAT/XLONG) are coordinates on that
   sphere; following common practice (wrf-python, salem) they are treated as
   numerically equal to WGS84 latitude/longitude, i.e. no datum shift.
2. The grid is regular in that CRS with spacing DX, DY (true at the standard
   latitude(s)). The grid origin is not stored explicitly, so it is fitted from
   the mass-point XLAT/XLONG: every point is projected and the origin is the
   mean of ``x - i*DX`` and ``y - j*DY``. The worst residual between the
   projected XLAT/XLONG and the fitted regular grid is kept as a check; if it
   exceeds a fraction of a grid cell the projection is rejected rather than
   producing a GeoTIFF that "opens" but is in the wrong place.
3. WRF arrays are stored south-to-north (row 0 = south). GeoTIFFs are
   north-up, so arrays are flipped and the affine transform points at the
   north-west *corner* of the north-west cell (half a cell beyond the outermost
   mass point).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from affine import Affine
from pyproj import CRS, Transformer

from modelatlasify.core.errors import GeoreferenceError
from modelatlasify.core.model import GridDefinition

WRF_EARTH_RADIUS_M = 6370000.0

#: Reject the fitted grid if any mass point lands farther than this fraction
#: of a grid cell from where XLAT/XLONG says it is.
MAX_RESIDUAL_FRACTION = 0.1

MAP_PROJ_NAMES = {
    0: "idealized (no projection)",
    1: "Lambert Conformal",
    2: "Polar Stereographic",
    3: "Mercator",
    6: "Latitude-Longitude (cylindrical equidistant)",
}


def _wrap180(lon: float) -> float:
    return ((float(lon) + 180.0) % 360.0) - 180.0


def wrf_crs(attrs: Mapping[str, Any]) -> CRS:
    """CRS for a WRF grid, from its global attributes."""
    try:
        map_proj = int(attrs["MAP_PROJ"])
    except KeyError as exc:
        raise GeoreferenceError("WRF file has no MAP_PROJ global attribute.") from exc

    def need(name: str) -> float:
        if name not in attrs:
            raise GeoreferenceError(
                f"MAP_PROJ={map_proj} requires the '{name}' global attribute, "
                "which is missing."
            )
        return float(attrs[name])

    sphere = f"+R={WRF_EARTH_RADIUS_M:.0f} +units=m +no_defs +type=crs"

    if map_proj == 1:  # Lambert conformal conic
        lat1, lat2 = need("TRUELAT1"), need("TRUELAT2")
        lon0 = _wrap180(need("STAND_LON"))
        # lat_0 only shifts the false origin, which is fitted anyway; use the
        # mother domain centre as WRF/WPS do.
        lat0 = float(attrs.get("MOAD_CEN_LAT", attrs.get("CEN_LAT", lat1)))
        proj = f"+proj=lcc +lat_1={lat1} +lat_2={lat2} +lat_0={lat0} +lon_0={lon0}"
    elif map_proj == 2:  # Polar stereographic; hemisphere from sign of TRUELAT1
        lat1 = need("TRUELAT1")
        lon0 = _wrap180(need("STAND_LON"))
        pole = 90.0 if lat1 >= 0 else -90.0
        proj = f"+proj=stere +lat_0={pole} +lat_ts={lat1} +lon_0={lon0}"
    elif map_proj == 3:  # Mercator
        lat1 = need("TRUELAT1")
        # Mercator x is linear in longitude, so any central meridian gives the
        # same grid once the origin is fitted. Centring it on the domain keeps
        # domains that straddle the antimeridian from wrapping.
        lon0 = _wrap180(need("CEN_LON"))
        proj = f"+proj=merc +lat_ts={lat1} +lon_0={lon0} +x_0=0 +y_0=0"
    else:
        name = MAP_PROJ_NAMES.get(map_proj, "unknown")
        raise GeoreferenceError(
            f"WRF MAP_PROJ={map_proj} ({name}) is not supported yet. Supported: "
            "1 (Lambert Conformal), 2 (Polar Stereographic), 3 (Mercator)."
        )
    return CRS.from_proj4(f"{proj} {sphere}")


def wrf_grid(
    attrs: Mapping[str, Any], xlat: np.ndarray, xlong: np.ndarray
) -> GridDefinition:
    """Build the north-up :class:`GridDefinition` for a WRF mass grid.

    ``xlat``/``xlong`` are the 2-D mass-point coordinates exactly as stored in
    the file (row 0 = south).
    """
    crs = wrf_crs(attrs)
    try:
        dx, dy = float(attrs["DX"]), float(attrs["DY"])
    except KeyError as exc:
        raise GeoreferenceError("WRF file is missing the DX/DY global attributes.") from exc
    if dx <= 0 or dy <= 0:
        raise GeoreferenceError(f"Invalid WRF grid spacing DX={dx}, DY={dy}.")

    xlat = np.asarray(xlat, dtype=np.float64)
    xlong = np.asarray(xlong, dtype=np.float64)
    if xlat.ndim != 2 or xlat.shape != xlong.shape:
        raise GeoreferenceError("XLAT/XLONG must be 2-D arrays of identical shape.")
    ny, nx = xlat.shape

    to_native = Transformer.from_crs(crs.geodetic_crs, crs, always_xy=True)
    x, y = to_native.transform(xlong, xlat)
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise GeoreferenceError(
            "Some XLAT/XLONG points cannot be projected with the CRS derived "
            f"from the WRF attributes ({crs.srs})."
        )

    jj, ii = np.mgrid[0:ny, 0:nx]
    x0 = float(np.mean(x - ii * dx))  # x of mass point (i=0, j=0), south-west
    y0 = float(np.mean(y - jj * dy))
    residual = np.hypot(x - (x0 + ii * dx), y - (y0 + jj * dy))
    max_err = float(residual.max())
    if max_err > MAX_RESIDUAL_FRACTION * min(dx, dy):
        raise GeoreferenceError(
            "The WRF grid is not regular in the CRS derived from its metadata: "
            f"XLAT/XLONG deviate from the fitted grid by up to {max_err:.0f} m "
            f"(DX={dx:.0f} m). Refusing to write mis-located GeoTIFFs. "
            f"CRS: {crs.srs}"
        )

    # North-up transform: top-left corner of the top-left cell.
    west = x0 - dx / 2.0
    north = y0 + (ny - 1) * dy + dy / 2.0
    transform = Affine(dx, 0.0, west, 0.0, -dy, north)

    map_proj = int(attrs["MAP_PROJ"])
    description = (
        f"WRF {MAP_PROJ_NAMES.get(map_proj, map_proj)} grid, {nx} x {ny} mass "
        f"points, DX={dx:g} m, DY={dy:g} m, on a sphere of radius "
        f"{WRF_EARTH_RADIUS_M:.0f} m"
    )
    lon_north_up = ((xlong[::-1] + 180.0) % 360.0) - 180.0
    return GridDefinition(
        crs=crs,
        transform=transform,
        width=nx,
        height=ny,
        lat=xlat[::-1].copy(),
        lon=lon_north_up,
        description=description,
        max_georeference_error_m=max_err,
    )
