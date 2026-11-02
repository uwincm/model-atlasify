"""Surface wind speed: definition and model-independent calculations."""

from __future__ import annotations

import numpy as np
from pyproj import Transformer

from modelatlasify.core.model import GridDefinition
from modelatlasify.core.variables import VariableDefinition, register

#: Wind vectors on surface wind speed maps: arrow length on the ground per
#: speed, the speed at which arrows stop growing, the arrow spacing, and the
#: fewest arrows across a map (smaller maps get a closer spacing).
VECTOR_KM_PER_KNOT = 100.0 / 10.0
VECTOR_MAX_KNOTS = 70.0
VECTOR_SPACING_KM = 250.0
VECTOR_MIN_ACROSS = 20
KNOTS_PER_M_S = 3600.0 / 1852.0

SURFACE_WIND_SPEED = register(
    VariableDefinition(
        name="surface_wind_speed",
        long_name="Surface wind speed",
        units="m s-1",
        standard_name="wind_speed",
        description=(
            "Near-surface (10 m) wind speed, sqrt(u^2 + v^2), instantaneous at the "
            "valid time."
        ),
        cmap="Spectral_r",
        levels=tuple(float(x) for x in range(0, 32, 2)),
        extend="max",
        max_zero_fraction=0.01,  # needs U and V both exactly 0
        vectors=("u10", "v10"),
        aliases=("wind", "wind_speed", "surface_wind"),
    )
)

U10 = register(
    VariableDefinition(
        name="u10",
        long_name="Eastward 10 m wind",
        units="m s-1",
        standard_name="eastward_wind",
        description=(
            "Eastward (earth-relative) component of the near-surface (10 m) wind, "
            "instantaneous at the valid time."
        ),
        cmap="RdBu_r",
        vmin=-30.0,
        vmax=30.0,
        extend="both",
        maps=False,
        aliases=("wind", "wind_vectors"),
    )
)

V10 = register(
    VariableDefinition(
        name="v10",
        long_name="Northward 10 m wind",
        units="m s-1",
        standard_name="northward_wind",
        description=(
            "Northward (earth-relative) component of the near-surface (10 m) wind, "
            "instantaneous at the valid time."
        ),
        cmap="RdBu_r",
        vmin=-30.0,
        vmax=30.0,
        extend="both",
        maps=False,
        aliases=("wind", "wind_vectors"),
    )
)


def wind_speed(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Magnitude of a horizontal wind vector.

    The result is independent of whether (u, v) are grid-relative or
    earth-relative, because rotating a vector does not change its length, so no
    map-rotation correction is needed for speed.
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    return np.hypot(u, v)


def destagger(data: np.ndarray, axis: int) -> np.ndarray:
    """Average a staggered (cell-face) field to cell centres along ``axis``.

    For an Arakawa-C grid the face values at i and i+1 bound mass point i, so
    the centred value is their mean; the result is one shorter along ``axis``.
    """
    data = np.asarray(data, dtype=np.float64)
    n = data.shape[axis]
    lower = np.take(data, np.arange(0, n - 1), axis=axis)
    upper = np.take(data, np.arange(1, n), axis=axis)
    return 0.5 * (lower + upper)


def grid_rotation(grid: GridDefinition) -> np.ndarray:
    """Angle (radians, counter-clockwise) from local east to the grid's +x
    (column) axis at every cell centre, shape (height, width), north-up.

    Computed from the grid's CRS by stepping a small distance along +x and
    measuring the direction of that step on the ground, so it holds for any
    projection (and for each position of a moving nest).
    """
    t = grid.transform
    rows, cols = np.mgrid[0:grid.height, 0:grid.width] + 0.5
    x = t.a * cols + t.b * rows + t.c
    y = t.d * cols + t.e * rows + t.f
    step = 0.01 * abs(t.a)
    to_lonlat = Transformer.from_crs(grid.crs, grid.crs.geodetic_crs, always_xy=True)
    lon0, lat0 = to_lonlat.transform(x, y)
    lon1, lat1 = to_lonlat.transform(x + step, y)
    dlon = (np.asarray(lon1) - np.asarray(lon0) + 180.0) % 360.0 - 180.0
    east = np.radians(dlon) * np.cos(np.radians(lat0))
    north = np.radians(np.asarray(lat1) - np.asarray(lat0))
    return np.arctan2(north, east)


def earth_relative(u: np.ndarray, v: np.ndarray, grid: GridDefinition
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Rotate grid-relative wind components (north-up arrays on ``grid``) to
    eastward/northward components. Speeds are unchanged."""
    theta = grid_rotation(grid)
    c, s = np.cos(theta), np.sin(theta)
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    return u * c - v * s, u * s + v * c
