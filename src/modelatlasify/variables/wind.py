"""Surface wind speed: definition and model-independent calculations."""

from __future__ import annotations

import numpy as np

from modelatlasify.core.variables import VariableDefinition, register

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
        aliases=("wind", "wind_speed", "surface_wind"),
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
