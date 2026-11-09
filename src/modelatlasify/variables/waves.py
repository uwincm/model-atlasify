"""Ocean surface wave variables (from a wave model such as DWM)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

from modelatlasify.core.variables import VariableDefinition, register

_SWH_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0)


def get_colormap_swh() -> colors.ListedColormap:
    """"turbo", one colour per interval of _SWH_LEVELS plus the "max" arrow
    (the colours turbo gives on those levels), with the lowest band (0-0.5 m)
    white so calm seas and the wind vectors over them stand out."""
    norm = colors.BoundaryNorm(_SWH_LEVELS, ncolors=256, extend="max")
    mids = [*((a + b) / 2 for a, b in zip(_SWH_LEVELS, _SWH_LEVELS[1:])), _SWH_LEVELS[-1] + 1]
    rgba = plt.get_cmap("turbo")(norm(np.array(mids)))
    rgba[0] = (1.0, 1.0, 1.0, 1.0)
    return colors.ListedColormap(rgba, name="turbo_swh")


_PERIOD_LEVELS = tuple(float(x) for x in range(0, 13, 1))  # 0-12 s; longer periods use the "max" arrow

SIGNIFICANT_WAVE_HEIGHT = register(
    VariableDefinition(
        name="significant_wave_height",
        long_name="Significant wave height",
        units="m",
        standard_name="sea_surface_wave_significant_height",
        description=(
            "Significant wave height (4 times the square root of the wave energy "
            "spectrum's zeroth moment), instantaneous at the valid time. Missing over land."
        ),
        cmap=get_colormap_swh(),
        levels=_SWH_LEVELS,
        extend="max",
        vectors=("u10", "v10"),  # 10 m wind vectors on the maps
        aliases=("swh", "hs", "wave_height", "waves"),
    )
)

MEAN_WAVE_PERIOD = register(
    VariableDefinition(
        name="mean_wave_period",
        long_name="Mean wave period",
        units="s",
        standard_name="sea_surface_wave_mean_period",
        description="Mean wave period of the wave spectrum, instantaneous at the valid time. "
                    "Missing over land.",
        cmap="viridis",
        levels=_PERIOD_LEVELS,
        extend="max",
        vectors=("u10", "v10"),  # 10 m wind vectors on the maps
        aliases=("mwp", "waves"),
    )
)

DOMINANT_WAVE_PERIOD = register(
    VariableDefinition(
        name="dominant_wave_period",
        long_name="Dominant wave period",
        units="s",
        standard_name="sea_surface_wave_period_at_variance_spectral_density_maximum",
        description="Period of the most energetic (peak) part of the wave spectrum, "
                    "instantaneous at the valid time. Missing over land.",
        cmap="viridis",
        levels=_PERIOD_LEVELS,
        extend="max",
        vectors=("u10", "v10"),  # 10 m wind vectors on the maps
        aliases=("dwp", "peak_wave_period", "waves"),
    )
)
