"""Ocean surface wave variables (from a wave model such as DWM)."""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

with warnings.catch_warnings():
    # cmocean 3.0.x builds its colour maps with an argument matplotlib 3.11
    # deprecates; harmless, and not ours to fix.
    warnings.filterwarnings("ignore", message="Passing 'N' to ListedColormap is deprecated")
    import cmocean

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
# Dominant (peak) periods run longer than mean periods (long-period swell).
_DOMINANT_PERIOD_LEVELS = tuple(float(x) for x in range(0, 19, 2))  # 0-18 s, every 2 s


def get_colormap_period(levels: tuple[float, ...], name: str) -> colors.ListedColormap:
    """cmocean "tempo" reversed (dark for short periods, light for long-period
    swell), one colour per interval of ``levels`` plus the "max" arrow.
    Colours are taken evenly along the full colour map with one extra step,
    and the last (tempo's near-white end) is dropped so the longest periods
    and the arrow stay visible against the pale background."""
    n = len(levels)  # intervals + the "max" arrow
    rgba = cmocean.cm.tempo_r(np.linspace(0.0, 1.0, n + 1))[:-1]
    return colors.ListedColormap(rgba, name=name)


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
        # cmocean "tempo" (Thyng et al. 2016), reversed, no white; as for dwp
        cmap=get_colormap_period(_PERIOD_LEVELS, "tempo_r_mwp"),
        levels=_PERIOD_LEVELS,
        extend="max",
        vectors=("u10", "v10"),  # 10 m wind vectors on the maps
        directions="mean_wave_direction",  # and which way the waves go
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
        # cmocean "tempo" (Thyng et al. 2016), reversed, no white
        cmap=get_colormap_period(_DOMINANT_PERIOD_LEVELS, "tempo_r_dwp"),
        levels=_DOMINANT_PERIOD_LEVELS,
        extend="max",
        vectors=("u10", "v10"),  # 10 m wind vectors on the maps
        directions="dominant_wave_direction",  # and which way the waves go
        aliases=("dwp", "peak_wave_period", "waves"),
    )
)

DOMINANT_WAVE_DIRECTION = register(
    VariableDefinition(
        name="dominant_wave_direction",
        long_name="Dominant wave direction",
        units="degree",
        standard_name="sea_surface_wave_from_direction_at_variance_spectral_density_maximum",
        description="Direction the most energetic (peak) part of the wave spectrum comes "
                    "from, in degrees clockwise from true north (0 = from the north, 90 = from "
                    "the east), instantaneous at the valid time. Missing over land. Drawn as "
                    "arrows on the dominant wave period maps.",
        cmap="twilight",  # cyclic
        vmin=0.0,
        vmax=360.0,
        extend="neither",
        maps=False,  # a direction reads better as arrows (on the dwp maps) than as colours
        aliases=("dwd", "peak_wave_direction", "waves"),
    )
)

MEAN_WAVE_DIRECTION = register(
    VariableDefinition(
        name="mean_wave_direction",
        long_name="Mean wave direction",
        units="degree",
        standard_name="sea_surface_wave_from_direction",
        description="Mean direction the wave spectrum comes from, in degrees clockwise from "
                    "true north (0 = from the north, 90 = from the east), instantaneous at the "
                    "valid time. Missing over land. Drawn as arrows on the mean wave period maps.",
        cmap="twilight",  # cyclic
        vmin=0.0,
        vmax=360.0,
        extend="neither",
        maps=False,  # a direction reads better as arrows (on the mwp maps) than as colours
        aliases=("mwd", "waves"),
    )
)
