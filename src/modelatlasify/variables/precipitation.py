"""Precipitation: definitions and model-independent calculations."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

from modelatlasify.core.variables import VariableDefinition, register


def get_colormap_precip():

    cmap = plt.cm.jet
    N_keep = 15
    cmap2 = colors.LinearSegmentedColormap.from_list(name='custom', colors=cmap(range(cmap.N)), N=N_keep)
    color_list1 = cmap2(range(cmap2.N))

    ## Modify individual colors here.
    color_list11 = cmap2(range(cmap2.N))
    color_list11[0] = [1,1,1,1] # Make first color white.
    color_list11[1] = [0.7,0.7,0.7, 1] # Make first color gray.
    color_list11[2] = [0.0, 0.9, 1.0, 1.0]
    color_list11[3] = [0.0/255.0, 156.0/255.0, 255.0/255.0, 1.0]
    color_list11[4] = [0.0/255.0, 58.0/255.0, 255.0/255.0, 1.0]
    color_list11[5] = [0.0, 0.5, 0.0, 1.]
    color_list11[6] = [0.0, 0.8, 0.0, 1.]
    color_list11[12] = color_list1[13]  # Move top to colors of jet color map down.
    color_list11[13] = color_list1[14] # Move top to colors of jet color map down.
    color_list11[14] = [1,0,1,1] # Magenta on top.

    ## Brighten
    import colorsys
    scale_l = 1.5
    max_l = 0.7
    for n in range(9, len(color_list11)):  ## Don't touch first two colors.
        h, l, s = colorsys.rgb_to_hls(*color_list11[n][0:3])
        color_list11[n][0:3] = colorsys.hls_to_rgb(h, min(max_l, l * scale_l), s = s)

    ## Add color map to the dict.
    CMAP = colors.LinearSegmentedColormap.from_list(name='custom', colors=color_list11, N=N_keep)

    return CMAP


PRECIPITATION = register(
    VariableDefinition(
        name="precipitation",
        long_name="Accumulated precipitation",
        units="mm",
        standard_name="precipitation_amount",
        cell_methods="time: sum",
        description=(
            "Total precipitation (liquid-water equivalent) accumulated from the "
            "start of the domain's integration (see start_datetime) to the valid time."
        ),
        cmap=get_colormap_precip(), #"YlGnBu",
        levels=(0.0, 0.1, 1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500),
        extend="max",
        transparent_below=0.1,
        constant_is_suspicious=False,
        aliases=("precip", "precipitation", "rain", "precipitation_accumulated"),
    )
)

PRECIPITATION_INTERVAL = register(
    VariableDefinition(
        name="precipitation_interval",
        long_name="Precipitation over output interval",
        units="mm",
        standard_name="precipitation_amount",
        cell_methods="time: sum",
        description=(
            "Total precipitation (liquid-water equivalent) that fell between the "
            "previous output time and the valid time, computed as the difference "
            "of consecutive accumulated totals."
        ),
        cmap=get_colormap_precip(), #"YlGnBu",
        levels=(0.0, 0.1, 0.25, 0.5, 1, 2, 3, 5, 7.5, 10, 15, 20, 30, 50, 75),
        extend="max",
        transparent_below=0.1,
        constant_is_suspicious=False,
        aliases=("precip", "precipitation", "rain", "precipitation_rate"),
    )
)

#: Differences more negative than this (mm) are treated as a broken accumulation
#: (e.g. a reset) rather than float round-off.
NEGATIVE_TOLERANCE_MM = 0.01


def interval_precipitation(
    accum_end: np.ndarray, accum_start: np.ndarray
) -> tuple[np.ndarray, int]:
    """Precipitation between two times from two accumulated totals.

    Returns ``(amount, n_negative)``. Small negative differences produced by
    float32 round-off (> -NEGATIVE_TOLERANCE_MM) are clipped to 0. Larger
    negative differences cannot be physical; they are set to NaN and counted
    so the caller can report them instead of silently hiding them.
    """
    diff = np.asarray(accum_end, dtype=np.float64) - np.asarray(accum_start, dtype=np.float64)
    bad = diff < -NEGATIVE_TOLERANCE_MM
    out = np.where(diff < 0, 0.0, diff)
    out[bad] = np.nan
    return out, int(np.count_nonzero(bad))
