"""2 m air temperature: definition and colour table."""

from __future__ import annotations

import numpy as np
from matplotlib import colors

from modelatlasify.core.variables import VariableDefinition, register

#: NCL colour table "t2m_29lev" (from the MeteoSwiss NCL library), 30 colours:
#: https://www.ncl.ucar.edu/Document/Graphics/ColorTables/t2m_29lev.shtml
T2M_29LEV_RGB = (
    (109, 227, 255), (175, 240, 255), (255, 196, 226), (255, 153, 204), (255, 0, 255),
    (128, 0, 128), (0, 0, 128), (70, 70, 255), (51, 102, 255), (133, 162, 255),
    (255, 255, 255), (204, 204, 204), (179, 179, 179), (153, 153, 153), (96, 96, 96),
    (128, 128, 0), (0, 92, 0), (0, 128, 0), (51, 153, 102), (157, 213, 0),
    (212, 255, 91), (255, 255, 0), (255, 184, 112), (255, 153, 0), (255, 102, 0),
    (255, 0, 0), (188, 75, 0), (171, 0, 56), (128, 0, 0), (163, 112, 255),
)

#: Every 1.5 C, counting down from 35 C to -10 C: 31 levels, 30 intervals, one
#: colour of T2M_29LEV_RGB per interval.
T2M_LEVELS = tuple(float(round(35.0 - 1.5 * k, 1)) for k in range(30, -1, -1))


def get_colormap_t2m() -> colors.ListedColormap:
    """t2m_29lev, one colour per interval of T2M_LEVELS; values below/above
    the levels get the first/last colour (the colour bar's end arrows)."""
    rgb = np.array(T2M_29LEV_RGB) / 255.0
    return colors.ListedColormap(rgb, name="t2m_29lev").with_extremes(under=rgb[0], over=rgb[-1])


T2M = register(
    VariableDefinition(
        name="t2m",
        long_name="2 m temperature",
        units="degC",
        standard_name="air_temperature",
        description=(
            "Air temperature at 2 m above the surface, instantaneous at the valid time. "
            "Maps also show the 10 m wind vectors (u10, v10)."
        ),
        cmap=get_colormap_t2m(),
        levels=T2M_LEVELS,
        extend="both",
        vectors=("u10", "v10"),
        aliases=("temperature", "t2", "2m_temperature"),
    )
)
