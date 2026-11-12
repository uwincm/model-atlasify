"""Pressure-level (upper-air) fields: relative humidity, geopotential height
and wind at 700 and 500 hPa.

Relative humidity maps show RH shaded with height contours and wind vectors;
geopotential height maps show height shaded with its own contours and wind
vectors (per-item maps only). The wind components are GeoTIFF-only.
"""

from __future__ import annotations

import matplotlib
import numpy as np
from matplotlib import colors

from modelatlasify.core.variables import ContourStyle, VariableDefinition, register

#: Pressure levels (hPa) with products.
PRESSURE_LEVELS_HPA = (700, 500)

#: The "rh" NCL colour table of the UW realtime WRF plots (rh.rgb), one colour
#: per interval of RH_LEVELS plus below/above: < 5 %, 5-10 %, ..., 90-95 %, > 95 %.
RH_RGB = (
    (250, 250, 250), (240, 240, 240), (230, 230, 230), (220, 220, 220), (230, 255, 230),
    (210, 255, 210), (180, 255, 180), (160, 255, 160), (120, 255, 120), (110, 255, 110),
    (100, 255, 100), (90, 255, 90), (70, 200, 70), (30, 200, 30), (10, 170, 10),
    (0, 160, 0), (0, 150, 20), (0, 150, 10), (0, 100, 50), (255, 255, 0),
)
#: 5 to 95 % every 5 %.
RH_LEVELS = tuple(float(x) for x in range(5, 100, 5))

#: Shading levels of geopotential height (m), one contour interval apart.
HEIGHT_LEVELS = {
    700: tuple(float(x) for x in range(2980, 3220, 20)),
    500: tuple(float(x) for x in range(5720, 5960, 20)),
}
#: Height contours: every 20 m (2 dam), labelled in dam, lightly smoothed.
HEIGHT_CONTOURS = ContourStyle(interval=20.0, label_scale=0.1, label_units="dam",
                               color="blue", smooth_passes=5)


def get_colormap_height() -> colors.Colormap:
    """The light (yellow to orange) part of YlOrBr, so blue contours stand out."""
    return colors.LinearSegmentedColormap.from_list(
        "height", matplotlib.colormaps["YlOrBr"](np.linspace(0.0, 0.6, 256)))


def get_colormap_rh() -> colors.ListedColormap:
    rgb = np.array(RH_RGB) / 255.0
    return colors.ListedColormap(rgb[1:-1], name="rh").with_extremes(under=rgb[0], over=rgb[-1])


def names(level: int) -> dict[str, str]:
    """Product names at one level: rh, height, u, v."""
    return {"rh": f"rh_{level}", "height": f"geopotential_height_{level}",
            "u": f"u_{level}", "v": f"v_{level}"}


for _lev in PRESSURE_LEVELS_HPA:
    _n = names(_lev)
    _common = ("upper_air", "pressure_levels", f"{_lev}hpa", f"{_lev}mb")
    _wind = (_n["u"], _n["v"])
    register(VariableDefinition(
        name=_n["rh"],
        long_name=f"{_lev} hPa relative humidity",
        units="%",
        standard_name="relative_humidity",
        description=(
            f"Relative humidity (with respect to liquid water) at {_lev} hPa, interpolated "
            "linearly in log pressure from the model levels; instantaneous at the valid "
            "time. Missing where the level is below ground. Maps also show "
            f"{_lev} hPa geopotential height contours and wind vectors."
        ),
        cmap=get_colormap_rh(),
        levels=RH_LEVELS,
        extend="both",
        overlay=_n["height"],
        vectors=_wind,
        constant_is_suspicious=True,
        aliases=(*_common, "rh", "relative_humidity"),
    ))
    register(VariableDefinition(
        name=_n["height"],
        long_name=f"{_lev} hPa geopotential height",
        units="m",
        standard_name="geopotential_height",
        description=(
            f"Geopotential height of the {_lev} hPa surface, interpolated linearly in log "
            "pressure from the model levels; instantaneous at the valid time. Missing "
            "where the level is below ground. Maps also show its contours and the "
            f"{_lev} hPa wind vectors."
        ),
        cmap=get_colormap_height(),
        levels=HEIGHT_LEVELS[_lev],
        extend="both",
        overlay=_n["height"],  # its own contours on top of the shading
        contour=HEIGHT_CONTOURS,
        vectors=_wind,
        composite=False,
        aliases=(*_common, "geopotential_height", "height"),
    ))
    for _comp, _dir in (("u", "Eastward"), ("v", "Northward")):
        register(VariableDefinition(
            name=_n[_comp],
            long_name=f"{_dir} {_lev} hPa wind",
            units="m s-1",
            standard_name=f"{_dir.lower()}_wind",
            description=(
                f"{_dir} (earth-relative) component of the wind at {_lev} hPa, interpolated "
                "linearly in log pressure from the model levels; instantaneous at the valid "
                "time. Missing where the level is below ground."
            ),
            cmap="RdBu_r",
            vmin=-40.0,
            vmax=40.0,
            extend="both",
            maps=False,
            aliases=(*_common, "upper_winds"),
        ))


# ------------------------------------------------------------ calculations
def relative_humidity(qv: np.ndarray, p_pa: np.ndarray, t_k: np.ndarray) -> np.ndarray:
    """RH (%) with respect to liquid water from the water vapour mixing ratio
    (kg kg-1), pressure (Pa) and temperature (K), as WRF's own diagnostics
    (wrf-python ``rh``, NCL ``wrf_rh``) compute it; limited to 0-100 %."""
    es = 611.2 * np.exp(17.67 * (t_k - 273.15) / (t_k - 29.65))  # Pa (Bolton 1980)
    qvs = 0.622 * es / (p_pa - (1.0 - 0.622) * es)
    return np.clip(100.0 * qv / qvs, 0.0, 100.0)


def interp_to_pressure(fields: list[np.ndarray], p_pa: np.ndarray, level_pa: float
                       ) -> list[np.ndarray]:
    """Interpolate 3-D fields (level, y, x) on the pressure ``p_pa`` (same
    shape, decreasing with level index) to the pressure ``level_pa``, linearly
    in log pressure. Columns where the level is below the lowest model level
    (below ground) or above the top are NaN."""
    lnp = np.log(p_pa)
    target = np.log(level_pa)
    nz = p_pa.shape[0]
    k = np.sum(p_pa >= level_pa, axis=0) - 1  # last level at or below (in height) the target
    ok = (k >= 0) & (k < nz - 1)
    kk = np.clip(k, 0, nz - 2)[None]
    lo = np.take_along_axis(lnp, kk, axis=0)[0]
    hi = np.take_along_axis(lnp, kk + 1, axis=0)[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        w = (lo - target) / (lo - hi)
    out = []
    for f in fields:
        a = np.take_along_axis(f, kk, axis=0)[0]
        b = np.take_along_axis(f, kk + 1, axis=0)[0]
        out.append(np.where(ok, a + w * (b - a), np.nan))
    return out
