"""Quick-look PNG maps of model fields.

The map is drawn in the grid's own projection, so the field is placed with
exactly the same CRS and affine transform as the GeoTIFF (no resampling and no
lat/lon approximation). The extent always comes from the data's grid.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import cartopy.crs as ccrs  # noqa: E402
import cartopy.feature as cfeature  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from pyproj import CRS  # noqa: E402

from modelatlasify.core.model import FieldResult, GridDefinition  # noqa: E402
from modelatlasify.core.variables import VariableDefinition  # noqa: E402

log = logging.getLogger(__name__)
_coastline_warning_shown = False


def cartopy_projection(crs: CRS) -> ccrs.Projection:
    """Cartopy projection equivalent to a (spherical) projected pyproj CRS.

    Supports the projections the model adapters currently emit; parameters are
    copied one-to-one so cartopy and the GeoTIFF agree exactly.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # "may lose information": fine for these
        p = crs.to_dict()
    if "R" in p:
        globe = ccrs.Globe(ellipse=None, semimajor_axis=p["R"], semiminor_axis=p["R"])
    else:
        globe = ccrs.Globe(ellipse=p.get("ellps", "WGS84"), datum=p.get("datum"))
    proj = p.get("proj")
    if proj == "merc":
        return ccrs.Mercator(
            central_longitude=p.get("lon_0", 0.0), latitude_true_scale=p.get("lat_ts", 0.0),
            false_easting=p.get("x_0", 0.0), false_northing=p.get("y_0", 0.0), globe=globe,
            min_latitude=-85.0, max_latitude=85.0,
        )
    if proj == "lcc":
        return ccrs.LambertConformal(
            central_longitude=p.get("lon_0", 0.0), central_latitude=p.get("lat_0", 0.0),
            standard_parallels=(p.get("lat_1"), p.get("lat_2", p.get("lat_1"))),
            false_easting=p.get("x_0", 0.0), false_northing=p.get("y_0", 0.0), globe=globe,
            cutoff=-30 if p.get("lat_1", 0) >= 0 else 30,
        )
    if proj == "stere":
        return ccrs.Stereographic(
            central_latitude=p.get("lat_0", 90.0), central_longitude=p.get("lon_0", 0.0),
            true_scale_latitude=p.get("lat_ts"), false_easting=p.get("x_0", 0.0),
            false_northing=p.get("y_0", 0.0), globe=globe,
        )
    raise ValueError(f"No cartopy equivalent implemented for projection '{proj}'.")


def _norm_and_cmap(variable: VariableDefinition, data: np.ndarray):
    cmap = plt.get_cmap(variable.cmap).with_extremes(bad=(0.0, 0.0, 0.0, 0.0))
    if variable.levels:
        return mcolors.BoundaryNorm(variable.levels, ncolors=cmap.N, extend=variable.extend), cmap
    finite = data[np.isfinite(data)]
    vmin = variable.vmin if variable.vmin is not None else (float(finite.min()) if finite.size else 0.0)
    vmax = variable.vmax if variable.vmax is not None else (float(finite.max()) if finite.size else 1.0)
    return mcolors.Normalize(vmin=vmin, vmax=vmax), cmap


def plot_field(
    path: Path, field: FieldResult, grid: GridDefinition, variable: VariableDefinition,
    title: str, subtitle: str = "", dpi: int = 110,
) -> Path:
    data = np.asarray(field.data, dtype=np.float64)
    masked = np.ma.masked_invalid(data)
    if variable.transparent_below is not None:
        masked = np.ma.masked_less(masked, variable.transparent_below)

    proj = cartopy_projection(grid.crs)
    xmin, ymin, xmax, ymax = grid.native_bounds
    aspect = (xmax - xmin) / (ymax - ymin)
    width_in = 11.0
    map_h = min(max(width_in / aspect, 3.0), 10.0)
    fig = plt.figure(figsize=(width_in, map_h + 1.6))
    ax = fig.add_axes([0.06, 0.20, 0.88, 0.66], projection=proj)
    ax.set_extent((xmin, xmax, ymin, ymax), crs=proj)
    ax.set_facecolor("#e9eef2")

    norm, cmap = _norm_and_cmap(variable, data)
    img = ax.imshow(
        masked, origin="upper", extent=(xmin, xmax, ymin, ymax), transform=proj,
        cmap=cmap, norm=norm, interpolation="nearest", zorder=1,
    )
    _add_geography(ax, extent_m=max(xmax - xmin, ymax - ymin))
    try:
        gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="0.4", alpha=0.6, linestyle="--")
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = gl.ylabel_style = {"size": 8}
    except Exception:  # noqa: BLE001 - labels unsupported for some projections
        ax.gridlines(linewidth=0.4, color="0.4", alpha=0.6, linestyle="--")

    cax = fig.add_axes([0.2, 0.08, 0.6, 0.03])
    cb = fig.colorbar(img, cax=cax, orientation="horizontal", extend=variable.extend,
                      spacing="uniform")
    cb.set_label(f"{variable.long_name} ({variable.units})", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    if variable.levels:
        cb.set_ticks(variable.levels)
        cb.set_ticklabels([f"{v:g}" for v in variable.levels])

    fig.text(0.06, 0.95, title, fontsize=12, fontweight="bold", va="top")
    if subtitle:
        fig.text(0.06, 0.905, subtitle, fontsize=9, va="top", color="0.25")
    fig.text(0.94, 0.95, "ModelAtlasify", fontsize=8, ha="right", va="top", color="0.5")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def _add_geography(ax, extent_m: float) -> None:
    """Land, coastlines and borders from Natural Earth (downloaded once by cartopy)."""
    global _coastline_warning_shown
    res = "10m" if extent_m < 800e3 else "50m" if extent_m < 6000e3 else "110m"
    features = [
        (cfeature.NaturalEarthFeature("physical", "land", res),
         dict(facecolor="#f4f1ea", edgecolor="none", zorder=0)),
        (cfeature.NaturalEarthFeature("physical", "coastline", res),
         dict(facecolor="none", edgecolor="0.15", linewidth=0.6, zorder=2)),
        (cfeature.BORDERS.with_scale(res),
         dict(facecolor="none", edgecolor="0.3", linewidth=0.4, zorder=2)),
    ]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # DownloadWarning noise
            for feature, _ in features:
                next(iter(feature.geometries()), None)  # force download/read now
    except Exception as exc:  # noqa: BLE001 - offline: plot without coastlines
        if not _coastline_warning_shown:
            log.warning("Could not load Natural Earth coastlines (%s); maps will not show them.", exc)
            _coastline_warning_shown = True
        return
    for feature, style in features:
        ax.add_feature(feature, **style)
