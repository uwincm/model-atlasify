"""Quick-look PNG maps of model fields.

The map is drawn in the grid's own projection, so the field is placed with
exactly the same CRS and affine transform as the GeoTIFF (no resampling and no
lat/lon approximation). The extent always comes from the data's grid (for
composites, the outermost grid; nested grids are placed on it exactly).
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
from pyproj import CRS, Transformer  # noqa: E402

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
    return _draw_map(path, [(field.data, grid)], variable, title, subtitle, dpi=dpi)


def plot_composite(
    path: Path, base: tuple[np.ndarray, GridDefinition],
    nests: list[tuple[np.ndarray | None, GridDefinition]], variable: VariableDefinition,
    title: str, subtitle: str = "", dpi: int = 110,
) -> Path:
    """Map of the base grid's whole area with nested grids drawn on top (finest
    last) and their boundaries outlined in black. A nest with ``data=None`` is
    drawn as an outline only."""
    return _draw_map(path, [base, *nests], variable, title, subtitle,
                     outlines=[g for _, g in nests], dpi=dpi)


def _draw_map(
    path: Path, layers: list[tuple[np.ndarray | None, GridDefinition]],
    variable: VariableDefinition, title: str, subtitle: str = "",
    outlines: list[GridDefinition] = (), dpi: int = 110,
) -> Path:
    """Draw ``layers`` (base grid first) over the base grid's extent."""
    base_grid = layers[0][1]
    proj = cartopy_projection(base_grid.crs)
    xmin, ymin, xmax, ymax = base_grid.native_bounds
    aspect = (xmax - xmin) / (ymax - ymin)
    width_in = 11.0
    map_h = min(max(width_in / aspect, 3.0), 10.0)
    fig = plt.figure(figsize=(width_in, map_h + 1.6))
    ax = fig.add_axes([0.06, 0.20, 0.88, 0.66], projection=proj)
    ax.set_extent((xmin, xmax, ymin, ymax), crs=proj)
    ax.set_facecolor("#e9eef2")

    drawn = [(np.asarray(d, dtype=np.float64), g) for d, g in layers if d is not None]
    norm, cmap = _norm_and_cmap(variable, np.concatenate([d.ravel() for d, _ in drawn]))
    img = None
    for k, (data, grid) in enumerate(drawn):
        masked = np.ma.masked_invalid(data)
        if variable.transparent_below is not None:
            masked = np.ma.masked_less(masked, variable.transparent_below)
        # Hide this layer wherever a finer layer is drawn on top of it.
        for _, finer in drawn[k + 1:]:
            masked = np.ma.masked_where(_inside(grid, finer), masked)
        style = dict(cmap=cmap, norm=norm, zorder=1 + 0.1 * k)
        extent = _extent_in(grid, base_grid.crs)
        if extent is not None:
            artist = ax.imshow(masked, origin="upper", extent=extent, transform=proj,
                               interpolation="nearest", **style)
        else:  # not axis-aligned in the base CRS: place each cell exactly
            xs, ys = _corners_in(grid, base_grid.crs)
            artist = ax.pcolormesh(xs, ys, masked, transform=proj, shading="flat", **style)
        img = img if img is not None else artist
    _add_geography(ax, extent_m=_grid_extent_m(base_grid))
    for grid in outlines:
        bx, by = _boundary_in(grid, base_grid.crs)
        ax.plot(bx, by, color="black", linewidth=1.2, transform=proj, zorder=3)
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


# ------------------------------------------------- placing grids on another
def _transformer(src: CRS, dst: CRS) -> Transformer | None:
    return None if src == dst else Transformer.from_crs(src, dst, always_xy=True)


def _apply(grid: GridDefinition, cols: np.ndarray, rows: np.ndarray):
    """Pixel (col, row) -> CRS (x, y) via the grid's affine transform."""
    t = grid.transform
    return t.a * cols + t.b * rows + t.c, t.d * cols + t.e * rows + t.f


def _corners_in(grid: GridDefinition, crs: CRS) -> tuple[np.ndarray, np.ndarray]:
    """Pixel-corner coordinates of ``grid`` in ``crs``, shape (height+1, width+1)."""
    rows, cols = np.mgrid[0:grid.height + 1, 0:grid.width + 1]
    xs, ys = _apply(grid, cols, rows)
    tr = _transformer(grid.crs, crs)
    if tr is not None:
        xs, ys = tr.transform(xs, ys)
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def _extent_in(grid: GridDefinition, crs: CRS) -> tuple[float, float, float, float] | None:
    """imshow extent (left, right, bottom, top) of ``grid`` in ``crs`` if the grid
    stays a north-up, axis-aligned regular grid there (e.g. same CRS, or WRF
    Mercator nests that differ only in central meridian); else None."""
    if _transformer(grid.crs, crs) is None:
        xmin, ymin, xmax, ymax = grid.native_bounds
        return (xmin, xmax, ymin, ymax)
    xs, ys = _corners_in(grid, crs)
    if not (np.all(np.isfinite(xs)) and np.all(np.isfinite(ys))):
        return None
    t = grid.transform
    tol = 1e-3 * min(abs(t.a), abs(t.e))
    left, top = xs[0, 0], ys[0, 0]
    dx = (xs[0, -1] - left) / grid.width
    dy = (ys[-1, 0] - top) / grid.height
    rows, cols = np.mgrid[0:grid.height + 1, 0:grid.width + 1]
    if (dx <= 0 or dy >= 0 or np.abs(xs - (left + cols * dx)).max() > tol
            or np.abs(ys - (top + rows * dy)).max() > tol):
        return None
    return (left, left + dx * grid.width, top + dy * grid.height, top)


def _inside(grid: GridDefinition, other: GridDefinition) -> np.ndarray:
    """Boolean (height, width): which pixel centres of ``grid`` lie within ``other``."""
    rows, cols = np.mgrid[0:grid.height, 0:grid.width]
    xs, ys = _apply(grid, cols + 0.5, rows + 0.5)
    tr = _transformer(grid.crs, other.crs)
    if tr is not None:
        xs, ys = tr.transform(xs, ys)
    xmin, ymin, xmax, ymax = other.native_bounds
    return (xs >= xmin) & (xs <= xmax) & (ys >= ymin) & (ys <= ymax)


def _boundary_in(grid: GridDefinition, crs: CRS, n: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Closed outer boundary of ``grid`` in ``crs``, densified to follow curvature."""
    xmin, ymin, xmax, ymax = grid.native_bounds
    s = np.linspace(0.0, 1.0, n, endpoint=False)
    xs = np.concatenate([xmin + s * (xmax - xmin), np.full(n, xmax),
                         xmax - s * (xmax - xmin), np.full(n, xmin), [xmin]])
    ys = np.concatenate([np.full(n, ymin), ymin + s * (ymax - ymin),
                         np.full(n, ymax), ymax - s * (ymax - ymin), [ymin]])
    tr = _transformer(grid.crs, crs)
    if tr is not None:
        xs, ys = tr.transform(xs, ys)
    return np.asarray(xs), np.asarray(ys)


def _geography_features(extent_m: float) -> list:
    """Natural Earth land, coastlines and borders at a scale suited to the extent."""
    res = "10m" if extent_m < 800e3 else "50m" if extent_m < 6000e3 else "110m"
    return [
        (cfeature.NaturalEarthFeature("physical", "land", res),
         dict(facecolor="#f4f1ea", edgecolor="none", zorder=0)),
        (cfeature.NaturalEarthFeature("physical", "coastline", res),
         dict(facecolor="none", edgecolor="0.15", linewidth=0.6, zorder=2)),
        (cfeature.BORDERS.with_scale(res),
         dict(facecolor="none", edgecolor="0.3", linewidth=0.4, zorder=2)),
    ]


def _load_geography(features: list) -> bool:
    """Force cartopy to download/read the features now; False if unavailable."""
    global _coastline_warning_shown
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # DownloadWarning noise
            for feature, _ in features:
                next(iter(feature.geometries()), None)
    except Exception as exc:  # noqa: BLE001 - offline: plot without coastlines
        if not _coastline_warning_shown:
            log.warning("Could not load Natural Earth coastlines (%s); maps will not show them.", exc)
            _coastline_warning_shown = True
        return False
    return True


def _grid_extent_m(grid: GridDefinition) -> float:
    xmin, ymin, xmax, ymax = grid.native_bounds
    return max(xmax - xmin, ymax - ymin)


def prefetch_geography(grid: GridDefinition) -> None:
    """Download the Natural Earth data a map of ``grid`` needs, once, before
    parallel workers start (concurrent first downloads can corrupt the cache)."""
    _load_geography(_geography_features(_grid_extent_m(grid)))


def _add_geography(ax, extent_m: float) -> None:
    """Land, coastlines and borders from Natural Earth (downloaded once by cartopy)."""
    features = _geography_features(extent_m)
    if not _load_geography(features):
        return
    for feature, style in features:
        ax.add_feature(feature, **style)
