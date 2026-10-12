"""Quick-look PNG maps of model fields.

A map shows a *view*: either a grid's whole area in the grid's own projection
(per-item maps and "full" composites) or a lon/lat sub-area in PlateCarree.
Fields are never resampled: in a grid's own projection they are placed with
exactly the GeoTIFF's CRS and affine transform; elsewhere every cell is drawn
as its true quadrilateral (pcolormesh on the projected cell corners).
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import cartopy.crs as ccrs  # noqa: E402
import cartopy.feature as cfeature  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from pyproj import CRS, Transformer  # noqa: E402

from modelatlasify.core.areas import SubArea  # noqa: E402
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


@dataclass(frozen=True)
class MapView:
    """What a map shows. ``crs`` is the native CRS for a grid's own-projection
    view, or None for a lon/lat view, whose x is longitude - ``center_lon``."""

    projection: ccrs.Projection
    extent: tuple[float, float, float, float]  # (x0, x1, y0, y1) in projection coords
    extent_m: float  # size on the ground, for choosing the coastline detail
    crs: CRS | None = None
    center_lon: float = 0.0


def full_view(grid: GridDefinition) -> MapView:
    """The grid's whole area in its own projection."""
    xmin, ymin, xmax, ymax = grid.native_bounds
    return MapView(cartopy_projection(grid.crs), (xmin, xmax, ymin, ymax),
                   _grid_extent_m(grid), crs=grid.crs)


def subarea_view(area: SubArea) -> MapView:
    """A lon/lat box in PlateCarree, centred on the box (antimeridian-safe)."""
    c = area.center_lon
    mid_lat = math.radians((area.south + area.north) / 2.0)
    r = 6371e3
    width_m = math.radians(area.east - area.west) * r * math.cos(mid_lat)
    height_m = math.radians(area.north - area.south) * r
    return MapView(ccrs.PlateCarree(central_longitude=c),
                   (area.west - c, area.east - c, area.south, area.north),
                   max(width_m, height_m), crs=None, center_lon=c)


def plot_field(
    path: Path, field: FieldResult, grid: GridDefinition, variable: VariableDefinition,
    title: str, subtitle: str = "", dpi: int = 110,
) -> Path:
    return _draw_map(path, full_view(grid), [(field.data, grid)], variable, title, subtitle,
                     dpi=dpi)


def plot_composite(
    path: Path, base: tuple[np.ndarray, GridDefinition],
    nests: list[tuple[np.ndarray | None, GridDefinition]], variable: VariableDefinition,
    title: str, subtitle: str = "", area: SubArea | None = None, dpi: int = 110,
) -> Path:
    """Map of the base grid's whole area (or of ``area``, in lon/lat) with nested
    grids drawn on top (finest last) and their boundaries outlined in black. A
    nest with ``data=None`` is drawn as an outline only."""
    view = full_view(base[1]) if area is None else subarea_view(area)
    return _draw_map(path, view, [base, *nests], variable, title, subtitle,
                     outlines=[g for _, g in nests], dpi=dpi)


def _draw_map(
    path: Path, view: MapView, layers: list[tuple[np.ndarray | None, GridDefinition]],
    variable: VariableDefinition, title: str, subtitle: str = "",
    outlines: list[GridDefinition] = (), dpi: int = 110,
) -> Path:
    """Draw ``layers`` (coarsest first) in ``view``."""
    proj = view.projection
    x0, x1, y0, y1 = view.extent
    aspect = (x1 - x0) / (y1 - y0)
    width_in = 11.0
    map_h = min(max(width_in / aspect, 3.0), 10.0)
    fig = plt.figure(figsize=(width_in, map_h + 1.6))
    ax = fig.add_axes([0.06, 0.20, 0.88, 0.66], projection=proj)
    ax.set_extent(view.extent, crs=proj)
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
        extent = _extent_in(grid, view.crs) if view.crs is not None else None
        if extent is not None:
            artist = ax.imshow(masked, origin="upper", extent=extent, transform=proj,
                               interpolation="nearest", **style)
        else:  # not axis-aligned in the view: place each cell exactly
            xs, ys = _corners_in_view(grid, view)
            window = _crop_window(xs, ys, view)
            if window is None:
                continue  # this layer is nowhere in the view
            r0, r1, c0, c1 = window
            artist = ax.pcolormesh(xs[r0:r1 + 1, c0:c1 + 1], ys[r0:r1 + 1, c0:c1 + 1],
                                   masked[r0:r1, c0:c1], transform=proj, shading="flat",
                                   **style)
        img = img if img is not None else artist
    if img is None:  # nothing in view; keep the colour bar
        img = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    _add_geography(ax, extent_m=view.extent_m)
    for grid in outlines:
        bx, by = _to_view(view, grid.crs, *_boundary_points(grid))
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


def _boundary_points(grid: GridDefinition, n: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Closed outer boundary of ``grid`` in its own CRS, densified to follow
    curvature once projected elsewhere."""
    xmin, ymin, xmax, ymax = grid.native_bounds
    s = np.linspace(0.0, 1.0, n, endpoint=False)
    xs = np.concatenate([xmin + s * (xmax - xmin), np.full(n, xmax),
                         xmax - s * (xmax - xmin), np.full(n, xmin), [xmin]])
    ys = np.concatenate([np.full(n, ymin), ymin + s * (ymax - ymin),
                         np.full(n, ymax), ymax - s * (ymax - ymin), [ymin]])
    return xs, ys


def _to_view(view: MapView, crs: CRS, xs: np.ndarray, ys: np.ndarray
             ) -> tuple[np.ndarray, np.ndarray]:
    """Points in ``crs`` -> the view's projection coordinates."""
    if view.crs is not None:
        tr = _transformer(crs, view.crs)
        if tr is not None:
            xs, ys = tr.transform(xs, ys)
        return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)
    lon, lat = Transformer.from_crs(crs, crs.geodetic_crs, always_xy=True).transform(xs, ys)
    # Longitude relative to the view centre, in [-180, 180): no wrap inside a view.
    x = (np.asarray(lon, dtype=np.float64) - view.center_lon + 180.0) % 360.0 - 180.0
    return x, np.asarray(lat, dtype=np.float64)


def _corners_in_view(grid: GridDefinition, view: MapView) -> tuple[np.ndarray, np.ndarray]:
    """Pixel-corner coordinates of ``grid`` in the view, shape (height+1, width+1)."""
    rows, cols = np.mgrid[0:grid.height + 1, 0:grid.width + 1]
    return _to_view(view, grid.crs, *_apply(grid, cols, rows))


def _crop_window(xs: np.ndarray, ys: np.ndarray, view: MapView
                 ) -> tuple[int, int, int, int] | None:
    """Cell window (row0, row1, col0, col1), end-exclusive, holding every cell
    that can overlap the view, given the cell-corner coordinates; None if the
    grid is nowhere in it. A cell overlapping the view has a corner within one
    cell size of it, so the view is widened by the largest cell first."""
    finite = np.isfinite(xs) & np.isfinite(ys)
    if not finite.any():
        return None
    with np.errstate(invalid="ignore"):
        cell = max(np.nanmax(np.abs(np.diff(xs, axis=1))), np.nanmax(np.abs(np.diff(ys, axis=0))),
                   np.nanmax(np.abs(np.diff(xs, axis=0))), np.nanmax(np.abs(np.diff(ys, axis=1))))
    x0, x1, y0, y1 = view.extent
    near = finite & (xs >= x0 - cell) & (xs <= x1 + cell) & (ys >= y0 - cell) & (ys <= y1 + cell)
    if not near.any():
        return None
    rows = np.flatnonzero(near.any(axis=1))
    cols = np.flatnonzero(near.any(axis=0))
    h, w = xs.shape[0] - 1, xs.shape[1] - 1
    return (max(rows[0] - 1, 0), min(rows[-1] + 1, h), max(cols[0] - 1, 0), min(cols[-1] + 1, w))


def view_overlaps(view: MapView, grid: GridDefinition) -> bool:
    """Whether any of ``grid`` is inside ``view``."""
    return _crop_window(*_corners_in_view(grid, view), view) is not None


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


def prefetch_geography(extent: GridDefinition | SubArea) -> None:
    """Download the Natural Earth data a map of a grid (or sub-area) needs, once,
    before parallel workers start (concurrent first downloads can corrupt the
    cache)."""
    extent_m = (_grid_extent_m(extent) if isinstance(extent, GridDefinition)
                else subarea_view(extent).extent_m)
    _load_geography(_geography_features(extent_m))


def _add_geography(ax, extent_m: float) -> None:
    """Land, coastlines and borders from Natural Earth (downloaded once by cartopy)."""
    features = _geography_features(extent_m)
    if not _load_geography(features):
        return
    for feature, style in features:
        ax.add_feature(feature, **style)
