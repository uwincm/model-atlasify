"""Quick-look PNG maps of model fields.

A map shows a *view*: either a grid's whole area in the grid's own projection
(per-item maps and "full" composites) or a lon/lat box in PlateCarree: a
``--plot-sub-areas`` box, or the whole area of a grid whose projection the
model did not record (``GridDefinition.lonlat_view_buffer_deg``).
Fields are never resampled: in a grid's own projection they are placed with
exactly the GeoTIFF's CRS and affine transform; elsewhere every cell is drawn
as its true quadrilateral (pcolormesh on the projected cell corners).
"""

from __future__ import annotations

import io
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

from modelatlasify.core import storage  # noqa: E402
from modelatlasify.core.areas import FULL_AREA, SubArea  # noqa: E402
from modelatlasify.core.model import FieldResult, GridDefinition  # noqa: E402
from modelatlasify.core.variables import VariableDefinition, VectorStyle  # noqa: E402
from modelatlasify.variables.wind import WIND_VECTORS  # noqa: E402

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
        if len(variable.levels) - 1 == cmap.N:
            # Exactly one colour per interval (e.g. t2m_29lev): out-of-range
            # values take the colour map's under/over colours.
            return mcolors.BoundaryNorm(variable.levels, ncolors=cmap.N), cmap
        return mcolors.BoundaryNorm(variable.levels, ncolors=cmap.N, extend=variable.extend), cmap
    finite = data[np.isfinite(data)]
    vmin = variable.vmin if variable.vmin is not None else (float(finite.min()) if finite.size else 0.0)
    vmax = variable.vmax if variable.vmax is not None else (float(finite.max()) if finite.size else 1.0)
    return mcolors.Normalize(vmin=vmin, vmax=vmax), cmap


#: Layers of one field, coarsest first: (data or None = outline only, grid).
Layers = list[tuple[np.ndarray | None, GridDefinition]]
#: A second field drawn over the first (e.g. precipitation over OLR).
Overlay = tuple[VariableDefinition, Layers]
#: Wind vectors, coarsest grid first: (eastward, northward) in m s-1, north-up.
VectorLayers = list[tuple[np.ndarray, np.ndarray, GridDefinition]]


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
    """The grid's whole area: in its own projection, or in a lon/lat box
    around it (plus a buffer) when ``grid.uses_lonlat_view``."""
    if grid.uses_lonlat_view:
        return subarea_view(lonlat_box(grid))
    xmin, ymin, xmax, ymax = grid.native_bounds
    return MapView(cartopy_projection(grid.crs), (xmin, xmax, ymin, ymax),
                   _grid_extent_m(grid), crs=grid.crs)


def outline_lonlat(grid: GridDefinition) -> tuple[np.ndarray, np.ndarray]:
    """The grid's closed outer edge in lon/lat (the model's true edge for a
    resampled grid)."""
    if grid.outline_lonlat is not None:
        lon, lat = grid.outline_lonlat
        return np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)
    lon, lat = Transformer.from_crs(grid.crs, grid.crs.geodetic_crs, always_xy=True).transform(
        *_boundary_points(grid))
    return np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)


def lonlat_box(grid: GridDefinition) -> SubArea:
    """Lon/lat box around the grid's outer edge, widened by
    ``grid.lonlat_view_buffer_deg`` on each side (antimeridian-safe)."""
    lon, lat = outline_lonlat(grid)
    lon = np.degrees(np.unwrap(np.radians(lon)))
    b = grid.lonlat_view_buffer_deg or 0.0
    west, east = float(lon.min()) - b, float(lon.max()) + b
    if east - west >= 360.0:
        west, east = -180.0, 180.0
    shift = 360.0 * math.floor((west + 180.0) / 360.0)
    return SubArea(FULL_AREA, west - shift, east - shift,
                   max(float(lat.min()) - b, -90.0), min(float(lat.max()) + b, 90.0))


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
    overlay: tuple[VariableDefinition, FieldResult | None] | None = None,
    vectors: VectorLayers | None = None, vector_style: VectorStyle | None = None,
) -> Path:
    """Map of one field over its grid's whole area. ``overlay`` is another
    field on the same grid drawn on top (None field = nothing to overlay);
    ``vectors`` = (eastward, northward) wind layers (on this grid or others,
    coarsest first), drawn as arrows."""
    over = None
    if overlay is not None and overlay[1] is not None:
        over = (overlay[0], [(overlay[1].data, grid)])
    # In a lon/lat box the grid's edge is inside the map: outline it.
    return _draw_map(path, full_view(grid), [(field.data, grid)], variable, title, subtitle,
                     outlines=[grid] if grid.uses_lonlat_view else [], dpi=dpi, overlay=over,
                     vectors=vectors, vector_style=vector_style)


def plot_composite(
    path: Path, base: tuple[np.ndarray, GridDefinition],
    nests: list[tuple[np.ndarray | None, GridDefinition]], variable: VariableDefinition,
    title: str, subtitle: str = "", area: SubArea | None = None, domain_edge: bool = False,
    dpi: int = 110, overlay: Overlay | None = None, vectors: VectorLayers | None = None,
    vector_style: VectorStyle | None = None,
) -> Path:
    """Map of the base grid's whole area (or of ``area``, in lon/lat) with nested
    grids drawn on top (finest last) and their boundaries outlined in black. A
    nest with ``data=None`` is drawn as an outline only. ``domain_edge`` also
    outlines the base grid itself: dashed, or solid like the nests when the
    base grid is drawn in a lon/lat box (``uses_lonlat_view``). ``overlay`` is
    a second field (base grid first, then the nests) drawn on top, and
    ``vectors`` wind arrows (each from the finest grid at its position)."""
    base_grid = base[1]
    view = full_view(base_grid) if area is None else subarea_view(area)
    solid_edge = base_grid.uses_lonlat_view
    return _draw_map(path, view, [base, *nests], variable, title, subtitle,
                     outlines=[g for _, g in nests] + ([base_grid] if solid_edge else []),
                     dashed_outlines=[base_grid] if domain_edge and not solid_edge else [],
                     dpi=dpi, overlay=overlay, vectors=vectors, vector_style=vector_style)


def _draw_map(
    path: Path, view: MapView, layers: Layers,
    variable: VariableDefinition, title: str, subtitle: str = "",
    outlines: list[GridDefinition] = (), dashed_outlines: list[GridDefinition] = (),
    dpi: int = 110, overlay: Overlay | None = None, vectors: VectorLayers | None = None,
    vector_style: VectorStyle | None = None,
) -> Path:
    """Draw ``layers`` (coarsest first) in ``view``, ``overlay`` and wind
    ``vectors`` on top."""
    proj = view.projection
    x0, x1, y0, y1 = view.extent
    aspect = (x1 - x0) / (y1 - y0)
    width_in = 11.0
    map_h = min(max(width_in / aspect, 3.0), 10.0)
    fig = plt.figure(figsize=(width_in, map_h + 1.6))
    ax = fig.add_axes([0.06, 0.20, 0.88, 0.66], projection=proj)
    ax.set_extent(view.extent, crs=proj)
    ax.set_facecolor("#e9eef2")

    img = _draw_layers(ax, view, layers, variable, zorder0=1.0)
    over_img = None
    if overlay is not None:
        over_img = _draw_layers(ax, view, overlay[1], overlay[0], zorder0=1.5)
    _add_geography(ax, extent_m=view.extent_m)
    for grids, style in ((outlines, "-"), (dashed_outlines, (0, (6, 4)))):
        for grid in grids:
            bx, by = _outline_in_view(view, grid)
            ax.plot(bx, by, color="black", linewidth=1.2, linestyle=style, transform=proj,
                    zorder=3)
    if vectors:
        _draw_vectors(fig, ax, view, vectors, vector_style or WIND_VECTORS)
    try:
        gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="0.4", alpha=0.6, linestyle="--")
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = gl.ylabel_style = {"size": 8}
    except Exception:  # noqa: BLE001 - labels unsupported for some projections
        ax.gridlines(linewidth=0.4, color="0.4", alpha=0.6, linestyle="--")

    if over_img is None:
        _colorbar(fig, [0.2, 0.08, 0.6, 0.03], img, variable)
    else:  # two colour bars side by side
        _colorbar(fig, [0.06, 0.08, 0.4, 0.03], img, variable)
        _colorbar(fig, [0.54, 0.08, 0.4, 0.03], over_img, overlay[0])

    fig.text(0.06, 0.95, title, fontsize=12, fontweight="bold", va="top")
    if subtitle:
        fig.text(0.06, 0.905, subtitle, fontsize=9, va="top", color="0.25")
    fig.text(0.94, 0.95, "ModelAtlasify", fontsize=8, ha="right", va="top", color="0.5")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    storage.write_bytes(path, buf.getvalue())  # a local file, or straight to S3
    return path


def _draw_layers(ax, view: MapView, layers: Layers, variable: VariableDefinition,
                 zorder0: float):
    """Draw one field's layers (coarsest first; each is hidden where a finer
    layer is drawn on top) and return a mappable for its colour bar."""
    proj = view.projection
    drawn = [(np.asarray(d, dtype=np.float64), g) for d, g in layers if d is not None]
    values = np.concatenate([d.ravel() for d, _ in drawn]) if drawn else np.array([np.nan])
    norm, cmap = _norm_and_cmap(variable, values)
    img = None
    for k, (data, grid) in enumerate(drawn):
        masked = np.ma.masked_invalid(data)
        if variable.transparent_below is not None:
            masked = np.ma.masked_less(masked, variable.transparent_below)
        if variable.transparent_above is not None:
            masked = np.ma.masked_greater(masked, variable.transparent_above)
        # Hide this layer wherever a finer layer is drawn on top of it.
        for _, finer in drawn[k + 1:]:
            masked = np.ma.masked_where(_inside(grid, finer), masked)
        style = dict(cmap=cmap, norm=norm, zorder=zorder0 + 0.01 * k)
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
    return img


def _draw_vectors(fig, ax, view: MapView, vectors: VectorLayers,
                  style: VectorStyle = WIND_VECTORS) -> None:
    """Arrows (wind, currents) on a square lattice spaced ``style.spacing_km``
    apart on the ground, or closer if that gives fewer than
    ``style.min_across`` arrows across the map (then exactly that many across,
    and as many rows as fit). Each is taken from the finest grid covering it
    (nearest cell). Arrows are ``style.km_per_unit`` long on the ground per
    unit of speed (scaled down in proportion when the spacing is closer, so
    arrows keep their size relative to the spacing), up to
    ``style.max_speed``, and point along the flow as it looks in the map's
    projection. With ``style.curly`` they are curly vectors instead (see
    :func:`_draw_curly`)."""
    x0, x1, y0, y1 = view.extent
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    _, north_c = _ground_axes(view, np.array([cx]), np.array([cy]))
    per_m = float(np.hypot(*north_c[:, 0]))  # view units per metre at the centre
    full_step = style.spacing_km * 1e3 * per_m
    step = min(full_step, (x1 - x0) / style.min_across)
    m_per_unit = style.km_per_unit * 1e3 * step / full_step  # ground metres per speed unit
    gx, gy = np.meshgrid(np.arange(x0 + step / 2, x1, step), np.arange(y0 + step / 2, y1, step))
    gx, gy = gx.ravel(), gy.ravel()
    lon, lat = _view_to_lonlat(view, gx, gy)
    if style.curly:
        _draw_curly(fig, ax, view, vectors, style, lon, lat, step, m_per_unit, per_m)
        return
    u = np.full(gx.shape, np.nan)
    v = np.full(gx.shape, np.nan)
    for ue, ve, grid in reversed(vectors):  # finest first
        todo = np.isnan(u) & np.isfinite(lon)
        if not todo.any():
            break
        x, y = Transformer.from_crs(grid.crs.geodetic_crs, grid.crs, always_xy=True).transform(
            lon[todo], lat[todo])
        t = grid.transform
        col = np.floor((np.asarray(x) - t.c) / t.a)
        row = np.floor((np.asarray(y) - t.f) / t.e)
        ok = np.isfinite(col) & np.isfinite(row) & (col >= 0) & (col < grid.width) \
            & (row >= 0) & (row < grid.height)
        idx = np.flatnonzero(todo)[ok]
        r, c = row[ok].astype(int), col[ok].astype(int)
        u[idx], v[idx] = np.asarray(ue)[r, c], np.asarray(ve)[r, c]
    speed = np.hypot(u, v)
    keep = np.isfinite(speed) & (speed > 0)
    if not keep.any():
        return
    gx, gy, u, v, speed = gx[keep], gy[keep], u[keep], v[keep], speed[keep]
    east, north = _ground_axes(view, gx, gy)
    length_m = np.minimum(speed * style.units_per_m_s, style.max_speed) * m_per_unit
    du = length_m * (u / speed * east[0] + v / speed * north[0])
    dv = length_m * (u / speed * east[1] + v / speed * north[1])
    q = ax.quiver(gx, gy, du, dv, angles="xy", scale_units="xy", scale=1, pivot="middle",
                  transform=view.projection, color="black", width=0.0016, headwidth=4,
                  headlength=4.5, headaxislength=4, zorder=2.5)
    # Key: the largest round speed whose arrow fits beside the colour bar.
    ax.apply_aspect()  # equal-aspect maps can be narrower than their axes box
    map_width_frac = ax.get_position().width  # as a fraction of the figure
    def key_frac(speed):
        return speed * m_per_unit * per_m / (x1 - x0) * map_width_frac
    key = next((s for s in style.key_speeds if key_frac(s) <= 0.08), style.key_speeds[-1])
    ax.quiverkey(q, 0.905, 0.095, key * m_per_unit * per_m,
                 f"{key:g} {style.units}\n(max {style.max_speed:g})", coordinates="figure",
                 labelpos="S", fontproperties={"size": 8})


#: Integration steps per curly vector (half upstream, half downstream).
CURLY_STEPS = 24


def _draw_curly(fig, ax, view: MapView, vectors: VectorLayers, style: VectorStyle,
                lon: np.ndarray, lat: np.ndarray, step: float, m_per_unit: float,
                per_m: float) -> None:
    """Curly vectors: through each lattice point (``lon``/``lat``), a short
    streamline of the flow, half upstream and half downstream of the point,
    with an arrowhead at its downstream end. Its length on the ground is what
    a straight arrow's would be (``m_per_unit`` metres per speed unit, at the
    point's speed, up to ``style.max_speed``). Each curve is traced in the
    finest grid covering its point (bilinear flow), and stops early at land
    or at that grid's edge."""
    x0, x1 = view.extent[:2]
    done = np.zeros(lon.shape, bool)
    lines, heads = [], []
    head_len = 0.18 * step  # view units
    for ue, ve, grid in reversed(vectors):  # finest first
        todo = ~done & np.isfinite(lon)
        if not todo.any():
            break
        ue, ve = np.asarray(ue, dtype=np.float64), np.asarray(ve, dtype=np.float64)
        col, row = _lonlat_to_cell(grid, lon[todo], lat[todo])
        speed = np.hypot(_bilinear(ue, col, row), _bilinear(ve, col, row))
        ok = np.isfinite(speed) & (speed > 0)
        done[np.flatnonzero(todo)[ok]] = True
        if not ok.any():
            continue
        col, row, speed = col[ok], row[ok], speed[ok]
        half = 0.5 * np.minimum(speed * style.units_per_m_s, style.max_speed) * m_per_unit
        up = _trace(ue, ve, grid, col, row, half, -1.0)
        down = _trace(ue, ve, grid, col, row, half, +1.0)
        cols = np.concatenate([up[0][:, ::-1], down[0][:, 1:]], axis=1)
        rows = np.concatenate([up[1][:, ::-1], down[1][:, 1:]], axis=1)
        t = grid.transform
        vx, vy = _to_view(view, grid.crs, (t.c + cols * t.a + rows * t.b).ravel(),
                          (t.f + cols * t.d + rows * t.e).ravel())
        for px, py in zip(vx.reshape(cols.shape), vy.reshape(cols.shape)):
            keep = np.isfinite(px) & np.isfinite(py)
            px, py = px[keep], py[keep]
            if px.size < 2:
                continue
            lines.append(np.column_stack([px, py]))
            dx, dy = px[-1] - px[-2], py[-1] - py[-2]
            seg = math.hypot(dx, dy)
            if seg == 0:
                continue
            hl = min(head_len, 0.5 * float(np.sum(np.hypot(np.diff(px), np.diff(py)))))
            ux, uy, hw = dx / seg, dy / seg, 0.45 * hl
            bx, by = px[-1] - hl * ux, py[-1] - hl * uy
            heads.append([(px[-1], py[-1]), (bx - hw * uy, by + hw * ux),
                          (bx + hw * uy, by - hw * ux)])
    if not lines:
        return
    from matplotlib.collections import LineCollection, PolyCollection
    from matplotlib.patches import FancyArrow

    ax.add_collection(LineCollection(lines, colors="black", linewidths=0.7,
                                     transform=view.projection, zorder=2.5))
    ax.add_collection(PolyCollection(heads, facecolors="black", edgecolors="none",
                                     transform=view.projection, zorder=2.5))
    # Key: a straight arrow as long as the curve of a round speed.
    ax.apply_aspect()
    map_width_frac = ax.get_position().width
    def key_frac(speed):
        return speed * m_per_unit * per_m / (x1 - x0) * map_width_frac
    key = next((s for s in style.key_speeds if key_frac(s) <= 0.08), style.key_speeds[-1])
    kl = key_frac(key)
    fig.add_artist(FancyArrow(0.905 - kl / 2, 0.11, kl, 0, width=0.0012, head_width=0.008,
                              head_length=min(0.006, kl / 2), length_includes_head=True,
                              color="black", transform=fig.transFigure))
    fig.text(0.905, 0.095, f"{key:g} {style.units} (curve length)\n(max {style.max_speed:g})",
             ha="center", va="top", fontsize=8)


def _lonlat_to_cell(grid: GridDefinition, lon: np.ndarray, lat: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Fractional (col, row) of lon/lat points in ``grid`` (cell corners are
    integers, so cell (r, c) spans [c, c+1) x [r, r+1))."""
    x, y = Transformer.from_crs(grid.crs.geodetic_crs, grid.crs, always_xy=True).transform(
        lon, lat)
    inv = ~grid.transform
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    return inv.a * x + inv.b * y + inv.c, inv.d * x + inv.e * y + inv.f


def _bilinear(field: np.ndarray, col: np.ndarray, row: np.ndarray) -> np.ndarray:
    """``field`` (cell values, north-up) at fractional (col, row), bilinear
    between cell centres; NaN outside the centres or next to a missing cell."""
    h, w = field.shape
    c, r = col - 0.5, row - 0.5
    c0, r0 = np.floor(c), np.floor(r)
    out = np.full(np.shape(col), np.nan)
    ok = np.isfinite(c0) & np.isfinite(r0) & (c0 >= 0) & (c0 < w - 1) & (r0 >= 0) & (r0 < h - 1)
    c0, r0 = c0[ok].astype(int), r0[ok].astype(int)
    fc, fr = c[ok] - c0, r[ok] - r0
    out[ok] = (field[r0, c0] * (1 - fc) * (1 - fr) + field[r0, c0 + 1] * fc * (1 - fr)
               + field[r0 + 1, c0] * (1 - fc) * fr + field[r0 + 1, c0 + 1] * fc * fr)
    return out


def _ground_metrics(grid: GridDefinition, col: np.ndarray, row: np.ndarray):
    """At fractional (col, row): ground metres per column step and per row
    step, and the angle (radians, counter-clockwise) from east to the +col
    direction. Rows run southward in north-up grids."""
    t = grid.transform
    to_ll = Transformer.from_crs(grid.crs, grid.crs.geodetic_crs, always_xy=True)
    d = 0.01
    def lonlat(c, r):
        lo, la = to_ll.transform(t.c + c * t.a + r * t.b, t.f + c * t.d + r * t.e)
        return np.radians(np.asarray(lo)), np.radians(np.asarray(la))
    lon0, lat0 = lonlat(col, row)
    lonc, latc = lonlat(col + d, row)
    lonr, latr = lonlat(col, row + d)
    r_earth = 6371e3
    def east_north(lo, la):
        dlon = (lo - lon0 + np.pi) % (2 * np.pi) - np.pi
        return dlon * np.cos(lat0) * r_earth, (la - lat0) * r_earth
    ec, nc = east_north(lonc, latc)
    er, nr = east_north(lonr, latr)
    return np.hypot(ec, nc) / d, np.hypot(er, nr) / d, np.arctan2(nc, ec)


def _trace(ue: np.ndarray, ve: np.ndarray, grid: GridDefinition, col: np.ndarray,
           row: np.ndarray, length_m: np.ndarray, sign: float):
    """Streamlines from (col, row), downstream (sign +1) or upstream (-1), of
    ground length ``length_m`` each, by midpoint (RK2) steps along the flow
    direction. Returns (cols, rows), each (n, CURLY_STEPS // 2 + 1), with NaN
    after a curve stops (land, grid edge, or no flow)."""
    n = CURLY_STEPS // 2
    ds = length_m / n
    cols = np.full((col.size, n + 1), np.nan)
    rows = np.full((col.size, n + 1), np.nan)
    cols[:, 0], rows[:, 0] = col, row
    c, r = col.copy(), row.copy()
    alive = np.ones(col.size, bool)

    def rate(c, r):  # (dcol, drow) per ground metre along the flow
        u, v = _bilinear(ue, c, r), _bilinear(ve, c, r)
        speed = np.hypot(u, v)
        m_col, m_row, theta = _ground_metrics(grid, c, r)
        with np.errstate(invalid="ignore", divide="ignore"):
            ex, ny = u / speed, v / speed
        along_col = ex * np.cos(theta) + ny * np.sin(theta)  # flow along +col
        along_up = -ex * np.sin(theta) + ny * np.cos(theta)  # flow along -row (north)
        return along_col / m_col, -along_up / m_row

    for k in range(1, n + 1):
        dc1, dr1 = rate(c, r)
        dc2, dr2 = rate(c + sign * 0.5 * ds * dc1, r + sign * 0.5 * ds * dr1)
        nc, nr = c + sign * ds * dc2, r + sign * ds * dr2
        alive &= np.isfinite(nc) & np.isfinite(nr)
        c, r = np.where(alive, nc, c), np.where(alive, nr, r)
        cols[alive, k], rows[alive, k] = c[alive], r[alive]
    return cols, rows


def _view_to_lonlat(view: MapView, x: np.ndarray, y: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray]:
    if view.crs is None:
        return x + view.center_lon, y
    lon, lat = Transformer.from_crs(view.crs, view.crs.geodetic_crs, always_xy=True).transform(x, y)
    return np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)


def _lonlat_to_view(view: MapView, lon: np.ndarray, lat: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray]:
    if view.crs is None:
        return (lon - view.center_lon + 180.0) % 360.0 - 180.0, lat
    x, y = Transformer.from_crs(view.crs.geodetic_crs, view.crs, always_xy=True).transform(lon, lat)
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)


def _ground_axes(view: MapView, x: np.ndarray, y: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray]:
    """View-coordinate displacement per metre due east and due north on the
    ground at view points (x, y); each is shape (2, n)."""
    r_earth = 6371e3
    d = 0.01  # degrees
    lon, lat = _view_to_lonlat(view, x, y)
    xe, ye = _lonlat_to_view(view, lon + d, lat)
    xn, yn = _lonlat_to_view(view, lon, lat + d)
    east_m = np.radians(d) * r_earth * np.cos(np.radians(lat))
    north_m = np.radians(d) * r_earth
    return (np.array([(xe - x) / east_m, (ye - y) / east_m]),
            np.array([(xn - x) / north_m, (yn - y) / north_m]))


def _colorbar(fig, rect: list[float], mappable, variable: VariableDefinition) -> None:
    cax = fig.add_axes(rect)
    cb = fig.colorbar(mappable, cax=cax, orientation="horizontal", extend=variable.extend,
                      spacing="uniform")
    cb.set_label(f"{variable.long_name} ({variable.units})", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    if variable.levels:
        # Label every level, or every other one when there are too many to fit.
        ticks = variable.levels if len(variable.levels) <= 20 else variable.levels[::2]
        cb.set_ticks(ticks)
        cb.set_ticklabels([f"{v:g}" for v in ticks])


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


def _outline_in_view(view: MapView, grid: GridDefinition) -> tuple[np.ndarray, np.ndarray]:
    """The grid's closed outer edge in the view's projection coordinates."""
    if grid.outline_lonlat is not None:
        return _to_view(view, grid.crs.geodetic_crs, *outline_lonlat(grid))
    return _to_view(view, grid.crs, *_boundary_points(grid))


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
    extent_m = (full_view(extent) if isinstance(extent, GridDefinition)
                else subarea_view(extent)).extent_m
    _load_geography(_geography_features(extent_m))


def _add_geography(ax, extent_m: float) -> None:
    """Land, coastlines and borders from Natural Earth (downloaded once by cartopy)."""
    features = _geography_features(extent_m)
    if not _load_geography(features):
        return
    for feature, style in features:
        ax.add_feature(feature, **style)
