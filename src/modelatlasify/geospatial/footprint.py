"""Longitude/latitude footprint of a projected grid, for STAC geometry/bbox.

The footprint is traced along the grid's *outer cell edges* (not the outermost
cell centres) in the native CRS, densified, and inverse-projected to lon/lat.
Domains that cross the antimeridian are split into a MultiPolygon and given a
bbox whose west edge is greater than its east edge, as required by GeoJSON
(RFC 7946 s. 3.1.9 / 5.2) and STAC. Domains containing a pole are expanded to
include it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyproj import Transformer
from shapely.geometry import MultiPolygon, Polygon, box, mapping
from shapely.affinity import translate

from modelatlasify.core.model import GridDefinition


@dataclass
class Footprint:
    geometry: dict  # GeoJSON Polygon or MultiPolygon, lon in [-180, 180]
    bbox: list[float]  # [west, south, east, north]; west > east across the antimeridian
    crosses_antimeridian: bool


def _edge_points(grid: GridDefinition, n_per_edge: int) -> tuple[np.ndarray, np.ndarray]:
    xmin, ymin, xmax, ymax = grid.native_bounds
    t = np.linspace(0.0, 1.0, n_per_edge, endpoint=False)
    xs = np.concatenate([xmin + (xmax - xmin) * t, np.full_like(t, xmax),
                         xmax - (xmax - xmin) * t, np.full_like(t, xmin)])
    ys = np.concatenate([np.full_like(t, ymin), ymin + (ymax - ymin) * t,
                         np.full_like(t, ymax), ymax - (ymax - ymin) * t])
    return xs, ys


def grid_footprint(grid: GridDefinition, n_per_edge: int = 100, precision: int = 6) -> Footprint:
    xs, ys = _edge_points(grid, n_per_edge)
    to_lonlat = Transformer.from_crs(grid.crs, grid.crs.geodetic_crs, always_xy=True)
    lon, lat = to_lonlat.transform(xs, ys)
    lon = np.asarray(lon)
    lat = np.asarray(lat)

    # Unwrap the ring so consecutive vertices never jump by ~360 degrees.
    unwrapped = np.degrees(np.unwrap(np.radians(lon)))
    # Net longitude travelled around the closed ring: 0 normally, +-360 if the
    # ring encircles a pole.
    steps = np.diff(np.append(lon, lon[0]))
    winding = int(np.round(np.sum((steps + 180.0) % 360.0 - 180.0) / 360.0))

    if winding != 0:
        # The ring circles a pole: the domain contains it.
        north = bool(np.mean(lat) > 0)
        lat_lim = 90.0 if north else -90.0
        lat_edge = float(np.min(lat)) if north else float(np.max(lat))
        poly = box(-180.0, min(lat_edge, lat_lim), 180.0, max(lat_edge, lat_lim))
        return Footprint(
            geometry=_round(mapping(poly), precision),
            bbox=[-180.0, round(float(poly.bounds[1]), precision), 180.0,
                  round(float(poly.bounds[3]), precision)],
            crosses_antimeridian=False,
        )

    # Shift so that the ring's centre longitude lies in [-180, 180).
    shift = 360.0 * np.floor((np.mean(unwrapped) + 180.0) / 360.0)
    unwrapped = unwrapped - shift
    ring = Polygon(np.column_stack([unwrapped, lat]))
    if not ring.is_valid:
        ring = ring.buffer(0)

    west, east = float(unwrapped.min()), float(unwrapped.max())
    south, north = float(lat.min()), float(lat.max())
    if west >= -180.0 and east <= 180.0:
        return Footprint(
            geometry=_round(mapping(ring), precision),
            bbox=[round(v, precision) for v in (west, south, east, north)],
            crosses_antimeridian=False,
        )

    parts = []
    for offset in (-360.0, 0.0, 360.0):
        piece = ring.intersection(box(-180.0 + offset, -90.0, 180.0 + offset, 90.0))
        if piece.is_empty:
            continue
        piece = translate(piece, xoff=-offset)
        parts.extend(piece.geoms if hasattr(piece, "geoms") else [piece])
    geom = MultiPolygon([p for p in parts if isinstance(p, Polygon) and not p.is_empty])
    bbox_west = west + 360.0 if west < -180.0 else west
    bbox_east = east - 360.0 if east > 180.0 else east
    return Footprint(
        geometry=_round(mapping(geom), precision),
        bbox=[round(v, precision) for v in (bbox_west, south, bbox_east, north)],
        crosses_antimeridian=True,
    )


def _round(geojson: dict, precision: int) -> dict:
    def r(obj):
        if isinstance(obj, (list, tuple)):
            return [r(o) for o in obj]
        if isinstance(obj, float):
            return round(obj, precision)
        return obj

    return {"type": geojson["type"], "coordinates": r(geojson["coordinates"])}
