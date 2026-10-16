"""Write small synthetic DWM output (output/umwmout_*.nc + namelists/main.nml).

The grid is Mercator on a 6 370 km sphere, centred on the antimeridian by
default (lon written in 0..360 like DWM), so tests can check that the grid is
recovered exactly. Fields are simple analytic patterns; a land block is NaN
with seamask = 0.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import netCDF4
import numpy as np
from pyproj import CRS, Transformer

R = 6370000.0


def mercator_lonlat(nx: int, ny: int, dx: float, cen_lon: float, cen_lat: float):
    """South-up lon (0..360) / lat of cell centres, and the true SW centre x/y."""
    crs = CRS.from_proj4(f"+proj=merc +lat_ts=0 +lon_0={cen_lon} +R={R:.0f} +units=m +no_defs")
    inv = Transformer.from_crs(crs, crs.geodetic_crs, always_xy=True)
    _, yc = Transformer.from_crs(crs.geodetic_crs, crs, always_xy=True).transform(cen_lon, cen_lat)
    x0 = -(nx - 1) / 2 * dx
    y0 = yc - (ny - 1) / 2 * dx
    jj, ii = np.mgrid[0:ny, 0:nx]
    lon, lat = inv.transform(x0 + ii * dx, y0 + jj * dx)
    return np.asarray(lon) % 360.0, np.asarray(lat), x0, y0


def write_dwm(run_dir: Path, times: list[datetime], *, nx: int = 40, ny: int = 24,
               dx: float = 50000.0, cen_lon: float = 180.0, cen_lat: float = 5.0,
               start: datetime | None = None, namelist: bool = True,
               suffix: str = ".nc") -> list[Path]:
    """Fields (south-up, i = column, j = row):
    swh = 1 + i / nx + hours since start; mwp = 5 + j / ny; dwp = 8 + i / nx.
    Columns 0-4 of rows 0-4 are land (NaN, seamask 0)."""
    out = run_dir / "output"
    out.mkdir(parents=True, exist_ok=True)
    start = start or times[0]
    lon, lat, _, _ = mercator_lonlat(nx, ny, dx, cen_lon, cen_lat)
    jj, ii = np.mgrid[0:ny, 0:nx]
    land = (ii < 5) & (jj < 5)
    paths = []
    for t in times:
        path = out / f"umwmout_{t:%Y-%m-%d_%H:%M:%S}{suffix}"
        with netCDF4.Dataset(path, "w") as nc:
            nc.createDimension("x", nx)
            nc.createDimension("y", ny)
            nc.createDimension("Time", None)
            hours = (t - start).total_seconds() / 3600.0
            fields = {
                "lon": (lon, "degrees East"), "lat": (lat, "degrees North"),
                "seamask": (np.where(land, 0, 1), "non-dimensional"),
                "swh": (1 + ii / nx + hours, "m"), "mwp": (5 + jj / ny, "s"),
                "dwp": (8 + ii / nx, "s"),
            }
            for name, (data, units) in fields.items():
                v = nc.createVariable(name, "i4" if name == "seamask" else "f4", ("Time", "y", "x"))
                v.units = units
                v.description = name
                if name in ("swh", "mwp", "dwp"):
                    data = np.where(land, np.nan, data)
                v[0] = data
        paths.append(path)
    if namelist:
        (run_dir / "namelists").mkdir(exist_ok=True)
        (run_dir / "namelists" / "main.nml").write_text(
            "&DOMAIN\n"
            f"  mm           = {nx}  ! Domain size in x\n"
            f"  nm           = {ny}\n"
            f"  startTimeStr = '{start:%Y-%m-%d %H:%M:%S}' ! Simulation start time\n"
            "  restart      = .false.\n/\n")
    return paths
