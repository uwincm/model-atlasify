"""Write small synthetic WRF-like history files for tests.

XLAT/XLONG are generated from a known projection and grid origin, so tests can
check that ModelAtlasify recovers the georeferencing exactly. Field values are
simple analytic patterns with known results.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import netCDF4
import numpy as np
from pyproj import CRS, Transformer

R = 6370000.0

PROJECTIONS = {
    1: dict(MAP_PROJ=1, MAP_PROJ_CHAR="Lambert Conformal", TRUELAT1=30.0, TRUELAT2=60.0,
            STAND_LON=-100.0, CEN_LAT=40.0, CEN_LON=-100.0, MOAD_CEN_LAT=40.0),
    2: dict(MAP_PROJ=2, MAP_PROJ_CHAR="Polar Stereographic", TRUELAT1=60.0, TRUELAT2=60.0,
            STAND_LON=-45.0, CEN_LAT=72.0, CEN_LON=-40.0, MOAD_CEN_LAT=72.0),
    3: dict(MAP_PROJ=3, MAP_PROJ_CHAR="Mercator", TRUELAT1=10.0, TRUELAT2=0.0,
            STAND_LON=-156.0, CEN_LAT=5.0, CEN_LON=-156.0, MOAD_CEN_LAT=5.0),
}


def true_crs(p: dict) -> CRS:
    if p["MAP_PROJ"] == 1:
        s = (f"+proj=lcc +lat_1={p['TRUELAT1']} +lat_2={p['TRUELAT2']} "
             f"+lat_0={p['MOAD_CEN_LAT']} +lon_0={p['STAND_LON']}")
    elif p["MAP_PROJ"] == 2:
        s = f"+proj=stere +lat_0=90 +lat_ts={p['TRUELAT1']} +lon_0={p['STAND_LON']}"
    else:
        s = f"+proj=merc +lat_ts={p['TRUELAT1']} +lon_0={p['CEN_LON']}"
    return CRS.from_proj4(f"{s} +R={R:.0f} +units=m +no_defs +type=crs")


def grid_lonlat(p: dict, nx: int, ny: int, dx: float, dy: float, shift=(0, 0)):
    """Mass-point lon/lat (south-up, like WRF) and the true SW mass point x/y.

    ``shift`` = (cols east, rows north) moves the grid by whole cells."""
    crs = true_crs(p)
    fwd = Transformer.from_crs(crs.geodetic_crs, crs, always_xy=True)
    inv = Transformer.from_crs(crs, crs.geodetic_crs, always_xy=True)
    xc, yc = fwd.transform(p["CEN_LON"], p["CEN_LAT"])
    x0 = xc - (nx - 1) / 2 * dx + shift[0] * dx
    y0 = yc - (ny - 1) / 2 * dy + shift[1] * dy
    jj, ii = np.mgrid[0:ny, 0:nx]
    lon, lat = inv.transform(x0 + ii * dx, y0 + jj * dy)
    lon = (np.asarray(lon) + 180.0) % 360.0 - 180.0
    return lon, np.asarray(lat), x0, y0


def write_wrfout(
    directory: Path,
    times: list[datetime],
    *,
    domain: int = 1,
    map_proj: int = 1,
    nx: int = 40,
    ny: int = 30,
    dx: float = 12000.0,
    sim_start: datetime | None = None,
    start_date: datetime | None = None,
    times_per_file: int = 1,
    include: tuple[str, ...] = ("U10", "V10", "U", "V", "RAINC", "RAINNC"),
    rain_units: str = "mm",
    bucket_mm: float = -1.0,
    precip_rate: float = 1.0,
    corrupt_xtime: tuple[datetime, ...] = (),
    move: tuple[int, int] = (0, 0),
    prec_acc_dt: float | None = None,
) -> list[Path]:
    """Write wrfout_dNN_* files. Returns paths.

    Fields (south-up as in WRF):
      U10 = 3 (1 + i/nx), V10 = 4 (1 + i/nx)  ->  speed 5 (1 + i/nx);
      U (staggered) = i_stag, V (staggered) = j_stag at every level;
      RAINNC = precip_rate * hours since sim_start * (1 + i/nx);
      RAINC  = 0.5 * hours since sim_start.

    ``prec_acc_dt`` (minutes) also writes PREC_ACC_C/PREC_ACC_NC: the RAINC/
    RAINNC increase since the last multiple of PREC_ACC_DT after sim_start.

    ``move`` = (cols east, rows north) per output time makes a moving nest:
    the k-th time is shifted by k * move cells, and the i in the patterns
    above is the column of the *first* position's lattice, so every pattern
    stays attached to the ground as the grid moves.
    """
    p = dict(PROJECTIONS[map_proj])
    sim_start = sim_start or times[0]
    start_date = start_date or sim_start
    if prec_acc_dt is not None:
        include = tuple(include) + ("PREC_ACC_C", "PREC_ACC_NC")
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for k in range(0, len(times), times_per_file):
        chunk = times[k:k + times_per_file]
        path = directory / f"wrfout_d{domain:02d}_{chunk[0]:%Y-%m-%d_%H:%M:%S}"
        with netCDF4.Dataset(path, "w", format="NETCDF4_CLASSIC") as nc:
            nc.createDimension("Time", None)
            nc.createDimension("DateStrLen", 19)
            nc.createDimension("west_east", nx)
            nc.createDimension("south_north", ny)
            nc.createDimension("west_east_stag", nx + 1)
            nc.createDimension("south_north_stag", ny + 1)
            nc.createDimension("bottom_top", 3)
            attrs = dict(
                TITLE=" OUTPUT FROM WRF V4.5.1 MODEL",
                SIMULATION_START_DATE=f"{sim_start:%Y-%m-%d_%H:%M:%S}",
                START_DATE=f"{start_date:%Y-%m-%d_%H:%M:%S}",
                SIMULATION_INITIALIZATION_TYPE="REAL-DATA CASE",
                **{"WEST-EAST_GRID_DIMENSION": nx + 1, "SOUTH-NORTH_GRID_DIMENSION": ny + 1,
                   "BOTTOM-TOP_GRID_DIMENSION": 4},
                DX=np.float32(dx), DY=np.float32(dx), GRID_ID=domain,
                PARENT_ID=domain - 1, MP_PHYSICS=8, CU_PHYSICS=1, BUCKET_MM=np.float32(bucket_mm),
                POLE_LAT=np.float32(90.0), POLE_LON=np.float32(0.0),
            )
            attrs.update({k2: (np.float32(v) if isinstance(v, float) else v) for k2, v in p.items()})
            if prec_acc_dt is not None:
                attrs["PREC_ACC_DT"] = np.float32(prec_acc_dt)
            nc.setncatts(attrs)

            tv = nc.createVariable("Times", "S1", ("Time", "DateStrLen"))
            xt = nc.createVariable("XTIME", "f4", ("Time",))
            xt.units = f"minutes since {sim_start:%Y-%m-%d %H:%M:%S}"
            v_lat = nc.createVariable("XLAT", "f4", ("Time", "south_north", "west_east"))
            v_lon = nc.createVariable("XLONG", "f4", ("Time", "south_north", "west_east"))
            v_lat.units, v_lon.units = "degree_north", "degree_east"
            fields = {}
            for name in include:
                dims = {
                    "U": ("Time", "bottom_top", "south_north", "west_east_stag"),
                    "V": ("Time", "bottom_top", "south_north_stag", "west_east"),
                }.get(name, ("Time", "south_north", "west_east"))
                fields[name] = nc.createVariable(name, "f4", dims)
                fields[name].units = rain_units if name.startswith(("RAIN", "I_RAIN", "PREC_ACC")) else (
                    "" if name.startswith("I_") else "m s-1")
            for n, t in enumerate(chunk):
                kk = k + n
                lon, lat, _, _ = grid_lonlat(p, nx, ny, dx, dx, (kk * move[0], kk * move[1]))
                tv[n, :] = np.array(list(f"{t:%Y-%m-%d_%H:%M:%S}"), dtype="S1")
                xt[n] = 0.0 if t in corrupt_xtime else (t - sim_start).total_seconds() / 60.0
                v_lat[n] = lat
                v_lon[n] = lon
                hours = (t - sim_start).total_seconds() / 3600.0
                ii = np.arange(nx)[None, :].repeat(ny, axis=0) + kk * move[0]
                if "U10" in fields:
                    fields["U10"][n] = 3.0 * (1.0 + ii / nx)
                if "V10" in fields:
                    fields["V10"][n] = 4.0 * (1.0 + ii / nx)
                if "U" in fields:
                    fields["U"][n] = np.broadcast_to(np.arange(nx + 1, dtype=float), (3, ny, nx + 1))
                if "V" in fields:
                    fields["V"][n] = np.broadcast_to(
                        np.arange(ny + 1, dtype=float)[:, None], (3, ny + 1, nx))
                rainnc = precip_rate * hours * (1.0 + ii / nx)
                if prec_acc_dt:  # 0 = switched off in WRF (fields stay 0)
                    minutes = hours * 60.0
                    acc_hours = (minutes - (minutes // prec_acc_dt) * prec_acc_dt) / 60.0
                    if minutes > 0 and acc_hours == 0:
                        acc_hours = prec_acc_dt / 60.0  # output at the end of a full window
                    fields["PREC_ACC_NC"][n] = precip_rate * acc_hours * (1.0 + ii / nx)
                    fields["PREC_ACC_C"][n] = np.full((ny, nx), 0.5 * acc_hours)
                elif prec_acc_dt is not None:
                    fields["PREC_ACC_NC"][n] = 0.0
                    fields["PREC_ACC_C"][n] = 0.0
                rainc = np.full((ny, nx), 0.5 * hours)
                if bucket_mm > 0:
                    if "I_RAINNC" in fields:
                        fields["I_RAINNC"][n] = np.floor(rainnc / bucket_mm)
                    if "I_RAINC" in fields:
                        fields["I_RAINC"][n] = np.floor(rainc / bucket_mm)
                    rainnc = np.mod(rainnc, bucket_mm)
                    rainc = np.mod(rainc, bucket_mm)
                if "RAINNC" in fields:
                    fields["RAINNC"][n] = rainnc
                if "RAINC" in fields:
                    fields["RAINC"][n] = rainc
        paths.append(path)
    return paths


def hourly(start: datetime, n: int, step_hours: float = 1.0) -> list[datetime]:
    return [start + timedelta(hours=step_hours * k) for k in range(n)]
