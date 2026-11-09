"""Synthetic HYCOM output: regional.grid.[ab] and archv/archs .a/.b files.

Grid: Mercator (sphere R = 6370 km) cell centres, ``nx`` x ``ny``, row 0 south.
Fields (layer 1 for 3-D ones), with i the column and j the row (south = 0):
  temp  = 20 + 10 i/nx  C,   salin = 30 + 5 j/ny
  u-vel. (west face of cell i) = 0.1 i,   u_btrop = 0.05  ->  total at cell centre
      0.1 (i + 0.5) + 0.05;
  v-vel. (south face of row j) = 0.02 j,  v_btrop = -0.01 -> centre 0.02 (j + 0.5) - 0.01.
The first ``land_cols`` columns are land: temp/salin missing, and the u faces
touching land (columns 0..land_cols) missing.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from pyproj import Transformer

WORD = 4096
UNDEF = np.float32(2.0 ** 100)
R = 6370000.0


def mercator_lonlat(nx: int, ny: int, dx_deg: float = 0.1, lon0: float = -90.0,
                    lat0: float = 20.0) -> tuple[np.ndarray, np.ndarray]:
    """Cell-centre lon/lat (ny, nx), row 0 south, regular in Mercator."""
    merc = Transformer.from_crs(f"+proj=longlat +R={R:.0f}",
                                f"+proj=merc +R={R:.0f} +lon_0=0", always_xy=True)
    _, y0 = merc.transform(lon0, lat0)
    dx = np.radians(dx_deg) * R
    x = np.radians(lon0) * R + dx * np.arange(nx)
    y = y0 + dx * np.arange(ny)
    xx, yy = np.meshgrid(x, y)
    lon, lat = merc.transform(xx, yy, direction="INVERSE")
    return np.asarray(lon), np.asarray(lat)


def _write_a(path: Path, fields: list[np.ndarray]) -> None:
    with open(path, "wb") as f:
        for data in fields:
            flat = np.asarray(data, dtype=np.float32).ravel()
            pad = -flat.size % WORD  # padded with zeros, written big-endian
            f.write(np.concatenate([flat, np.zeros(pad, np.float32)]).astype(">f4").tobytes())


def write_grid(directory: Path, nx: int = 30, ny: int = 20, pang: float = 0.0) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    lon, lat = mercator_lonlat(nx, ny)
    names = ["plon", "plat", "pang"]
    data = [lon, lat, np.full((ny, nx), pang)]
    _write_a(directory / "regional.grid.a", data)
    lines = [f"{nx:5d}    'idm   ' = longitudinal array size",
             f"{ny:5d}    'jdm   ' = latitudinal  array size",
             "    0    'mapflg' = map flag (-1=unknown,0=mercator,2=uniform,4=f-plane)"]
    lines += [f"{n}:  min,max = {np.min(d):15.5f} {np.max(d):15.5f}" for n, d in zip(names, data)]
    (directory / "regional.grid.b").write_text("\n".join(lines) + "\n")


def model_day(t: datetime) -> float:
    return 1.0 + (t.replace(tzinfo=None) - datetime(1901, 1, 1)).total_seconds() / 86400.0


def write_archive(directory: Path, t: datetime, kind: str = "archv", nx: int = 30,
                  ny: int = 20, land_cols: int = 2, temp_offset: float = 0.0,
                  day_error: timedelta = timedelta(0)) -> Path:
    """One archive (``archv`` with 2 layers, or ``archs`` with layer 1) at ``t``."""
    i = np.arange(nx)[None, :].repeat(ny, 0).astype(float)
    j = np.arange(ny)[:, None].repeat(nx, 1).astype(float)
    land = i < land_cols
    temp = np.where(land, UNDEF, 20.0 + 10.0 * i / nx + temp_offset)
    salin = np.where(land, UNDEF, 30.0 + 5.0 * j / ny)
    u = np.where(i <= land_cols, UNDEF, 0.1 * i)
    v = np.where(land, UNDEF, 0.02 * j)
    ubt = np.where(land, UNDEF, 0.05)
    vbt = np.where(land, UNDEF, -0.01)
    zeros = np.zeros((ny, nx))
    records = [("montg1", 6, zeros), ("srfhgt", 0, zeros), ("u_btrop", 0, ubt),
               ("v_btrop", 0, vbt)]
    for k in (1, 2) if kind == "archv" else (1,):
        records += [("u-vel.", k, u), ("v-vel.", k, v), ("thknss", k, zeros + 9806.0),
                    ("temp", k, temp), ("salin", k, salin)]
    stem = f"{kind}.{t:%Y}_{t.timetuple().tm_yday:03d}_{t:%H}"
    a_path, b_path = directory / f"{stem}.a", directory / f"{stem}.b"
    _write_a(a_path, [d for _, _, d in records])
    day = model_day(t + day_error)
    head = ["HYCOM synthetic test archive", "", "", "1234567890",
            "   22    'iversn' = hycom version number x10",
            "  201    'iexpt ' = experiment number x10",
            "    3    'yrflag' = days in year flag",
            f"{nx:5d}    'idm   ' = longitudinal array size",
            f"{ny:5d}    'jdm   ' = latitudinal  array size",
            "field       time step  model day  k  dens        min              max"]
    head += [f"{n:<8} = {100:10d} {day:10.3f} {k:2d} 17.000  0.0E+00  1.0E+00"
             for n, k, _ in records]
    b_path.write_text("\n".join(head) + "\n")
    return a_path


def write_hycom(directory: Path, times: list[datetime], kind: str = "archv", **kw) -> None:
    write_grid(directory, **{k: v for k, v in kw.items() if k in ("nx", "ny", "pang")})
    for t in times:
        write_archive(directory, t, kind=kind,
                      **{k: v for k, v in kw.items() if k in ("nx", "ny", "land_cols")})
