"""Reading HYCOM ``.a``/``.b`` files (archives and ``regional.grid``).

Adapted from pyhycom (https://github.com/uwincm/pyhycom, ``pyhycom.py``:
``getDims``, ``getFieldIndex``, ``getField``, ``date_from_filename``,
``forday``), reduced to what ModelAtlasify needs and changed to read only the
requested records (by byte offset) instead of whole files.

pyhycom is distributed under the MIT License:

    Copyright (c) 2025 Unified Wave Interface Coupled Model

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

File format (HYCOM "a/b"): the ``.b`` file is a text header listing the
records; the ``.a`` file holds each record as ``idm * jdm`` big-endian float32
values (row 0 = southernmost), padded to a multiple of 4096 values. Missing
values (land) are 2**100.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

#: Records are padded to a multiple of this many 4-byte values.
WORD = 4096
#: HYCOM's fill value for missing data (land).
UNDEF = np.float32(2.0 ** 100)
#: yrflag = 3: model day 1.0 is 1901-01-01 00 UTC.
YRFLAG3_EPOCH = datetime(1901, 1, 1, tzinfo=timezone.utc)
FILENAME_TIME = re.compile(r"(\d{4})_(\d{3})_(\d{2})")


@dataclass(frozen=True)
class Record:
    """One record listed in a ``.b`` file."""

    name: str  # e.g. "temp", "u-vel.", "u_btrop", "plon"
    layer: int  # archive layer k (1 = top; 0 = a 2-D surface field); 0 for grids
    model_day: float | None  # archives only


@dataclass(frozen=True)
class Header:
    """What ModelAtlasify needs from a ``.b`` file."""

    idm: int  # number of columns (x)
    jdm: int  # number of rows (y)
    records: tuple[Record, ...]
    yrflag: int | None = None  # archives only
    mapflg: int | None = None  # regional.grid only

    def index(self, name: str, layer: int | None = None) -> int:
        """Position of the first record called ``name`` (and in ``layer``, if
        given) in the file; KeyError if there is none."""
        for i, r in enumerate(self.records):
            if r.name == name and (layer is None or r.layer == layer):
                return i
        raise KeyError(f"no '{name}'" + ("" if layer is None else f" in layer {layer}"))

    def has(self, name: str, layer: int | None = None) -> bool:
        try:
            self.index(name, layer)
        except KeyError:
            return False
        return True


def _int_setting(lines: list[str], key: str) -> int | None:
    for line in lines:
        if f"'{key}" in line:
            return int(line.split()[0])
    return None


def read_header(b_path: Path) -> Header:
    """Parse an archive (``archv``/``archs``) or ``regional.grid`` ``.b`` file."""
    lines = Path(b_path).read_text(errors="replace").splitlines()
    idm, jdm = _int_setting(lines, "idm"), _int_setting(lines, "jdm")
    if idm is None or jdm is None:
        raise ValueError(f"{Path(b_path).name}: no idm/jdm; not a HYCOM .b file?")
    records = []
    archive = any(line.startswith("field ") for line in lines)
    if archive:  # "name = step model_day k dens min max", after the "field ..." line
        start = next(i for i, line in enumerate(lines) if line.startswith("field ")) + 1
        for line in lines[start:]:
            if "=" not in line:
                continue
            name, rest = line.split("=", 1)
            parts = rest.split()
            records.append(Record(name.strip(), int(parts[2]), float(parts[1])))
    else:  # regional.grid: "plon:  min,max = ..." after idm/jdm/mapflg
        for line in lines:
            m = re.match(r"^\s*(\w+)\s*:", line)
            if m:
                records.append(Record(m[1], 0, None))
    return Header(idm, jdm, tuple(records), yrflag=_int_setting(lines, "yrflag"),
                  mapflg=_int_setting(lines, "mapflg"))


def read_record(a_path: Path, header: Header, index: int) -> np.ndarray:
    """Record ``index`` of ``a_path`` as float64 (jdm, idm), south row first,
    with missing values as NaN. Only that record is read."""
    n = header.idm * header.jdm
    padded = -(-n // WORD) * WORD
    data = np.fromfile(a_path, dtype=">f4", count=n, offset=index * padded * 4)
    if data.size != n:
        raise ValueError(f"{Path(a_path).name} is too short for record {index + 1} "
                         "(truncated file?)")
    out = data.astype(np.float64).reshape(header.jdm, header.idm)
    out[data.reshape(header.jdm, header.idm) >= UNDEF] = np.nan
    return out


def model_day_to_datetime(model_day: float, yrflag: int) -> datetime:
    """UTC time of a HYCOM model day; only ``yrflag = 3`` (actual calendar) is
    supported."""
    if yrflag != 3:
        raise ValueError(f"yrflag = {yrflag} is not supported (only 3, the actual calendar)")
    seconds = round((model_day - 1.0) * 86400.0)
    return YRFLAG3_EPOCH + timedelta(seconds=seconds)


def filename_time(name: str) -> datetime | None:
    """Valid time in a name like ``archv.2011_245_01.a`` (year, day of year,
    hour), or None."""
    m = FILENAME_TIME.search(name)
    if not m:
        return None
    year, day, hour = map(int, m.groups())
    return datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=day - 1, hours=hour)
