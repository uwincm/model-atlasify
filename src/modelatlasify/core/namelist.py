"""Minimal Fortran namelist reading (shared by the model adapters)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def read_namelist(path: Path) -> dict[str, Any]:
    """The simple ``key = value`` entries of a Fortran namelist (comments dropped)."""
    out: dict[str, Any] = {}
    for line in path.read_text(errors="replace").splitlines():
        line = line.split("!", 1)[0].strip()
        m = re.match(r"^([A-Za-z_]\w*)\s*=\s*(.+?)\s*,?$", line)
        if not m:
            continue
        key, raw = m[1], m[2].strip()
        if raw[:1] in "'\"":
            value: Any = raw.strip("'\"")
        elif raw.lower() in (".t.", ".true."):
            value = True
        elif raw.lower() in (".f.", ".false."):
            value = False
        else:
            try:
                value = int(raw)
            except ValueError:
                try:
                    value = float(raw.lower().replace("d", "e"))
                except ValueError:
                    value = raw
        out[key] = value
    return out
