"""Named lon/lat sub-areas for zoomed composite maps."""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Name of the whole-domain plot area; not available for sub-areas.
FULL_AREA = "full"
_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True)
class SubArea:
    """A lon/lat box. ``west`` is in [-180, 180) and ``east`` > ``west``; a box
    crossing the antimeridian has ``east`` > 180 (e.g. 170 .. 190)."""

    name: str
    west: float
    east: float
    south: float
    north: float

    @property
    def center_lon(self) -> float:
        return (self.west + self.east) / 2.0


def parse_sub_areas(text: str) -> list[SubArea]:
    """Parse ``NAME,LON1,LON2,LAT1,LAT2[,NAME,...]`` (raises ValueError).

    Longitudes may be in [-180, 180] or [0, 360]. The box runs eastward from
    LON1 to LON2, so LON2 < LON1 means it crosses the antimeridian.
    """
    parts = [p.strip() for p in text.split(",")]
    if len(parts) % 5:
        raise ValueError(f"expected groups of 5 (NAME,LON1,LON2,LAT1,LAT2), got {len(parts)} items")
    areas: list[SubArea] = []
    for k in range(0, len(parts), 5):
        name, *nums = parts[k:k + 5]
        if not _NAME.match(name):
            raise ValueError(f"sub-area name '{name}' must use only letters, digits, '_' and '-'")
        if name == FULL_AREA:
            raise ValueError(f"'{FULL_AREA}' is reserved for the whole-domain maps")
        if name in {a.name for a in areas}:
            raise ValueError(f"sub-area '{name}' is given twice")
        try:
            lon1, lon2, lat1, lat2 = (float(v) for v in nums)
        except ValueError:
            raise ValueError(f"sub-area '{name}': LON1,LON2,LAT1,LAT2 must be numbers") from None
        if not (-90.0 <= lat1 <= 90.0 and -90.0 <= lat2 <= 90.0) or lat1 >= lat2:
            raise ValueError(f"sub-area '{name}': need -90 <= LAT1 < LAT2 <= 90")
        if not (-180.0 <= lon1 <= 360.0 and -180.0 <= lon2 <= 360.0):
            raise ValueError(f"sub-area '{name}': longitudes must be within [-180, 360]")
        west = (lon1 + 180.0) % 360.0 - 180.0
        span = (lon2 - lon1) % 360.0
        if span == 0.0:
            raise ValueError(f"sub-area '{name}': LON1 and LON2 must differ")
        areas.append(SubArea(name, west, west + span, lat1, lat2))
    return areas
