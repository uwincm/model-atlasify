"""Model-independent definitions of the variables ModelAtlasify produces.

A :class:`VariableDefinition` says *what* a product is (name, units, meaning,
how to display it). *How* a particular model provides it lives in that model's
adapter, which typically uses helpers from :mod:`modelatlasify.variables`.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class VariableDefinition:
    name: str  # file/asset key, e.g. "surface_wind_speed"
    long_name: str  # human-readable, e.g. "Surface wind speed"
    units: str  # UDUNITS-style, e.g. "m s-1"
    description: str
    standard_name: str | None = None  # CF standard name, if one applies
    cell_methods: str | None = None  # CF cell_methods, e.g. "time: sum"
    # Plot defaults.
    cmap: str = "viridis"
    vmin: float | None = None
    vmax: float | None = None
    # Levels for a discrete colour scale (overrides vmin/vmax when given).
    levels: tuple[float, ...] | None = None
    extend: str = "max"
    # Values at or below this are drawn transparent (e.g. no rain).
    transparent_below: float | None = None
    # A spatially constant field (e.g. wind identically 0) almost always means
    # zero-filled/corrupt input. False for fields that are legitimately
    # constant sometimes (precipitation is all 0 at initialization).
    constant_is_suspicious: bool = True
    # More exact zeros than this fraction of points also indicates zero-filled
    # blocks (None = no limit, e.g. precipitation is exactly 0 over dry areas).
    max_zero_fraction: float | None = None
    aliases: tuple[str, ...] = field(default_factory=tuple)


_REGISTRY: dict[str, VariableDefinition] = {}


def register(defn: VariableDefinition) -> VariableDefinition:
    _REGISTRY[defn.name] = defn
    return defn


def get_variable(name: str) -> VariableDefinition:
    return _REGISTRY[name]


def all_variables() -> dict[str, VariableDefinition]:
    return dict(_REGISTRY)


def resolve_names(requested: list[str]) -> list[str]:
    """Map user-facing names/aliases (e.g. "wind") to registered variable names.

    Aliases can expand to several variables ("precipitation" gives both the
    accumulated and the interval product).
    """
    out: list[str] = []
    for raw in requested:
        key = raw.strip().lower()
        if not key:
            continue
        matches = [
            v.name for v in _REGISTRY.values() if key == v.name or key in v.aliases
        ]
        if not matches:
            known = sorted({a for v in _REGISTRY.values() for a in (v.name, *v.aliases)})
            raise KeyError(f"Unknown variable '{raw}'. Known names: {', '.join(known)}")
        for m in matches:
            if m not in out:
                out.append(m)
    return out
