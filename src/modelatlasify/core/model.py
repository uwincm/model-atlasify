"""Model-agnostic data structures and the model adapter interface.

A *model adapter* knows how to find one model's output on disk and how to turn
its native fields into ModelAtlasify's standard variables. Everything
downstream of the adapter (GeoTIFF writing, plotting, STAC) only sees the types
defined here, so adding a new model means writing a new adapter, not touching
the rest of the pipeline.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from affine import Affine
from pyproj import CRS


@dataclass(frozen=True)
class GridDefinition:
    """A regular grid in a projected CRS.

    ``data`` arrays handed to the rest of the pipeline are always oriented
    north-up (row 0 is the northernmost row), matching ``transform``, which
    maps (col, row) pixel *corners* to CRS coordinates.
    """

    crs: CRS
    transform: Affine
    width: int
    height: int
    # (height, width), north-up. None where they are not kept (per-time grids
    # of moving nests): crs + transform describe the grid exactly anyway.
    lat: np.ndarray | None = None  # degrees north
    lon: np.ndarray | None = None  # degrees east in [-180, 180)
    description: str = ""
    max_georeference_error_m: float | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def native_bounds(self) -> tuple[float, float, float, float]:
        """(xmin, ymin, xmax, ymax) of the grid's outer pixel edges in the CRS."""
        t = self.transform
        xs = (t.c, t.c + t.a * self.width)
        ys = (t.f, t.f + t.e * self.height)
        return (min(xs), min(ys), max(xs), max(ys))

    def pixel_offset(self, other: GridDefinition) -> tuple[int, int] | None:
        """(rows, cols) by which ``other`` is shifted relative to this grid, i.e.
        pixel (r, c) of this grid is pixel (r + rows, c + cols) of ``other``;
        None unless both are the same lattice shifted by whole cells."""
        a, b = self.transform, other.transform
        if (self.crs != other.crs or self.shape != other.shape
                or not np.allclose((a.a, a.b, a.d, a.e), (b.a, b.b, b.d, b.e))):
            return None
        cols = (a.c - b.c) / b.a
        rows = (a.f - b.f) / b.e
        if abs(cols - round(cols)) > 0.01 or abs(rows - round(rows)) > 0.01:
            return None
        return int(round(rows)), int(round(cols))


def shift_to_grid(data: np.ndarray, src: GridDefinition, dst: GridDefinition) -> np.ndarray:
    """Re-index a north-up field on ``src`` onto ``dst``, a whole-cell shift of
    the same lattice (e.g. a moving nest at a later time). Points of ``dst``
    outside ``src`` are NaN."""
    offset = dst.pixel_offset(src)
    if offset is None:
        raise ValueError("grids are not whole-cell shifts of the same lattice")
    dr, dc = offset
    h, w = dst.shape
    out = np.full((h, w), np.nan)
    r0, r1 = max(0, -dr), min(h, h - dr)
    c0, c1 = max(0, -dc), min(w, w - dc)
    if r0 < r1 and c0 < c1:
        out[r0:r1, c0:c1] = data[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
    return out


@dataclass
class ModelDataset:
    """One spatially consistent time series of model output.

    For WRF this is a single domain (e.g. d01) of a single run. Several datasets
    with the same ``run_id`` (e.g. d01 and d02) belong to the same model run.

    A domain whose position changes over time (a moving nest) has one grid
    per valid time in ``time_grids``; use :meth:`grid_at`. ``grid`` is then
    the grid at the first valid time.
    """

    model: str  # e.g. "WRF"
    run_id: str
    dataset_id: str  # unique within the run, e.g. "d01"
    title: str
    files: list[Path]
    valid_times: list[datetime]  # timezone-aware UTC, sorted, unique
    reference_time: datetime | None  # initialization / simulation start
    grid: GridDefinition
    metadata: dict[str, Any] = field(default_factory=dict)  # model-specific, JSON-serializable
    source_dir: Path | None = None
    #: The file holding each valid time (parallel to ``valid_times``).
    time_sources: list[Path] = field(default_factory=list)
    #: The run's folder under the output directory (POSIX, relative), set by
    #: the pipeline from the input path; ``run_id`` is the same path joined
    #: with "--".
    run_path: str = ""
    #: Grid at each valid time (parallel to ``valid_times``) for moving
    #: domains; None when the grid is fixed.
    time_grids: list[GridDefinition] | None = field(default=None, repr=False)
    #: Opaque, adapter-private bookkeeping (e.g. which file/index holds each time).
    adapter_state: Any = field(default=None, repr=False)

    @property
    def run_name(self) -> str:
        """Short, human-readable run name: the run directory's own name."""
        return self.run_path.rsplit("/", 1)[-1] if self.run_path else self.run_id

    @property
    def moving(self) -> bool:
        return self.time_grids is not None

    def grid_at(self, time_index: int) -> GridDefinition:
        return self.grid if self.time_grids is None else self.time_grids[time_index]

    def distinct_grids(self) -> list[GridDefinition]:
        """Every position the grid takes, in time order, without repeats."""
        if self.time_grids is None:
            return [self.grid]
        seen: dict[tuple, GridDefinition] = {}
        for g in self.time_grids:
            seen.setdefault(tuple(g.transform), g)
        return list(seen.values())


@dataclass
class FieldResult:
    """A computed 2-D field for one variable at one valid time."""

    variable: str
    data: np.ndarray | None  # float, (height, width), north-up, NaN = missing
    # (None once written out, e.g. in records returned by worker processes)
    valid_time: datetime
    # For period quantities (e.g. precipitation over an output interval).
    start_time: datetime | None = None
    end_time: datetime | None = None
    # Free-form provenance: which native variables were used and how.
    provenance: dict[str, Any] = field(default_factory=dict)


class ModelAdapter(abc.ABC):
    """Interface every model adapter implements."""

    #: Short model name, e.g. "WRF".
    name: str = ""

    @abc.abstractmethod
    def discover(self, directory: Path) -> list[ModelDataset]:
        """Find this model's output in ``directory`` and group it into datasets."""

    @abc.abstractmethod
    def supported_variables(self) -> list[str]:
        """Standard variable names this adapter can produce."""

    @abc.abstractmethod
    def compute(
        self, dataset: ModelDataset, variable: str, time_index: int
    ) -> FieldResult | None:
        """Compute ``variable`` at ``dataset.valid_times[time_index]``.

        Returns ``None`` when the variable is legitimately undefined at that time
        (e.g. interval precipitation at the first output time). Raises
        :class:`~modelatlasify.core.errors.VariableUnavailableError` when the
        required model fields are missing.
        """

    def prepare(self, dataset: ModelDataset, variables: list[str]) -> None:  # noqa: B027
        """Optional hook, run once per dataset in the main process before any
        :meth:`compute` call. Adapters use it for work that depends on earlier
        times (e.g. accumulation checks) and store the results in
        ``dataset.adapter_state``, so parallel workers need not redo it."""

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release any open file handles."""
