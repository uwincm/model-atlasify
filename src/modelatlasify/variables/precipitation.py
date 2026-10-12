"""Precipitation: definitions and model-independent calculations."""

from __future__ import annotations

import numpy as np

from modelatlasify.core.variables import VariableDefinition, register

PRECIPITATION = register(
    VariableDefinition(
        name="precipitation",
        long_name="Accumulated precipitation",
        units="mm",
        standard_name="precipitation_amount",
        cell_methods="time: sum",
        description=(
            "Total precipitation (liquid-water equivalent) accumulated from the "
            "start of the domain's integration (see start_datetime) to the valid time."
        ),
        cmap="YlGnBu",
        levels=(0.1, 1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500),
        extend="max",
        transparent_below=0.1,
        constant_is_suspicious=False,
        aliases=("precip", "precipitation", "rain", "precipitation_accumulated"),
    )
)

PRECIPITATION_INTERVAL = register(
    VariableDefinition(
        name="precipitation_interval",
        long_name="Precipitation over output interval",
        units="mm",
        standard_name="precipitation_amount",
        cell_methods="time: sum",
        description=(
            "Total precipitation (liquid-water equivalent) that fell between the "
            "previous output time and the valid time, computed as the difference "
            "of consecutive accumulated totals."
        ),
        cmap="YlGnBu",
        levels=(0.1, 0.25, 0.5, 1, 2, 3, 5, 7.5, 10, 15, 20, 30, 50, 75),
        extend="max",
        transparent_below=0.1,
        constant_is_suspicious=False,
        aliases=("precip", "precipitation", "rain", "precipitation_rate"),
    )
)

#: Differences more negative than this (mm) are treated as a broken accumulation
#: (e.g. a reset) rather than float round-off.
NEGATIVE_TOLERANCE_MM = 0.01


def interval_precipitation(
    accum_end: np.ndarray, accum_start: np.ndarray
) -> tuple[np.ndarray, int]:
    """Precipitation between two times from two accumulated totals.

    Returns ``(amount, n_negative)``. Small negative differences produced by
    float32 round-off (> -NEGATIVE_TOLERANCE_MM) are clipped to 0. Larger
    negative differences cannot be physical; they are set to NaN and counted
    so the caller can report them instead of silently hiding them.
    """
    diff = np.asarray(accum_end, dtype=np.float64) - np.asarray(accum_start, dtype=np.float64)
    bad = diff < -NEGATIVE_TOLERANCE_MM
    out = np.where(diff < 0, 0.0, diff)
    out[bad] = np.nan
    return out, int(np.count_nonzero(bad))
