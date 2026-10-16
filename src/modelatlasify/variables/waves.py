"""Ocean surface wave variables (from a wave model such as DWM)."""

from __future__ import annotations

from modelatlasify.core.variables import VariableDefinition, register

_PERIOD_LEVELS = tuple(float(x) for x in range(0, 13, 1))  # 0-12 s; longer periods use the "max" arrow

SIGNIFICANT_WAVE_HEIGHT = register(
    VariableDefinition(
        name="significant_wave_height",
        long_name="Significant wave height",
        units="m",
        standard_name="sea_surface_wave_significant_height",
        description=(
            "Significant wave height (4 times the square root of the wave energy "
            "spectrum's zeroth moment), instantaneous at the valid time. Missing over land."
        ),
        cmap="turbo",
        levels=(0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0),
        extend="max",
        aliases=("swh", "hs", "wave_height", "waves"),
    )
)

MEAN_WAVE_PERIOD = register(
    VariableDefinition(
        name="mean_wave_period",
        long_name="Mean wave period",
        units="s",
        standard_name="sea_surface_wave_mean_period",
        description="Mean wave period of the wave spectrum, instantaneous at the valid time. "
                    "Missing over land.",
        cmap="viridis",
        levels=_PERIOD_LEVELS,
        extend="max",
        aliases=("mwp", "waves"),
    )
)

DOMINANT_WAVE_PERIOD = register(
    VariableDefinition(
        name="dominant_wave_period",
        long_name="Dominant wave period",
        units="s",
        standard_name="sea_surface_wave_period_at_variance_spectral_density_maximum",
        description="Period of the most energetic (peak) part of the wave spectrum, "
                    "instantaneous at the valid time. Missing over land.",
        cmap="viridis",
        levels=_PERIOD_LEVELS,
        extend="max",
        aliases=("dwp", "peak_wave_period", "waves"),
    )
)
