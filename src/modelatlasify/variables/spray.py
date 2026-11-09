"""Sea spray variables (from an atmosphere model with spray-mediated fluxes,
e.g. WRF in UWIN-CM with ``sprayMediatedHeatFluxes = .T.``)."""

from __future__ import annotations

from modelatlasify.core.variables import VariableDefinition, register

#: 1-2-5 steps per decade, 1e-7 to 1e-2 kg m-2 s-1. With the evenly spaced bins
#: of a BoundaryNorm colour bar this is a logarithmic colour scale. Strong
#: storms reach ~0.1; those values use the "max" arrow.
_MASS_FLUX_LEVELS = tuple(float(f"{m}e{e}") for e in range(-7, -2) for m in (1, 2, 5)) + (1e-2,)

SPRAY_MASS_FLUX = register(
    VariableDefinition(
        name="spray_mass_flux",
        long_name="Sea spray mass flux",
        units="kg m-2 s-1",
        description=(
            "Mass flux of sea spray droplets from the ocean surface, instantaneous "
            "at the valid time. 0 where the wind is too weak for spray; missing "
            "where the spray is not computed (e.g. outside the wave model's domain)."
        ),
        cmap="viridis",
        levels=_MASS_FLUX_LEVELS,
        extend="max",
        transparent_below=_MASS_FLUX_LEVELS[0],
        constant_is_suspicious=False,  # 0 everywhere at initialization or in light winds
        vectors=("u10", "v10"),  # 10 m wind vectors on the maps
        aliases=("spray", "mspr", "spray_mass"),
    )
)

SPRAY_LATENT_HEAT_FLUX = register(
    VariableDefinition(
        name="spray_latent_heat_flux",
        long_name="Spray-mediated latent heat flux",
        units="W m-2",
        description=(
            "Latent heat flux carried by sea spray (upward positive), instantaneous "
            "at the valid time."
        ),
        cmap="YlOrRd",
        levels=tuple(float(x) for x in range(0, 32, 2)),
        extend="max",
        transparent_below=0.01,
        constant_is_suspicious=False,
        vectors=("u10", "v10"),  # 10 m wind vectors on the maps
        aliases=("spray", "hlspr", "spray_lhf"),
    )
)

SPRAY_SENSIBLE_HEAT_FLUX = register(
    VariableDefinition(
        name="spray_sensible_heat_flux",
        long_name="Spray-mediated sensible heat flux",
        units="W m-2",
        description=(
            "Sensible heat flux carried by sea spray (upward positive), instantaneous "
            "at the valid time."
        ),
        cmap="YlOrRd",
        levels=tuple(round(0.2 * k, 1) for k in range(16)),
        extend="max",
        transparent_below=0.001,
        constant_is_suspicious=False,
        vectors=("u10", "v10"),  # 10 m wind vectors on the maps
        aliases=("spray", "hsspr", "spray_shf"),
    )
)
