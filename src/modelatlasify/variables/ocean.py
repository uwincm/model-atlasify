"""Ocean surface variables (from an ocean model such as HYCOM)."""

from __future__ import annotations

import warnings

with warnings.catch_warnings():
    # cmocean 3.0.x builds its colour maps with an argument matplotlib 3.11
    # deprecates; harmless, and not ours to fix.
    warnings.filterwarnings("ignore", message="Passing 'N' to ListedColormap is deprecated")
    import cmocean

from modelatlasify.core.variables import VariableDefinition, VectorStyle, register

#: Surface current vectors: twice as dense as the wind's (one every 125 km, at
#: least 40 across a map), 125 km per 1 m s-1 at that spacing (so a 1 m s-1
#: arrow is one spacing long), growing up to 2 m s-1. Typical currents of a
#: few tenths of a m s-1 give arrows a few tenths of the spacing.
CURRENT_VECTORS = VectorStyle(spacing_km=125.0, min_across=40, km_per_unit=125.0,
                              max_speed=2.0, units="m/s", units_per_m_s=1.0,
                              key_speeds=(1.0, 0.5, 0.2, 0.1, 0.05))
_CURRENTS = ("surface_u", "surface_v")

SST = register(
    VariableDefinition(
        name="sst",
        long_name="Sea surface temperature",
        units="degC",
        standard_name="sea_surface_temperature",
        description=(
            "Temperature of the ocean model's top layer, instantaneous at the valid time. "
            "Missing over land."
        ),
        cmap=cmocean.cm.thermal,  # cmocean (Thyng et al. 2016) "thermal"
        levels=tuple(float(x) for x in range(10, 33)),  # 10-32 C every 1 C
        extend="both",
        vectors=_CURRENTS,
        vector_style=CURRENT_VECTORS,
        aliases=("ocean", "sea_surface_temperature"),
    )
)

SURFACE_SALINITY = register(
    VariableDefinition(
        name="surface_salinity",
        long_name="Sea surface salinity",
        units="psu",
        standard_name="sea_surface_salinity",
        description=(
            "Salinity of the ocean model's top layer (practical salinity), instantaneous "
            "at the valid time. Missing over land."
        ),
        cmap=cmocean.cm.haline,  # cmocean (Thyng et al. 2016) "haline"
        levels=tuple(30.0 + 0.5 * k for k in range(17)),  # 30-38 every 0.5
        extend="both",
        vectors=_CURRENTS,
        vector_style=CURRENT_VECTORS,
        constant_is_suspicious=True,
        aliases=("sss", "salinity", "ocean"),
    )
)

SURFACE_CURRENTS = register(
    VariableDefinition(
        name="surface_currents",
        long_name="Surface current speed",
        units="m s-1",
        standard_name="sea_water_speed",
        description=(
            "Speed of the total (baroclinic + barotropic) current in the ocean model's top "
            "layer, sqrt(surface_u^2 + surface_v^2), instantaneous at the valid time. "
            "Missing over land."
        ),
        cmap="YlOrRd",
        levels=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 1.75, 2.0),
        extend="max",
        vectors=_CURRENTS,
        vector_style=CURRENT_VECTORS,
        max_zero_fraction=0.5,
        aliases=("currents", "ocean"),
    )
)

SURFACE_U = register(
    VariableDefinition(
        name="surface_u",
        long_name="Eastward surface current",
        units="m s-1",
        standard_name="eastward_sea_water_velocity",
        description=(
            "Eastward (zonal) component of the total current in the ocean model's top "
            "layer, instantaneous at the valid time. Missing over land."
        ),
        cmap="RdBu_r",
        vmin=-2.0,
        vmax=2.0,
        extend="both",
        maps=False,
        constant_is_suspicious=False,
        aliases=("currents", "ocean"),
    )
)

SURFACE_V = register(
    VariableDefinition(
        name="surface_v",
        long_name="Northward surface current",
        units="m s-1",
        standard_name="northward_sea_water_velocity",
        description=(
            "Northward (meridional) component of the total current in the ocean model's "
            "top layer, instantaneous at the valid time. Missing over land."
        ),
        cmap="RdBu_r",
        vmin=-2.0,
        vmax=2.0,
        extend="both",
        maps=False,
        constant_is_suspicious=False,
        aliases=("currents", "ocean"),
    )
)
