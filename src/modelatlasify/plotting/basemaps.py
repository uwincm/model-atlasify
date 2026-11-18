"""Web-map tile backgrounds for maps (``VariableDefinition.basemap``).

Kept free of heavy imports so the command line can list the choices.
"""

from __future__ import annotations

#: Basemap name -> (XYZ tile URL, attribution printed on the map).
BASEMAPS: dict[str, tuple[str, str]] = {
    # Esri World Imagery: satellite/aerial imagery only (no roads or labels).
    "esri_world_imagery": (
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/"
        "{z}/{y}/{x}",
        "Imagery: Esri, Maxar, Earthstar Geographics, and the GIS User Community"),
    # Google Maps satellite imagery with roads and labels, the same tiles
    # QGIS's "Google Hybrid" XYZ basemap uses.
    "google_hybrid": ("https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
                      "Imagery \N{COPYRIGHT SIGN} Google"),
}
#: The default for maps that have a basemap.
DEFAULT_BASEMAP = "esri_world_imagery"
#: Basemap choice that keeps the plain land/sea background.
NO_BASEMAP = "none"
