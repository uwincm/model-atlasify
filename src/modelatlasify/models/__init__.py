"""Model adapters, looked up by name."""

from __future__ import annotations

from modelatlasify.core.model import ModelAdapter


def get_adapter(name: str) -> ModelAdapter:
    key = name.lower()
    if key == "wrf":
        from modelatlasify.models.wrf import WRFAdapter

        return WRFAdapter()
    if key in ("dwm", "umwm"):  # UMWM is DWM's former name
        from modelatlasify.models.dwm import DWMAdapter

        return DWMAdapter()
    raise KeyError(f"Unknown model '{name}'. Available: {', '.join(AVAILABLE_MODELS)}")


AVAILABLE_MODELS = ("wrf", "dwm")
