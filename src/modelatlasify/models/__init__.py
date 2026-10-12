"""Model adapters, looked up by name."""

from __future__ import annotations

from modelatlasify.core.model import ModelAdapter


def get_adapter(name: str) -> ModelAdapter:
    key = name.lower()
    if key == "wrf":
        from modelatlasify.models.wrf import WRFAdapter

        return WRFAdapter()
    raise KeyError(f"Unknown model '{name}'. Available: wrf")


AVAILABLE_MODELS = ("wrf",)
