"""Exceptions with messages meant to be shown to CLI users."""


class ModelAtlasifyError(Exception):
    """Base class for expected, user-facing errors."""


class DiscoveryError(ModelAtlasifyError):
    """No usable model output was found, or it could not be grouped."""


class VariableUnavailableError(ModelAtlasifyError):
    """A requested variable cannot be computed from the available model fields."""


class GeoreferenceError(ModelAtlasifyError):
    """The model grid's projection could not be determined or verified."""


class FieldQualityError(ModelAtlasifyError):
    """A field was computed but failed a sanity check (e.g. corrupt input data).

    The pipeline skips such products and records why, instead of publishing them.
    """
