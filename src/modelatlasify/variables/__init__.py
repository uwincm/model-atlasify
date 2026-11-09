"""Standard variable definitions. Importing this package registers them all.

The import order is the registration order, which sets the row order of
model-images.html.
"""

from modelatlasify.variables import precipitation, spray, waves, wind  # noqa: F401
from modelatlasify.variables import temperature  # noqa: F401, I001 - row after wind
