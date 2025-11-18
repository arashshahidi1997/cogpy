from ..core.wave import *

# cogpy/wave/__init__.py
from cogpy.core import wave as _impl  # lets us grab __all__ cleanly
from cogpy.core.wave import *  # re-export the public API

__all__ = getattr(_impl, "__all__", [])  # optional: keep curated __all__
