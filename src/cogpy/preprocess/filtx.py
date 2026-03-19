"""Backward-compatible shim — imports are now in ``filtering/`` subpackage.

All public names are re-exported here so existing imports continue to work::

    from cogpy.preprocess.filtx import bandpassx  # still works
    from cogpy.preprocess.filtering import bandpassx  # preferred
"""

from .filtering import *  # noqa: F401,F403
from .filtering import __all__, _apply_full_array, _fs_scalar  # noqa: F401
