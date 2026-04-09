"""I/O subpackage — file readers, writers, and format converters.

Generic modules (usable by anyone):
    converters, save_utils, sidecars

Lab-internal modules (assume specific recording setups and directory
layouts used in the Bhatt Lab; included for convenience but not part
of the stable public API):
    ecog_io, ecephys_io, ieeg_io, ieeg_sidecars,
    load_meta, xml_io, xml_anat_map

Example
-------
::

    from cogpy.io import ieeg_io, ecephys_io
"""

from lazy_loader import attach
from typing import TYPE_CHECKING

__getattr__, __dir__, __all__ = attach(
    __name__,
    submodules=[
        "converters",
        "ecephys_io",
        "ecog_io",
        "ieeg_io",
        "ieeg_sidecars",
        "load_meta",
        "save_utils",
        "sidecars",
        "xml_anat_map",
        "xml_io",
    ],
)

if TYPE_CHECKING:
    from . import (
        converters,
        ecephys_io,
        ecog_io,
        ieeg_io,
        ieeg_sidecars,
        load_meta,
        save_utils,
        sidecars,
        xml_anat_map,
        xml_io,
    )
