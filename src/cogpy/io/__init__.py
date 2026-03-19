"""I/O subpackage.

Input and output functions for various file formats.

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
