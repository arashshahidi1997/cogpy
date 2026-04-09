"""Sample ECoG datasets for testing and tutorials.

Functions: ``load_sample``, ``load_raw_sample`` — load bundled sample recordings.
GUI bundles: ``ieeg_grid_bundle``, ``spectrogram_bursts_bundle`` — pre-packaged
data + layout for interactive visualization demos.
"""

from .load import load_sample, load_raw_sample
from .gui_bundles import ieeg_grid_bundle, spectrogram_bursts_bundle

# from .make import

__all__ = [
    "load_sample",
    "load_raw_sample",
    "ieeg_grid_bundle",
    "spectrogram_bursts_bundle",
]
