"""Waveform detection, extraction, feature analysis, and travelling-wave methods.

Waveform analysis: ``detect``, ``features``, ``process``, ``plot``, ``utils``.
Travelling-wave analysis: ``phase_gradient``, ``optical_flow``, ``vector_field``,
``generalized_phase``, ``kw_spectrum``, ``beamforming``, ``multitaper_nd``,
``surrogates``, ``synthetic``.
"""

from lazy_loader import attach
from typing import TYPE_CHECKING

__getattr__, __dir__, __all__ = attach(
    __name__,
    submodules=[
        "detect",
        "features",
        "plot",
        "process",
        "utils",
        # Travelling-wave analysis
        "_types",
        "synthetic",
        "phase_gradient",
        "kw_spectrum",
        "optical_flow",
        "vector_field",
        "surrogates",
        "beamforming",
        "multitaper_nd",
        "generalized_phase",
    ],
)

if TYPE_CHECKING:
    from . import (
        detect,
        features,
        plot,
        process,
        utils,
        _types,
        synthetic,
        phase_gradient,
        kw_spectrum,
        optical_flow,
        vector_field,
        surrogates,
        beamforming,
        multitaper_nd,
        generalized_phase,
    )
