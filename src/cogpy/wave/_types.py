"""Shared data containers for travelling-wave analysis."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
from numpy.typing import NDArray


class PatternType(Enum):
    """Spatial pattern classification for velocity fields."""

    planar = "planar"
    rotating = "rotating"
    spiral = "spiral"
    source = "source"
    sink = "sink"
    mixed = "mixed"
    uncertain = "uncertain"


@dataclass
class Geometry:
    """Spatial layout of recording array.

    Parameters
    ----------
    dx : float or None
        Grid spacing along first spatial axis (AP).
    dy : float or None
        Grid spacing along second spatial axis (ML).
    coords : ndarray of shape (N, 2) or None
        Arbitrary electrode positions for irregular arrays.

    Examples
    --------
    >>> g = Geometry.regular(0.4, 0.4)
    >>> g.is_regular
    True
    >>> g = Geometry.irregular(np.array([[0, 0], [1, 0], [0.5, 0.8]]))
    >>> g.is_regular
    False
    """

    dx: Optional[float] = None
    dy: Optional[float] = None
    coords: Optional[NDArray] = None

    @classmethod
    def regular(cls, dx: float, dy: float | None = None) -> Geometry:
        """Create a regular-grid geometry.

        Parameters
        ----------
        dx : float
            Grid spacing along AP axis.
        dy : float, optional
            Grid spacing along ML axis. Defaults to *dx*.
        """
        if dy is None:
            dy = dx
        return cls(dx=dx, dy=dy)

    @classmethod
    def irregular(cls, coords: NDArray) -> Geometry:
        """Create an irregular-array geometry.

        Parameters
        ----------
        coords : ndarray, shape (N, 2)
            Electrode (x, y) positions.
        """
        coords = np.asarray(coords, dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError("coords must have shape (N, 2)")
        return cls(coords=coords)

    @property
    def is_regular(self) -> bool:
        """True when the geometry describes a uniform grid."""
        return self.dx is not None and self.dy is not None and self.coords is None


@dataclass
class WaveEstimate:
    """Result of a single wave-parameter estimation.

    Parameters
    ----------
    direction : float
        Propagation direction in radians (0 = positive AP).
    speed : float
        Propagation speed in spatial-units / second.
    frequency : float
        Temporal frequency in Hz.
    wavenumber : float or None
        Spatial wavenumber magnitude (1 / wavelength).
    wavelength : float or None
        Spatial wavelength in the same units as the geometry.
    pattern_type : PatternType
        Classification of the spatial pattern.
    confidence : float
        Scalar confidence score in [0, 1].
    fit_quality : float
        Goodness-of-fit score in [0, 1] (e.g. PGD).
    support_mask : ndarray or None
        Boolean mask indicating which spatial locations contributed.
    """

    direction: float
    speed: float
    frequency: float
    wavenumber: Optional[float] = None
    wavelength: Optional[float] = None
    pattern_type: PatternType = PatternType.uncertain
    confidence: float = 0.0
    fit_quality: float = 0.0
    support_mask: Optional[NDArray] = None

    def __post_init__(self):
        # Auto-fill wavelength / wavenumber if one is provided.
        if (
            self.wavenumber is not None
            and self.wavelength is None
            and self.wavenumber > 0
        ):
            self.wavelength = 1.0 / self.wavenumber
        elif (
            self.wavelength is not None
            and self.wavenumber is None
            and self.wavelength > 0
        ):
            self.wavenumber = 1.0 / self.wavelength
