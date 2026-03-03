"""
Time window controller for TensorScope.

Manages the current visible time window (t0, t1) with validation and bounds checking.
"""

from __future__ import annotations

import param


class TimeWindowCtrl(param.Parameterized):
    """
    Controller for visible time window.

    Manages (t0, t1) window with:
    - Bounds checking (stays within data limits)
    - Ordering validation (t0 < t1)
    - Snap-to-bounds behavior

    This is authoritative state for the visible time range.
    Views can derive from this (e.g., set Bokeh x_range) or report back to it.

    Parameters
    ----------
    bounds : tuple[float, float]
        Data time limits (min, max)
    window : tuple[float, float]
        Current visible window (t0, t1)
    snap : bool
        Whether to snap to bounds when window extends beyond

    Examples
    --------
    >>> ctrl = TimeWindowCtrl(bounds=(0.0, 10.0))
    >>> ctrl.set_window(2.0, 5.0)
    >>> ctrl.window
    (2.0, 5.0)
    >>> ctrl.recenter(7.0, width_s=2.0)
    >>> ctrl.window
    (6.0, 8.0)
    """

    bounds = param.Range(default=(0.0, 1.0), doc="Data time limits")
    window = param.Range(default=(0.0, 1.0), doc="Current visible window (t0, t1)")
    snap = param.Boolean(default=True, doc="Snap to bounds")

    def __init__(self, bounds: tuple[float, float] = (0.0, 1.0), **params):
        """Initialize with data bounds."""
        super().__init__(bounds=bounds, window=bounds, **params)

    def set_bounds(self, t_min: float, t_max: float) -> None:
        """
        Set data time bounds.

        Parameters
        ----------
        t_min, t_max : float
            New bounds
        """
        t_min_f = float(t_min)
        t_max_f = float(t_max)
        if t_min_f >= t_max_f:
            raise ValueError(f"bounds must have t_min < t_max, got {t_min_f}, {t_max_f}")
        self.bounds = (t_min_f, t_max_f)

        if bool(self.snap):
            self._clip_window_to_bounds()

    def set_window(self, t0: float, t1: float) -> None:
        """
        Set visible window.

        Parameters
        ----------
        t0, t1 : float
            Window start and end times
        """
        t0_f = float(t0)
        t1_f = float(t1)
        if t0_f >= t1_f:
            raise ValueError(f"window must have t0 < t1, got {t0_f}, {t1_f}")

        if bool(self.snap):
            t_min, t_max = (float(self.bounds[0]), float(self.bounds[1]))
            t0_f = max(t_min, min(t0_f, t_max))
            t1_f = max(t_min, min(t1_f, t_max))

            if t0_f >= t1_f:
                span = max(0.0, t_max - t_min)
                eps = min(1e-3, span / 1000.0) if span > 0 else 1e-3
                t1_f = min(t_max, t0_f + eps)
                if t0_f >= t1_f:
                    t0_f = max(t_min, t1_f - eps)
                if t0_f >= t1_f:
                    t0_f, t1_f = t_min, t_max

        self.window = (t0_f, t1_f)

    def recenter(self, t_center: float, width_s: float) -> None:
        """
        Center window at time with given width.

        Parameters
        ----------
        t_center : float
            Center time
        width_s : float
            Window width in seconds
        """
        half_width = float(width_s) / 2.0
        self.set_window(float(t_center) - half_width, float(t_center) + half_width)

    def _clip_window_to_bounds(self) -> None:
        """Clip current window to bounds."""
        t0, t1 = self.window
        self.set_window(float(t0), float(t1))

