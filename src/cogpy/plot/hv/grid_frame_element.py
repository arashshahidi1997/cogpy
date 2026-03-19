"""
grid_frame_element.py
=====================
Spatial AP×ML frame viewer driven by a TimeHair.

Shows one scalar frame (one value per electrode) at the time selected by
a ``TimeHair``. Three reduction modes:

- ``"instantaneous"``: snap to the nearest sample, no windowing.
- ``"rms"``: RMS over ``window_s`` seconds centred on ``t``.
- ``"mean"``: arithmetic mean over ``window_s`` seconds centred on ``t``.

When a ``ProcessingChain`` is provided, windowed modes call
``chain.get_window(t0, t1)`` before reducing, so the displayed frame
reflects the same processing applied in the multichannel viewer.

Typical usage
-------------
    hair  = TimeHair()
    frame = GridFrameElement(sig_grid, hair, mode="rms", chain=chain)
    pn.Row(frame.panel(), viewer.panel()).servable()
"""
from __future__ import annotations

import numpy as np
import xarray as xr
from cogpy.utils.imports import import_optional
pn = import_optional("panel")

from .topomap import TopoMap

__all__ = ["GridFrameElement"]


class GridFrameElement:
    """
    Spatial AP×ML frame viewer driven by a :class:`TimeHair`.

    Parameters
    ----------
    sig_grid : xr.DataArray
        Must have dims ``("time", "AP", "ML")`` (order flexible).
    hair : TimeHair
        Shared time parameter.  Changes to ``hair.t`` trigger frame updates.
    mode : {"instantaneous", "rms", "mean"}
        Scalar reduction mode.
    window_s : float
        Half-window in seconds used for ``"rms"`` and ``"mean"`` modes.
    chain : ProcessingChain | None
        When provided, windowed modes process the slice through the chain
        before reducing.  The chain must have been constructed from the
        stacked ``(time, channel)`` form of the same recording.
    colormap : str
        Passed to :class:`TopoMap`.
    symmetric : bool
        Passed to :class:`TopoMap`.  Useful for signed scalars (e.g. mean).
    title : str
        Plot title prefix.  Mode is appended automatically.
    **topomap_kwargs
        Forwarded to :class:`TopoMap` (e.g. ``width``, ``height``,
        ``electrode_radius``).
    """

    def __init__(
        self,
        sig_grid: xr.DataArray,
        hair,
        *,
        mode: str = "instantaneous",
        window_s: float = 0.1,
        chain=None,
        colormap: str = "viridis",
        symmetric: bool = False,
        title: str = "Spatial frame",
        **topomap_kwargs,
    ):
        if mode not in {"instantaneous", "rms", "mean"}:
            raise ValueError(
                f"mode must be 'instantaneous', 'rms', or 'mean'; got {mode!r}"
            )
        for dim in ("time", "AP", "ML"):
            if dim not in sig_grid.dims:
                raise ValueError(
                    f"sig_grid missing dim {dim!r}, got dims={tuple(sig_grid.dims)}"
                )

        self._sig = sig_grid.transpose("time", "AP", "ML")
        self._hair = hair
        self._mode = str(mode)
        self._window_s = float(window_s)
        self._chain = chain
        self._t_vals = np.asarray(self._sig.coords["time"].values, dtype=np.float64)

        full_title = f"{title}  [{mode}]"
        init_frame = np.asarray(self._sig.isel(time=0).values, dtype=float)
        ap_coords = np.asarray(self._sig.coords["AP"].values)
        ml_coords = np.asarray(self._sig.coords["ML"].values)

        self._tmap = TopoMap(
            init_frame,
            ap_coords=ap_coords,
            ml_coords=ml_coords,
            colormap=colormap,
            symmetric=symmetric,
            title=full_title,
            **topomap_kwargs,
        )

        # Wire hair watcher — fires on every t change.
        self._hair_watcher = hair.param.watch(self._on_t, "t")

        # Trigger immediately if t is already set.
        if hair.t is not None:
            self._on_t()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @property
    def topomap(self) -> TopoMap:
        """The underlying :class:`TopoMap` (for tap callbacks, etc.)."""
        return self._tmap

    def panel(self) -> pn.viewable.Viewable:
        """Return the Panel pane wrapping the spatial frame."""
        return self._tmap.panel()

    def dispose(self) -> None:
        """Detach watchers to avoid leaks when layers are removed."""
        try:
            self._hair.param.unwatch(self._hair_watcher)
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_t(self, event=None) -> None:
        t = self._hair.t
        if t is None:
            return
        try:
            frame = self._compute_frame(float(t))
        except Exception:  # noqa: BLE001
            return
        self._tmap.update(frame)

    def _compute_frame(self, t: float) -> np.ndarray:
        if self._mode == "instantaneous":
            return self._frame_instantaneous(t)

        half = self._window_s / 2.0
        t0, t1 = t - half, t + half

        if self._chain is not None:
            return self._frame_windowed_chain(t0, t1, t)
        return self._frame_windowed_raw(t0, t1, t)

    def _frame_instantaneous(self, t: float) -> np.ndarray:
        idx = int(np.argmin(np.abs(self._t_vals - t)))
        return np.asarray(self._sig.isel(time=idx).values, dtype=float)

    def _frame_windowed_raw(self, t0: float, t1: float, t_mid: float) -> np.ndarray:
        win = self._sig.sel(time=slice(t0, t1))
        if win.sizes["time"] == 0:
            return self._frame_instantaneous(t_mid)
        if self._mode == "rms":
            return np.sqrt(np.asarray((win ** 2).mean("time").values, dtype=float))
        return np.asarray(win.mean("time").values, dtype=float)

    def _frame_windowed_chain(self, t0: float, t1: float, t_mid: float) -> np.ndarray:
        win = self._chain.get_window(t0, t1)
        if win.sizes["time"] == 0:
            return self._frame_instantaneous(t_mid)

        if "AP" in win.dims and "ML" in win.dims:
            win_grid = win.transpose("time", "AP", "ML")
            if self._mode == "rms":
                return np.sqrt(
                    np.asarray((win_grid ** 2).mean("time").values, dtype=float)
                )
            return np.asarray(win_grid.mean("time").values, dtype=float)

        # Flat (time, channel) form — reduce then reconstruct grid.
        if self._mode == "rms":
            reduced = np.sqrt((win ** 2).mean("time"))
        else:
            reduced = win.mean("time")
        return self._channel_to_grid(reduced)

    def _channel_to_grid(self, reduced: xr.DataArray) -> np.ndarray:
        """
        Reconstruct a 2D (n_ap, n_ml) array from a flat (channel,) DataArray
        that carries AP and ML as auxiliary coordinates.
        """
        if "AP" not in reduced.coords or "ML" not in reduced.coords:
            raise ValueError(
                "Chain output missing AP/ML coords; cannot reconstruct spatial frame."
            )

        ap_u = np.asarray(self._sig.coords["AP"].values)
        ml_u = np.asarray(self._sig.coords["ML"].values)
        ap_to_i = {v: i for i, v in enumerate(ap_u)}
        ml_to_i = {v: i for i, v in enumerate(ml_u)}

        frame = np.full((len(ap_u), len(ml_u)), np.nan)
        ap_vals = reduced.coords["AP"].values
        ml_vals = reduced.coords["ML"].values
        vals = np.asarray(reduced.values, dtype=float)

        for ch_i in range(len(vals)):
            ai = ap_to_i.get(ap_vals[ch_i])
            mi = ml_to_i.get(ml_vals[ch_i])
            if ai is not None and mi is not None:
                frame[ai, mi] = vals[ch_i]

        return frame
