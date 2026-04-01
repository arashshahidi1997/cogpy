from __future__ import annotations

import time
from typing import Literal

import numpy as np
import xarray as xr

from .multichannel_viewer import MultichannelViewer
from .topomap import TopoMap

__all__ = ["link_topomap_to_viewer"]


Scalar = Literal["rms", "mean", "std", "max"]


def link_topomap_to_viewer(
    topomap: TopoMap,
    viewer: MultichannelViewer,
    sig_grid: xr.DataArray,  # ("time","ML","AP")
    time_dim: str = "time",
    *,
    scalar: Scalar = "rms",
    debounce_ms: int = 150,
) -> None:
    """
    Wire viewer's RangeX stream → recompute scalar over visible window → topomap.update().

    The link is one-way: viewer drives topomap, not vice versa.
    Works by subscribing to viewer._range_stream events.
    """
    if scalar not in {"rms", "mean", "std", "max"}:
        raise ValueError("scalar must be one of: 'rms', 'mean', 'std', 'max'")
    if int(debounce_ms) < 0:
        raise ValueError("debounce_ms must be >= 0")

    # Ensure stream exists.
    viewer.panel()
    if (
        not hasattr(viewer, "_range_stream") or viewer._range_stream is None
    ):  # noqa: SLF001
        raise ValueError(
            "viewer has no _range_stream; call viewer.panel() before linking"
        )

    if (
        time_dim not in sig_grid.dims
        or "ML" not in sig_grid.dims
        or "AP" not in sig_grid.dims
    ):
        raise ValueError(
            f"sig_grid must have dims including ('{time_dim}','ML','AP'); got dims={tuple(sig_grid.dims)}"
        )

    n_ap = int(sig_grid.sizes["AP"])
    n_ml = int(sig_grid.sizes["ML"])
    if topomap.n_ap != n_ap or topomap.n_ml != n_ml:
        raise ValueError(
            "topomap shape must match sig_grid shape: "
            f"topomap=({topomap.n_ap},{topomap.n_ml}) sig_grid=({n_ap},{n_ml})"
        )

    debounce_s = float(debounce_ms) / 1000.0
    last_t = 0.0

    def _compute_and_update(t0: float, t1: float) -> None:
        # Window slice in physical time coords.
        win = sig_grid.sel(**{time_dim: slice(float(t0), float(t1))})
        if int(win.sizes.get(time_dim, 0)) < 2:
            return

        if scalar == "mean":
            reduced = win.mean(dim=time_dim)
        elif scalar == "std":
            reduced = win.std(dim=time_dim)
        elif scalar == "max":
            reduced = win.max(dim=time_dim)
        else:
            # True RMS: sqrt(mean(x^2))
            reduced = np.sqrt((win * win).mean(dim=time_dim))

        # Ensure (AP, ML) for TopoMap.update (expects n_ap x n_ml).
        reduced = reduced.transpose("AP", "ML")
        arr = np.asarray(reduced.values, dtype=float)
        topomap.update(arr)

    def _on_range_event(**kwargs) -> None:
        nonlocal last_t
        xrng = kwargs.get("x_range", None)
        if xrng is None or xrng == (None, None):
            return
        t0, t1 = xrng
        if t0 is None or t1 is None:
            return
        now = time.monotonic()
        if debounce_s > 0 and (now - last_t) < debounce_s:
            return
        last_t = now
        _compute_and_update(float(t0), float(t1))

    # Subscribe to the HoloViews stream directly.
    stream = viewer._range_stream  # noqa: SLF001
    if hasattr(stream, "add_subscriber"):
        stream.add_subscriber(_on_range_event)
    else:
        raise ValueError("viewer._range_stream does not support add_subscriber()")

    # Initialize once using the current range, if available.
    try:
        lo, hi = stream.x_range
        if lo is not None and hi is not None:
            _compute_and_update(float(lo), float(hi))
    except Exception:  # noqa: BLE001
        pass
