"""
TensorScope view system (v3.0).

Views are pure projections of tensors (read-only).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np
import xarray as xr

from ..state import SelectionState, TensorNode

__all__ = [
    "View",
    "register_view",
    "get_available_views",
    "ViewRegistry",
    "TimeseriesView",
    "SpatialMapView",
    "PSDAverageView",
    "PSDSpatialView",
]


class View(ABC):
    """Base class for tensor views."""

    name: str = "view"

    @abstractmethod
    def render(self, tensor: xr.DataArray, selection: SelectionState):
        raise NotImplementedError


def _canon_dims(dims: tuple[str, ...]) -> tuple[str, ...]:
    """
    Normalize common TensorScope dimension permutations.

    Notes
    -----
    Some upstream transforms (e.g. `specx.psdx`) append `freq` as the last dim,
    so we canonicalize by dim *set* for view discovery.
    """

    dset = frozenset(str(d) for d in dims)
    if dset == frozenset({"time", "AP", "ML"}):
        return ("time", "AP", "ML")
    if dset == frozenset({"time", "channel"}):
        return ("time", "channel")
    if dset == frozenset({"freq", "AP", "ML"}):
        return ("freq", "AP", "ML")
    if dset == frozenset({"freq", "channel"}):
        return ("freq", "channel")
    if dset == frozenset({"time", "freq", "AP", "ML"}):
        return ("time", "freq", "AP", "ML")
    if dset == frozenset({"time", "freq", "channel"}):
        return ("time", "freq", "channel")
    return tuple(str(d) for d in dims)


VIEW_REGISTRY: dict[tuple[str, ...], list[type[View]]] = {}


def register_view(dims: tuple[str, ...]) -> Callable[[type[View]], type[View]]:
    """Decorator to register a view for a dimension signature."""

    key = _canon_dims(tuple(dims))

    def decorator(cls: type[View]) -> type[View]:
        VIEW_REGISTRY.setdefault(key, []).append(cls)
        return cls

    return decorator


def get_available_views(tensor_node: TensorNode) -> list[type[View]]:
    """Get view classes available for a tensor based on its dimensions."""

    return VIEW_REGISTRY.get(_canon_dims(tensor_node.dims), [])


@register_view(("time", "AP", "ML"))
class TimeseriesView(View):
    """Stacked timeseries (channels flattened from AP×ML)."""

    name = "timeseries"

    def render(self, tensor: xr.DataArray, selection: SelectionState):
        import holoviews as hv

        hv.extension("bokeh")

        from cogpy.core.spectral.psd_utils import stack_spatial_dims

        stacked = stack_spatial_dims(tensor)
        if "time" not in stacked.dims:
            return hv.Div("<b>No 'time' dim</b>")

        n_ch = int(min(int(stacked.sizes.get("channel", 1)), 32))
        curves = []
        for i in range(n_ch):
            ch_data = stacked.isel(channel=i)
            t = np.asarray(ch_data["time"].values, dtype=float)
            y = np.asarray(ch_data.values, dtype=float)
            offset = float(i) * 10.0
            curves.append(
                hv.Curve((t, y + offset), kdims=["time"], vdims=["amplitude"]).opts(
                    color="blue", alpha=0.7
                )
            )

        vline = hv.VLine(float(selection.time)).opts(color="red", line_dash="dashed", line_width=2)
        return (
            hv.Overlay([*curves, vline])
            .opts(width=600, height=400, xlabel="Time (s)", ylabel="Channel (stacked)", title="Timeseries")
        )


@register_view(("time", "AP", "ML"))
class SpatialMapView(View):
    """2D spatial heatmap at selected time (AP×ML)."""

    name = "spatial_map"

    def render(self, tensor: xr.DataArray, selection: SelectionState):
        import holoviews as hv

        hv.extension("bokeh")

        if "time" not in tensor.dims:
            return hv.Div("<b>No 'time' dim</b>")
        if not (("AP" in tensor.dims) and ("ML" in tensor.dims)):
            return hv.Div("<b>No (AP, ML) dims</b>")

        slice_data = tensor.sel(time=float(selection.time), method="nearest").transpose("AP", "ML")
        vals = np.asarray(slice_data.values, dtype=float)

        n_ap = int(slice_data.sizes.get("AP", vals.shape[0] if vals.ndim > 0 else 0))
        n_ml = int(slice_data.sizes.get("ML", vals.shape[1] if vals.ndim > 1 else 0))
        ap_idx = np.arange(n_ap)
        ml_idx = np.arange(n_ml)

        return hv.Image((ml_idx, ap_idx, vals), kdims=["ML", "AP"], vdims=["amplitude"]).opts(
            cmap="RdBu_r",
            colorbar=True,
            tools=["hover"],
            xlabel="ML (index)",
            ylabel="AP (index)",
            title=f"Spatial Map @ {float(selection.time):.3f} s",
            width=420,
            height=420,
            aspect="equal",
            invert_yaxis=True,
        )


@register_view(("freq", "AP", "ML"))
class PSDAverageView(View):
    """Mean PSD curve (+/- std) across spatial dims."""

    name = "psd_average"

    def render(self, tensor: xr.DataArray, selection: SelectionState):
        import holoviews as hv

        hv.extension("bokeh")

        if "freq" not in tensor.dims:
            return hv.Div("<b>No 'freq' dim</b>")

        avg_dims = [d for d in ("AP", "ML", "channel") if d in tensor.dims]
        if avg_dims:
            psd_mean = tensor.mean(dim=avg_dims)
            psd_std = tensor.std(dim=avg_dims)
        else:
            psd_mean = tensor
            psd_std = xr.zeros_like(tensor)

        f = np.asarray(psd_mean["freq"].values, dtype=float)
        y = np.asarray(psd_mean.values, dtype=float)
        s = np.asarray(psd_std.values, dtype=float)

        curve = hv.Curve((f, y), kdims=["freq"], vdims=["power"]).opts(
            invert_axes=True,
            xlabel="Power",
            ylabel="Frequency (Hz)",
            width=280,
            height=420,
            color="blue",
            line_width=2,
        )

        band = hv.Area((f, y - s, y + s), kdims=["freq"], vdims=["lower", "upper"]).opts(
            invert_axes=True,
            color="blue",
            alpha=0.2,
            line_alpha=0.0,
        )

        hline = hv.HLine(float(selection.freq)).opts(color="red", line_dash="dashed", line_width=2)
        return band * curve * hline


@register_view(("freq", "AP", "ML"))
class PSDSpatialView(View):
    """2D spatial heatmap at selected frequency (AP×ML)."""

    name = "psd_spatial"

    def render(self, tensor: xr.DataArray, selection: SelectionState):
        import holoviews as hv

        hv.extension("bokeh")

        if "freq" not in tensor.dims:
            return hv.Div("<b>No 'freq' dim</b>")
        if not (("AP" in tensor.dims) and ("ML" in tensor.dims)):
            return hv.Div("<b>No (AP, ML) dims</b>")

        slice_data = tensor.sel(freq=float(selection.freq), method="nearest").transpose("AP", "ML")
        vals = np.asarray(slice_data.values, dtype=float)

        n_ap = int(slice_data.sizes.get("AP", vals.shape[0] if vals.ndim > 0 else 0))
        n_ml = int(slice_data.sizes.get("ML", vals.shape[1] if vals.ndim > 1 else 0))
        ap_idx = np.arange(n_ap)
        ml_idx = np.arange(n_ml)

        return hv.Image((ml_idx, ap_idx, vals), kdims=["ML", "AP"], vdims=["power"]).opts(
            cmap="hot",
            colorbar=True,
            tools=["hover"],
            xlabel="ML (index)",
            ylabel="AP (index)",
            title=f"Spatial PSD @ {float(selection.freq):.1f} Hz",
            width=420,
            height=420,
            aspect="equal",
            invert_yaxis=True,
        )


class ViewRegistry:
    """Registry wrapper for accessing views."""

    @staticmethod
    def get_views(tensor_node: TensorNode) -> list[type[View]]:
        return get_available_views(tensor_node)
