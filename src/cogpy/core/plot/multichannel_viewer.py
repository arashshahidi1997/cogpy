"""
multichannel_viewer.py
======================
Clean, grid-unaware multichannel time series viewer.

The only job of this module is: given a (channel, time) numpy array
and a list of channel indices to show, render them as stacked traces
with a pannable overview strip.

No xarray, no ChannelGrid, no param — just numpy + HoloViews + Panel.

Usage
-----
    import panel as pn
    from multichannel_viewer import MultichannelViewer

    pn.extension("bokeh")

    viewer = MultichannelViewer(sig_z, t_vals, ch_labels)
    viewer.show_channels([0, 1, 2, 3, 4, 5, 6, 7])
    viewer.panel().servable()
"""
from __future__ import annotations

import numpy as np
import holoviews as hv
from holoviews import streams
from holoviews.plotting.links import RangeToolLink
import panel as pn
from tsdownsample import MinMaxLTTBDownsampler

from .theme import BG, BG_PANEL, BLUE, PALETTE, TEXT

__all__ = ["MultichannelViewer"]

_ds = MinMaxLTTBDownsampler()

_PALETTE = PALETTE


def _make_contiguous(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a)


def _downsample(t: np.ndarray, y: np.ndarray, n_out: int):
    if len(t) <= n_out:
        return t, y
    idx = _ds.downsample(_make_contiguous(t), _make_contiguous(y), n_out=n_out)
    return t[idx], y[idx]


def _find_indices(t_vals: np.ndarray, t0: float, t1: float):
    i0 = int(np.searchsorted(t_vals, t0, side="left"))
    i1 = int(np.searchsorted(t_vals, t1, side="right"))
    return max(i0, 0), min(i1, len(t_vals))


class MultichannelViewer:
    """
    Stacked multichannel time series viewer.

    Parameters
    ----------
    sig_z : np.ndarray
        Shape (n_ch, n_time). Should be z-scored or otherwise normalised
        so offset spacing is meaningful. Must be C-contiguous.
    t_vals : np.ndarray
        Shape (n_time,). Time axis in seconds.
    ch_labels : list[str]
        Length n_ch. Display label for each channel (used as y-tick).
    initial_window_s : float
        Initial visible time window width in seconds.
    min_window_s, max_window_s : float
        Window slider bounds.
    detail_px : int
        Max rendered points per channel (MinMaxLTTB budget).
    overview_px : int
        Points in the static overview strip.
    offset_scale : float
        Vertical spacing between traces in data units (z-score units if z-scored).
    width, detail_height, overview_height : int
        Layout pixel dimensions.
    time_dim : str
        Label for the time axis.
    title : str
        Plot title.
    """

    def __init__(
        self,
        sig_z: np.ndarray,
        t_vals: np.ndarray,
        ch_labels: list[str],
        *,
        initial_window_s: float = 10.0,
        min_window_s: float = 0.5,
        max_window_s: float = 120.0,
        detail_px: int = 2000,
        overview_px: int = 10_000,
        offset_scale: float = 3.0,
        width: int = 1100,
        detail_height: int = 500,
        overview_height: int = 120,
        time_dim: str = "time",
        title: str = "Multichannel viewer",
    ):
        assert sig_z.ndim == 2, "sig_z must be (n_ch, n_time)"
        assert sig_z.shape[1] == len(t_vals), "sig_z columns must match t_vals length"
        assert sig_z.shape[0] == len(ch_labels), "sig_z rows must match ch_labels length"

        self._sig_z    = np.ascontiguousarray(sig_z)
        self._t_vals   = np.asarray(t_vals, dtype=np.float64)
        self._ch_labels = list(ch_labels)
        self._n_ch     = sig_z.shape[0]

        self._iw   = initial_window_s
        self._minw = min_window_s
        self._maxw = max_window_s
        self._dpx  = detail_px
        self._opx  = overview_px
        self._offs = offset_scale
        self._w    = width
        self._dh   = detail_height
        self._oh   = overview_height
        self._tdim = time_dim
        self._title = title

        self._t0 = float(self._t_vals[0])
        self._t1 = float(self._t_vals[-1])

        # Current channel indices to display — mutated by show_channels()
        self._active_ix: list[int] = list(range(min(8, self._n_ch)))

        self._built = False

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def show_channels(self, indices: list[int]) -> None:
        """
        Set which channels to display and refresh.

        Parameters
        ----------
        indices : list[int]
            Row indices into sig_z. Order determines vertical stacking
            (first index → top trace).
        """
        self._active_ix = [i for i in indices if 0 <= i < self._n_ch]
        if self._built:
            self._apply()

    def panel(self) -> pn.viewable.Viewable:
        """Build and return the Panel layout. Call once."""
        if self._built:
            return self._layout

        # Ensure a plotting backend is loaded before applying .opts(...)
        # (HoloViews raises if no extension has been loaded yet).
        try:
            hv.extension("bokeh")
        except Exception:  # noqa: BLE001
            pass

        self._range_stream = streams.RangeX(
            x_range=(self._t0, self._t0 + self._iw)
        )

        # Overview — static mean trace
        mean_sig = self._sig_z.mean(axis=0)
        t_ov, y_ov = _downsample(self._t_vals, mean_sig, self._opx)
        self._overview = hv.Curve(
            (t_ov, y_ov), kdims=self._tdim, vdims="amp"
        ).opts(
            width=self._w, height=self._oh,
            color=BLUE, line_width=0.8,
            xlabel="", ylabel="",
            toolbar=None, default_tools=[],
            title="Overview  —  drag to navigate",
            yaxis=None,
        )

        # Detail DynamicMap
        self._detail_dmap = hv.DynamicMap(
            self._build_detail, streams=[self._range_stream]
        )

        RangeToolLink(
            self._overview, self._detail_dmap,
            axes=["x", "x"],
            boundsx=(self._t0, self._t0 + self._iw),
        )

        # Window slider
        self._window_slider = pn.widgets.FloatSlider(
            name="Window (s)", value=self._iw,
            start=self._minw, end=self._maxw, step=0.5,
            sizing_mode="fixed", width=240,
        )
        self._window_slider.param.watch(self._on_window, "value")

        self._layout = pn.Column(
            pn.pane.Markdown(f"## {self._title}", styles={"color": TEXT}),
            pn.Row(
                self._window_slider,
                styles={"background": BG_PANEL, "padding": "8px", "border-radius": "6px"},
            ),
            self._detail_dmap,
            self._overview,
            styles={"background": BG, "padding": "16px"},
        )

        self._built = True
        return self._layout

    # ------------------------------------------------------------------
    # Internal rendering
    # ------------------------------------------------------------------

    def _build_detail(self, x_range) -> hv.Element:
        """Called by DynamicMap on every stream event."""
        if x_range is None or x_range == (None, None):
            t0, t1 = self._t0, self._t0 + self._iw
        else:
            t0 = max(float(x_range[0]), self._t0)
            t1 = min(float(x_range[1]), self._t1)

        i0, i1   = _find_indices(self._t_vals, t0, t1)
        t_win    = self._t_vals[i0:i1]
        ch_ixs   = list(self._active_ix)   # snapshot
        n_vis    = len(ch_ixs)

        if n_vis == 0 or i1 <= i0:
            return hv.Curve([], kdims=self._tdim, vdims="amp").opts(
                width=self._w, height=self._dh, framewise=True
            )

        # Build one NdOverlay with integer keys — avoids label-based caching
        # that causes HoloViews to silently drop new curves on re-render.
        curves = {}
        yticks = []
        for rank, ch in enumerate(ch_ixs):
            y_win      = self._sig_z[ch, i0:i1]
            t_ds, y_ds = _downsample(t_win, y_win, self._dpx)
            offset     = (n_vis - 1 - rank) * self._offs
            curves[rank] = hv.Curve(
                (t_ds, y_ds + offset),
                kdims=self._tdim, vdims="amp",
            ).opts(
                color=_PALETTE[rank % len(_PALETTE)],
                line_width=1,
            )
            yticks.append((offset, self._ch_labels[ch]))

        return hv.NdOverlay(curves, kdims="rank").opts(
            hv.opts.NdOverlay(
                width=self._w, height=self._dh,
                show_legend=False, toolbar="above",
                xlabel=f"{self._tdim} (s)", ylabel="",
                yticks=yticks,
                title=self._title,
                framewise=True,
                # Allow y-tick labels to change when channels change.
                shared_axes=False,
            )
        )

    def _apply(self) -> None:
        """Push current x_range as an event to force re-render."""
        lo, hi = self._range_stream.x_range or (self._t0, self._t0 + self._iw)
        self._range_stream.event(x_range=(float(lo), float(hi)))

    def _on_window(self, event) -> None:
        lo, hi = self._range_stream.x_range or (self._t0, self._t0 + self._iw)
        center = (lo + hi) / 2
        half   = event.new / 2
        new_lo = max(self._t0, center - half)
        new_hi = min(self._t1, center + half)
        self._range_stream.event(x_range=(new_lo, new_hi))
