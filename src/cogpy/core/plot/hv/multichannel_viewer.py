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

from .theme import BG, BG_PANEL, BLUE, PALETTE, TEAL, TEXT, TEXT_SMALL

__all__ = ["MultichannelViewer"]

_ds = MinMaxLTTBDownsampler()

_PALETTE = PALETTE


def _hide_yaxis_tick_labels(plot, element) -> None:
    """
    NeuroScope2-style: keep y ticks (for visual reference / zoom context),
    but hide tick labels (channel identity is shown inline).

    HoloViews does not expose a simple "hide tick labels but keep axis" option, so
    we apply it via a Bokeh hook.
    """
    try:
        for ax in plot.state.yaxis:
            ax.major_label_text_font_size = "0pt"
            ax.major_label_text_alpha = 0.0
    except Exception:
        return


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
        show_channel_labels: bool = True,
        framewise: bool = True,
        chain=None,
    ):
        assert sig_z.ndim == 2, "sig_z must be (n_ch, n_time)"
        assert sig_z.shape[1] == len(t_vals), "sig_z columns must match t_vals length"
        assert sig_z.shape[0] == len(ch_labels), "sig_z rows must match ch_labels length"

        self._sig_z    = np.ascontiguousarray(sig_z)
        self._t_vals   = np.asarray(t_vals, dtype=np.float64)
        self._ch_labels = list(ch_labels)
        self._n_ch     = sig_z.shape[0]
        self._chain = chain

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
        self._show_channel_labels = bool(show_channel_labels)
        self._framewise = bool(framewise)

        self._t0 = float(self._t_vals[0])
        self._t1 = float(self._t_vals[-1])

        # Current channel indices to display — mutated by show_channels()
        self._active_ix: list[int] = list(range(min(8, self._n_ch)))

        self._range_stream = None
        self._current_t_range = (self._t0, min(self._t0 + self._iw, self._t1))
        self._hair = None  # set by add_time_hair()
        self._hair_watcher = None
        self._chain_watchers = []

        if self._chain is not None:
            self._wire_chain(self._chain)

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

    @property
    def range_stream(self):
        """Read-only access to the HoloViews RangeX stream (public API)."""
        return self._range_stream

    def panel(self, *, fresh: bool = False) -> pn.viewable.Viewable:
        """Build and return the Panel layout.

        Notes
        -----
        In notebooks, displaying the same Panel object multiple times can raise
        Bokeh "Models must be owned by only a single document" errors. Use
        ``fresh=True`` (or create a new viewer instance) when re-displaying in a
        new output cell.
        """
        if self._built:
            return self._layout.clone() if bool(fresh) else self._layout

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
        finite = y_ov[np.isfinite(y_ov)]
        if finite.size:
            y0 = float(finite.min())
            y1 = float(finite.max())
            pad = 0.05 * (y1 - y0) if y1 > y0 else 1.0
            ylim = (y0 - pad, y1 + pad)
        else:
            ylim = (-1.0, 1.0)
        self._overview = hv.Curve(
            (t_ov, y_ov), kdims=self._tdim, vdims="amp"
        ).opts(
            width=self._w, height=self._oh,
            color=BLUE, line_width=0.8,
            xlabel="", ylabel="",
            toolbar=None, default_tools=[],
            title="Overview  —  drag to navigate",
            yaxis=None,
            ylim=ylim,
            framewise=False,
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

        self._framewise_chk = pn.widgets.Checkbox(
            name="Auto-scale Y",
            value=bool(self._framewise),
            width=120,
        )

        def _on_framewise(event=None) -> None:
            self._framewise = bool(getattr(event, "new", self._framewise_chk.value))
            self._apply()

        self._framewise_chk.param.watch(_on_framewise, "value")

        self._layout = pn.Column(
            pn.pane.Markdown(f"## {self._title}", styles={"color": TEXT}),
            pn.Row(
                self._window_slider,
                self._framewise_chk,
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

    def _wire_chain(self, chain):
        """Re-render current window when any processing param changes."""
        params_to_watch = [
            "cmr_on",
            "bandpass_on",
            "bandpass_lo",
            "bandpass_hi",
            "bandpass_order",
            "spatial_median_on",
            "spatial_median_size",
            "zscore_on",
            "zscore_robust",
        ]
        try:
            h = chain.param.watch(self._on_chain_change, params_to_watch)
            self._chain_watchers.append(h)
        except Exception:  # noqa: BLE001
            return

    def _on_chain_change(self, *events):
        if self._range_stream is None:
            return
        lo, hi = self._current_t_range
        self._range_stream.event(x_range=(float(lo), float(hi)))

    def _build_detail(self, x_range) -> hv.Element:
        """Called by DynamicMap on every stream event."""
        if x_range is None or x_range == (None, None):
            t0, t1 = self._t0, self._t0 + self._iw
        else:
            t0 = max(float(x_range[0]), self._t0)
            t1 = min(float(x_range[1]), self._t1)

        i0, i1   = _find_indices(self._t_vals, t0, t1)
        ch_ixs   = list(self._active_ix)   # snapshot
        n_vis    = len(ch_ixs)

        # Always return the same element type (Overlay) so DynamicMap updates are stable.
        if n_vis == 0 or i1 <= i0:
            empty_overlay = hv.NdOverlay({}, kdims="rank").opts(
                hv.opts.NdOverlay(
                    width=self._w,
                    height=self._dh,
                    show_legend=False,
                    toolbar="above",
                    xlabel=f"{self._tdim} (s)",
                    ylabel="",
                    title=self._title,
                    framewise=True,
                    shared_axes=False,
                    axiswise=True,
                    yaxis="left",
                    hooks=[_hide_yaxis_tick_labels],
                )
            )
            empty_labels = hv.Labels(
                {self._tdim: [], "amp": [], "text": []},
                kdims=[self._tdim, "amp"],
                vdims=["text"],
            ).opts(
                text_align="left",
                text_baseline="middle",
                text_font_size=TEXT_SMALL,
                text_color=TEXT,
                text_alpha=0.7,
                xoffset=6,
            )
            return self._with_hair(empty_overlay * empty_labels)

        t0_ix = float(self._t_vals[i0])
        t1_ix = float(self._t_vals[min(i1, len(self._t_vals) - 1)])
        self._current_t_range = (t0_ix, t1_ix)

        if self._chain is not None:
            win = self._chain.get_window(t0_ix, t1_ix, channels=list(ch_ixs))
            t_win = np.asarray(win[self._tdim].values, dtype=np.float64)
            sig_window = np.asarray(win.values).T  # (n_active, n_samples)
        else:
            t_win = self._t_vals[i0:i1]
            sig_window = None

        # Build one NdOverlay with integer keys — avoids label-based caching
        # that causes HoloViews to silently drop new curves on re-render.
        curves = {}
        y_offsets = []
        y_labels = []
        for rank, ch in enumerate(ch_ixs):
            if sig_window is not None:
                y_win = sig_window[rank]
            else:
                y_win = self._sig_z[ch, i0:i1]
            t_ds, y_ds = _downsample(t_win, y_win, self._dpx)
            offset     = (n_vis - 1 - rank) * self._offs
            curves[rank] = hv.Curve(
                (t_ds, y_ds + offset),
                kdims=self._tdim, vdims="amp",
            ).opts(
                color=_PALETTE[rank % len(_PALETTE)],
                line_width=1,
            )
            y_offsets.append(float(offset))
            y_labels.append(str(self._ch_labels[ch]))

        traces = hv.NdOverlay(curves, kdims="rank").opts(
            hv.opts.NdOverlay(
                width=self._w, height=self._dh,
                show_legend=False, toolbar="above",
                xlabel=f"{self._tdim} (s)", ylabel="",
                title=self._title,
                framewise=bool(self._framewise),
                shared_axes=False,
                axiswise=True,
                yaxis="left",
                hooks=[_hide_yaxis_tick_labels],
            )
        )

        if not self._show_channel_labels:
            return self._with_hair(traces)

        # Inline channel labels (NeuroScope2-style): draw text at each trace offset.
        #
        # Important: place the text *inside* the visible x-range to avoid clipping.
        x_left = float(t_win[0]) if len(t_win) else float(self._t0)
        x_span = float(t1 - t0)
        x_pos = x_left + (0.01 * x_span if x_span > 0 else 0.0)
        labels_el = hv.Labels(
            {self._tdim: [x_pos] * n_vis, "amp": y_offsets, "text": y_labels},
            kdims=[self._tdim, "amp"],
            vdims=["text"],
        ).opts(
            text_align="left",
            text_baseline="middle",
            text_font_size=TEXT_SMALL,
            text_color=TEXT,
            text_alpha=0.8,
        )

        return self._with_hair(traces * labels_el)

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

    def add_time_hair(self, hair) -> None:
        """
        Show a vertical time-hair line in the detail traces driven by a
        :class:`TimeHair`.

        The VLine is baked into each render of the detail DynamicMap so that
        the ``RangeToolLink`` and ``RangeX`` stream remain stable — no
        overlay swapping, no Bokeh figure replacement.

        Every change to ``hair.t`` triggers a lightweight re-render of the
        current window (same data slice, processing unchanged, just a new
        VLine position).  This is acceptable for click-based or player-driven
        updates.  Avoid wiring a continuously-dragged slider when the chain
        includes an expensive filter.

        Can be called before or after :meth:`panel`.

        Parameters
        ----------
        hair : TimeHair
            Shared time parameter.  ``hair.t`` drives the VLine position.
        """
        if self._hair is not None and self._hair_watcher is not None:
            try:
                self._hair.param.unwatch(self._hair_watcher)
            except Exception:  # noqa: BLE001
                pass

        self._hair = hair
        self._hair_watcher = hair.param.watch(self._on_hair_t, "t")

    def _with_hair(self, element: hv.Element) -> hv.Element:
        """Append VLine to element if a time hair is set and has a position."""
        if self._hair is None:
            return element
        t = getattr(self._hair, "t", None)
        if t is None:
            return element
        x = t.item() if hasattr(t, "item") else t
        vline = hv.VLine(float(x)).opts(color=TEAL, line_width=2, alpha=0.9)
        return element * vline

    def _on_hair_t(self, event=None) -> None:
        """Re-render current window when hair.t changes."""
        if self._range_stream is None:
            return
        lo, hi = self._current_t_range
        self._range_stream.event(x_range=(float(lo), float(hi)))

    def attach_time_hair_to_overview(self, hair, *, time_kdim: str | None = None, **attach_kwargs):
        """Attach tap/click behavior + hair overlay to the overview curve.

        This is the recommended way to make clicks in the overview update
        ``hair.t`` (and therefore drive any listeners such as the detail VLine
        or other linked widgets).

        Parameters
        ----------
        hair : TimeHair
            Shared time parameter controller.
        time_kdim : str | None
            Time dimension name for tapping. Defaults to this viewer's
            ``time_dim``.
        **attach_kwargs
            Passed through to ``hair.attach(...)``.

        Returns
        -------
        holoviews object
            The attached overview element. If the panel layout is already
            built, the overview slot is updated in-place.
        """
        time_kdim = self._tdim if time_kdim is None else str(time_kdim)
        if not getattr(self, "_built", False):
            self.panel()

        attached = hair.attach(self._overview, time_kdim=time_kdim, **attach_kwargs)
        if self._built:
            # Overview slot (index 3).
            self._layout[3] = attached
        return attached

    def dispose(self) -> None:
        """Best-effort cleanup of watchers/streams to avoid leaks in layered apps."""
        if self._hair is not None and self._hair_watcher is not None:
            try:
                self._hair.param.unwatch(self._hair_watcher)
            except Exception:  # noqa: BLE001
                pass
        self._hair = None
        self._hair_watcher = None

        if self._chain is not None and self._chain_watchers:
            for h in list(self._chain_watchers):
                try:
                    self._chain.param.unwatch(h)
                except Exception:  # noqa: BLE001
                    pass
        self._chain_watchers = []

        self._range_stream = None
        try:
            self._layout = None
        except Exception:  # noqa: BLE001
            pass
        self._overview = None
        self._detail_dmap = None
        self._built = False
