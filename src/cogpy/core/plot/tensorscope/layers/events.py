"""Event visualization layers."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import panel as pn

from .base import TensorLayer


class EventTableLayer(TensorLayer):
    """
    Event table display layer.

    Shows events in a Tabulator widget with:
    - Sortable/filterable columns
    - Click row → jump to event time
    - Prev/Next navigation
    """

    def __init__(self, state, stream_name: str):
        super().__init__(state)

        self.layer_id = f"event_table_{stream_name}"
        self.title = f"Events: {stream_name}"
        self.stream_name = str(stream_name)

        try:
            pn.extension("tabulator")
        except Exception:  # noqa: BLE001
            pass

        stream = state.event_registry.get(self.stream_name)
        if stream is None:
            raise ValueError(f"Event stream {self.stream_name!r} not found in registry")
        self.stream = stream

        self._table = pn.widgets.Tabulator(
            stream.df,
            show_index=False,
            theme="midnight",
            sizing_mode="stretch_both",
            selectable=1,
            layout="fit_columns",
        )
        self._add_data_ref(self._table)

        # Wire up row selection → jump to time
        try:
            self._table.on_click(self._on_row_click)
        except Exception:  # noqa: BLE001
            pass

        self._prev_btn = pn.widgets.Button(name="◀ Prev", width=80, button_type="primary")
        self._next_btn = pn.widgets.Button(name="Next ▶", width=80, button_type="primary")
        self._add_data_ref(self._prev_btn)
        self._add_data_ref(self._next_btn)
        self._prev_btn.on_click(self._on_prev)
        self._next_btn.on_click(self._on_next)

        self._info = pn.pane.Markdown(f"**{len(stream)} events**", sizing_mode="stretch_width")
        self._add_data_ref(self._info)

    def _on_row_click(self, event):
        if getattr(event, "row", None) is None:
            return
        row_data = self.stream.df.iloc[int(event.row)]
        event_time = float(row_data[self.stream.time_col])
        self.state.current_time = event_time

    def _on_prev(self, _event=None):
        if self.state.current_time is None:
            return
        prev_event = self.stream.get_prev_event(float(self.state.current_time))
        if prev_event is not None:
            self.state.current_time = float(prev_event[self.stream.time_col])

    def _on_next(self, _event=None):
        if self.state.current_time is None:
            return
        next_event = self.stream.get_next_event(float(self.state.current_time))
        if next_event is not None:
            self.state.current_time = float(next_event[self.stream.time_col])

    def panel(self):
        if self._panel is None:
            self._panel = pn.Column(
                pn.Row(self._prev_btn, self._next_btn, self._info, sizing_mode="stretch_width"),
                self._table,
                sizing_mode="stretch_both",
            )
        return self._panel


class EventOverlayLayer(TensorLayer):
    """
    Event overlay layer (v2.6.2).

    Creates HoloViews overlays for different view types:
    - Spatial: event points at (ML, AP)
    - Temporal: vertical lines at event time
    - Spectrogram: event points at (time, freq)

    Notes
    -----
    TensorScope currently has two visualization stacks:
    1) The "layers" stack (GridFrameElement/MultichannelViewer)
    2) The v2.2 ViewFactory stack (pure HoloViews)

    This layer provides HoloViews overlays that can be multiplied onto
    ViewFactory views (recommended) and also previewed standalone.
    """

    def __init__(self, state, stream_name: str):
        super().__init__(state)
        self.layer_id = f"event_overlay_{stream_name}"
        self.title = f"Event Overlay: {stream_name}"
        self.stream_name = str(stream_name)
        self.max_markers = 200

        stream = state.event_registry.get(self.stream_name)
        if stream is None:
            raise ValueError(f"Event stream {self.stream_name!r} not found")
        self.stream = stream

    def _time_window(self) -> tuple[float, float]:
        tw = getattr(self.state, "time_window", None)
        if tw is not None and hasattr(tw, "window"):
            try:
                t0, t1 = tw.window
                return (float(t0), float(t1))
            except Exception:  # noqa: BLE001
                pass
        # Fallback: use full event range (or 0..1 for empty).
        if len(self.stream.df) == 0:
            return (0.0, 1.0)
        t0 = float(self.stream.df[self.stream.time_col].min())
        t1 = float(self.stream.df[self.stream.time_col].max())
        if not np.isfinite(t0) or not np.isfinite(t1) or t0 == t1:
            return (0.0, 1.0)
        return (t0, t1)

    def _spatial_stream(self):
        import holoviews as hv

        hv.extension("bokeh")

        space = getattr(self.state, "spatial_space", None)
        if space is None or not hasattr(space, "create_stream"):
            return hv.streams.Stream.define("SpatialSel", AP=0.0, ML=0.0)()

        try:
            stream = space.create_stream()
        except Exception:  # noqa: BLE001
            stream = hv.streams.Stream.define("SpatialSel", AP=0.0, ML=0.0)()
        return stream

    def create_spatial_overlay(self):
        """Overlay events as points in ML×AP space."""
        import holoviews as hv

        hv.extension("bokeh")

        window_stream = hv.streams.Params(self.state.time_window, parameters=["window"])

        def _render(window=None):
            t0, t1 = (window if window is not None else self._time_window())
            df = self.stream.get_events_in_window(float(t0), float(t1))
            if df.empty:
                return hv.Points([], kdims=["ML", "AP"])
            if ("AP" not in df.columns) or ("ML" not in df.columns):
                return hv.Points([], kdims=["ML", "AP"])

            if len(df) > self.max_markers:
                df = df.sample(n=self.max_markers, random_state=0)

            cols = ["ML", "AP", self.stream.time_col, self.stream.id_col]
            if "value" in df.columns:
                cols.append("value")
            df2 = df[cols].copy()

            vdims = [self.stream.time_col, self.stream.id_col]
            if "value" in df2.columns:
                vdims.append("value")

            pts = hv.Points(df2, kdims=["ML", "AP"], vdims=vdims)

            color = "value" if "value" in df2.columns else self.stream.style.color
            opts = dict(
                marker=self.stream.style.marker,
                color=color,
                alpha=float(self.stream.style.alpha),
                size=8,
                line_width=float(self.stream.style.line_width),
                tools=["hover"],
            )
            if color == "value":
                opts.update(cmap="viridis", colorbar=True)
            return pts.opts(**opts)

        return hv.DynamicMap(_render, streams=[window_stream])

    def create_temporal_overlay(self):
        """Overlay events as vertical lines at event time."""
        import holoviews as hv

        hv.extension("bokeh")

        window_stream = hv.streams.Params(self.state.time_window, parameters=["window"])
        spatial_stream = self._spatial_stream()

        def _render(window=None, **spatial_sel):
            t0, t1 = (window if window is not None else self._time_window())
            df = self.stream.get_events_in_window(float(t0), float(t1))
            if df.empty:
                return hv.Overlay([])

            # If events include spatial columns, optionally filter by selection.
            if ("AP" in df.columns) and ("ML" in df.columns):
                ap = spatial_sel.get("AP", None)
                ml = spatial_sel.get("ML", None)
                if ap is not None:
                    df = df[df["AP"] == int(round(float(ap)))]
                if ml is not None:
                    df = df[df["ML"] == int(round(float(ml)))]

            if len(df) > self.max_markers:
                df = df.sample(n=self.max_markers, random_state=0)

            color = self.stream.style.color
            alpha = float(self.stream.style.alpha)
            lw = float(self.stream.style.line_width)

            vlines = [
                hv.VLine(float(row[self.stream.time_col])).opts(color=color, alpha=alpha * 0.6, line_width=lw)
                for _, row in df.iterrows()
            ]
            return hv.Overlay(vlines) if vlines else hv.Overlay([])

        return hv.DynamicMap(_render, streams=[window_stream, spatial_stream])

    def create_spectrogram_overlay(self):
        """Overlay events as points in time×freq space (optionally filtered by AP/ML)."""
        import holoviews as hv

        hv.extension("bokeh")

        window_stream = hv.streams.Params(self.state.time_window, parameters=["window"])
        spatial_stream = self._spatial_stream()

        def _render(window=None, **spatial_sel):
            t0, t1 = (window if window is not None else self._time_window())
            df = self.stream.get_events_in_window(float(t0), float(t1))
            if df.empty:
                return hv.Points([], kdims=[self.stream.time_col, "freq"])

            if "freq" not in df.columns:
                return hv.Points([], kdims=[self.stream.time_col, "freq"])

            if ("AP" in df.columns) and ("ML" in df.columns):
                ap = spatial_sel.get("AP", None)
                ml = spatial_sel.get("ML", None)
                if ap is not None:
                    df = df[df["AP"] == int(round(float(ap)))]
                if ml is not None:
                    df = df[df["ML"] == int(round(float(ml)))]

            if len(df) > self.max_markers:
                df = df.sample(n=self.max_markers, random_state=0)

            cols = [self.stream.time_col, "freq", self.stream.id_col]
            if "value" in df.columns:
                cols.append("value")
            df2 = df[cols].copy()

            vdims = [self.stream.id_col]
            if "value" in df2.columns:
                vdims.append("value")

            pts = hv.Points(df2, kdims=[self.stream.time_col, "freq"], vdims=vdims).opts(
                color=self.stream.style.color,
                marker=self.stream.style.marker,
                alpha=float(self.stream.style.alpha),
                size=8,
                line_width=float(self.stream.style.line_width),
                tools=["hover"],
            )
            return pts

        return hv.DynamicMap(_render, streams=[window_stream, spatial_stream])

    def panel(self):
        if self._panel is None:
            try:
                style = asdict(self.stream.style)
            except Exception:  # noqa: BLE001
                style = {"color": getattr(self.stream.style, "color", None)}

            self._panel = pn.Column(
                pn.pane.Markdown(
                    f"### Event Overlay: `{self.stream_name}`\n\n"
                    f"- **Events:** {len(self.stream)}\n"
                    f"- **Style:** {style}\n\n"
                    "Preview overlays (for ViewFactory-based views):",
                    sizing_mode="stretch_width",
                ),
                pn.Tabs(
                    ("Spatial", pn.pane.HoloViews(self.create_spatial_overlay(), sizing_mode="stretch_both")),
                    ("Temporal", pn.pane.HoloViews(self.create_temporal_overlay(), sizing_mode="stretch_both")),
                    ("Spectrogram", pn.pane.HoloViews(self.create_spectrogram_overlay(), sizing_mode="stretch_both")),
                    sizing_mode="stretch_both",
                ),
                sizing_mode="stretch_both",
            )
        return self._panel
