"""Event visualization layers."""

from __future__ import annotations

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
    Event overlay layer (Phase 4: basic implementation).

    Phase 4 delivers a proof-of-concept placeholder. Full overlay integration
    with MultichannelViewer/TopoMap is planned for later.
    """

    def __init__(self, state, stream_name: str):
        super().__init__(state)
        self.layer_id = f"event_overlay_{stream_name}"
        self.title = f"Event Overlay: {stream_name}"
        self.stream_name = str(stream_name)

        stream = state.event_registry.get(self.stream_name)
        if stream is None:
            raise ValueError(f"Event stream {self.stream_name!r} not found")
        self.stream = stream

    def panel(self):
        if self._panel is None:
            self._panel = pn.pane.Markdown(
                f"**Event Overlay: {self.stream_name}**\n\n"
                f"- {len(self.stream)} events\n"
                f"- Color: {self.stream.style.color}\n\n"
                f"*Phase 4: Overlay logic TBD (integrate with TimeseriesLayer)*"
            )
        return self._panel

