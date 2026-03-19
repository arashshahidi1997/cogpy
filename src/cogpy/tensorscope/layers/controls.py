"""Control panel layers for channel selection and processing."""

from __future__ import annotations

import panel as pn

from cogpy.plot.hv.channel_grid_widget import ChannelGridWidget

from .base import TensorLayer


class ChannelSelectorLayer(TensorLayer):
    """Channel grid selector layer wrapping ChannelGridWidget."""

    def __init__(self, state):
        super().__init__(state)
        self.layer_id = "channel_selector"
        self.title = "Channel Selection"

        self._widget = self._add_data_ref(ChannelGridWidget.from_grid(state.channel_grid, cell_size=18))

    def panel(self):
        if self._panel is None:
            self._panel = pn.Accordion(
                (self.title, self._widget.panel()),
                active=[0],
                sizing_mode="stretch_width",
            )
        return self._panel


class ProcessingControlsLayer(TensorLayer):
    """Processing controls layer wrapping ProcessingChain.controls()."""

    def __init__(self, state):
        super().__init__(state)
        self.layer_id = "processing_controls"
        self.title = "Processing"
        self._controls = self._add_data_ref(state.processing.controls())

    def panel(self):
        if self._panel is None:
            self._panel = pn.Card(self._controls, title=self.title, collapsed=False)
        return self._panel
