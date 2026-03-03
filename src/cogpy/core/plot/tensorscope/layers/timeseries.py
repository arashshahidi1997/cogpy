"""Timeseries visualization layer wrapping MultichannelViewer."""

from __future__ import annotations

import numpy as np

from cogpy.core.plot.multichannel_viewer import MultichannelViewer
from cogpy.core.plot.tensorscope.schema import flatten_grid_to_channels

from .base import TensorLayer


class TimeseriesLayer(TensorLayer):
    """
    Timeseries traces layer.

    Wraps MultichannelViewer to display stacked traces.
    Updates when channel selection changes; optional time hair overlay.
    """

    def __init__(self, state, show_hair: bool = True):
        super().__init__(state)
        self.layer_id = "timeseries"
        self.title = "Timeseries"
        self.show_hair = bool(show_hair)

        modality = state.data_registry.get("grid_lfp") if state.data_registry is not None else None
        if modality is None:
            raise ValueError("No grid_lfp modality registered")

        sig_z, t_vals, ch_labels = self._viewer_inputs_from_modality(modality)

        self._viewer = self._add_data_ref(
            MultichannelViewer(
                sig_z=sig_z,
                t_vals=t_vals,
                ch_labels=ch_labels,
                title=self.title,
                chain=state.processing,
            )
        )

        if self.show_hair and getattr(state, "time_hair", None) is not None:
            self._viewer.add_time_hair(state.time_hair)
            self._viewer.attach_time_hair_to_overview(state.time_hair)

        if getattr(state, "channel_grid", None) is not None:
            self._watch(state.channel_grid, self._on_selection_change, "selected")
            self._update_displayed_channels()

    @staticmethod
    def _viewer_inputs_from_modality(modality):
        cache = getattr(modality, "_tensorscope_viewer_cache", None)
        if cache is not None:
            return cache

        flat = flatten_grid_to_channels(modality.data)
        sig_z = np.ascontiguousarray(np.asarray(flat.values).T)
        t_vals = np.asarray(flat.time.values)

        ap_vals = np.asarray(flat.AP.values)
        ml_vals = np.asarray(flat.ML.values)
        ch_labels = [f"Ch{i} (AP{ap},ML{ml})" for i, (ap, ml) in enumerate(zip(ap_vals, ml_vals))]

        cache = (sig_z, t_vals, ch_labels)
        setattr(modality, "_tensorscope_viewer_cache", cache)
        return cache

    def _on_selection_change(self, _event=None):
        self._update_displayed_channels()

    def _update_displayed_channels(self):
        selected_flat = list(self.state.selected_channels_flat)
        if selected_flat:
            self._viewer.show_channels(selected_flat)

    def panel(self):
        if self._panel is None:
            self._panel = self._viewer.panel()
        return self._panel

