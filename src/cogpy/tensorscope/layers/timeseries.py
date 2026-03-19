"""Timeseries visualization layer wrapping MultichannelViewer."""

from __future__ import annotations

import numpy as np

from cogpy.plot.hv.multichannel_viewer import MultichannelViewer
from cogpy.tensorscope.schema import flatten_grid_to_channels

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
        self._sig_z = sig_z
        self._t_vals = t_vals
        self._ch_labels = ch_labels

        self._viewer = self._add_data_ref(
            MultichannelViewer(
                sig_z=self._sig_z,
                t_vals=self._t_vals,
                ch_labels=self._ch_labels,
                title=self.title,
                chain=state.processing,
                detail_height=600,
                overview_height=160,
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
        sig_z = np.ascontiguousarray(np.asarray(flat.values).T).copy()
        t_vals = np.asarray(flat.time.values).copy()

        ap_vals = np.asarray(flat.AP.values)
        ml_vals = np.asarray(flat.ML.values)
        ch_labels = [f"Ch{i} (AP{ap},ML{ml})" for i, (ap, ml) in enumerate(zip(ap_vals, ml_vals))]

        assert sig_z.ndim == 2, f"sig_z must be 2D, got {sig_z.ndim}D"
        assert t_vals.ndim == 1, f"t_vals must be 1D, got {t_vals.ndim}D"
        assert sig_z.shape[1] == len(t_vals), (
            f"sig_z time dimension {sig_z.shape[1]} != t_vals length {len(t_vals)}"
        )
        assert len(ch_labels) == sig_z.shape[0], (
            f"Label count {len(ch_labels)} != channel count {sig_z.shape[0]}"
        )

        cache = (sig_z, t_vals, ch_labels)
        setattr(modality, "_tensorscope_viewer_cache", cache)
        return cache

    def _on_selection_change(self, _event=None):
        self._update_displayed_channels()

    def _update_displayed_channels(self):
        selected_flat = list(self.state.selected_channels_flat)
        n_channels = int(self._sig_z.shape[0])

        if selected_flat:
            valid = [i for i in selected_flat if 0 <= i < n_channels]
            if valid:
                self._viewer.show_channels(valid)
            else:
                self._viewer.show_channels(list(range(n_channels)))
            return

        # No selection: show a reasonable default.
        self._viewer.show_channels(list(range(min(16, n_channels))))

    def panel(self):
        if self._panel is None:
            self._panel = self._viewer.panel()
        return self._panel
