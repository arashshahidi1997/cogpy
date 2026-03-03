"""Time navigation layer wrapping PlayerWithRealTime."""

from __future__ import annotations

from cogpy.core.plot.time_player import PlayerWithRealTime

from .base import TensorLayer


class TimeNavigatorLayer(TensorLayer):
    """Time navigation controls layer wrapping PlayerWithRealTime."""

    def __init__(self, state):
        super().__init__(state)
        self.layer_id = "time_navigator"
        self.title = "Time Navigation"

        modality = state.data_registry.get("grid_lfp") if state.data_registry is not None else None
        if modality is None:
            raise ValueError("No grid_lfp modality registered")

        time_vals = list(modality.data.time.values)
        self._player = self._add_data_ref(PlayerWithRealTime(time_vals))

        self._watch(self._player.t_player, self._on_player_time, "value")

    def _on_player_time(self, event):
        self.state.current_time = event.new

    def panel(self):
        if self._panel is None:
            self._panel = self._player.view
        return self._panel

