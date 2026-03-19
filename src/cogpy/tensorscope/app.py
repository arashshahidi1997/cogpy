"""
TensorScope application (v3.0).

Tensor-centric UI with tabs and an adaptive sidebar.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import panel as pn
import xarray as xr

from .state import TensorNode, TensorScopeState
from .views import get_available_views

__all__ = ["TensorScopeApp"]


class TensorScopeApp:
    """
    TensorScope application (v3.0).

    Features
    --------
    - Tensor registry
    - Tensor tabs
    - Dimension-based view discovery
    - Linked selection via `SelectionState`
    """

    def __init__(self) -> None:
        self.state = TensorScopeState()

    def add_tensor(
        self,
        name: str,
        data: xr.DataArray,
        *,
        source: str | None = None,
        transform: str = "signal",
        params: dict[str, Any] | None = None,
    ) -> None:
        node = TensorNode(
            name=str(name),
            data=data,
            source=None if source is None else str(source),
            transform=str(transform),
            params={} if params is None else dict(params),
        )
        self.state.tensors.add(node)

    def add_psd_tensor(
        self,
        name: str,
        *,
        source: str,
        window: float = 1.0,
        nperseg: int = 256,
        method: str = "welch",
    ) -> None:
        from cogpy.spectral.psd_utils import compute_psd_window

        src = self.state.tensors.get(source)
        psd = compute_psd_window(
            src.data,
            t_center=float(self.state.selection.time),
            window_size=float(window),
            nperseg=int(nperseg),
            method=str(method),
        )

        # Canonicalize dims for view discovery and semantic clarity.
        if "freq" in psd.dims:
            spatial = [d for d in ("AP", "ML", "channel") if d in psd.dims]
            psd = psd.transpose("freq", *spatial)

        self.add_tensor(
            name=str(name),
            data=psd,
            source=str(source),
            transform="psd",
            params={"window": float(window), "nperseg": int(nperseg), "method": str(method)},
        )

    def add_spectrogram_tensor(self, name: str, *, source: str, nperseg: int = 256, noverlap: int = 128) -> None:
        raise NotImplementedError("Spectrogram tensor is deferred (v3.0 only implements signal + PSD).")

    def build(self) -> pn.template.FastListTemplate:
        pn.extension()

        tensor_names = self.state.tensors.list()
        if not tensor_names:
            template = pn.template.FastListTemplate(
                title="TensorScope v3.0",
                sidebar=[pn.pane.Markdown("Add tensors via `TensorScopeApp.add_tensor()` before building.")],
                main=[pn.pane.Markdown("No tensors registered.")],
            )
            return template

        # Keep active tensor consistent with tabs.
        if self.state.active_tensor not in self.state.tensors:
            self.state.active_tensor = tensor_names[0]

        tabs = pn.Tabs(*[(name, self._create_tensor_panel(name)) for name in tensor_names], dynamic=True)

        def _sync_active(_event) -> None:
            try:
                idx = int(tabs.active)
            except Exception:  # noqa: BLE001
                return
            if 0 <= idx < len(tensor_names):
                try:
                    self.state.set_active_tensor(tensor_names[idx])
                except Exception:  # noqa: BLE001
                    pass

        tabs.param.watch(_sync_active, "active")

        template = pn.template.FastListTemplate(
            title="TensorScope v3.0",
            sidebar_width=360,
            sidebar=[self._create_sidebar()],
            main=[tabs],
        )
        return template

    def _create_tensor_panel(self, tensor_name: str) -> pn.Row:
        node = self.state.tensors.get(tensor_name)
        view_classes = get_available_views(node)

        if not view_classes:
            return pn.Row(pn.pane.Markdown(f"No views available for dims: `{node.dims}`"))

        panes: list[pn.viewable.Viewable] = []

        for view_cls in view_classes[:2]:  # fixed layout in v3.0
            view = view_cls()

            def _render(_time, _freq, _ap, _ml, _channel, *, _node=node, _view=view):
                return _view.render(_node.data, self.state.selection)

            bound = pn.bind(
                _render,
                self.state.selection.param.time,
                self.state.selection.param.freq,
                self.state.selection.param.ap,
                self.state.selection.param.ml,
                self.state.selection.param.channel,
            )
            panes.append(pn.pane.HoloViews(bound, sizing_mode="stretch_both"))

        return pn.Row(*panes, sizing_mode="stretch_both")

    def _create_sidebar(self) -> pn.Column:
        sel = self.state.selection

        # Selection controls (v3.0 minimal): time + freq always visible.
        time_slider = pn.widgets.FloatSlider.from_param(sel.param.time, name="Time (s)")
        freq_slider = pn.widgets.FloatSlider.from_param(sel.param.freq, name="Freq (Hz)")

        # Give reasonable default ranges if bounds are open-ended.
        if time_slider.end is None or not np.isfinite(float(time_slider.end)):  # type: ignore[name-defined]
            time_slider.start = 0.0
            time_slider.end = 10.0
        if freq_slider.end is None or not np.isfinite(float(freq_slider.end)):  # type: ignore[name-defined]
            freq_slider.start = 0.0
            freq_slider.end = 150.0

        lineage = pn.bind(lambda _t: f"**Active tensor:** `{self.state.active_tensor}`", sel.param.time)
        return pn.Column(
            pn.pane.Markdown("### Tensor Info"),
            pn.pane.Markdown(lineage),
            pn.layout.Divider(),
            pn.pane.Markdown("### Selection"),
            time_slider,
            freq_slider,
            sizing_mode="stretch_width",
        )
