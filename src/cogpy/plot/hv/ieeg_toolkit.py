from __future__ import annotations

from typing import Literal

from cogpy.utils.imports import import_optional

pn = import_optional("panel")
param = import_optional("param")

from cogpy.datasets.gui_bundles import ieeg_grid_bundle

from .channel_grid import ChannelGrid
from .channel_grid_widget import ChannelGridWidget
from .grid_indexing import apml_from_flat_index
from .ieeg_viewer import ieeg_viewer
from .selection_policy import top_n_correlation, top_n_variance

__all__ = ["ieeg_toolkit_app"]


class _AutoSelectState(param.Parameterized):
    metric = param.Selector(default="variance", objects=["variance", "correlation"])
    n = param.Integer(default=16, bounds=(1, 512))
    seed_channel = param.Integer(default=0, bounds=(0, None))
    apply_to_grid = param.Action(lambda self: self.param.trigger("apply_to_grid"))


def ieeg_toolkit_app(
    *,
    mode: Literal["small", "large"] = "small",
    seed: int = 0,
    initial_window_s: float = 5.0,
) -> pn.viewable.Viewable:
    """
    Servable Panel app for the iEEG grid + stacked-trace viewer.

    This is thin glue around:
      - `cogpy.datasets.gui_bundles.ieeg_grid_bundle`
      - `ChannelGridWidget`
      - `ieeg_viewer`
    """
    pn.extension("bokeh")

    bundle = ieeg_grid_bundle(mode=mode, seed=seed)

    grid = ChannelGrid(n_ap=bundle.n_ap, n_ml=bundle.n_ml)
    grid_w = ChannelGridWidget.from_grid(
        grid,
        cell_values=bundle.rms_apml,
        ap_coords=bundle.ap_coords,
        ml_coords=bundle.ml_coords,
        atlas_image=bundle.atlas_image,
    )

    viewer = ieeg_viewer(
        bundle.sig_tc,
        channel_grid=grid,
        n_ml=bundle.n_ml,
        initial_window_s=float(initial_window_s),
        title=f"iEEG Toolkit ({mode}, seed={seed})",
    )

    # Auto-selection controls (Phase 4.2)
    state = _AutoSelectState()
    state.n = min(int(state.n), bundle.n_ap * bundle.n_ml)
    state.seed_channel = 0

    metric_select = pn.widgets.Select(
        name="Metric", options=["variance", "correlation"], value=state.metric
    )
    n_slider = pn.widgets.IntSlider(
        name="Top-N", start=1, end=min(256, bundle.n_ap * bundle.n_ml), value=16
    )
    seed_slider = pn.widgets.IntSlider(
        name="Seed ch (flat)", start=0, end=bundle.n_ap * bundle.n_ml - 1, value=0
    )
    apply_btn = pn.widgets.Button(
        name="Select on grid", button_type="primary", width=140
    )

    def _apply_autoselect(_):
        # Use current viewer window, if available
        t0 = t1 = None
        try:
            lo, hi = viewer.viewer._range_stream.x_range  # set once panel() is built
            t0, t1 = float(lo), float(hi)
        except Exception:  # noqa: BLE001
            pass

        n = int(n_slider.value)
        if metric_select.value == "variance":
            chosen = top_n_variance(bundle.sig_tc, n=n, t0=t0, t1=t1)
        else:
            chosen = top_n_correlation(
                bundle.sig_tc,
                seed_channel=int(seed_slider.value),
                n=n,
                t0=t0,
                t1=t1,
            )

        pairs = set(apml_from_flat_index(ix, bundle.n_ml) for ix in chosen)
        grid.select_manual(pairs)

    apply_btn.on_click(_apply_autoselect)

    controls = pn.Column(
        pn.pane.Markdown("### Auto-select"),
        metric_select,
        n_slider,
        seed_slider,
        apply_btn,
        styles={"background": "#1e1e2e", "padding": "10px", "border-radius": "8px"},
        width=320,
    )

    return pn.Row(
        pn.Column(grid_w.panel(), controls),
        viewer.panel(),
        sizing_mode="fixed",
    )
