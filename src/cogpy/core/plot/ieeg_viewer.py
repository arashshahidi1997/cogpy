"""
ieeg_viewer.py
==============
Grid-aware iEEG viewer built on top of MultichannelViewer.

For a grid-unaware viewer use MultichannelViewer directly.

Usage
-----
    # Standalone (no grid)
    from ieeg_viewer import ieeg_viewer
    viewer = ieeg_viewer(sig_tc)
    viewer.panel().servable()

    # Grid-wired
    from channel_grid import ChannelGrid
    grid = ChannelGrid(n_ap=16, n_ml=16)
    viewer = ieeg_viewer(sig_tc, channel_grid=grid, n_ml=16)
    viewer.panel().servable()

    # Full layout with grid widget
    from channel_grid_widget import ChannelGridWidget
    w = ChannelGridWidget.from_grid(grid)
    pn.Row(w.panel(), viewer.panel()).servable()
"""
from __future__ import annotations

import numpy as np
import panel as pn
import xarray as xr

from .multichannel_viewer import MultichannelViewer
from .grid_indexing import flat_indices_from_selected, apml_from_flat_index_order

__all__ = ["ieeg_viewer"]


def ieeg_viewer(
    data: xr.DataArray,
    *,
    time_dim: str = "time",
    channel_dim: str = "channel",
    initial_window_s: float = 10.0,
    min_window_s: float = 0.5,
    max_window_s: float = 120.0,
    detail_px: int = 2000,
    overview_px: int = 10_000,
    n_channels_default: int = 8,
    offset_scale: float = 3.0,
    width: int = 1100,
    detail_height: int = 500,
    overview_height: int = 120,
    title: str | None = None,
    # Grid wiring
    channel_grid=None,
    n_ml: int | None = None,
    flat_order: str = "row-major",
) -> "IEEGViewer":
    """
    Build a grid-aware iEEG viewer.

    Parameters
    ----------
    data : xr.DataArray
        Dims (channel, time) or (time, channel). Time in seconds.
    channel_grid : ChannelGrid | None
        When provided, channel display is driven by grid.selected.
        Requires n_ml.
    n_ml : int | None
        Grid ML columns, for flat index computation (ap * n_ml + ml).
    (all other params passed through to MultichannelViewer)

    Returns
    -------
    IEEGViewer
        Has a .panel() method returning the Panel layout,
        and .viewer attribute exposing the MultichannelViewer.
    """
    if channel_grid is not None and n_ml is None:
        raise ValueError("n_ml required when channel_grid is provided")
    if channel_grid is not None and flat_order not in {"row-major", "col-major"}:
        raise ValueError("flat_order must be 'row-major' or 'col-major'")

    for dim in (time_dim, channel_dim):
        if dim not in data.dims:
            raise ValueError(f"{dim!r} not in data.dims={tuple(data.dims)}")

    # Canonical (channel, time) numpy
    arr     = data.transpose(channel_dim, time_dim)
    t_vals  = np.asarray(arr[time_dim].values, dtype=np.float64)
    ch_vals = list(arr[channel_dim].values)
    n_ch    = len(ch_vals)

    raw   = np.asarray(arr.values, dtype=np.float64)
    means = raw.mean(axis=1, keepdims=True)
    stds  = raw.std(axis=1, keepdims=True) + 1e-12
    sig_z = np.ascontiguousarray((raw - means) / stds)

    ch_labels = [str(v) for v in ch_vals]

    viewer = MultichannelViewer(
        sig_z, t_vals, ch_labels,
        initial_window_s=initial_window_s,
        min_window_s=min_window_s,
        max_window_s=max_window_s,
        detail_px=detail_px,
        overview_px=overview_px,
        offset_scale=offset_scale,
        width=width,
        detail_height=detail_height,
        overview_height=overview_height,
        time_dim=time_dim,
        title=title or f"{data.name or 'iEEG'}  ({n_ch} ch, {t_vals[-1]-t_vals[0]:.1f} s)",
    )

    # Default selection
    viewer.show_channels(list(range(min(n_channels_default, n_ch))))

    n_ap = int(getattr(channel_grid, "n_ap", 0)) if channel_grid is not None else None
    return IEEGViewer(
        viewer,
        channel_grid=channel_grid,
        n_ap=n_ap,
        n_ml=n_ml,
        n_ch=n_ch,
        flat_order=flat_order,
    )


class IEEGViewer:
    """
    Thin wrapper that owns the Panel layout and optional grid wiring.
    Access the underlying MultichannelViewer via .viewer.
    """

    def __init__(self, viewer: MultichannelViewer, *, channel_grid, n_ap, n_ml, n_ch, flat_order: str):
        self.viewer = viewer
        self._grid  = channel_grid
        self._n_ap  = n_ap
        self._n_ml  = n_ml
        self._n_ch  = n_ch
        self._flat_order = flat_order
        self._built = False

    def panel(self) -> pn.viewable.Viewable:
        viewer_panel = self.viewer.panel()

        if self._grid is None:
            return viewer_panel

        # Grid-wired controls
        n_sel_md = pn.pane.Markdown(
            f"**{self.viewer._n_ch}** channels",
            styles={"color": "#cdd6f4", "font-size": "11px"},
        )
        apply_btn = pn.widgets.Button(
            name="Apply selection", button_type="primary", width=140,
        )
        sort_sel = pn.widgets.Select(
            name="Sort",
            options=["none", "AP", "ML", "variance"],
            value="none",
            width=140,
        )

        # Track pending selection separately from applied
        _pending: list[int] = list(self.viewer._active_ix)

        def _flat(selected):
            if self._n_ml is None:
                return []
            n_ap = self._n_ap
            if n_ap is None or n_ap == 0:
                # Best-effort fallback; needed for col-major mapping.
                n_ap = int(getattr(self._grid, "n_ap", 0)) if self._grid is not None else 0
            if n_ap <= 0:
                raise ValueError("n_ap must be available when channel_grid is provided")
            return flat_indices_from_selected(
                selected,
                n_ap=int(n_ap),
                n_ml=int(self._n_ml),
                order=str(self._flat_order),
                n_ch=int(self._n_ch),
            )

        def _on_grid(event):
            _pending[:] = _flat(event.new)
            n_sel_md.object = (
                f"**{len(_pending)}** pending  "
                f"_(applied: {len(self.viewer._active_ix)})_"
            )

        self._grid.param.watch(_on_grid, "selected")

        # Initialise pending from current grid state
        _pending[:] = _flat(self._grid.selected)

        def _sorted(indices: list[int]) -> list[int]:
            key = str(sort_sel.value or "none")
            if key == "none":
                return list(indices)
            if key == "AP":
                if self._n_ap is None or self._n_ml is None:
                    return list(indices)
                return sorted(
                    indices,
                    key=lambda ix: apml_from_flat_index_order(
                        int(ix), n_ap=int(self._n_ap), n_ml=int(self._n_ml), order=str(self._flat_order)
                    ),
                )
            if key == "ML":
                if self._n_ap is None or self._n_ml is None:
                    return list(indices)
                return sorted(
                    indices,
                    key=lambda ix: (
                        apml_from_flat_index_order(
                            int(ix), n_ap=int(self._n_ap), n_ml=int(self._n_ml), order=str(self._flat_order)
                        )[1],
                        apml_from_flat_index_order(
                            int(ix), n_ap=int(self._n_ap), n_ml=int(self._n_ml), order=str(self._flat_order)
                        )[0],
                    ),
                )
            if key == "variance":
                try:
                    lo, hi = self.viewer._range_stream.x_range
                    t0, t1 = float(lo), float(hi)
                except Exception:  # noqa: BLE001
                    t0, t1 = self.viewer._t0, self.viewer._t1

                # Compute variance over current window using precomputed z-scored numpy.
                t_vals = self.viewer._t_vals
                i0 = int(np.searchsorted(t_vals, t0, side="left"))
                i1 = int(np.searchsorted(t_vals, t1, side="right"))
                i0 = max(i0, 0)
                i1 = min(i1, len(t_vals))
                if i1 <= i0:
                    i0, i1 = 0, len(t_vals)
                sig = self.viewer._sig_z
                v = np.var(sig[:, i0:i1], axis=1)
                return sorted(indices, key=lambda ix: float(v[int(ix)]), reverse=True)

            return list(indices)

        def _on_apply(_):
            self.viewer.show_channels(_sorted(list(_pending)))
            n_sel_md.object = f"**{len(_pending)}** channels"

        apply_btn.on_click(_on_apply)

        controls = pn.Row(
            n_sel_md, sort_sel, apply_btn,
            styles={"background": "#1e1e2e", "padding": "8px", "border-radius": "6px"},
        )

        # Insert controls above the viewer's own controls
        return pn.Column(controls, viewer_panel)

    def servable(self):
        return self.panel().servable()
