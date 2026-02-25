"""
channel_grid_widget.py
======================
Panel widget wrapping ChannelGrid with an interactive Bokeh grid display.

Fixes vs previous version
--------------------------
* Manual selection now sticks: uses source.patch() instead of replacing
  source.data wholesale (which was resetting Bokeh's selection state and
  causing the toggle to fight itself).
* Sparse mode replaces the old subgrid mode.
* Optional atlas background image support (modes: "crop" or "full").

Usage
-----
    import panel as pn
    from channel_grid import ChannelGrid
    from channel_grid_widget import ChannelGridWidget

    pn.extension()

    w = ChannelGridWidget(n_ap=16, n_ml=16)
    w.panel().servable()

    # With signal background (per-electrode scalar, e.g. RMS)
    rms = sig.std(dim="time").transpose("AP", "ML").values  # (n_ap, n_ml)
    w = ChannelGridWidget(n_ap=16, n_ml=16, cell_values=rms)

    # With atlas image
    import numpy as np
    atlas_img = np.array(...)   # (H, W, 3) or (H, W, 4) uint8
    w = ChannelGridWidget(
        n_ap=16, n_ml=16,
        ap_coords=np.linspace(-4, 1, 16),   # physical mm, bregma-relative
        ml_coords=np.linspace(-4, 4, 16),
        atlas_image=atlas_img,
        atlas_mode="crop",   # or "full"
        bl_distance=7.5,     # bregma-lambda distance in mm, default scale
    )
"""

from __future__ import annotations

import numpy as np
import panel as pn
import param
from bokeh.models import ColumnDataSource, HoverTool, TapTool
from bokeh.plotting import figure

from .channel_grid import ChannelGrid

__all__ = ["ChannelGridWidget"]

_COL_UNSELECTED = "#2c2c3e"
_COL_SELECTED   = "#4fc3f7"
_COL_BORDER     = "#1a1a2e"
_COL_TEXT       = "#cdd6f4"
_BG             = "#181825"


class ChannelGridWidget(param.Parameterized):
    """
    Interactive Panel widget wrapping a ChannelGrid.

    Parameters
    ----------
    n_ap, n_ml : int
        Grid dimensions.
    cell_size : int
        Pixel size of each grid cell.
    cell_values : np.ndarray | None
        Optional (n_ap, n_ml) per-electrode scalar (e.g. RMS, variance).
        Normalized to [0,1] and used as background brightness for unselected
        cells, so you can see signal amplitude while navigating.
    ap_coords, ml_coords : np.ndarray | None
        Physical coordinates (mm, bregma-relative) for each AP row / ML col.
        Required for atlas overlay.
    atlas_image : np.ndarray | None
        RGB or RGBA uint8 image of the rat dorsal cortex atlas.
    atlas_mode : str
        "crop"  — crop the atlas to the electrode extent, show grid on top.
        "full"  — show full atlas, place grid at correct physical location.
    bl_distance : float
        Bregma-lambda distance in mm used to scale the atlas image.
        Default 7.5 mm (rat average).
    """

    grid: ChannelGrid = param.Parameter()

    def __init__(
        self,
        n_ap: int = 16,
        n_ml: int = 16,
        *,
        cell_size: int = 28,
        cell_values: np.ndarray | None = None,
        bad_mask: np.ndarray | None = None,
        # atlas
        ap_coords: np.ndarray | None = None,
        ml_coords: np.ndarray | None = None,
        atlas_image: np.ndarray | None = None,
        atlas_mode: str = "crop",          # "crop" | "full"
        bl_distance: float = 7.5,
        **params,
    ):
        params.setdefault("grid", ChannelGrid(n_ap=n_ap, n_ml=n_ml))
        super().__init__(**params)

        self._cell_size   = cell_size
        self._cell_values = self._validate_cell_values(cell_values, self.grid.n_ap, self.grid.n_ml)
        self._bad_mask    = self._validate_bad_mask(bad_mask, self.grid.n_ap, self.grid.n_ml)

        # atlas
        self._ap_coords   = np.asarray(ap_coords) if ap_coords is not None else None
        self._ml_coords   = np.asarray(ml_coords) if ml_coords is not None else None
        self._atlas_image = atlas_image
        self._atlas_mode  = atlas_mode
        self._bl_distance = float(bl_distance)

        self._source   = self._build_source()
        self._fig      = self._build_figure()
        self._controls = self._build_controls()

        self.grid.param.watch(self._on_grid_change, "selected")
        self.grid.param.watch(self._on_mode_change, "mode")

    # ------------------------------------------------------------------ #
    # Constructors                                                         #
    # ------------------------------------------------------------------ #
    @classmethod
    def from_grid(
        cls,
        grid: ChannelGrid,
        *,
        cell_size: int = 28,
        cell_values: np.ndarray | None = None,
        bad_mask: np.ndarray | None = None,
        ap_coords: np.ndarray | None = None,
        ml_coords: np.ndarray | None = None,
        atlas_image: np.ndarray | None = None,
        atlas_mode: str = "crop",
        bl_distance: float = 7.5,
    ) -> "ChannelGridWidget":
        return cls(
            grid.n_ap, grid.n_ml,
            grid=grid,
            cell_size=cell_size,
            cell_values=cell_values,
            bad_mask=bad_mask,
            ap_coords=ap_coords,
            ml_coords=ml_coords,
            atlas_image=atlas_image,
            atlas_mode=atlas_mode,
            bl_distance=bl_distance,
        )

    # ------------------------------------------------------------------ #
    # Public                                                               #
    # ------------------------------------------------------------------ #
    def panel(self) -> pn.viewable.Viewable:
        return pn.Column(
            self._controls,
            pn.pane.Bokeh(self._fig),
            styles={"background": _BG, "padding": "10px", "border-radius": "8px"},
        )

    def update_cell_values(self, values: np.ndarray) -> None:
        """Push a new per-electrode background array and redraw."""
        self._cell_values = self._validate_cell_values(values, self.grid.n_ap, self.grid.n_ml)
        self._patch_colors()

    def update_bad_mask(self, bad_mask: np.ndarray | None) -> None:
        """Push a new bad-channel mask and redraw overlay."""
        self._bad_mask = self._validate_bad_mask(bad_mask, self.grid.n_ap, self.grid.n_ml)
        self._patch_bad()

    # ------------------------------------------------------------------ #
    # Data source                                                          #
    # ------------------------------------------------------------------ #
    def _build_source(self) -> ColumnDataSource:
        g   = self.grid
        sel = g.selected
        cv  = self._cell_values
        bad = self._bad_mask

        aps, mls, colors, alphas, labels, bad_alpha = [], [], [], [], [], []
        for ap in range(g.n_ap):
            for ml in range(g.n_ml):
                selected = (ap, ml) in sel
                aps.append(ap)
                mls.append(ml)
                colors.append(_COL_SELECTED if selected else _COL_UNSELECTED)
                alphas.append(self._cell_alpha(ap, ml, selected, cv))
                labels.append(f"AP={ap}, ML={ml}")
                is_bad = bool(bad[ap, ml]) if bad is not None else False
                bad_alpha.append(1.0 if is_bad else 0.0)

        return ColumnDataSource(dict(ap=aps, ml=mls, color=colors, alpha=alphas, label=labels, bad_alpha=bad_alpha))

    def _cell_alpha(self, ap, ml, selected, cv):
        if selected:
            return 1.0
        if cv is None:
            return 0.45
        return 0.25 + 0.65 * float(cv[ap, ml])

    # ------------------------------------------------------------------ #
    # Bokeh figure                                                         #
    # ------------------------------------------------------------------ #
    def _build_figure(self) -> figure:
        g  = self.grid
        cs = self._cell_size
        pw = g.n_ml * cs + 50
        ph = g.n_ap * cs + 50

        fig = figure(
            width=pw, height=ph,
            x_range=(-0.5, g.n_ml - 0.5),
            y_range=(-0.5, g.n_ap - 0.5),
            tools="tap",
            toolbar_location=None,
        )
        fig.background_fill_color      = _BG
        fig.border_fill_color          = _BG
        fig.grid.visible               = False
        fig.axis.axis_line_color       = "#3a3a5c"
        fig.axis.major_tick_line_color = "#3a3a5c"
        fig.axis.minor_tick_line_color = None
        fig.axis.major_label_text_color     = _COL_TEXT
        fig.axis.major_label_text_font_size = "9px"
        fig.xaxis.axis_label = "ML"
        fig.yaxis.axis_label = "AP"
        fig.axis.axis_label_text_color = _COL_TEXT

        # Atlas background (drawn before electrode rects so it sits behind)
        self._maybe_add_atlas(fig)

        fig.rect(
            x="ml", y="ap",
            width=0.88, height=0.88,
            source=self._source,
            fill_color="color",
            fill_alpha="alpha",
            line_color=_COL_BORDER,
            line_width=1,
        )

        # Bad-channel overlay (X marker) — always present, alpha toggled by bad_alpha.
        fig.scatter(
            x="ml",
            y="ap",
            source=self._source,
            marker="x",
            size=12,
            line_color="#f38ba8",
            line_width=2,
            fill_alpha=0.0,
            line_alpha="bad_alpha",
        )

        fig.add_tools(HoverTool(tooltips=[("", "@label")]))
        self._source.selected.on_change("indices", self._on_tap)

        return fig

    def _maybe_add_atlas(self, fig: figure) -> None:
        """Overlay atlas image if provided."""
        if self._atlas_image is None:
            return

        img  = self._atlas_image
        g    = self.grid
        mode = self._atlas_mode

        # Ensure RGBA uint32 for Bokeh image_rgba
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        if img.ndim == 3 and img.shape[2] == 3:
            alpha = np.full((*img.shape[:2], 1), 180, dtype=np.uint8)
            img   = np.concatenate([img, alpha], axis=2)
        # Bokeh wants (H, W) uint32 in RGBA packed
        img_u32 = img.view(np.uint32).reshape(img.shape[:2])

        if mode == "crop" or self._ap_coords is None or self._ml_coords is None:
            # Fill the whole plot area — simple stretch to grid extent
            fig.image_rgba(
                image=[img_u32],
                x=-0.5, y=-0.5,
                dw=g.n_ml, dh=g.n_ap,
                level="image",
            )
        else:
            # "full" mode: place the image at correct physical coords
            # Map physical mm → grid index space using bl_distance scale
            # 1 grid unit = (ap_range / (n_ap-1)) mm
            ap_min = float(self._ap_coords.min())
            ap_max = float(self._ap_coords.max())
            ml_min = float(self._ml_coords.min())
            ml_max = float(self._ml_coords.max())

            # How many mm per grid step
            ap_step = (ap_max - ap_min) / max(g.n_ap - 1, 1)
            ml_step = (ml_max - ml_min) / max(g.n_ml - 1, 1)

            # Atlas physical extent (assume atlas covers bregma ± bl_distance/2)
            half_bl = self._bl_distance / 2
            # x = ML axis, y = AP axis
            x0_mm = -half_bl;  x1_mm = half_bl
            y0_mm = -half_bl;  y1_mm = half_bl

            # Convert mm → grid index coords
            x0 = (x0_mm - ml_min) / ml_step - 0.5
            x1 = (x1_mm - ml_min) / ml_step - 0.5
            y0 = (y0_mm - ap_min) / ap_step - 0.5
            y1 = (y1_mm - ap_min) / ap_step - 0.5

            fig.image_rgba(
                image=[img_u32],
                x=x0, y=y0,
                dw=(x1 - x0), dh=(y1 - y0),
                level="image",
            )

    # ------------------------------------------------------------------ #
    # Controls                                                             #
    # ------------------------------------------------------------------ #
    def _build_controls(self) -> pn.viewable.Viewable:
        g = self.grid

        self._mode_btn = pn.widgets.RadioButtonGroup(
            options=["row", "column", "sparse", "neighborhood", "manual"],
            value=g.mode,
            button_type="light",
            sizing_mode="stretch_width",
        )
        self._mode_btn.param.watch(self._on_mode_btn, "value")

        # Secondary widgets
        self._row_slider = pn.widgets.IntSlider(
            name="AP row", value=g.row, start=0, end=g.n_ap - 1
        )
        self._col_slider = pn.widgets.IntSlider(
            name="ML col", value=g.column, start=0, end=g.n_ml - 1
        )
        self._stride_slider = pn.widgets.IntSlider(
            name="Stride", value=g.sparse_stride, start=1, end=max(g.n_ap, g.n_ml) // 2
        )
        self._offset_slider = pn.widgets.IntSlider(
            name="Offset", value=g.sparse_offset, start=0, end=max(g.n_ap, g.n_ml) - 1
        )
        self._radius_slider = pn.widgets.IntSlider(
            name="Radius", value=g.neighborhood_radius, start=0, end=max(g.n_ap, g.n_ml) // 2
        )
        self._n_selected_md = pn.pane.Markdown(
            self._selected_label(),
            styles={"color": _COL_TEXT, "font-size": "11px"},
        )

        # Wire secondary widgets → grid
        self._row_slider.param.watch(lambda e: setattr(g, "row", e.new), "value")
        self._col_slider.param.watch(lambda e: setattr(g, "column", e.new), "value")
        self._stride_slider.param.watch(lambda e: setattr(g, "sparse_stride", e.new), "value")
        self._offset_slider.param.watch(lambda e: setattr(g, "sparse_offset", e.new), "value")
        self._radius_slider.param.watch(lambda e: setattr(g, "neighborhood_radius", e.new), "value")

        self._secondary = pn.Row()
        self._refresh_secondary()

        return pn.Column(
            self._mode_btn,
            self._secondary,
            self._n_selected_md,
            styles={"margin-bottom": "6px"},
        )

    def _secondary_for_mode(self, mode: str) -> pn.viewable.Viewable:
        dim = {"color": _COL_TEXT, "font-size": "11px"}
        if mode == "row":
            return self._row_slider
        if mode == "column":
            return self._col_slider
        if mode == "sparse":
            return pn.Row(self._stride_slider, self._offset_slider)
        if mode == "neighborhood":
            return pn.Row(
                self._radius_slider,
                pn.pane.Markdown("_Click grid to set center_", styles=dim),
            )
        if mode == "manual":
            return pn.pane.Markdown("_Click cells to toggle selection_", styles=dim)
        return pn.Row()

    def _refresh_secondary(self):
        self._secondary.objects = [self._secondary_for_mode(self.grid.mode)]

    # ------------------------------------------------------------------ #
    # Callbacks                                                            #
    # ------------------------------------------------------------------ #
    def _on_tap(self, attr, old, new):
        if not new:
            return
        src = self._source.data
        ap  = int(src["ap"][new[0]])
        ml  = int(src["ml"][new[0]])
        g   = self.grid

        # Always clear Bokeh's own selection — we own the visual state
        self._source.selected.indices = []

        mode = g.mode
        if mode == "row":
            g.row = ap
            self._row_slider.value = ap
        elif mode == "column":
            g.column = ml
            self._col_slider.value = ml
        elif mode == "sparse":
            pass  # tap has no meaning in sparse mode; sliders drive it
        elif mode == "neighborhood":
            g.neighborhood_center = (ap, ml)
        elif mode == "manual":
            # Toggle via ChannelGrid helper — does NOT replace manual_selection
            # wholesale, so Bokeh source state is never out of sync
            g.toggle_manual(ap, ml)

    def _on_mode_btn(self, event):
        self.grid.mode = event.new

    def _on_mode_change(self, event):
        # Keep button in sync if mode was set programmatically on the grid
        self._mode_btn.value = event.new
        self._refresh_secondary()

    def _on_grid_change(self, event):
        self._patch_colors()
        self._n_selected_md.object = self._selected_label()

    # ------------------------------------------------------------------ #
    # Color update — patch only, never replace source.data wholesale      #
    # ------------------------------------------------------------------ #
    def _patch_colors(self):
        g   = self.grid
        sel = g.selected
        cv  = self._cell_values
        src = self._source.data

        color_patches = []
        alpha_patches = []
        for i, (ap, ml) in enumerate(zip(src["ap"], src["ml"])):
            selected = (int(ap), int(ml)) in sel
            color_patches.append((i, _COL_SELECTED if selected else _COL_UNSELECTED))
            alpha_patches.append((i, self._cell_alpha(int(ap), int(ml), selected, cv)))

        self._source.patch({"color": color_patches, "alpha": alpha_patches})

    def _patch_bad(self):
        bad = self._bad_mask
        src = self._source.data
        patches = []
        if bad is None:
            patches = [(i, 0.0) for i in range(len(src["ap"]))]
        else:
            for i, (ap, ml) in enumerate(zip(src["ap"], src["ml"])):
                patches.append((i, 1.0 if bool(bad[int(ap), int(ml)]) else 0.0))
        self._source.patch({"bad_alpha": patches})

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #
    def _selected_label(self) -> str:
        return f"**{self.grid.n_selected}** channels selected"

    @staticmethod
    def _validate_cell_values(cv, n_ap, n_ml):
        if cv is None:
            return None
        cv = np.asarray(cv, dtype=float)
        if cv.shape != (n_ap, n_ml):
            raise ValueError(f"cell_values must have shape ({n_ap}, {n_ml}), got {cv.shape}")
        lo, hi = np.nanmin(cv), np.nanmax(cv)
        return (cv - lo) / (hi - lo) if hi > lo else np.full_like(cv, 0.5)

    @staticmethod
    def _validate_bad_mask(bad_mask, n_ap, n_ml):
        if bad_mask is None:
            return None
        bad = np.asarray(bad_mask, dtype=bool)
        if bad.shape != (n_ap, n_ml):
            raise ValueError(f"bad_mask must have shape ({n_ap}, {n_ml}), got {bad.shape}")
        return bad

    def __repr__(self) -> str:
        return f"ChannelGridWidget({self.grid!r})"
