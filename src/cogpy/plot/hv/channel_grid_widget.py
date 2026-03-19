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

from collections.abc import Callable
import numpy as np
import panel as pn
import param
from bokeh.models import ColumnDataSource, HoverTool, TapTool
from bokeh.plotting import figure

from .channel_grid import ChannelGrid
from .theme import BG, BORDER, TEAL, TEXT, TEXT_MED, style_figure
from .topomap import TopoMap
from cogpy.datasets.schemas import AtlasImageOverlay

__all__ = ["ChannelGridWidget"]

_COL_UNSELECTED = "#2c2c3e"
_COL_SELECTED   = TEAL
_COL_BORDER     = "#1a1a2e"
_COL_TEXT       = TEXT
_BG             = BG


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
        topomap: TopoMap | None = None,
        atlas_overlay: AtlasImageOverlay | None = None,
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
        self._topomap     = topomap
        self._cell_values = None if topomap is not None else self._validate_cell_values(cell_values, self.grid.n_ap, self.grid.n_ml)
        self._bad_mask    = self._validate_bad_mask(bad_mask, self.grid.n_ap, self.grid.n_ml)

        self._tap_callbacks: list[Callable[[dict], None]] = []

        # atlas
        self._ap_coords   = np.asarray(ap_coords) if ap_coords is not None else None
        self._ml_coords   = np.asarray(ml_coords) if ml_coords is not None else None
        if atlas_overlay is not None and atlas_image is not None:
            raise ValueError("Provide only one of atlas_overlay or atlas_image")
        self._atlas_overlay = atlas_overlay
        if atlas_overlay is not None:
            self._atlas_image = atlas_overlay.image
            self._atlas_mode = "full" if (self._ap_coords is not None and self._ml_coords is not None) else "crop"
            self._bl_distance = float(atlas_overlay.bl_distance)
        else:
            self._atlas_image = atlas_image
            self._atlas_mode = atlas_mode
            self._bl_distance = float(bl_distance)

        self._source   = self._build_source()
        self._fig      = self._build_figure()
        self._controls = self._build_controls()

        self.grid.param.watch(self._on_grid_change, "selected")
        self.grid.param.watch(self._on_mode_change, "mode")

        if self._topomap is None:
            self._source.selected.on_change("indices", self._on_tap)
        else:
            # In notebooks, showing the TopoMap separately and then embedding it as a
            # background will try to attach the same Bokeh models to a second
            # document, which Bokeh forbids.
            if getattr(self._topomap.figure, "document", None) is not None:
                raise RuntimeError(
                    "TopoMap figure is already attached to a Bokeh document. "
                    "When using TopoMap as the ChannelGridWidget background, do not "
                    "also display the same TopoMap separately. Create a fresh TopoMap "
                    "instance (or display only the widget)."
                )
            if int(self._topomap.n_ap) != int(self.grid.n_ap) or int(self._topomap.n_ml) != int(self.grid.n_ml):
                raise ValueError(
                    "topomap shape must match grid shape: "
                    f"grid=({self.grid.n_ap},{self.grid.n_ml}) topomap=({self._topomap.n_ap},{self._topomap.n_ml})"
                )
            self._topomap.on_tap(self._on_topomap_tap)

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
        topomap: TopoMap | None = None,
        atlas_overlay: AtlasImageOverlay | None = None,
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
            topomap=topomap,
            atlas_overlay=atlas_overlay,
            ap_coords=ap_coords,
            ml_coords=ml_coords,
            atlas_image=atlas_image,
            atlas_mode=atlas_mode,
            bl_distance=bl_distance,
        )

    @classmethod
    def from_topomap(
        cls,
        topomap: TopoMap,
        grid: ChannelGrid,
        *,
        cell_size: int = 28,
        bad_mask: np.ndarray | None = None,
        atlas_overlay: AtlasImageOverlay | None = None,
    ) -> "ChannelGridWidget":
        """
        Construct a widget whose background is driven by the provided TopoMap.
        """
        return cls(
            grid.n_ap,
            grid.n_ml,
            grid=grid,
            cell_size=cell_size,
            cell_values=None,
            bad_mask=bad_mask,
            topomap=topomap,
            atlas_overlay=atlas_overlay,
        )

    # ------------------------------------------------------------------ #
    # Public                                                               #
    # ------------------------------------------------------------------ #
    def panel(self) -> pn.viewable.Viewable:
        return pn.Column(
            self._controls,
            pn.pane.Bokeh(self._fig),
            styles={"background": BG, "padding": "10px", "border-radius": "8px"},
        )

    def on_tap(self, callback: Callable[[dict], None]) -> None:
        """
        Register a tap callback. Called with dict:
          {"ap_idx": int, "ml_idx": int, "ap": float|None, "ml": float|None}
        """
        self._tap_callbacks.append(callback)

    def update_cell_values(self, values: np.ndarray) -> None:
        """Push a new per-electrode background array and redraw."""
        if self._topomap is not None:
            self._topomap.update(values)
            return
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

        aps, mls, ap_idx, ml_idx, colors, alphas, labels, bad_alpha = [], [], [], [], [], [], [], []

        # TopoMap-wired mode draws overlays in physical coords (ap/ml) but uses
        # integer indices for selection logic.
        if self._topomap is not None:
            ap_coords = np.asarray(self._topomap.ap_coords, dtype=float)
            ml_coords = np.asarray(self._topomap.ml_coords, dtype=float)
            ap_step = float(np.median(np.abs(np.diff(ap_coords)))) if len(ap_coords) > 1 else 1.0
            ml_step = float(np.median(np.abs(np.diff(ml_coords)))) if len(ml_coords) > 1 else 1.0
            self._overlay_ap_step = ap_step
            self._overlay_ml_step = ml_step

        for ap in range(g.n_ap):
            for ml in range(g.n_ml):
                selected = (ap, ml) in sel
                ap_idx.append(ap)
                ml_idx.append(ml)

                if self._topomap is None:
                    aps.append(ap)
                    mls.append(ml)
                    colors.append(_COL_SELECTED if selected else _COL_UNSELECTED)
                    alphas.append(self._cell_alpha(ap, ml, selected, cv))
                else:
                    aps.append(float(self._topomap.ap_coords[ap]))
                    mls.append(float(self._topomap.ml_coords[ml]))
                    colors.append(_COL_SELECTED)
                    alphas.append(1.0 if selected else 0.0)

                labels.append(f"AP={ap}, ML={ml}")
                is_bad = bool(bad[ap, ml]) if bad is not None else False
                bad_alpha.append(1.0 if is_bad else 0.0)

        return ColumnDataSource(
            dict(
                ap=aps,
                ml=mls,
                ap_idx=ap_idx,
                ml_idx=ml_idx,
                color=colors,
                alpha=alphas,
                label=labels,
                bad_alpha=bad_alpha,
            )
        )

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

        if self._topomap is not None:
            fig = self._topomap.figure
            fig.width = pw
            fig.height = ph

            ap_step = float(getattr(self, "_overlay_ap_step", 1.0))
            ml_step = float(getattr(self, "_overlay_ml_step", 1.0))

            # Atlas background (drawn before selection overlays so it sits behind).
            self._maybe_add_atlas(fig)

            # Selected-only overlay (unselected alpha=0).
            fig.rect(
                x="ml",
                y="ap",
                width=0.88 * ml_step,
                height=0.88 * ap_step,
                source=self._source,
                fill_color=_COL_SELECTED,
                fill_alpha="alpha",
                line_color=_COL_BORDER,
                line_width=1.5,
            )

            # Bad-channel overlay (X marker) — alpha toggled by bad_alpha.
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

            return fig

        fig = figure(
            width=pw, height=ph,
            x_range=(-0.5, g.n_ml - 0.5),
            y_range=(-0.5, g.n_ap - 0.5),
            tools="tap",
            toolbar_location=None,
        )
        style_figure(fig, xlabel="ML", ylabel="AP", toolbar=False)

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
        # Tap wiring handled in __init__ depending on whether topomap is used.

        return fig

    def _on_topomap_tap(self, info: dict) -> None:
        ap = int(info["ap_idx"])
        ml = int(info["ml_idx"])
        g = self.grid

        mode = g.mode
        if mode == "row":
            g.row = ap
            self._row_slider.value = ap
        elif mode == "column":
            g.column = ml
            self._col_slider.value = ml
        elif mode == "sparse":
            pass
        elif mode == "neighborhood":
            g.neighborhood_center = (ap, ml)
        elif mode == "manual":
            g.toggle_manual(ap, ml)

        self._emit_tap(ap, ml)

    def _emit_tap(self, ap: int, ml: int) -> None:
        if not self._tap_callbacks:
            return
        ap_mm = None
        ml_mm = None
        if self._topomap is not None:
            ap_mm = float(self._topomap.ap_coords[int(ap)])
            ml_mm = float(self._topomap.ml_coords[int(ml)])
        elif self._ap_coords is not None and self._ml_coords is not None:
            ap_mm = float(self._ap_coords[int(ap)])
            ml_mm = float(self._ml_coords[int(ml)])
        info = {"ap_idx": int(ap), "ml_idx": int(ml), "ap": ap_mm, "ml": ml_mm}
        for cb in list(self._tap_callbacks):
            cb(info)

    def _maybe_add_atlas(self, fig: figure) -> None:
        """Overlay atlas image if provided."""
        if self._atlas_image is None:
            return

        img  = self._atlas_image
        g    = self.grid
        mode = self._atlas_mode

        # Ensure RGBA uint32 for Bokeh image_rgba
        if img.dtype != np.uint8:
            img = np.asarray(img)
            # Common case: matplotlib.image.imread returns float in [0, 1].
            if np.issubdtype(img.dtype, np.floating):
                finite = img[np.isfinite(img)]
                if finite.size and float(finite.max()) <= 1.0:
                    img = img * 255.0
            img = np.clip(img, 0, 255).astype(np.uint8)
        if img.ndim == 3 and img.shape[2] == 3:
            alpha = np.full((*img.shape[:2], 1), 180, dtype=np.uint8)
            img   = np.concatenate([img, alpha], axis=2)
        # Bokeh wants (H, W) uint32 in RGBA packed
        img_u32 = img.view(np.uint32).reshape(img.shape[:2])
        # Bokeh's image glyphs map array row 0 to the *bottom* of the image in
        # data space, while most image loaders (and Matplotlib's default imshow)
        # treat row 0 as the *top*. Flip vertically so pixel-to-mm registrations
        # (made in image coordinates) align in Bokeh.
        img_u32 = np.flipud(img_u32)

        # TopoMap-wired mode uses physical coords on the figure axes.
        if self._topomap is not None:
            ap_coords = np.asarray(self._topomap.ap_coords, dtype=float)
            ml_coords = np.asarray(self._topomap.ml_coords, dtype=float)
            ap_min = float(np.nanmin(ap_coords))
            ap_max = float(np.nanmax(ap_coords))
            ml_min = float(np.nanmin(ml_coords))
            ml_max = float(np.nanmax(ml_coords))

            if self._atlas_overlay is not None:
                # Use explicit atlas extents in physical coords.
                y0, y1 = map(float, self._atlas_overlay.ap_extent)  # AP axis
                x0, x1 = map(float, self._atlas_overlay.ml_extent)  # ML axis
            else:
                # Fill electrode extent (physical coords).
                x0, x1 = ml_min, ml_max
                y0, y1 = ap_min, ap_max

            fig.image_rgba(
                image=[img_u32],
                x=float(x0),
                y=float(y0),
                dw=float(x1 - x0),
                dh=float(y1 - y0),
                level="image",
            )
            return

        # Non-TopoMap mode uses grid index space on axes.
        if mode == "crop" or self._ap_coords is None or self._ml_coords is None:
            # Fill the whole plot area — simple stretch to grid extent.
            fig.image_rgba(image=[img_u32], x=-0.5, y=-0.5, dw=g.n_ml, dh=g.n_ap, level="image")
            return

        # "full" mode: place the image at correct physical coords, mapped into grid index space.
        ap_min = float(self._ap_coords.min())
        ap_max = float(self._ap_coords.max())
        ml_min = float(self._ml_coords.min())
        ml_max = float(self._ml_coords.max())

        # How many mm per grid step.
        ap_step = (ap_max - ap_min) / max(g.n_ap - 1, 1)
        ml_step = (ml_max - ml_min) / max(g.n_ml - 1, 1)

        # Atlas physical extent:
        # - If an AtlasImageOverlay is provided, use its explicit extents.
        # - Otherwise, fall back to the older bregma-lambda heuristic.
        if self._atlas_overlay is not None:
            y0_mm, y1_mm = map(float, self._atlas_overlay.ap_extent)  # AP axis
            x0_mm, x1_mm = map(float, self._atlas_overlay.ml_extent)  # ML axis
        else:
            half_bl = self._bl_distance / 2
            x0_mm = -half_bl
            x1_mm = half_bl
            y0_mm = -half_bl
            y1_mm = half_bl

        # Convert mm → grid index coords.
        x0 = (x0_mm - ml_min) / ml_step - 0.5
        x1 = (x1_mm - ml_min) / ml_step - 0.5
        y0 = (y0_mm - ap_min) / ap_step - 0.5
        y1 = (y1_mm - ap_min) / ap_step - 0.5

        fig.image_rgba(image=[img_u32], x=x0, y=y0, dw=(x1 - x0), dh=(y1 - y0), level="image")

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
            styles={"color": TEXT, "font-size": TEXT_MED},
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
        dim = {"color": TEXT, "font-size": TEXT_MED}
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

        self._emit_tap(ap, ml)

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
        for i, (ap, ml) in enumerate(zip(src["ap_idx"], src["ml_idx"])):
            selected = (int(ap), int(ml)) in sel
            if self._topomap is None:
                color_patches.append((i, _COL_SELECTED if selected else _COL_UNSELECTED))
                alpha_patches.append((i, self._cell_alpha(int(ap), int(ml), selected, cv)))
            else:
                color_patches.append((i, _COL_SELECTED))
                alpha_patches.append((i, 1.0 if selected else 0.0))

        self._source.patch({"color": color_patches, "alpha": alpha_patches})

    def _patch_bad(self):
        bad = self._bad_mask
        src = self._source.data
        patches = []
        if bad is None:
            patches = [(i, 0.0) for i in range(len(src["ap"]))]
        else:
            for i, (ap, ml) in enumerate(zip(src["ap_idx"], src["ml_idx"])):
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
