"""
topomap.py
==========
AP×ML heatmap of a per-electrode scalar.

Accepts any (AP, ML) numpy array or xarray.DataArray — the scalar can be
precomputed RMS, time-windowed power from a GridSpectrogram4D, z-score,
correlation, or anything else that reduces to one value per electrode.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import xarray as xr
from cogpy.utils.imports import import_optional
pn = import_optional("panel")
bokeh_models = import_optional("bokeh.models")
from bokeh.models import BasicTicker, ColorBar, ColumnDataSource, HoverTool, LinearColorMapper
from bokeh.models import FixedTicker
from bokeh.plotting import figure

from .theme import BG, BORDER, COLORMAPS, TEXT, style_figure

__all__ = ["TopoMap"]


class TopoMap:
    """
    AP×ML heatmap of a per-electrode scalar.

    Parameters
    ----------
    values
        Shape (n_ap, n_ml).
    ap_coords, ml_coords
        Physical coordinates (mm) for axes. If None, uses integer indices.
    colormap
        One of COLORMAPS keys.
    symmetric
        If True, clamps range symmetrically around zero (±max|value|).
    cell_size
        Cell size used to derive default figure dimensions.
    width, height
        Override figure dimensions.
    title
        Plot title.
    electrode_radius
        Radius as fraction of min(AP step, ML step).
    show_values
        Overlay numeric value text on each cell (cluttered for large grids).
    """

    def __init__(
        self,
        values: np.ndarray,
        *,
        ap_coords: np.ndarray | None = None,
        ml_coords: np.ndarray | None = None,
        colormap: str = "viridis",
        symmetric: bool = False,
        style: str = "electrodes",
        cell_size: int = 30,
        width: int | None = None,
        height: int | None = None,
        title: str = "TopoMap",
        electrode_radius: float = 0.38,
        show_values: bool = False,
    ):
        values = np.asarray(values, dtype=float)
        if values.ndim != 2:
            raise ValueError(f"values must be 2D (n_ap, n_ml), got shape {values.shape}")

        self._n_ap, self._n_ml = values.shape
        self._ap_coords = self._as_float_coords(ap_coords, self._n_ap, "ap_coords")
        self._ml_coords = self._as_float_coords(ml_coords, self._n_ml, "ml_coords")

        if colormap not in COLORMAPS:
            raise ValueError(f"colormap must be one of {list(COLORMAPS)}, got {colormap!r}")

        if style not in {"electrodes", "image"}:
            raise ValueError("style must be 'electrodes' or 'image'")

        self._colormap = colormap
        self._symmetric = bool(symmetric)
        self._style = str(style)
        self._cell_size = int(cell_size)
        self._electrode_radius = float(electrode_radius)
        self._show_values = bool(show_values)
        self._title = str(title)
        self._tap_callbacks: list[Callable[[dict], None]] = []

        pw = int(width) if width is not None else (self._n_ml * self._cell_size + 100)
        ph = int(height) if height is not None else (self._n_ap * self._cell_size + 80)

        self._source, self._mapper = self._build_source_and_mapper(values)
        self._fig = self._build_figure(pw, ph)
        self._layout = None

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_dataarray(
        cls,
        da: xr.DataArray,
        *,
        ap_dim: str = "AP",
        ml_dim: str = "ML",
        **kwargs,
    ) -> "TopoMap":
        for dim in (ap_dim, ml_dim):
            if dim not in da.dims:
                raise ValueError(f"DataArray missing dim {dim!r}, got {tuple(da.dims)}")

        da_t = da.transpose(ap_dim, ml_dim)
        values = np.asarray(da_t.values, dtype=float)
        ap_coords = np.asarray(da_t[ap_dim].values)
        ml_coords = np.asarray(da_t[ml_dim].values)
        return cls(values, ap_coords=ap_coords, ml_coords=ml_coords, **kwargs)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=float)
        if values.shape != (self._n_ap, self._n_ml):
            raise ValueError(f"values shape {values.shape} does not match ({self._n_ap}, {self._n_ml})")
        lo, hi = self._color_range(values)
        self._mapper.low = lo
        self._mapper.high = hi

        flat = values.ravel()
        self._source.patch(
            {
                "value": [(i, float(v)) for i, v in enumerate(flat)],
                "value_str": [(i, f"{float(v):.3g}") for i, v in enumerate(flat)],
            }
        )
        # Image-style uses a separate image source.
        if getattr(self, "_img_source", None) is not None:
            self._img_source.data = {"image": [values.astype(float)]}

    def on_tap(self, callback: Callable[[dict], None]) -> None:
        self._tap_callbacks.append(callback)

    def panel(self) -> pn.viewable.Viewable:
        if self._layout is None:
            self._layout = pn.pane.Bokeh(self._fig)
        return self._layout

    @property
    def figure(self) -> figure:
        return self._fig

    @property
    def n_ap(self) -> int:
        return int(self._n_ap)

    @property
    def n_ml(self) -> int:
        return int(self._n_ml)

    @property
    def ap_coords(self) -> np.ndarray:
        return np.asarray(self._ap_coords)

    @property
    def ml_coords(self) -> np.ndarray:
        return np.asarray(self._ml_coords)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    @staticmethod
    def _as_float_coords(coords: np.ndarray | None, n: int, name: str) -> np.ndarray:
        if coords is None:
            return np.arange(n, dtype=float)
        arr = np.asarray(coords)
        if arr.shape != (n,):
            raise ValueError(f"{name} must have shape ({n},), got {arr.shape}")
        try:
            return arr.astype(float)
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"{name} must be numeric; got dtype {arr.dtype}") from e

    def _color_range(self, values: np.ndarray) -> tuple[float, float]:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return 0.0, 1.0
        lo, hi = float(finite.min()), float(finite.max())
        if self._symmetric:
            bound = max(abs(lo), abs(hi))
            return -bound, bound
        if lo == hi:
            return lo - 1.0, hi + 1.0
        return lo, hi

    def _build_source_and_mapper(self, values: np.ndarray) -> tuple[ColumnDataSource, LinearColorMapper]:
        ap_idx_grid, ml_idx_grid = np.meshgrid(np.arange(self._n_ap), np.arange(self._n_ml), indexing="ij")
        ap_phys_grid, ml_phys_grid = np.meshgrid(self._ap_coords, self._ml_coords, indexing="ij")

        # Display convention: AP index starts at the top (opposite physical direction when AP increases upward).
        ap_idx_display = (self._n_ap - 1 - ap_idx_grid)

        flat_vals = values.ravel()
        lo, hi = self._color_range(values)

        source = ColumnDataSource(
            dict(
                ap_idx=ap_idx_grid.ravel().tolist(),
                ap_idx_display=ap_idx_display.ravel().tolist(),
                ml_idx=ml_idx_grid.ravel().tolist(),
                ap=ap_phys_grid.ravel().tolist(),
                ml=ml_phys_grid.ravel().tolist(),
                value=[float(v) for v in flat_vals.tolist()],
                value_str=[f"{float(v):.3g}" for v in flat_vals.tolist()],
            )
        )

        mapper = LinearColorMapper(palette=COLORMAPS[self._colormap], low=lo, high=hi, nan_color="#2a2a2a")
        return source, mapper

    def _build_figure(self, pw: int, ph: int) -> figure:
        ap_phys = self._ap_coords
        ml_phys = self._ml_coords

        ap_pad = (float(ap_phys[-1]) - float(ap_phys[0])) * 0.08 if len(ap_phys) > 1 else 1.0
        ml_pad = (float(ml_phys[-1]) - float(ml_phys[0])) * 0.08 if len(ml_phys) > 1 else 1.0

        fig = figure(
            width=pw,
            height=ph,
            x_range=(float(ml_phys.min()) - ml_pad, float(ml_phys.max()) + ml_pad),
            y_range=(float(ap_phys.min()) - ap_pad, float(ap_phys.max()) + ap_pad),
            tools="tap",
            toolbar_location=None,
            title=self._title,
        )
        style_figure(fig, xlabel="ML", ylabel="AP", toolbar=False)
        fig.title.text_color = TEXT
        fig.title.text_font_size = "13px"

        if self._style == "electrodes":
            ap_step = float(np.diff(ap_phys).mean()) if len(ap_phys) > 1 else 1.0
            ml_step = float(np.diff(ml_phys).mean()) if len(ml_phys) > 1 else 1.0
            radius = self._electrode_radius * min(abs(ap_step), abs(ml_step))

            fig.circle(
                x="ml",
                y="ap",
                radius=radius,
                source=self._source,
                fill_color={"field": "value", "transform": self._mapper},
                fill_alpha=0.92,
                line_color="#1a1a2e",
                line_width=0.8,
            )
        else:
            # Continuous heatmap image in physical coordinates.
            # Note: image expects rows correspond to y (AP) increasing upward.
            initial = np.array(self._source.data["value"], dtype=float).reshape(self._n_ap, self._n_ml)
            self._img_source = ColumnDataSource({"image": [initial]})
            x0 = float(ml_phys.min())
            y0 = float(ap_phys.min())
            dw = float(ml_phys.max() - ml_phys.min()) if len(ml_phys) > 1 else 1.0
            dh = float(ap_phys.max() - ap_phys.min()) if len(ap_phys) > 1 else 1.0
            fig.image(
                image="image",
                x=x0,
                y=y0,
                dw=dw,
                dh=dh,
                source=self._img_source,
                color_mapper=self._mapper,
            )

        if self._show_values:
            fig.text(
                x="ml",
                y="ap",
                text="value_str",
                source=self._source,
                text_align="center",
                text_baseline="middle",
                text_font_size="7px",
                text_color="#ffffff",
                text_alpha=0.75,
            )

        color_bar = ColorBar(
            color_mapper=self._mapper,
            ticker=BasicTicker(desired_num_ticks=6),
            label_standoff=8,
            border_line_color=None,
            location=(0, 0),
            background_fill_color=BG,
            major_label_text_color=TEXT,
            major_label_text_font_size="9px",
        )
        fig.add_layout(color_bar, "right")

        # Show ML/AP indices as axis tick labels, positioned at physical coords.
        fig.xaxis.ticker = FixedTicker(ticks=[float(v) for v in ml_phys.tolist()])
        fig.yaxis.ticker = FixedTicker(ticks=[float(v) for v in ap_phys.tolist()])
        fig.xaxis.major_label_overrides = {float(v): str(i) for i, v in enumerate(ml_phys.tolist())}
        fig.yaxis.major_label_overrides = {float(v): str(self._n_ap - 1 - i) for i, v in enumerate(ap_phys.tolist())}

        fig.add_tools(
            HoverTool(
                tooltips=[
                    ("AP", "@ap{0.2f} mm  (row @ap_idx_display)"),
                    ("ML", "@ml{0.2f} mm  (col @ml_idx)"),
                    ("value", "@value{0.4f}"),
                ]
            )
        )

        self._source.selected.on_change("indices", self._on_tap)
        return fig

    def _on_tap(self, attr, old, new) -> None:
        if not new:
            return

        # We own selection state; always clear Bokeh's selection.
        idx = int(new[0])
        self._source.selected.indices = []

        if not self._tap_callbacks:
            return

        d = self._source.data
        info = {
            "ap_idx": int(d["ap_idx"][idx]),
            "ap_idx_display": int(d["ap_idx_display"][idx]),
            "ml_idx": int(d["ml_idx"][idx]),
            "ap": float(d["ap"][idx]),
            "ml": float(d["ml"][idx]),
            "value": float(d["value"][idx]),
        }
        for cb in list(self._tap_callbacks):
            cb(info)

    def __repr__(self) -> str:
        return f"TopoMap({self._n_ap}×{self._n_ml} colormap={self._colormap!r} symmetric={self._symmetric})"
