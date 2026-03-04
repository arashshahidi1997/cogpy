"""
Interactive 4D orthogonal slicer (core variant).

This module defines :class:`~cogpy.core.plot.orthoslicer.OrthoSlicer`, a small
Panel/HoloViews viewer for a 4D ``xarray.DataArray`` that you conceptually treat
as ``(t, z, y, x)``:

- ``view_xy``: show an XY image at the current (t, z)
- ``view_tz``: show a TZ image at the current (x, y)

What makes this *core* variant different from the other orthoslicer modules in
this package:

- It focuses on the basic interaction model: tap-to-move crosshairs + optional
  Datashader rasterization for large arrays.
- It does **not** try to persist zoom/pan state across updates (see
  ``orthoslicer_zoom.py``) and it does **not** provide a linked time-window
  RangeTool (see ``orthoslicer_rangercopy.py`` / ``orthoslicer_ranger.py``).
- It does not add local-mesh / faceting helpers (see ``orthoslicer_facet.py``).
"""

import warnings

warnings.warn(
    f"{__name__} is deprecated. "
    "For new projects, use TensorScope (cogpy.core.plot.tensorscope). "
    "For maintenance of existing orthoslicer code, prefer orthoslicer_rangercopy.py.",
    DeprecationWarning,
    stacklevel=2,
)

import numpy as np
import xarray as xr
import holoviews as hv
import panel as pn
import param
import datashader as ds
from holoviews.operation.datashader import rasterize
from holoviews import streams
from ..plot.time_player import PlayerWithRealTime

hv.extension("bokeh")


def nearest_sel(dimx, val):
    return dimx.sel({dimx.name: val}, method="nearest").item()


class OrthoSlicer(param.Parameterized):
    # Declare parameters at class level
    use_datashader = param.Boolean(True, label="Use Datashader")
    t = param.Number(label="Time")
    x = param.Number(label="X")
    y = param.Number(label="Y")
    z = param.Number(label="Z")
    clim = param.Range(label="Color Limits")

    def __init__(
        self, array, dt=None, dz=None, dy=None, dx=None, clim_quantile=True, **params
    ):
        super().__init__(**params)

        if not isinstance(array, xr.DataArray):
            raise ValueError("Input must be an xarray.DataArray.")

        self.array = array
        self.clim_quantile = clim_quantile
        self.dt = dt
        self.dz = dz
        self.dy = dy
        self.dx = dx
        self.vdim = hv.Dimension(
            array.name, label=array.name, unit=array.attrs.get("units", "")
        )

        # Prepare metadata and dimensions
        self.setup()

        # Streams to update with clicked coordinates
        self.tap_xy = streams.Tap(x=self.x, y=self.y)
        self.tap_tz = streams.Tap(x=self.t, y=self.z)
        self.tap_xy.add_subscriber(self._on_tap_xy)
        self.tap_tz.add_subscriber(self._on_tap_tz)

        # t player
        self.set_controls()

    def setup(self):
        # Prepare metadata and dimensions
        self.prepare_meta()

        # Standardize to match expected dimension names t, z, y, x
        self.standardize_coords()

        # Bounds from standardized array
        self.set_param_bounds()

        # Contrast limits (values + bounds) to adjust color mapping
        self.set_contrast_limits()

    def _on_tap_xy(self, x, y):
        if x is not None and y is not None:
            self.param.update(
                x=nearest_sel(self.array.x, x), y=nearest_sel(self.array.y, y)
            )

    def _on_tap_tz(self, x, y):
        if x is not None and y is not None:
            self.param.update(
                t=nearest_sel(self.array.t, x), z=nearest_sel(self.array.z, y)
            )

    @param.depends("t", "z", "x", "y", "clim", "use_datashader")
    def view_xy(self):
        width, height = 800, 400
        img = self.array.sel(t=self.t, z=self.z, method="nearest")
        base = hv.Image(
            img, kdims=[self.hvdims["x"], self.hvdims["y"]], vdims=[self.vdim]
        )
        if self.use_datashader:
            base = rasterize(
                base,
                aggregator=ds.mean("val"),
                width=width,
                height=height,
                dynamic=False,
            )
        base = base.opts(
            cmap="Viridis",
            colorbar=True,
            framewise=True,
            title=self._title_xy(),
            clim=self.clim,
            width=width,
            height=height,
            tools=["tap", "pan", "wheel_zoom", "box_zoom", "reset"],
        )
        cross = hv.VLine(self.x).opts(
            color="white", line_width=1, alpha=0.8
        ) * hv.HLine(self.y).opts(color="white", line_width=1, alpha=0.8)
        self.tap_xy.source = base
        adjoined = base.hist() * cross
        return adjoined

    @param.depends("x", "y", "t", "z", "clim", "use_datashader")
    def view_tz(self):
        width, height = 400, 400
        img = self.array.sel(x=self.x, y=self.y, method="nearest")
        base = hv.Image(
            img, kdims=[self.hvdims["t"], self.hvdims["z"]], vdims=[self.vdim]
        )
        if self.use_datashader:
            base = rasterize(
                base,
                aggregator=ds.mean("val"),
                width=width,
                height=height,
                dynamic=False,
            )
        base = base.opts(
            cmap="Viridis",
            colorbar=True,
            framewise=True,
            title=self._title_tz(),
            clim=self.clim,
            width=width,
            height=height,
            tools=["tap", "pan", "wheel_zoom", "box_zoom", "reset"],
        )
        cross = hv.VLine(self.t).opts(
            color="white", line_width=1, alpha=0.8
        ) * hv.HLine(self.z).opts(color="white", line_width=1, alpha=0.8)
        combined = base * cross
        self.tap_tz.source = combined
        return combined

    # Make these instance methods (or use @staticmethod and call via class)
    def prepare_meta(self):
        # Collect mapping + metadata from provided tuples
        dims = dict(x=self.dx, y=self.dy, z=self.dz, t=self.dt)

        rename_map = {}
        original_name_map = {}
        self.hvdims = {}

        for std_name, spec in dims.items():
            if spec is None:
                continue
            src, hvdim = spec
            rename_map[src] = std_name
            original_name_map[std_name] = src
            self.hvdims[std_name] = hvdim  # hv.Dimension carries label + unit

            # update param label from hvdim
            if hvdim.label and std_name in self.param:
                self.param[std_name].label = hvdim.label

        self._rename_map = rename_map
        self._original_name = original_name_map
        self._units = {std_name: hvdim.unit for std_name, hvdim in self.hvdims.items()}
        self._labels = {
            std_name: hvdim.label for std_name, hvdim in self.hvdims.items()
        }

        # make it explicit with attributes
        self.lx, self.ly, self.lz, self.lt = (
            self._labels["x"],
            self._labels["y"],
            self._labels["z"],
            self._labels["t"],
        )
        self.ox, self.oy, self.oz, self.ot = (
            self._original_name["x"],
            self._original_name["y"],
            self._original_name["z"],
            self._original_name["t"],
        )
        self.ux, self.uy, self.uz, self.ut = (
            self._units["x"],
            self._units["y"],
            self._units["z"],
            self._units["t"],
        )

    def standardize_coords(self):
        self.array = self.array.rename(
            self._rename_map
        )  # e.g. {'time':'t','freq':'z','ml':'x','ap':'y'}
        self.array = self.array.transpose("t", "z", "y", "x")
        self.array.name = "val"

        for dim in ("t", "z", "x", "y"):
            if dim in self.param:
                lbl = self._labels.get(dim)
                if lbl:
                    self.param[dim].label = lbl

    def get_bounds(self):
        b = {
            dim: (self.array[dim].min().item(), self.array[dim].max().item())
            for dim in self.array.dims
        }
        b[self.array.name] = (self.array.min().item(), self.array.max().item())
        return b

    def set_contrast_limits(self):
        vmin, vmax = self.bounds["val"]
        if self.clim_quantile:
            clim_init = (
                float(np.percentile(self.array.values, 2)),
                float(np.percentile(self.array.values, 98)),
            )
        else:
            clim_init = (float(vmin), float(vmax))
        self.clim = clim_init  # set initial clim
        self.param["clim"].bounds = (float(vmin), float(vmax))
        self.param["clim"].default = clim_init

    def set_param_bounds(self):
        self.bounds = self.get_bounds()

        # Set parameter bounds and starting values
        for dim in ("t", "x", "y", "z"):
            lo, hi = self.bounds[dim]
            self.param[dim].bounds = (float(lo), float(hi))
            # start at lower bound (or use midpoint if you prefer)
            setattr(self, dim, float(lo))

    def set_controls(self):
        # Set up the t player
        self.set_t_player()

        # Discrete coords for spatial axes
        x_vals = list(map(float, self.array.x.values))
        y_vals = list(map(float, self.array.y.values))
        z_vals = list(map(float, self.array.z.values))

        # A small live readout for t (reactive via pn.bind)
        self.param_controls = pn.Param(
            self,
            parameters=["x", "y", "z", "clim", "use_datashader"],  # no "t" here
            widgets={
                "x": pn.widgets.DiscreteSlider(options=x_vals, name=self.ox),
                "y": pn.widgets.DiscreteSlider(options=y_vals, name=self.oy),
                "z": pn.widgets.DiscreteSlider(options=z_vals, name=self.oz),
                "clim": pn.widgets.RangeSlider,
                "use_datashader": pn.widgets.Checkbox,
            },
            show_name=False,
            width=320,
        )

    def display_controls(self):
        return pn.Column(
            pn.pane.Markdown("### Time"),
            self.t_player_widget.view,
            pn.layout.Divider(),
            pn.pane.Markdown("### Controls"),
            self.param_controls,
        )

    def set_t_player(self):
        self.t_player_widget = PlayerWithRealTime(
            self.array.t.values, interval_bounds=(50, 2000)
        )
        self.t_player = self.t_player_widget.t_player
        self.t_player.param.watch(lambda e: setattr(self, "t", e.new), "value")
        self.param.watch(lambda e: setattr(self.t_player, "value", e.new), "t")

    def _fmt_val(self, dim, val):
        u = self._units.get(dim, "")
        if u:
            return f"{val:.2f} {u}"
        return f"{val:.2f}"

    def _title_xy(self):
        # "ML-AP @ freq=… Hz, time=… s" (using your labels + units)
        return f"{self.ox}-{self.oy} @ {self.oz}={self._fmt_val('z', self.z)}, {self.ot}={self._fmt_val('t', self.t)}"

    def _title_tz(self):
        # "time-freq @ ML=… , AP=…"
        return f"{self.ot}-{self.oz} @ {self.ox}={self._fmt_val('x', self.x)}, {self.oy}={self._fmt_val('y', self.y)}"


def nearest_sel(dimx, val):
    return dimx.sel({dimx.name: val}, method="nearest").item()
