import panel as pn
import param


class TimeHair(param.Parameterized):
    """
    Clickable time hair (vertical line) for HoloViews plots.

    Attach to a ``hv.Curve`` (or any element with a time kdim) to add a movable
    vertical line that follows click/tap events. The selected time is stored in
    ``.t`` and can be linked to other widgets/logic via Param watchers.
    """

    t = param.Parameter(default=None, doc="Selected time coordinate.")
    snap = param.Boolean(
        default=True,
        doc="If True, snap taps to the nearest available time coordinate.",
    )

    def __init__(self, *, t=None, snap=True, snap_values=None, **params):
        super().__init__(**params)
        self.t = t
        self.snap = bool(snap)
        self._snap_values = snap_values  # optional explicit snapping grid
        self._tap_streams = []

    @staticmethod
    def _dimension_names(element) -> list[str]:
        names = []
        for d in getattr(element, "kdims", []) + getattr(element, "vdims", []):
            names.append(getattr(d, "name", str(d)))
        return names

    @staticmethod
    def _build_snapper(values):
        import numpy as np

        vals = np.asarray(values)
        if vals.size == 0:
            return None

        if vals.dtype == object:
            # Try to coerce common cases (python datetime, floats-as-objects).
            try:
                vals = vals.astype("datetime64[ns]")
            except Exception:
                try:
                    vals = vals.astype(float)
                except Exception:
                    return None

        # Work with a sorted view for fast nearest lookup.
        if np.issubdtype(vals.dtype, np.datetime64):
            order = np.argsort(vals.astype("datetime64[ns]").astype("int64"))
        else:
            order = np.argsort(vals.astype(float))
        vals_sorted = vals[order]

        if np.issubdtype(vals_sorted.dtype, np.datetime64):
            key = vals_sorted.astype("datetime64[ns]").astype("int64")

            def snap(x):
                try:
                    x64 = np.datetime64(x, "ns").astype("int64")
                except Exception:
                    return x
                idx = int(np.searchsorted(key, x64))
                if idx <= 0:
                    return vals_sorted[0]
                if idx >= key.size:
                    return vals_sorted[-1]
                return vals_sorted[idx - 1] if (x64 - key[idx - 1]) <= (key[idx] - x64) else vals_sorted[idx]

            return snap

        key = vals_sorted.astype(float)

        def snap(x):
            try:
                xf = float(x)
            except Exception:
                return x
            idx = int(np.searchsorted(key, xf))
            if idx <= 0:
                return float(vals_sorted[0])
            if idx >= key.size:
                return float(vals_sorted[-1])
            return float(vals_sorted[idx - 1]) if (xf - key[idx - 1]) <= (key[idx] - xf) else float(vals_sorted[idx])

        return snap

    def _infer_snap_values(self, obj, time_kdim: str):
        if self._snap_values is not None:
            return self._snap_values

        import holoviews as hv

        # Find the first element that actually has the requested kdim.
        for el in obj.traverse(lambda x: x, specs=hv.Element):
            kdim_names = [getattr(d, "name", str(d)) for d in getattr(el, "kdims", [])]
            if time_kdim in kdim_names:
                try:
                    vals = el.dimension_values(time_kdim)
                except Exception:
                    continue
                # Avoid pathological memory use for huge arrays.
                if getattr(vals, "size", 0) and int(vals.size) <= 2_000_000:
                    self._snap_values = vals
                    return vals
        return None

    def attach(
        self,
        obj,
        *,
        time_kdim: str = "time",
        ensure_tap_tools: bool = True,
        tools: list[str] | None = None,
        active_tools: list[str] | None = None,
        line_color: str = "white",
        line_width: int = 2,
        line_alpha: float = 0.9,
    ):
        """
        Return a new HoloViews object with an interactive time hair.

        Parameters
        ----------
        obj
            A ``hv.Element`` (e.g. ``hv.Curve``) or a container (``Layout``,
            ``Overlay``, ``DynamicMap``) containing elements with ``time_kdim``.
        time_kdim
            Name of the time key dimension to attach tap behavior to.
        ensure_tap_tools
            If True, applies a basic toolset including ``tap`` so clicks work
            even if the incoming element did not declare tools.
        """
        import holoviews as hv
        from holoviews import streams

        snap_values = self._infer_snap_values(obj, time_kdim) if bool(self.snap) else None
        snapper = self._build_snapper(snap_values) if snap_values is not None and bool(self.snap) else None

        if self.t is None and snap_values is not None and getattr(snap_values, "size", 0):
            self.t = snap_values[0].item() if hasattr(snap_values[0], "item") else snap_values[0]

        params_stream = streams.Params(self, ["t"])

        def _hair(t=None, **_):
            if t is None:
                # Hidden placeholder to keep overlay type stable.
                return hv.VLine(0).opts(alpha=0.0, line_width=0)
            return hv.VLine(t).opts(
                color=str(line_color),
                line_width=int(line_width),
                alpha=float(line_alpha),
            )

        hair_dm = hv.DynamicMap(_hair, streams=[params_stream])

        if tools is None:
            tools = ["tap", "pan", "wheel_zoom", "box_zoom", "reset"]
        if active_tools is None:
            active_tools = ["tap"]

        def _on_tap(**kwargs):
            x = kwargs.get("x")
            if x is None:
                return
            self.t = snapper(x) if snapper is not None else x

        def _decorate(el):
            kdim_names = [getattr(d, "name", str(d)) for d in getattr(el, "kdims", [])]
            if time_kdim not in kdim_names:
                return el

            el2 = el
            if bool(ensure_tap_tools):
                el2 = el2.opts(tools=list(tools), active_tools=list(active_tools))

            tap = streams.Tap(source=el2, x=None, y=None)
            tap.add_subscriber(_on_tap)
            self._tap_streams.append(tap)
            return el2 * hair_dm

        if isinstance(obj, hv.Element):
            return _decorate(obj)
        return obj.map(_decorate, specs=hv.Element)


def add_time_hair(
    obj,
    *,
    time_kdim: str = "time",
    t=None,
    snap: bool = True,
    return_controller: bool = False,
    **attach_kwargs,
):
    """
    Convenience wrapper around :class:`TimeHair`.

    Returns
    -------
    hv object (and optionally the TimeHair controller)
    """
    controller = TimeHair(t=t, snap=snap)
    out = controller.attach(obj, time_kdim=time_kdim, **attach_kwargs)
    return (out, controller) if bool(return_controller) else out


class AxisHair(param.Parameterized):
    """Clickable 1D hair (vertical or horizontal) for HoloViews plots.

    This is a generalization of :class:`TimeHair` that can attach to any kdim
    and can draw either a vertical (x) or horizontal (y) line. Clicking/tapping
    updates ``.value``.
    """

    value = param.Parameter(default=None, doc="Selected coordinate value.")
    snap = param.Boolean(
        default=True,
        doc="If True, snap taps to the nearest available coordinate value.",
    )
    orientation = param.ObjectSelector(
        default="v",
        objects=["v", "h"],
        doc="Hair orientation: 'v' uses tap.x and draws VLine; 'h' uses tap.y and draws HLine.",
    )

    def __init__(self, *, value=None, snap=True, orientation="v", snap_values=None, **params):
        super().__init__(**params)
        self.value = value
        self.snap = bool(snap)
        self.orientation = str(orientation)
        self._snap_values = snap_values
        self._tap_streams = []

    def _infer_snap_values(self, obj, kdim: str):
        if self._snap_values is not None:
            return self._snap_values

        import holoviews as hv

        for el in obj.traverse(lambda x: x, specs=hv.Element):
            kdim_names = [getattr(d, "name", str(d)) for d in getattr(el, "kdims", [])]
            if kdim not in kdim_names:
                continue
            try:
                vals = el.dimension_values(kdim)
            except Exception:
                continue
            if getattr(vals, "size", 0) and int(vals.size) <= 2_000_000:
                self._snap_values = vals
                return vals
        return None

    def attach(
        self,
        obj,
        *,
        kdim: str,
        ensure_tap_tools: bool = True,
        tools: list[str] | None = None,
        active_tools: list[str] | None = None,
        line_color: str = "white",
        line_width: int = 2,
        line_alpha: float = 0.9,
    ):
        import holoviews as hv
        from holoviews import streams

        if tools is None:
            tools = ["tap", "pan", "wheel_zoom", "box_zoom", "reset"]
        if active_tools is None:
            active_tools = ["tap"]

        snap_values = self._infer_snap_values(obj, kdim) if bool(self.snap) else None
        snapper = (
            TimeHair._build_snapper(snap_values)
            if snap_values is not None and bool(self.snap)
            else None
        )

        if self.value is None and snap_values is not None and getattr(snap_values, "size", 0):
            self.value = snap_values[0].item() if hasattr(snap_values[0], "item") else snap_values[0]

        params_stream = streams.Params(self, ["value"])

        def _hair(value=None, **_):
            if value is None:
                placeholder = hv.VLine(0) if self.orientation == "v" else hv.HLine(0)
                return placeholder.opts(alpha=0.0, line_width=0)
            if self.orientation == "v":
                el = hv.VLine(value)
            else:
                el = hv.HLine(value)
            return el.opts(color=str(line_color), line_width=int(line_width), alpha=float(line_alpha))

        hair_dm = hv.DynamicMap(_hair, streams=[params_stream])

        tap_key = "x" if self.orientation == "v" else "y"

        def _on_tap(**kwargs):
            v = kwargs.get(tap_key)
            if v is None:
                return
            self.value = snapper(v) if snapper is not None else v

        def _decorate(el):
            kdim_names = [getattr(d, "name", str(d)) for d in getattr(el, "kdims", [])]
            if kdim not in kdim_names:
                return el

            el2 = el
            if bool(ensure_tap_tools):
                el2 = el2.opts(tools=list(tools), active_tools=list(active_tools))

            tap = streams.Tap(source=el2, x=None, y=None)
            tap.add_subscriber(_on_tap)
            self._tap_streams.append(tap)
            return el2 * hair_dm

        if isinstance(obj, hv.Element):
            return _decorate(obj)
        return obj.map(_decorate, specs=hv.Element)


def add_axis_hair(
    obj,
    *,
    kdim: str,
    value=None,
    snap: bool = True,
    orientation: str = "v",
    return_controller: bool = False,
    **attach_kwargs,
):
    """Convenience wrapper around :class:`AxisHair`."""
    controller = AxisHair(value=value, snap=snap, orientation=orientation)
    out = controller.attach(obj, kdim=kdim, **attach_kwargs)
    return (out, controller) if bool(return_controller) else out


class PlayerWithRealTime(param.Parameterized):
    interval = param.Integer(default=200, bounds=(50, 2000), doc="ms per step")
    speed = param.Number(default=5, bounds=(0.1, 20), doc="Real-time speed multiplier")
    real_time = param.Boolean(default=True, doc="Show speed vs interval control")

    @staticmethod
    def _calculate_bounds(dt_sec, interval_bounds=(50, 2000)):
        """Calculate speed bounds based on interval bounds and time step"""
        min_interval, max_interval = interval_bounds
        # Speed is inversely related to interval: speed = (dt_sec * 1000) / interval
        min_speed = round((dt_sec * 1000) / max_interval, 2)
        max_speed = round((dt_sec * 1000) / min_interval, 2)
        default_interval = 200  # Default interval in ms
        default_speed = round((dt_sec * 1000) / default_interval, 1)

        return {
            "interval_bounds": interval_bounds,
            "speed_bounds": (min_speed, max_speed),
            "default_interval": default_interval,
            "default_speed": default_speed,
        }

    def __init__(self, t_values, interval_bounds=(50, 2000), **params):
        assert len(t_values) >= 2, "Need at least two t values"
        self._t = list(t_values)
        self._dt_sec = self._t[1] - self._t[0]

        bounds_info = self._calculate_bounds(self._dt_sec, interval_bounds)

        # Update bounds and defaults dynamically
        self.param.interval.bounds = bounds_info["interval_bounds"]
        self.interval = bounds_info["default_interval"]

        self.param.speed.bounds = bounds_info["speed_bounds"]
        self.speed = bounds_info["default_speed"]

        super().__init__(**params)

        self._updating = False  # Prevent recursive updates

        # Sync initial values
        self._sync_from_interval()

        # Create the player
        self.t_player = pn.widgets.DiscretePlayer(
            name="t (s)",
            options=self._t,
            value=self._t[0],
            interval=self.interval,
            loop_policy="loop",
        )

        # Simple readout of current t
        self.readout = pn.pane.Str(width=140)
        self._update_readout()

        # Watch for player value changes to update readout
        self.t_player.param.watch(lambda event: self._update_readout(), "value")

        # UI components
        self.interval_slider = pn.widgets.IntSlider.from_param(
            self.param.interval, name="Interval (ms)"
        )

        self.speed_input = pn.widgets.FloatInput.from_param(
            self.param.speed, name="Speed (×)", step=0.1, format="0.00"
        )

        self.real_time_chk = pn.widgets.Checkbox.from_param(
            self.param.real_time, name="Real-time mode"
        )

        # Watch for parameter changes to keep them in sync
        self.param.watch(self._on_interval_change, "interval")
        self.param.watch(self._on_speed_change, "speed")
        self.param.watch(self._on_mode_change, "real_time")

        # Set initial visibility
        self._update_visibility()

        self.view = pn.Row(
            self.t_player,
            pn.Column(
                self.readout,
                self.real_time_chk,
                pn.Row(self.speed_input, self.interval_slider),
            ),
        )

    def _update_readout(self):
        """Update the readout display"""
        self.readout.object = f"t = {self.t_player.value:.3f} s"

    def _sync_from_speed(self):
        """Update interval based on current speed"""
        if self._updating:
            return
        self._updating = True
        new_interval = int((self._dt_sec * 1000) / max(self.speed, 1e-9))
        # Respect the interval bounds
        min_interval, max_interval = self.param.interval.bounds
        self.interval = max(min_interval, min(max_interval, new_interval))
        self._updating = False

    def _sync_from_interval(self):
        """Update speed based on current interval"""
        if self._updating:
            return
        self._updating = True
        new_speed = (self._dt_sec * 1000) / max(self.interval, 1e-9)
        # Respect the speed bounds and round to 1 decimal place
        min_speed, max_speed = self.param.speed.bounds
        self.speed = round(max(min_speed, min(max_speed, new_speed)), 1)
        self._updating = False

    def _update_visibility(self):
        """Show/hide controls based on real_time mode"""
        if self.real_time:
            self.speed_input.visible = True
            self.interval_slider.visible = False
        else:
            self.speed_input.visible = False
            self.interval_slider.visible = True

    def _on_interval_change(self, event=None):
        """When interval changes, sync speed and update player"""
        self._sync_from_interval()
        self.t_player.interval = self.interval

    def _on_speed_change(self, event=None):
        """When speed changes, sync interval and update player"""
        self._sync_from_speed()
        self.t_player.interval = self.interval

    def _on_mode_change(self, event=None):
        """When mode changes, update visibility"""
        self._update_visibility()


def example_1():
    # Example: regular time sampling
    t_values_seconds = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]  # 0.2s steps

    player = PlayerWithRealTime(t_values_seconds, interval_bounds=(50, 2000))
    # show the player in a Panel app
    pn.extension()
    pn.panel(player.view).show(title="Time Player with Real-Time Control")


if __name__ == "__main__":
    # example_1()

    # create chunked dask data array with long time

    import dask.array as da

    # Create a long time dimension
    time = da.linspace(0, 10, 10000, chunks=1000)
    data = da.random.random((10, 10, 100, 10000), chunks=(10, 10, 100, 1000))

    # Create an xarray DataArray
    coords = {
        "time": time,
        "x": ("x", da.linspace(-5, 5, 10)),
        "y": ("y", da.linspace(-5, 5, 10)),
        "z": ("z", da.linspace(-5, 5, 100)),
    }
    array = xr.DataArray(data, coords=coords, dims=["x", "y", "z", "time"])
