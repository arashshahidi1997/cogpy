import panel as pn
import param


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
