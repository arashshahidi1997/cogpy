"""Event-triggered average layer (v2.6.3)."""

from __future__ import annotations

import numpy as np

__all__ = ["EventTriggeredAverageLayer"]


class EventTriggeredAverageLayer:
    """
    Event-triggered average (ETA) around event peak times.

    This is a lightweight HoloViews view constructor intended for TensorScope
    exploration workflows (v2.6.3).
    """

    def __init__(
        self,
        state,
        stream_name: str,
        *,
        pre: float = 0.2,
        post: float = 0.2,
        max_events: int = 200,
        use_visible_window: bool = False,
    ) -> None:
        self.state = state
        self.stream_name = str(stream_name)
        self.pre = float(pre)
        self.post = float(post)
        self.max_events = int(max_events)
        self.use_visible_window = bool(use_visible_window)

    def _spatial_stream(self):
        import holoviews as hv

        hv.extension("bokeh")
        space = getattr(self.state, "spatial_space", None)
        if space is None or not hasattr(space, "create_stream"):
            return hv.streams.Stream.define("SpatialSel", AP=0.0, ML=0.0)()
        try:
            return space.create_stream()
        except Exception:  # noqa: BLE001
            return hv.streams.Stream.define("SpatialSel", AP=0.0, ML=0.0)()

    def create_view(self):
        import holoviews as hv

        hv.extension("bokeh")

        time_window_stream = None
        if self.use_visible_window and getattr(self.state, "time_window", None) is not None:
            time_window_stream = hv.streams.Params(self.state.time_window, parameters=["window"])

        spatial_stream = self._spatial_stream()

        def _render(window=None, **spatial_sel):
            stream = self.state.event_registry.get(self.stream_name) if self.state.event_registry else None
            signal = self.state.signal_registry.get_active() if self.state.signal_registry else None
            if stream is None or signal is None or len(stream.df) == 0:
                return hv.Curve([], kdims=["time_rel"], vdims=["amplitude"]).opts(
                    title="Event-triggered average (no events)",
                    xlabel="Time relative to event (s)",
                    ylabel="Amplitude",
                )

            df = stream.df
            if self.use_visible_window and window is not None:
                t0, t1 = window
                df = stream.get_events_in_window(float(t0), float(t1))

            # Optional spatial filtering if event table provides AP/ML.
            if ("AP" in df.columns) and ("ML" in df.columns):
                ap = spatial_sel.get("AP", None)
                ml = spatial_sel.get("ML", None)
                if ap is not None:
                    df = df[df["AP"] == int(round(float(ap)))]
                if ml is not None:
                    df = df[df["ML"] == int(round(float(ml)))]

            if df.empty:
                return hv.Curve([], kdims=["time_rel"], vdims=["amplitude"]).opts(
                    title="Event-triggered average (no events after filtering)",
                    xlabel="Time relative to event (s)",
                    ylabel="Amplitude",
                )

            # Select a single trace at the current AP/ML selection if possible.
            data = signal.data
            if ("AP" in data.dims) and ("ML" in data.dims):
                n_ap = int(data.sizes.get("AP", 0))
                n_ml = int(data.sizes.get("ML", 0))
                ap_i = int(np.clip(int(round(float(spatial_sel.get("AP", 0.0)))), 0, max(n_ap - 1, 0)))
                ml_i = int(np.clip(int(round(float(spatial_sel.get("ML", 0.0)))), 0, max(n_ml - 1, 0)))
                trace = data.isel(AP=ap_i, ML=ml_i)
            else:
                # Fallback: reduce over non-time dims.
                reduce_dims = [d for d in data.dims if d != "time"]
                trace = data.mean(dim=reduce_dims) if reduce_dims else data

            t_vals = np.asarray(trace["time"].values, dtype=float)
            if t_vals.size < 2:
                return hv.Curve([], kdims=["time_rel"], vdims=["amplitude"]).opts(
                    title="Event-triggered average (invalid time axis)",
                    xlabel="Time relative to event (s)",
                    ylabel="Amplitude",
                )

            dt = float(np.median(np.diff(t_vals)))
            if not np.isfinite(dt) or dt <= 0:
                dt = float(np.diff(t_vals[:2])[0]) if t_vals.size >= 2 else 1e-3
            n = int(round((self.pre + self.post) / dt)) + 1
            rel_t = np.linspace(-self.pre, self.post, n, dtype=float)

            t_min = float(t_vals[0])
            t_max = float(t_vals[-1])

            event_times = np.asarray(df[stream.time_col].to_numpy(), dtype=float)
            event_times = event_times[np.isfinite(event_times)]
            if event_times.size == 0:
                return hv.Curve([], kdims=["time_rel"], vdims=["amplitude"]).opts(
                    title="Event-triggered average (no finite event times)",
                    xlabel="Time relative to event (s)",
                    ylabel="Amplitude",
                )

            if event_times.size > self.max_events:
                event_times = np.random.RandomState(0).choice(event_times, size=self.max_events, replace=False)

            y = np.asarray(trace.values, dtype=float)
            y = y.reshape(-1)
            if y.size != t_vals.size:
                # Unexpected shape: force 1D.
                y = np.asarray(np.ravel(y), dtype=float)
                if y.size != t_vals.size:
                    return hv.Curve([], kdims=["time_rel"], vdims=["amplitude"]).opts(
                        title="Event-triggered average (non-1D trace)",
                        xlabel="Time relative to event (s)",
                        ylabel="Amplitude",
                    )

            windows = []
            for t0 in event_times:
                if (t0 - self.pre) < t_min or (t0 + self.post) > t_max:
                    continue
                xq = t0 + rel_t
                windows.append(np.interp(xq, t_vals, y))

            if not windows:
                return hv.Curve([], kdims=["time_rel"], vdims=["amplitude"]).opts(
                    title="Event-triggered average (no in-bounds windows)",
                    xlabel="Time relative to event (s)",
                    ylabel="Amplitude",
                )

            w = np.asarray(windows, dtype=float)
            mu = np.nanmean(w, axis=0)
            sd = np.nanstd(w, axis=0)

            curve = hv.Curve((rel_t, mu), kdims=["time_rel"], vdims=["amplitude"]).opts(
                width=600,
                height=260,
                color="#2a6fdb",
                line_width=2,
                tools=["hover"],
                xlabel="Time relative to event (s)",
                ylabel="Amplitude",
                title=f"Event-triggered average (n={w.shape[0]})",
            )
            band = hv.Area((rel_t, mu - sd, mu + sd), kdims=["time_rel"], vdims=["lower", "upper"]).opts(
                color="#2a6fdb",
                alpha=0.2,
                line_width=0,
            )
            v0 = hv.VLine(0.0).opts(color="#ff0000", alpha=0.7, line_dash="dashed", line_width=2)
            return band * curve * v0

        streams = [spatial_stream]
        if time_window_stream is not None:
            streams.insert(0, time_window_stream)
        return hv.DynamicMap(_render, streams=streams)

