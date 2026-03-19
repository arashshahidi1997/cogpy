"""
Orthoslicer module: time-frequency-spatial exploration (v2.4).

Computes a multitaper spectrogram using `spectrogramx()` and provides
four linked views via CoordinateSpace streams:
- Time profile (time @ freq + spatial)
- Freq profile (freq @ time + spatial)
- TF heatmap (time×freq @ spatial)
- Spatial map (AP×ML @ time + freq)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..transforms.base import CoordinateSpace
from .base import ViewPresetModule

__all__ = ["MODULE", "create_orthoslicer_module"]


@dataclass(frozen=True)
class _OrthoParams:
    nperseg: int = 256
    noverlap: int = 128
    bandwidth: float = 4.0
    max_seconds: float = 5.0


def _compute_spectrogram(sig_data, params: _OrthoParams):
    from cogpy.spectral.specx import spectrogramx

    if "time" not in sig_data.dims:
        raise ValueError("Signal data must have a 'time' dimension")
    if not (("AP" in sig_data.dims) and ("ML" in sig_data.dims)):
        raise ValueError("Orthoslicer requires grid data with dims (time, AP, ML)")

    t_vals = np.asarray(sig_data["time"].values, dtype=float)
    t0 = float(t_vals[0])
    t1 = float(t_vals[-1])
    if params.max_seconds and (t1 - t0) > float(params.max_seconds):
        t1 = t0 + float(params.max_seconds)

    win = sig_data.sel(time=slice(t0, t1))
    return spectrogramx(
        win,
        axis="time",
        bandwidth=float(params.bandwidth),
        nperseg=int(params.nperseg),
        noverlap=int(params.noverlap),
    )


def create_orthoslicer_module(
    *,
    nperseg: int = 256,
    noverlap: int = 128,
    bandwidth: float = 4.0,
    max_seconds: float = 5.0,
) -> ViewPresetModule:
    params = _OrthoParams(
        nperseg=int(nperseg),
        noverlap=int(noverlap),
        bandwidth=float(bandwidth),
        max_seconds=float(max_seconds),
    )

    def _activate(state):
        import holoviews as hv

        hv.extension("bokeh")

        reg = getattr(state, "signal_registry", None)
        sig = reg.get_active() if reg is not None else None
        if sig is None:
            return hv.Div("<b>No active signal</b>")

        try:
            spec = _compute_spectrogram(sig.data, params)
        except ImportError as exc:
            return hv.Div(f"<b>spectrogramx unavailable</b><br>{exc}")
        except Exception as exc:  # noqa: BLE001
            return hv.Div(f"<b>Error computing spectrogram</b><br>{exc}")

        if not (("AP" in spec.dims) and ("ML" in spec.dims) and ("freq" in spec.dims) and ("time" in spec.dims)):
            return hv.Div(f"<b>Unexpected spectrogram dims</b><br>{tuple(spec.dims)!r}")

        spatial_space = getattr(state, "spatial_space", None)
        if spatial_space is None or not (spatial_space.has_dim("AP") and spatial_space.has_dim("ML")):
            return hv.Div("<b>State has no spatial_space with dims AP/ML</b>")

        temporal_space = CoordinateSpace("temporal", dims={"time"})
        spectral_space = CoordinateSpace("spectral", dims={"freq"})

        spatial_stream = spatial_space.create_stream()
        temporal_stream = temporal_space.create_stream()
        spectral_stream = spectral_space.create_stream()

        n_ap = int(spec.sizes["AP"])
        n_ml = int(spec.sizes["ML"])

        ap0 = spatial_space.get_selection("AP")
        ml0 = spatial_space.get_selection("ML")
        if ap0 is None:
            ap0 = n_ap // 2 if n_ap > 0 else 0
        if ml0 is None:
            ml0 = n_ml // 2 if n_ml > 0 else 0

        time0 = float(np.asarray(spec["time"].values, dtype=float)[len(spec["time"]) // 2])
        freq0 = float(np.asarray(spec["freq"].values, dtype=float)[len(spec["freq"]) // 2])

        spatial_space.set_selection("AP", int(ap0))
        spatial_space.set_selection("ML", int(ml0))
        temporal_space.set_selection("time", float(time0))
        spectral_space.set_selection("freq", float(freq0))

        tf_tap = hv.streams.Tap(x=None, y=None)
        spatial_tap = hv.streams.Tap(x=None, y=None)

        def _tf_view(AP=0, ML=0, x=None, y=None):
            ap_i = int(np.clip(int(AP), 0, max(n_ap - 1, 0)))
            ml_i = int(np.clip(int(ML), 0, max(n_ml - 1, 0)))

            if (x is not None) and (y is not None):
                x_f = float(x)
                y_f = float(y)
                if temporal_space.get_selection("time") != x_f:
                    temporal_space.set_selection("time", x_f)
                if spectral_space.get_selection("freq") != y_f:
                    spectral_space.set_selection("freq", y_f)

            tf = spec.isel(AP=ap_i, ML=ml_i).transpose("freq", "time")
            time_vals = np.asarray(tf["time"].values, dtype=float)
            freq_vals = np.asarray(tf["freq"].values, dtype=float)
            vals = np.asarray(tf.values, dtype=float)

            return hv.Image(
                (time_vals, freq_vals, vals),
                kdims=["time", "freq"],
                vdims=["Power (a.u.)"],
            ).opts(
                cmap="viridis",
                colorbar=True,
                width=520,
                height=320,
                xlabel="Time (s)",
                ylabel="Freq (Hz)",
                tools=["hover", "tap"],
                title=f"TF @ (AP={ap_i}, ML={ml_i})",
            )

        def _spatial_view(time=None, freq=None, x=None, y=None):
            if (x is not None) and (y is not None):
                ml_i = int(np.clip(int(np.round(float(x))), 0, max(n_ml - 1, 0)))
                ap_i = int(np.clip(int(np.round(float(y))), 0, max(n_ap - 1, 0)))
                if spatial_space.get_selection("ML") != ml_i:
                    spatial_space.set_selection("ML", ml_i)
                if spatial_space.get_selection("AP") != ap_i:
                    spatial_space.set_selection("AP", ap_i)

            t = float(time) if time is not None else float(time0)
            f = float(freq) if freq is not None else float(freq0)

            frame = spec.sel(time=t, freq=f, method="nearest").transpose("AP", "ML")
            vals = np.asarray(frame.values, dtype=float)

            ap_idx = np.arange(n_ap)
            ml_idx = np.arange(n_ml)
            img = hv.Image(
                (ml_idx, ap_idx, vals),
                kdims=["ML", "AP"],
                vdims=["Power (a.u.)"],
            ).opts(
                cmap="viridis",
                colorbar=True,
                width=360,
                height=360,
                xlabel="ML (index)",
                ylabel="AP (index)",
                tools=["hover", "tap"],
                aspect="equal",
                title="Spatial @ (time,freq)",
            )

            ap_sel = spatial_space.get_selection("AP")
            ml_sel = spatial_space.get_selection("ML")
            if (ap_sel is not None) and (ml_sel is not None):
                marker = hv.Points([(float(ml_sel), float(ap_sel))], kdims=["ML", "AP"]).opts(
                    color="yellow",
                    marker="x",
                    size=15,
                    line_width=3,
                )
                return img * marker
            return img

        def _time_profile(freq=None, AP=0, ML=0):
            ap_i = int(np.clip(int(AP), 0, max(n_ap - 1, 0)))
            ml_i = int(np.clip(int(ML), 0, max(n_ml - 1, 0)))
            f = float(freq) if freq is not None else float(freq0)

            trace = spec.isel(AP=ap_i, ML=ml_i).sel(freq=f, method="nearest")
            t_vals = np.asarray(trace["time"].values, dtype=float)
            y_vals = np.asarray(trace.values, dtype=float)
            return hv.Curve((t_vals, y_vals), kdims=["time"], vdims=["Power (a.u.)"]).opts(
                width=520,
                height=220,
                tools=["hover"],
                xlabel="Time (s)",
                ylabel="Power",
                title=f"Time @ f≈{float(trace['freq'].values):.1f}Hz",
            )

        def _freq_profile(time=None, AP=0, ML=0):
            ap_i = int(np.clip(int(AP), 0, max(n_ap - 1, 0)))
            ml_i = int(np.clip(int(ML), 0, max(n_ml - 1, 0)))
            t = float(time) if time is not None else float(time0)

            trace = spec.isel(AP=ap_i, ML=ml_i).sel(time=t, method="nearest")
            f_vals = np.asarray(trace["freq"].values, dtype=float)
            y_vals = np.asarray(trace.values, dtype=float)
            return hv.Curve((f_vals, y_vals), kdims=["freq"], vdims=["Power (a.u.)"]).opts(
                width=520,
                height=220,
                tools=["hover"],
                xlabel="Freq (Hz)",
                ylabel="Power",
                title=f"Freq @ t≈{float(trace['time'].values):.2f}s",
            )

        time_view = hv.DynamicMap(_time_profile, streams=[spectral_stream, spatial_stream])
        freq_view = hv.DynamicMap(_freq_profile, streams=[temporal_stream, spatial_stream])
        tf_view = hv.DynamicMap(_tf_view, streams=[spatial_stream, tf_tap])
        spatial_view = hv.DynamicMap(_spatial_view, streams=[temporal_stream, spectral_stream, spatial_tap])

        top = (time_view + freq_view).cols(2)
        bottom = (tf_view + spatial_view).cols(2)
        return (top + bottom).cols(1)

    return ViewPresetModule(
        name="orthoslicer",
        description="Spectrogram orthoslicer (time×freq×AP×ML) with linked streams",
        specs=[],
        layout="custom",
        activate_fn=_activate,
    )


MODULE = create_orthoslicer_module()

