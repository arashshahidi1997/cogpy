"""
TensorScope v3.0 Minimal Example.

Demonstrates:
- Tensor registry
- Tensor tabs (Signal/PSD)
- Linked selection
- View discovery

Run:
    panel serve v3_minimal_app.py --show
"""

import numpy as np
import panel as pn
import xarray as xr

from cogpy.core.plot.tensorscope.app import TensorScopeApp

pn.extension()


def _make_signal(*, fs: int = 1000, duration: float = 10.0, n_ap: int = 8, n_ml: int = 8) -> xr.DataArray:
    rng = np.random.default_rng(0)
    n_samples = int(fs * duration)
    time = np.linspace(0, duration, n_samples)
    ap_coords = np.arange(n_ap)
    ml_coords = np.arange(n_ml)

    signal_data = np.zeros((n_samples, n_ap, n_ml), dtype=float)
    for i, ap in enumerate(ap_coords):
        for j, ml in enumerate(ml_coords):
            phase = rng.uniform(0, 2 * np.pi)
            signal_data[:, i, j] = (
                np.sin(2 * np.pi * 10 * time + phase)
                + 0.5 * np.sin(2 * np.pi * 40 * time + phase)
                + 0.3 * np.sin(2 * np.pi * 60 * time + phase)
                + 0.2 * rng.standard_normal(n_samples)
            )
            signal_data[:, i, j] *= 1.0 + 0.1 * float(ap) + 0.1 * float(ml)

    da = xr.DataArray(
        signal_data,
        dims=("time", "AP", "ML"),
        coords={"time": time, "AP": ap_coords, "ML": ml_coords},
        name="signal",
        attrs={"fs": float(fs)},
    )
    return da


signal = _make_signal()

app = TensorScopeApp()
app.add_tensor("signal", signal, transform="signal")
app.add_psd_tensor(name="psd", source="signal", window=1.0, nperseg=256, method="welch")

app.state.selection.time = 5.0
app.state.selection.freq = 40.0
app.state.selection.ap = 4
app.state.selection.ml = 4

template = app.build()
template.servable()

