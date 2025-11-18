import numpy as np
import xarray as xr


def make_dataset():
    t_vals = np.linspace(0, 10, 50)
    f_vals = np.linspace(0, 5, 25)
    ml_vals = np.linspace(0, 1, 10)
    ap_vals = np.linspace(0, 1, 10)

    arr_spectrogram = 100 * np.outer(np.cos(t_vals), np.sin(f_vals))  # (time, freq)
    arr_specmap = 100 * np.outer(np.sin(ml_vals), np.cos(ap_vals))  # (ml, ap)

    arr = (
        arr_specmap[:, :, np.newaxis, np.newaxis]
        * arr_spectrogram[np.newaxis, np.newaxis, :, :]
    )
    da = xr.DataArray(
        arr,
        dims=("ml", "ap", "time", "freq"),
        coords={"ml": ml_vals, "ap": ap_vals, "time": t_vals, "freq": f_vals},
        name="val",
    )
    return da
