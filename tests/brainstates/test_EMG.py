import pandas as pd
import xarray as xr
from cogpy.brainstates.EMG import synthetic_data, compute_emg_proxy

def test_compute_emg_proxy():
    x_data, coords_df, fs = synthetic_data()
    min_distance = 400.0  # µm
    emg_proxy_df, corr_da = compute_emg_proxy(
        x_data, fs, coords_df, min_distance, window_size=0.5, window_step=0.25, return_corr_da=True
    )
    assert isinstance(emg_proxy_df, pd.DataFrame), "emg_proxy_df should be a DataFrame"
    assert 'time' in emg_proxy_df.columns, "emg_proxy_df should contain 'time' column"
    assert 'emg_proxy' in emg_proxy_df.columns, "emg_proxy_df should contain 'emg_proxy' column"
    assert 'emg_proxy_std' in emg_proxy_df.columns, "emg_proxy_df should contain 'emg_proxy_std' column"
    assert isinstance(corr_da, xr.DataArray), "corr_da should be an xarray DataArray"
    assert corr_da.shape == (39, 32, 32)
    assert corr_da.dims == ('time', 'ch1', 'ch2'), "corr_da should have dimensions (time, ch1, ch2)"
    print("All tests passed!")