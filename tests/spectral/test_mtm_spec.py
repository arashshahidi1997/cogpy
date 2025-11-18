# import pytest
# import numpy as np
# import xarray as xr
# from pathlib import Path
# import shutil
# import ghostipy as gsp
# from cogpy.spectral.mtm_spec import specgram_coords, mtm_spectrogram_dask

# script_dir = Path(__file__).resolve().parent
# temp_dir = Path(script_dir/'temp')
# temp_dir.mkdir(parents=True, exist_ok=True)

# def test_specgram_coords():
#     fs = 1000
#     nperseg = 512
#     noverlap = 256
#     bandwidth = 2
#     # Test that specgram_coords matches ghostipy's mtm_spectrogram output
#     x = np.random.randn(1_000_000)
#     S_dummy, f_gsp, t_gsp = gsp.mtm_spectrogram(x, fs=fs, nperseg=nperseg, noverlap=noverlap, bandwidth=bandwidth)

#     step = nperseg - noverlap
#     f_calc, t_calc, M, nwin  = specgram_coords(fs, nperseg, noverlap, len(x))

#     assert np.allclose(f_calc, f_gsp), "Frequency arrays do not match!"
#     assert np.allclose(t_calc, t_gsp), "Time arrays do not match!"
#     assert M == len(f_gsp), "Number of frequency bins does not match!"
#     assert nwin == len(t_gsp), "Number of time bins does not match!"
#     print("specgram_coords test passed!")


# def test_mtm_spectrogram_dask():
#     zarr_path = temp_dir / "mtm_spectrogram.zarr"
#     if zarr_path.exists():
#         shutil.rmtree(zarr_path)
#     x = xr.DataArray(np.random.randn(4, 100_000), dims=("channel", "time"))
#     mtm_kwargs = {"fs": 1000, "nperseg": 256, "noverlap": 128, "bandwidth": 4}
#     S_xr = mtm_spectrogram_dask(x, **mtm_kwargs)
#     S_xr.to_zarr(zarr_path, consolidated=True, zarr_format=2)

#     # compute one channel
#     S_channel, f, t = gsp.mtm_spectrogram(x[0].values, **mtm_kwargs)

#     # load from zarr
#     S_xr_loaded = xr.open_dataarray(zarr_path, engine="zarr", chunks="auto")

#     # # Check that the loaded data matches the original
#     assert np.allclose(S_xr_loaded[0].load().data, S_channel), "mtm_spectrogram_dask loaded data does not match direct ghostipy output!"

#     # delete the zarr file
#     shutil.rmtree(zarr_path)
