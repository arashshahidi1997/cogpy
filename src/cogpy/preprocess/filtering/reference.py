"""Common reference removal for multichannel xarray signals."""

import numpy as np
import xarray as xr


def cmrx(
    sigx: xr.DataArray,
    *,
    channel_dims: tuple[str, ...] | None = None,
    skipna: bool = True,
) -> xr.DataArray:
    """Common median reference: subtract median across channels at each time sample.

    For grid input (AP+ML dims present): subtracts median over (AP, ML).
    For stacked input (channel dim present): subtracts median over channel.
    Pass channel_dims explicitly to override auto-detection.

    This matches NeuroScope2's "median filter" preprocessing:
        ephys.traces = ephys.traces - median(ephys.traces, 2)

    See Also
    --------
    cogpy.preprocess.filtering.normalization.zscorex : Z-score normalization.
    cogpy.preprocess.filtering.spatial.gaussian_spatialx : Spatial Gaussian lowpass.

    Examples
    --------
    >>> import cogpy
    >>> sigx = cogpy.datasets.load_sample()  # (AP, ML, time)
    >>> ref = cmrx(sigx)  # subtracts median over (AP, ML) at each time point
    >>> ref.shape == sigx.shape
    True
    """
    if channel_dims is None:
        if "AP" in sigx.dims and "ML" in sigx.dims:
            channel_dims = ("AP", "ML")
        elif "channel" in sigx.dims:
            channel_dims = ("channel",)
        else:
            raise ValueError(
                "cmrx could not auto-detect channel dims. Expected ('AP','ML') grid dims or a 'channel' dim; "
                f"got sigx.dims={tuple(sigx.dims)}. Pass channel_dims=... explicitly."
            )

    channel_dims = tuple(channel_dims)
    for d in channel_dims:
        if d not in sigx.dims:
            raise ValueError(
                f"cmrx expected channel dim {d!r} in sigx.dims={tuple(sigx.dims)}"
            )

    axis = tuple(range(-len(channel_dims), 0))
    if bool(skipna):
        med = xr.apply_ufunc(
            lambda x: np.nanmedian(x, axis=axis),
            sigx,
            input_core_dims=[list(channel_dims)],
            output_core_dims=[[]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[np.result_type(sigx.dtype, np.float64)],
        )
    else:
        med = xr.apply_ufunc(
            lambda x: np.median(x, axis=axis),
            sigx,
            input_core_dims=[list(channel_dims)],
            output_core_dims=[[]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[np.result_type(sigx.dtype, np.float64)],
        )

    out = sigx - med
    out.attrs = dict(sigx.attrs)
    out.attrs.update(
        {"filter_type": "common_median_reference", "channel_dims": channel_dims}
    )
    out.name = sigx.name
    return out
