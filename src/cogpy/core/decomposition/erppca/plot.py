import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


def plot_maxfreq_slice_loadings(
    maxfreq_slice, factor_labels, nrow=4, ncol=4, figsize=(12, 12)
):
    # imshow_kwargs = dict(vmin=np.min(maxfreq_slice), vmax=np.max(maxfreq_slice), cmap='jet')
    imshow_kwargs = dict(cmap="jet")
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
    for ifactor, ax in enumerate(axes.flatten()):
        maxfreq_slice.sel(factor=ifactor).plot(ax=ax, **imshow_kwargs)
        ax.invert_yaxis()
        ax.set_title(f"factor {ifactor} {factor_labels.loc[ifactor].freqmax:.2f}Hz")
    plt.tight_layout()
    return fig, axes


def plot_maxfreq_slc(ss, imshow_kwargs=None):
    # imshow_kwargs = dict(vmin=np.min(maxfreq_slice), vmax=np.max(maxfreq_slice), cmap='jet')
    if imshow_kwargs is None:
        imshow_kwargs = dict(cmap="jet", vmin=0)
    fig, axes = plt.subplots(12, 5, figsize=(12, 24))
    for ifactor, ax in enumerate(axes.flatten()):
        ss.ldx_slc_maxfreq.sel(factor=ifactor).plot(ax=ax, **imshow_kwargs)
        ax.set(
            title=f"fac{ifactor}, {ss.ldx_df.loc[ifactor].freqmax:.0f}Hz",
            xlabel="ML",
            ylabel="AP",
        )
        ax.scatter(ss.ldx_df.loc[ifactor].ML, ss.ldx_df.loc[ifactor].AP, color="b")
    return fig, axes
