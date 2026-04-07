"""Plotting functions for decomposition results (PCA, SpatSpec)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# PCA scree / view plots (formerly core/decomposition/decomposition.py)
# ---------------------------------------------------------------------------


def scree_plot(pca: PCA, cutoff=0.95, ax=None):
    if ax is None:
        ax = plt.gca()

    dim, cumvar = explain_var_dim(pca, cutoff)

    ax.plot(pca.explained_variance_ratio_)
    f = interp1d(np.arange(len(cumvar)), pca.explained_variance_ratio_)
    section = np.linspace(0, dim, 100)
    ax.fill_between(section, f(section), color="orange", alpha=0.5)
    ax.axvline(dim, color="orange")
    ax.set(
        xlabel="PC dim",
        ylabel="expl. var. ratio",
        title="Principal Component Analysis",
        xticks=list(plt.xticks()[0])[1:-1] + [dim],
    )

    ax2 = ax.twinx()
    ax2.plot(cumvar, color="r", alpha=0.8)
    ax2.tick_params(axis="y", labelcolor="r")
    ax2.hlines(
        cutoff, dim, pca.n_components_ + 10, colors="k", linestyles="--", alpha=0.5
    )
    ax2.set(xlim=[0, pca.n_components_])
    ax2.text(
        dim - 1,
        0.8 * cutoff,
        f"{100*cutoff}% explained variance",
        rotation=90,
        fontsize=8,
    )


def explain_var_dim(pca: PCA, cutoff=0.95):
    """Return (dim, cumvar) where dim is the first PC exceeding *cutoff*."""
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    dim = np.argmax((cumvar - cutoff) > 0)
    return dim, cumvar


def plot_view_grid(x, dim, title="", text="PC", half_view=True, **scatter_kwargs):
    fig, ax = plt.subplots(dim, dim)

    text_list = [f"{text}{i+1}" for i in range(dim)] if isinstance(text, str) else text

    for (ix, iy), iax in np.ndenumerate(np.arange(dim * dim).reshape(dim, dim)):
        _ax = ax.flatten()[iax]
        if ix >= iy or not half_view:
            _ax.scatter(x[:, iy], x[:, ix], **scatter_kwargs)
            _ax.set(xticks=[], yticks=[])
            if iy == 0:
                _ax.set_ylabel(f"{text_list[ix]}")
            if ix == dim - 1:
                _ax.set_xlabel(f"{text_list[iy]}")
        else:
            _ax.axis("off")


def plot_view_3D(x, y, z, **scatter3D_kwargs):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.scatter3D(x, y, z, **scatter3D_kwargs)
    ax.set(xlabel="x", ylabel="y", zlabel="z", xticks=[], yticks=[], zticks=[])


# ---------------------------------------------------------------------------
# SpatSpec loading plots (formerly erppca/plot.py)
# ---------------------------------------------------------------------------


def plot_maxfreq_slice_loadings(
    maxfreq_slice, factor_labels, nrow=4, ncol=4, figsize=(12, 12)
):
    imshow_kwargs = dict(cmap="jet")
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
    for ifactor, ax in enumerate(axes.flatten()):
        maxfreq_slice.sel(factor=ifactor).plot(ax=ax, **imshow_kwargs)
        ax.invert_yaxis()
        ax.set_title(f"factor {ifactor} {factor_labels.loc[ifactor].freqmax:.2f}Hz")
    plt.tight_layout()
    return fig, axes


def plot_maxfreq_slc(ss, imshow_kwargs=None):
    if imshow_kwargs is None:
        imshow_kwargs = dict(cmap="jet", vmin=0)
    fig, axes = plt.subplots(12, 5, figsize=(12, 24))
    for ifactor, ax in enumerate(axes.flatten()):
        ss.ldx_slc_maxfreq.sel(factor=ifactor).plot.imshow(
            cmap="jet", vmin=0, ax=ax
        )
        ax.set(
            title=f"fac{ifactor}, {ss.ldx_df.loc[ifactor].freqmax:.0f}Hz",
            xlabel="ML",
            ylabel="AP",
        )
        ax.scatter(ss.ldx_df.loc[ifactor].ML, ss.ldx_df.loc[ifactor].AP, color="b")
    return fig, axes


# ---------------------------------------------------------------------------
# Factor matching plots (formerly in erppca/match.py)
# ---------------------------------------------------------------------------


def plot_matched_facs(ss_match_cc, title, file):
    """Plot matched factors ldx_slc_maxfreq across recordings."""
    nmatch_fac = len(ss_match_cc.factor)
    nrec = len(ss_match_cc.rec)

    fig, ax = plt.subplots(nmatch_fac, nrec, figsize=(25, 5 * nmatch_fac))
    for irec in range(nrec):
        for ifac in range(nmatch_fac):
            axis = ax[ifac, irec]
            ss_match_cc.ldx_slc_maxfreq.sel(factor=ifac)[irec].plot.imshow(
                cmap="jet", vmin=0, ax=axis
            )

    fig.suptitle(title, y=0.999, fontweight="bold")
    plt.tight_layout()
    fig.savefig(file)
