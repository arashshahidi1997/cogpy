"""HoloViews visualizations for decomposition results.

All functions return static HoloViews objects (``Image``, ``Layout``,
``HoloMap``) that render on static sites. No ``DynamicMap`` usage.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from cogpy.utils.imports import import_optional

hv = import_optional("holoviews")


def loading_spatial_layout(
    ldx_slc_maxfreq: xr.DataArray,
    ldx_df,
    *,
    cmap: str = "RdBu_r",
    symmetric: bool = True,
    width: int = 200,
    height: int = 200,
) -> hv.Layout:
    """Grid layout of spatial loading maps at each factor's peak frequency.

    Parameters
    ----------
    ldx_slc_maxfreq : xr.DataArray
        Spatial loadings at peak frequency, shape ``(factor, h, w)``.
    ldx_df : pd.DataFrame
        Loading summary table with ``freqmax``, ``AP``, ``ML`` columns.
    cmap : str
        Colormap name.
    symmetric : bool
        If True, center colormap at zero.
    width, height : int
        Per-panel size in pixels.

    Returns
    -------
    hv.Layout
        Grid of ``hv.Image`` panels, one per factor.
    """
    panels = []
    for ifac in ldx_slc_maxfreq.factor.values:
        slc = ldx_slc_maxfreq.sel(factor=ifac)
        peak_freq = ldx_df.loc[ifac, "freqmax"]
        peak_ap = ldx_df.loc[ifac, "AP"]
        peak_ml = ldx_df.loc[ifac, "ML"]

        clim = (
            (-float(np.abs(slc).max()), float(np.abs(slc).max()))
            if symmetric
            else (None, None)
        )

        img = hv.Image(
            slc,
            kdims=["w", "h"],
            label=f"F{ifac} ({peak_freq:.1f} Hz)",
        ).opts(
            cmap=cmap,
            clim=clim,
            colorbar=True,
            width=width,
            height=height,
            aspect="equal",
            invert_yaxis=True,
            title=f"Factor {ifac} — {peak_freq:.1f} Hz",
            tools=["hover"],
        )

        # Mark peak electrode
        peak_pt = hv.Points(
            [(peak_ml, peak_ap)],
        ).opts(
            color="black",
            marker="x",
            size=12,
            line_width=2,
        )

        panels.append(img * peak_pt)

    return hv.Layout(panels).cols(4)


def loading_spectral_profiles(
    ldx_slc_maxch: xr.DataArray,
    ldx_df,
    *,
    width: int = 250,
    height: int = 180,
) -> hv.Layout:
    """Overlay of spectral profiles at each factor's peak electrode.

    Parameters
    ----------
    ldx_slc_maxch : xr.DataArray
        Loading spectra at peak channel, shape ``(factor, freq)``.
    ldx_df : pd.DataFrame
        Loading summary table with ``freqmax`` column.

    Returns
    -------
    hv.Layout
        Grid of ``hv.Curve`` panels.
    """
    panels = []
    for ifac in ldx_slc_maxch.factor.values:
        profile = ldx_slc_maxch.sel(factor=ifac)
        peak_freq = ldx_df.loc[ifac, "freqmax"]

        curve = hv.Curve(
            (profile.freq.values, profile.values),
            kdims=["Frequency (Hz)"],
            vdims=["Loading"],
        ).opts(width=width, height=height, title=f"Factor {ifac}")

        vline = hv.VLine(peak_freq).opts(
            color="red",
            line_dash="dashed",
            alpha=0.7,
        )

        panels.append(curve * vline)

    return hv.Layout(panels).cols(4)


def score_traces(
    scx: xr.DataArray,
    ldx_df=None,
    *,
    gain: float = 1.0,
    width: int = 800,
    height: int = 500,
) -> hv.Overlay:
    """Stacked factor score traces with adjustable gain.

    Each factor is offset vertically by its index. The ``gain`` parameter
    scales trace amplitudes without changing the gap between traces,
    making it easy to see low-amplitude variation.

    Parameters
    ----------
    scx : xr.DataArray
        Factor scores ``(time, factor)``.
    ldx_df : pd.DataFrame or None
        If given, labels include peak frequency.
    gain : float
        Amplitude multiplier applied to each trace. Increasing gain
        makes small variations visible without changing trace spacing.
    width, height : int
        Plot dimensions in pixels.

    Returns
    -------
    hv.Overlay
        All traces overlaid with vertical offsets and y-axis labels.
    """
    factors = scx.factor.values
    nfac = len(factors)
    t = scx.time.values

    # Normalise each trace to unit std, then apply gain
    curves = []
    yticks = []
    for i, ifac in enumerate(factors):
        trace = scx.sel(factor=ifac).values.copy()
        std = np.nanstd(trace)
        if std > 0:
            trace = trace / std
        offset = nfac - 1 - i  # top-to-bottom stacking
        y = trace * gain + offset

        if ldx_df is not None:
            peak_freq = ldx_df.loc[ifac, "freqmax"]
            label = f"F{ifac} ({peak_freq:.0f} Hz)"
        else:
            label = f"Factor {ifac}"

        yticks.append((offset, label))
        curves.append(
            hv.Curve(
                (t, y),
                kdims=["Time (s)"],
                vdims=["Score"],
                label=label,
            )
        )

    overlay = hv.Overlay(curves).opts(
        width=width,
        height=height,
        yticks=yticks,
        ylabel="",
        title=f"Factor scores (gain={gain:.1f})",
        show_legend=False,
    )
    return overlay


def score_traces_holomap(
    scx: xr.DataArray,
    ldx_df=None,
    *,
    gains: tuple[float, ...] = (0.3, 0.5, 1.0, 2.0, 4.0),
    width: int = 800,
    height: int = 500,
) -> hv.HoloMap:
    """HoloMap of stacked score traces at different gain levels.

    Each frame is a ``score_traces`` overlay at a given gain. Use the
    widget slider to adjust gain interactively on a static site.

    Parameters
    ----------
    scx : xr.DataArray
        Factor scores ``(time, factor)``.
    ldx_df : pd.DataFrame or None
        If given, labels include peak frequency.
    gains : tuple of float
        Gain levels to include in the HoloMap.

    Returns
    -------
    hv.HoloMap
        Keyed by gain value.
    """
    frames = {}
    for g in gains:
        frames[g] = score_traces(
            scx,
            ldx_df,
            gain=g,
            width=width,
            height=height,
        )
    return hv.HoloMap(frames, kdims=["Gain"])


def factor_holomap(
    ldx: xr.DataArray,
    ldx_df,
    *,
    cmap: str = "RdBu_r",
    symmetric: bool = True,
    width: int = 350,
    height: int = 300,
) -> hv.HoloMap:
    """HoloMap of spatial loadings across frequency, keyed by factor.

    Suitable for static rendering (no server needed). Each frame shows
    the spatial loading at the peak frequency for one factor.

    Parameters
    ----------
    ldx : xr.DataArray
        Full loadings ``(factor, h, w, freq)``.
    ldx_df : pd.DataFrame
        Loading summary with ``freqmax``, ``AP``, ``ML``.

    Returns
    -------
    hv.HoloMap
        Keyed by factor index.
    """
    frames = {}
    for ifac in ldx.factor.values:
        peak_freq = ldx_df.loc[ifac, "freqmax"]
        slc = ldx.sel(factor=ifac, freq=peak_freq, method="nearest")

        clim = (
            (-float(np.abs(slc).max()), float(np.abs(slc).max()))
            if symmetric
            else (None, None)
        )

        img = hv.Image(
            slc,
            kdims=["w", "h"],
        ).opts(
            cmap=cmap,
            clim=clim,
            colorbar=True,
            width=width,
            height=height,
            aspect="equal",
            invert_yaxis=True,
            title=f"Factor {ifac} — {peak_freq:.1f} Hz",
            tools=["hover"],
        )

        peak_ap = ldx_df.loc[ifac, "AP"]
        peak_ml = ldx_df.loc[ifac, "ML"]
        peak_pt = hv.Points(
            [(peak_ml, peak_ap)],
        ).opts(color="black", marker="x", size=12, line_width=2)

        frames[ifac] = img * peak_pt

    return hv.HoloMap(frames, kdims=["Factor"])
