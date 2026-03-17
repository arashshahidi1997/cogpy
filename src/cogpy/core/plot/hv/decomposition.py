"""HoloViews visualizations for decomposition results.

All functions return static HoloViews objects (``Image``, ``Layout``,
``HoloMap``) that render on static sites. No ``DynamicMap`` usage.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

try:
    import holoviews as hv
except ImportError as e:
    raise ImportError(
        "holoviews is required for cogpy.core.plot.hv.decomposition. "
        "Install with: pip install holoviews"
    ) from e


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

        clim = (-float(np.abs(slc).max()), float(np.abs(slc).max())) if symmetric else (None, None)

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
            color="red", line_dash="dashed", alpha=0.7,
        )

        panels.append(curve * vline)

    return hv.Layout(panels).cols(4)


def score_traces(
    scx: xr.DataArray,
    ldx_df=None,
    *,
    width: int = 800,
    height_per_factor: int = 80,
) -> hv.Layout:
    """Stacked factor score time series.

    Parameters
    ----------
    scx : xr.DataArray
        Factor scores ``(time, factor)``.
    ldx_df : pd.DataFrame or None
        If given, labels include peak frequency.

    Returns
    -------
    hv.Layout
        Vertically stacked ``hv.Curve`` panels.
    """
    panels = []
    for ifac in scx.factor.values:
        trace = scx.sel(factor=ifac)
        if ldx_df is not None:
            peak_freq = ldx_df.loc[ifac, "freqmax"]
            label = f"F{ifac} ({peak_freq:.0f} Hz)"
        else:
            label = f"Factor {ifac}"

        curve = hv.Curve(
            (trace.time.values, trace.values),
            kdims=["Time (s)"],
            vdims=["Score"],
            label=label,
        ).opts(
            width=width,
            height=height_per_factor,
            title=label,
        )
        panels.append(curve)

    return hv.Layout(panels).cols(1)


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

        clim = (-float(np.abs(slc).max()), float(np.abs(slc).max())) if symmetric else (None, None)

        img = hv.Image(
            slc,
            kdims=["w", "h"],
        ).opts(
            cmap=cmap,
            clim=clim,
            colorbar=True,
            width=width,
            height=height,
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
