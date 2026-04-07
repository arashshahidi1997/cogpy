"""Static-compatible HoloViews plot primitives for signal analysis.

All functions return static HoloViews elements (Curve, Image, Overlay,
Layout, HoloMap) — never DynamicMap — so they work in both live notebooks
and static HTML export.
"""

from __future__ import annotations


import numpy as np
import xarray as xr


def psd_curve(
    psd: np.ndarray,
    freqs: np.ndarray,
    *,
    label: str = "",
    logx: bool = True,
    logy: bool = True,
    **curve_opts,
):
    """
    PSD as a HoloViews Curve.

    Parameters
    ----------
    psd : (n_freq,) float
        Power spectral density.
    freqs : (n_freq,) float
        Frequency axis (Hz).
    label : str
        Curve label (for legend).
    logx, logy : bool
        Log scale on x/y axes.

    Returns
    -------
    hv.Curve
    """
    import holoviews as hv

    psd = np.asarray(psd, dtype=float)
    freqs = np.asarray(freqs, dtype=float)
    c = hv.Curve((freqs, psd), kdims=["Frequency (Hz)"], vdims=["Power"], label=label)
    opts = dict(logx=logx, logy=logy, width=600, height=300)
    opts.update(curve_opts)
    return c.opts(**opts)


def psd_overlay(
    psds: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    logx: bool = True,
    logy: bool = True,
    title: str = "",
    **overlay_opts,
):
    """
    Overlay multiple PSDs for comparison (e.g. before/after).

    Parameters
    ----------
    psds : dict[str, (psd, freqs)]
        Mapping of ``{label: (psd_array, freq_array)}``.
    title : str
        Plot title.

    Returns
    -------
    hv.Overlay
    """
    import holoviews as hv

    curves = []
    for label, (psd, freqs) in psds.items():
        curves.append(psd_curve(psd, freqs, label=label, logx=logx, logy=logy))
    ov = hv.Overlay(curves)
    opts = dict(legend_position="top_right", title=title, width=600, height=300)
    opts.update(overlay_opts)
    return ov.opts(**opts)


def psd_with_lines(
    psd: np.ndarray,
    freqs: np.ndarray,
    line_freqs: np.ndarray,
    *,
    label: str = "PSD",
    line_label: str = "detected line",
    logx: bool = True,
    logy: bool = True,
    title: str = "",
):
    """
    PSD curve with vertical lines marking detected spectral peaks.

    Parameters
    ----------
    psd : (n_freq,) float
    freqs : (n_freq,) float
    line_freqs : (K,) float
        Frequencies to mark with vertical lines.

    Returns
    -------
    hv.Overlay
    """
    import holoviews as hv

    c = psd_curve(psd, freqs, label=label, logx=logx, logy=logy)
    vlines = hv.Overlay(
        [
            hv.VLine(f).opts(color="red", line_dash="dashed", alpha=0.6, line_width=1)
            for f in np.asarray(line_freqs).ravel()
        ]
    )
    return (c * vlines).opts(title=title, width=600, height=300)


def spatial_heatmap(
    data: xr.DataArray | np.ndarray,
    *,
    ap_dim: str = "AP",
    ml_dim: str = "ML",
    title: str = "",
    cmap: str = "viridis",
    symmetric: bool = False,
    colorbar: bool = True,
    width: int = 350,
    height: int = 300,
    **image_opts,
):
    """
    Static heatmap of a 2D (AP, ML) scalar field.

    Parameters
    ----------
    data : xr.DataArray with dims (AP, ML) or (n_ap, n_ml) ndarray
        Scalar values per electrode.
    title : str
        Plot title.
    symmetric : bool
        If True, center colormap at zero (good for signed quantities).

    Returns
    -------
    hv.Image
    """
    import holoviews as hv

    if isinstance(data, xr.DataArray):
        arr = data.values
    else:
        arr = np.asarray(data, dtype=float)

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")

    n_ap, n_ml = arr.shape
    img = hv.Image(
        (np.arange(n_ml), np.arange(n_ap), arr),
        kdims=[ml_dim, ap_dim],
        vdims=["value"],
    )
    opts = dict(
        cmap=cmap,
        colorbar=colorbar,
        width=width,
        height=height,
        title=title,
        invert_yaxis=True,
    )
    if symmetric:
        vmax = float(np.nanmax(np.abs(arr)))
        opts["clim"] = (-vmax, vmax)
        if cmap == "viridis":
            opts["cmap"] = "RdBu_r"
    opts.update(image_opts)
    return img.opts(**opts)


def spatial_heatmap_grid(
    maps: dict[str, np.ndarray | xr.DataArray],
    *,
    ap_dim: str = "AP",
    ml_dim: str = "ML",
    cmap: str = "viridis",
    symmetric: bool = False,
    width: int = 250,
    height: int = 220,
    cols: int = 4,
):
    """
    Grid layout of multiple spatial heatmaps (e.g. per-band power, per-metric).

    Parameters
    ----------
    maps : dict[str, (AP, ML) array]
        Mapping of ``{title: 2d_array}``.
    cols : int
        Number of columns in the layout grid.

    Returns
    -------
    hv.Layout
    """
    import holoviews as hv

    plots = []
    for title, data in maps.items():
        plots.append(
            spatial_heatmap(
                data,
                ap_dim=ap_dim,
                ml_dim=ml_dim,
                title=title,
                cmap=cmap,
                symmetric=symmetric,
                width=width,
                height=height,
            )
        )
    return hv.Layout(plots).cols(cols)


def triggered_waveform(
    template: np.ndarray,
    std: np.ndarray | None = None,
    *,
    lag_axis: np.ndarray | None = None,
    label: str = "",
    title: str = "",
    fill_alpha: float = 0.25,
    width: int = 400,
    height: int = 250,
):
    """
    Plot a triggered average waveform with optional ±1 std band.

    Parameters
    ----------
    template : (n_lag,) float
        Mean/median triggered waveform.
    std : (n_lag,) float, optional
        Standard deviation across events.
    lag_axis : (n_lag,) float, optional
        Lag values in seconds.  Defaults to sample indices.
    label : str
        Curve label.

    Returns
    -------
    hv.Overlay
        Curve (± Area band if std provided).
    """
    import holoviews as hv

    template = np.asarray(template, dtype=float)
    n = template.size
    x = np.asarray(lag_axis, dtype=float) if lag_axis is not None else np.arange(n)

    curve = hv.Curve((x, template), kdims=["Lag (s)"], vdims=["Amplitude"], label=label)
    elements = [curve]

    if std is not None:
        std = np.asarray(std, dtype=float)
        band = hv.Area(
            (x, template - std, template + std),
            kdims=["Lag (s)"],
            vdims=["lower", "upper"],
        ).opts(alpha=fill_alpha, line_alpha=0)
        elements = [band, curve]

    ov = hv.Overlay(elements).opts(title=title, width=width, height=height)
    return ov


def triggered_waveform_grid(
    templates: xr.DataArray,
    stds: xr.DataArray | None = None,
    *,
    ap_dim: str = "AP",
    ml_dim: str = "ML",
    lag_dim: str = "lag",
    width: int = 150,
    height: int = 120,
):
    """
    Grid of triggered waveforms laid out by (AP, ML) electrode position.

    Parameters
    ----------
    templates : xr.DataArray with dims (AP, ML, lag)
        Per-channel triggered average.
    stds : xr.DataArray with dims (AP, ML, lag), optional
        Per-channel triggered std.

    Returns
    -------
    hv.HoloMap or hv.Layout
        Grid of waveform plots, one per electrode.
    """
    import holoviews as hv

    if lag_dim not in templates.dims:
        raise ValueError(f"templates must have {lag_dim!r} dim")
    if ap_dim not in templates.dims or ml_dim not in templates.dims:
        raise ValueError(f"templates must have {ap_dim!r} and {ml_dim!r} dims")

    lag_vals = np.asarray(templates.coords[lag_dim].values, dtype=float)
    ap_vals = templates.coords[ap_dim].values
    ml_vals = templates.coords[ml_dim].values

    plots = {}
    for iap, ap in enumerate(ap_vals):
        for iml, ml in enumerate(ml_vals):
            tmpl = templates.sel({ap_dim: ap, ml_dim: ml}).values
            std_ch = None
            if stds is not None:
                std_ch = stds.sel({ap_dim: ap, ml_dim: ml}).values

            c = hv.Curve((lag_vals, tmpl), kdims=["lag"], vdims=["amp"])
            elements = [c]
            if std_ch is not None:
                band = hv.Area(
                    (lag_vals, tmpl - std_ch, tmpl + std_ch),
                    kdims=["lag"],
                    vdims=["lower", "upper"],
                ).opts(alpha=0.2, line_alpha=0)
                elements = [band, c]

            ov = hv.Overlay(elements).opts(
                width=width,
                height=height,
                title=f"AP={iap} ML={iml}",
                xaxis=None,
                yaxis=None,
            )
            plots[(int(iap), int(iml))] = ov

    hmap = hv.HoloMap(plots, kdims=[ap_dim, ml_dim])
    return hmap.layout().cols(len(ml_vals))


def event_histogram(
    values: np.ndarray,
    *,
    bins: int | np.ndarray = 50,
    xlabel: str = "",
    ylabel: str = "Count",
    title: str = "",
    width: int = 500,
    height: int = 250,
):
    """
    Histogram of scalar values (e.g. inter-event intervals, lags, amplitudes).

    Parameters
    ----------
    values : (N,) float
    bins : int or array
        Number of bins or explicit bin edges.

    Returns
    -------
    hv.Histogram
    """
    import holoviews as hv

    values = np.asarray(values, dtype=float)
    freqs, edges = np.histogram(values, bins=bins)
    h = hv.Histogram((edges, freqs), kdims=[xlabel or "value"], vdims=[ylabel])
    return h.opts(title=title, width=width, height=height)


def lag_histogram(
    counts: np.ndarray,
    bin_edges: np.ndarray,
    *,
    title: str = "Cross-correlogram",
    width: int = 500,
    height: int = 250,
):
    """
    Plot a cross-correlogram from ``event_lag_histogram`` output.

    Parameters
    ----------
    counts : (n_bins,) int
        From :func:`cogpy.events.match.event_lag_histogram`.
    bin_edges : (n_bins + 1,) float
        Bin edges in seconds.

    Returns
    -------
    hv.Histogram
    """
    import holoviews as hv

    h = hv.Histogram((bin_edges, counts), kdims=["Lag (s)"], vdims=["Count"])
    return h.opts(title=title, width=width, height=height)


def factor_loading_grid(
    loadings: xr.DataArray,
    *,
    factor_dim: str = "factor",
    ap_dim: str = "AP",
    ml_dim: str = "ML",
    freq_dim: str = "freq",
    cmap: str = "RdBu_r",
    symmetric: bool = True,
    width: int = 200,
    height: int = 180,
    cols: int | None = None,
):
    """
    Grid of spatial loading maps, one per factor, at peak frequency.

    Parameters
    ----------
    loadings : xr.DataArray with dims (factor, AP, ML, freq)
        Spatio-spectral factor loadings.
    factor_dim : str
        Name of the factor dimension.
    freq_dim : str
        Name of the frequency dimension.  The map at the frequency of
        maximum absolute loading is shown for each factor.

    Returns
    -------
    hv.Layout
        Grid of heatmaps, one per factor.
    """
    import holoviews as hv

    if factor_dim not in loadings.dims:
        raise ValueError(
            f"loadings must have {factor_dim!r} dim, got {list(loadings.dims)}"
        )

    factors = loadings.coords[factor_dim].values
    n_fac = len(factors)
    if cols is None:
        cols = min(n_fac, 5)

    plots = []
    for fac in factors:
        ld_fac = loadings.sel({factor_dim: fac})

        # Find peak frequency (max absolute loading)
        if freq_dim in ld_fac.dims:
            abs_ld = np.abs(ld_fac).mean(dim=(ap_dim, ml_dim))
            peak_freq_idx = int(abs_ld.argmax().values)
            peak_freq = float(ld_fac.coords[freq_dim].values[peak_freq_idx])
            spatial_map = ld_fac.isel({freq_dim: peak_freq_idx}).values
            title = f"F{fac} @ {peak_freq:.1f} Hz"
        else:
            spatial_map = ld_fac.values
            title = f"F{fac}"

        plots.append(
            spatial_heatmap(
                spatial_map,
                ap_dim=ap_dim,
                ml_dim=ml_dim,
                title=title,
                cmap=cmap,
                symmetric=symmetric,
                width=width,
                height=height,
            )
        )

    return hv.Layout(plots).cols(cols)


def drift_plot(
    matched_times: np.ndarray,
    matched_lags: np.ndarray,
    coeffs: np.ndarray | None = None,
    *,
    title: str = "Event lag drift",
    width: int = 600,
    height: int = 250,
):
    """
    Scatter plot of matched lags vs time, with optional polynomial fit.

    Parameters
    ----------
    matched_times : (K,) float
        Event times from signal A.
    matched_lags : (K,) float
        Corresponding lags (B - A) in seconds.
    coeffs : (degree+1,) float, optional
        Polynomial coefficients from :func:`cogpy.events.match.estimate_drift`.

    Returns
    -------
    hv.Overlay
    """
    import holoviews as hv

    points = hv.Scatter(
        (matched_times, matched_lags),
        kdims=["Time (s)"],
        vdims=["Lag (s)"],
    ).opts(size=2, alpha=0.4, color="steelblue")

    elements = [points]

    if coeffs is not None and not np.any(np.isnan(coeffs)):
        t_fit = np.linspace(float(matched_times.min()), float(matched_times.max()), 200)
        lag_fit = np.polyval(coeffs, t_fit)
        fit_curve = hv.Curve(
            (t_fit, lag_fit),
            kdims=["Time (s)"],
            vdims=["Lag (s)"],
            label="fit",
        ).opts(color="red", line_width=2)
        elements.append(fit_curve)

    return hv.Overlay(elements).opts(title=title, width=width, height=height)


def signal_trace(
    signal: xr.DataArray | np.ndarray,
    *,
    time: np.ndarray | None = None,
    label: str = "",
    title: str = "",
    width: int = 700,
    height: int = 200,
    **curve_opts,
):
    """
    Simple time-series trace as a HoloViews Curve.

    Parameters
    ----------
    signal : 1D array or xr.DataArray with time dim
    time : (N,) float, optional
        Time axis.  Inferred from xarray coords if available.

    Returns
    -------
    hv.Curve
    """
    import holoviews as hv

    if isinstance(signal, xr.DataArray):
        if time is None and "time" in signal.coords:
            time = np.asarray(signal.coords["time"].values, dtype=float)
        signal = signal.values
    signal = np.asarray(signal, dtype=float).ravel()
    if time is None:
        time = np.arange(len(signal))
    time = np.asarray(time, dtype=float).ravel()

    c = hv.Curve((time, signal), kdims=["Time (s)"], vdims=["Amplitude"], label=label)
    opts = dict(width=width, height=height, title=title)
    opts.update(curve_opts)
    return c.opts(**opts)


def signal_traces_overlay(
    signals: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    title: str = "",
    width: int = 700,
    height: int = 250,
):
    """
    Overlay multiple time-series traces (e.g. before/after comparison).

    Parameters
    ----------
    signals : dict[str, (time, data)]
        Mapping of ``{label: (time_array, signal_array)}``.

    Returns
    -------
    hv.Overlay
    """
    import holoviews as hv

    curves = []
    for label, (t, y) in signals.items():
        curves.append(signal_trace(np.asarray(y), time=np.asarray(t), label=label))
    return hv.Overlay(curves).opts(
        title=title,
        width=width,
        height=height,
        legend_position="top_right",
    )
