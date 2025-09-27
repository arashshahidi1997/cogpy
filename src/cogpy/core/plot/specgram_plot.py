import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from ..utils.wrappers import ax_plot
from IPython.display import display, clear_output
import ipywidgets as widgets
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def plot4d_mtx(mtx: xr.DataArray, imshowkw=True):
    vmin = np.nanmin(mtx.data)
    vmax = np.nanmax(mtx.data)
    imshow_kwargs = dict(vmin=vmin, vmax=vmax, cmap='jet')
    num_freq = mtx.sizes['freq']
    num_time = mtx.sizes['time']
    
    fig, ax = plt.subplots(num_time, num_freq, figsize=(20, 20))

    for iframe in range(num_time):
        rec_mtx_frame = mtx.isel(time=iframe)
        for ifreq in range(num_freq):
            axis = ax[iframe, ifreq]
            if imshowkw:
                axis.imshow(rec_mtx_frame.isel(freq=ifreq), **imshow_kwargs)
            axis.imshow(rec_mtx_frame.isel(freq=ifreq))
            axis.set_xticks([])
            axis.set_yticks([])

    for ifreq in range(num_freq):
        axis = ax[0, ifreq]
        axis.set_title(f'{mtx.freq[ifreq].values:.1f} Hz', fontsize=12)

    for iframe in range(num_time):
        axis = ax[iframe, 0]
        axis.set_ylabel(f'{mtx.time[iframe].values:.1f} s', fontsize=12)

    plt.show()

@ax_plot
def add_lines(dim_sizes, ax=None, direction='h', **hlinekwargs):
    nt, nh = dim_sizes
    for ti_ in range(1,nt):
        if direction=='h':
            ax.axhline(ti_*nh-0.5, **hlinekwargs)
        if direction=='v':
            ax.axvline(ti_*nh-0.5, **hlinekwargs)

def get_raveled_ticks(dim_sizes):
    nt, nh = dim_sizes
    return np.arange(nh//2, nh * nt, nh)

def arrstr(arr, decimals):
    return np.vectorize(lambda x: f"{x:.{decimals}f}")(arr)

# %% plot
def scroll_spec_png(sig, spec, ich, dst_file):
    # Create output widget
    out = widgets.Output()

    # Create figure and subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    # Add signal trace
    ax1.plot(sig.times, sig._arr[ich], label='Signal')
    ax1.legend()

    # Add spectrogram plot
    im = ax2.imshow(np.log10(spec.specgram[ich]), aspect='auto', origin='lower', cmap='viridis', extent=[spec.times[0], spec.times[-1], spec.freqs[0], spec.freqs[-1]])
    ax2.set_ylabel('Frequency')

    # Set x-axis range for signal and spectrogram
    signal_xrange = [0, sig.dur]
    spectrogram_xrange = [0, spec.nT]

    # Create slider
    slider = widgets.FloatSlider(value=0, min=sig.times[0], max=sig.times[1], step=.1, description='Time Index:')

    # Define update function for slider value change
    def update_slider(change):
        x_range = [signal_xrange[0] + slider.value, signal_xrange[1] + slider.value]
        ax1.set_xlim(x_range)
        ax2.set_xlim(x_range)
        with out:
            clear_output(wait=True)
            display(fig)

    # Register update function to slider value change
    slider.observe(update_slider, 'value')

    # Display the figure and slider
    with out:
        display(fig, slider)

    # Save the figure as HTML
    fig.savefig(dst_file)

def scroll_spec_html(sig, spec, ich, dst_file):
    # Generate example signal and spectrogram data
    # sig = np.random.randn(1000)
    # spectrogram = np.random.rand(100, 250)
    # times = np.linspace(0, 1000, 250)
    # Create subplots with shared x-axis
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)

    # Add signal trace
    fig.add_trace(go.Scatter(x=sig.times, y=sig._arr[ich], name='Signal'), row=1, col=1)

    # Add spectrogram trace
    fig.add_trace(go.Heatmap(x=spec.times, y=spec.freqs, z=np.log10(spec.specgram[ich]), colorscale='Viridis'), row=2, col=1)

    # Add slider to layout
    fig.update_layout(
        xaxis=dict(rangeslider=dict(visible=True),
                                 type="linear"),
        # sliders=[slider],
        height=600,
    )
    # Show the figure
    fig.write_html(dst_file)

