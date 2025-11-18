import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import ipywidgets as widgets
import os
from matplotlib.animation import FuncAnimation
import pickle as pkl
import quantities as pq
from ..utils.wrappers import ax_plot


class DATAX:
    def __init__(self, data=None, time_dim="time", ml_dim="ML", ap_dim="AP", fs=625.0):
        """
        Base class for handling xarray DataArray with default attributes.

        Parameters:
        - data: xarray.DataArray or None (default: None)
        - time_dim: str, name of the time dimension (default: 'time')
        - ml_dim: str, name of the ML dimension (default: 'ML')
        - ap_dim: str, name of the AP dimension (default: 'AP')
        - fs: float, sampling frequency (default: 625.0 Hz)
        """

        if data is None:
            time = np.linspace(0, 133.4, 83356)  # Default time values
            self.xarr = xr.DataArray(
                np.random.rand(83356, 32, 16),  # Default random data
                dims=[time_dim, ml_dim, ap_dim],
                coords={time_dim: time},
                attrs={"fs": fs},  # Sampling frequency
            )
        else:
            self.xarr = data

        self.fs = self.xarr.attrs.get("fs", fs)
        self.lmin = self.xarr.min(dim=[ml_dim, ap_dim])  # Default min extrema
        self.lmax = self.xarr.max(dim=[ml_dim, ap_dim])  # Default max extrema
        self.mask2D = np.zeros((32, 16), dtype=bool)  # Default bad channel mask


# Refactored FramePlot for xarray compatibility
class FramePlot(DATAX):
    plotting_shape_context = ("ML", "AP", "time")

    def take_frame(self, t):
        """Extract a single frame at time index `t` using xarray."""
        return self.xarr.isel(time=t)

    def plot_signal(self, t=0, **kwargs):
        """Plot a signal frame at time `t`."""
        frame = self.take_frame(t)
        plot_signal(frame, **kwargs)

    def plot_extrema(self, t=0, **kwargs):
        """Plot extrema points for frame `t`."""
        plot_extrema([self.lmin.isel(time=t), self.lmax.isel(time=t)], **kwargs)

    def plot_waves(self, t=0, **kwargs):
        """Plot waves at time `t`."""
        plot_waves([self.lmin.isel(time=t), self.lmax.isel(time=t)], **kwargs)

    def plot_contour(self, t=0, **kwargs):
        """Plot contour of signal at time `t`."""
        frame = self.take_frame(t)
        plot_contour(frame, **kwargs)

    def plot_bad_channels(self, **kwargs):
        """Plot bad channels mask."""
        plot_bad_channels(self.mask2D, **kwargs)

    def widget(self, plot_func, ax=None, autoplay=True, **plot_kwargs):
        """Interactive widget for signal visualization."""
        widget(self.xarr, plot_func, ax=ax, autoplay=autoplay, **plot_kwargs)

    def save_animation(self):
        pass

    def tplayer(self, interval=300, step=1):
        return widgets.Play(
            min=0, max=self.xarr.sizes["time"] - 1, step=step, interval=interval
        )

    def tslider(self, step=1):
        return widgets.IntSlider(0, 0, self.xarr.sizes["time"] - 1, step=step)


# Updating animate function for xarray
def animate(data, extent=[0, 16, 16, 0], save_dst=None):
    """
    Create an animation from xarray.DataArray.

    Parameters
    ----------
    data : xarray.DataArray
        A 3D xarray DataArray with dimensions (time, ML, AP).

    extent : list
        The spatial extent of the data [xmin, xmax, ymin, ymax].

    save_dst : str, optional
        Path to save the animation. If None, animation is displayed in notebook.

    Returns
    -------
    ani : matplotlib.animation.FuncAnimation
        The animation object.
    """

    fig, ax = plt.subplots()

    def update(frame):
        ax.clear()
        ax.imshow(data.isel(time=frame), cmap="hot", extent=extent)
        ax.set_title(f"Frame: {frame}")

    ani = FuncAnimation(fig, update, frames=len(data.time), interval=50, blit=False)

    if save_dst:
        ani.save(save_dst, writer="ffmpeg")

    return ani


# Updated widget_array function for xarray
def widget_array(
    arr_list, plot_func, interval=300, s=4, title=None, autoplay=True, **kwargs
):
    """
    Create an interactive widget for visualizing xarray.DataArray.

    Parameters
    ----------
    arr_list : list of xarray.DataArray
        List of xarray.DataArray objects to visualize.

    plot_func : function
        Function used to plot each frame.

    interval : int, optional
        Interval in ms for animation (default 300ms).

    s : int, optional
        Size of subplot figures (default 4).

    title : list of str, optional
        Titles for each subplot.

    autoplay : bool, optional
        Whether to autoplay animation (default True).
    """

    arr_list = list(arr_list) if hasattr(arr_list, "__iter__") else arr_list
    h, w = len(arr_list), len(arr_list[0])

    slider = (
        widgets.Play(
            min=0, max=arr_list[0].sizes["time"] - 1, step=1, interval=interval
        )
        if autoplay
        else widgets.IntSlider(0, 0, arr_list[0].sizes["time"] - 1)
    )

    fig, axes = plt.subplots(h, w, figsize=(w * s, h * s))

    @widgets.interact(t=slider)
    def plot_frame(t):
        for i, arr in enumerate(arr_list):
            for j, ax in enumerate(axes.flat):
                ax.clear()
                plot_func(arr.isel(time=t), ax=ax, **kwargs)
                if title:
                    ax.set_title(title[i * w + j])

    plt.show()


# Example plot functions compatible with xarray
@ax_plot
def plot_signal(arr, ax=None, color_bar=False, **kwargs):
    """Plot a signal using xarray's built-in plotting function."""
    arr.plot(ax=ax, **kwargs)
    if color_bar:
        plt.colorbar(ax=ax)


@ax_plot
def plot_extrema(ext, ax=None, cmin="r", cmax="b", **kwargs):
    dfl, dfm = ext
    ax.scatter(dfl.w, dfl.h, c=cmin, **kwargs)
    ax.scatter(dfm.w, dfm.h, c=cmax, **kwargs)


@ax_plot
def plot_waves(ext, ax=None, cmin="r", cmax="b", **kwargs):
    for df, c in zip(ext, [cmin, cmax]):
        for w, h, clu in df[["w", "h", "Clu"]].values:
            ax.text(w, h, clu, c=c, ha="center", va="center", **kwargs)


@ax_plot
def plot_contour(arr, ax=None, **kwargs):
    """Plot a contour plot using xarray."""
    ax.contour(arr, **kwargs)


@ax_plot
def plot_bad_channels(mask, ax=None, s=2, c="purple", **kwargs):
    """Plot bad channels."""
    ax.scatter(*np.where(mask.T), s=s, c=c, **kwargs)


# Save animation frames
def save_frames(directory, frames):
    """Save frames as images in a directory."""
    for i, frame in enumerate(frames):
        plt.imsave(os.path.join(directory, f"frame_{i:04d}.png"), frame, cmap="hot")


# Convert frames to video using ffmpeg
def save_movie(directory="."):
    """Convert saved frames into a movie."""
    os.system(
        f"ffmpeg -r 30 -i {directory}/frame_%04d.png -vcodec libx264 -y {directory}/movie.mp4"
    )


@ax_plot
def widget(arr, plot_func, ax=None, autoplay=True, **kwargs):
    """
    Note!
    %matplotlib inline
    %matplotlib widget
    ---
    arr: array (h, w, t)
    plot_func: plot_func(GridSignal, t, ax, **kwargs) | array of plot_func 1d or 2d
    if both sig and plot_func are 2d arrays then their shape should be identical. Each GridSignal is plotted by the corresponding plot_func

        autoplay: True: player, False: slider

    """
    if autoplay:
        slider = widgets.Play(0, 0, arr.shape[-1] - 1, interval=300)

    else:
        slider = widgets.IntSlider(0, 0, arr.shape[-1] - 1)

    @widgets.interact(t=slider)
    def plot_frame(t):
        ax.clear()
        plot_func(arr[..., t], **kwargs)
        plt.draw()


# Interactive widgets
grid_button = widgets.ToggleButton(value=False, description="Grid", icon="check")
extrema_button = widgets.ToggleButton(value=True, description="Extrema", icon="check")
bad_chan_button = widgets.ToggleButton(
    value=False, description="Bad Channels", icon="check"
)
autoplay_button = widgets.ToggleButton(
    value=False, description="Autoplay", icon="check"
)
contour_button = widgets.ToggleButton(value=True, description="Contour", icon="check")
