import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
import pickle as pkl
import os
from matplotlib.animation import FuncAnimation
import quantities as pq
from ..utils.wrappers import ax_plot

# plot


class FramePlot:
    plotting_shape_context = ("h", "w", "t")

    def plot_context(plot_func):
        def wrapped_(self, *args, **kwargs):
            with self.reshape_context(FramePlot.plotting_shape_context) as sig:
                return plot_func(sig, *args, **kwargs)

        return wrapped_

    def take_frame(self, t):
        if torch.is_tensor(self._arr):
            frame = self._arr.select("t", t)

        else:
            frame = np.take(self._arr, t, axis=-1)

        return frame

    @plot_context
    def plot_signal(self, t=0, **kwargs):
        frame = self.take_frame(t=t)
        plot_signal(frame, **kwargs)
        # ax.set(title=self.title)

    @plot_context
    def plot_extrema(self, t=0, **kwargs):
        plot_extrema([self.lmin[t], self.lmax[t]], **kwargs)
        # ax.set(title=self.title)

    @plot_context
    def plot_waves(self, t=0, **kwargs):
        plot_waves([self.lmin[t], self.lmax[t]], **kwargs)

    @plot_context
    def plot_contour(self, t=0, **kwargs):
        frame = self.take_frame(t=t)
        plot_contour(frame, **kwargs)
        # ax.set(title=self.title)

    @plot_context
    def plot_bad_channels(self, **kwargs):
        plot_bad_channels(self.mask2D, **kwargs)

    @plot_context
    def widget(self, plot_func, ax=None, autoplay=True, **plot_kwargs):
        widget(self._arr, plot_func, ax=ax, autoplay=autoplay, **plot_kwargs)

    @plot_context
    def widget_array(
        self, plot_func, interval=300, s=4, title=None, autoplay=True, **plot_kwargs
    ):
        widget_array(
            self._arr,
            plot_func,
            interval=interval,
            s=s,
            title=title,
            autoplay=autoplay,
            **plot_kwargs,
        )

    def save_animation(self):
        pass

    def tplayer(self, interval=300, step=1):
        if isinstance(step, pq.Quantity):
            step = int(step.rescale(pq.s).magnitude * self.fs)
        return widgets.Play(0, 0, self.dur - 1, step=step, interval=interval)

    def tslider(self, step=1):
        return widgets.IntSlider(0, 0, self.dur - 1, step=step)


def animate(data, extent=[0, 16, 16, 0], save_dst=None):
    """
    Plot a grid animation using matplotlib

    Parameters
    ----------
    data : np.array
        A 3D numpy array with shape (rows, cols, samples) representing the grid data

    extent : list
        A list with 4 elements representing the extent of the grid data [xmin, xmax, ymin, ymax]

    save_dst : str
        The destination path to save the animation. If None, the animation will be displayed in the notebook

    Returns
    -------
    ani : matplotlib.animation.FuncAnimation
        The animation object

    to save the animation as a video file, you need to have ffmpeg installed on your machine.
    You can install it by running the following command in your terminal:
    $ conda install -c conda-forge ffmpeg

    run the following command to save the animation:
    ani.save('path/to/save/animation.mp4', writer='ffmpeg')
    """

    fig, ax = plt.subplots()

    def update(frame):
        # clear the previous plot
        ax.clear()
        # plot the 16x16 grid at the current time frame
        ax.imshow(data[:, :, frame], cmap="hot", extent=extent)
        # set the title of the plot
        ax.set_title("Frame: {}".format(frame))

    # create the animation
    ani = FuncAnimation(fig, update, frames=100, interval=50, blit=False)
    if save_dst is not None:
        ani.save(save_dst, writer="ffmpeg")
    return ani


@ax_plot
def plot_bad_channels(mask, s=2, ax=None, c="purple", **kwargs):
    ax.scatter(*np.where(mask.T), s=s, c=c, **kwargs)


@ax_plot
def plot_signal(arr, color_bar=False, ax=None, **kwargs):
    """
    arr:
    ax:
    color_bar: bool
    **kwargs: imshow kwargs
    """
    im = ax.imshow(arr, **kwargs)
    if color_bar:
        plt.colorbar(im)


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


# def plot_extrema_old(ext, **kwargs):
#     ext_modes = ['CoM_thr', 'CoM_dist']
#     c_order = ['green', 'blue', 'orange', 'red']
#     i = 0
#     for s in ext:
#         for m in ext_modes:
#             ax.scatter(ext[s][m].w, ext[s][m].h, c=c_order[i], **kwargs)
#             i += 1


@ax_plot
def plot_contour(arr, ax=None, **kwargs):
    ax.contour(arr)


def squeeze2D(input):
    output = np.array([input], dtype=object).squeeze()
    if output.ndim < 2:
        output = np.expand_dims(output.reshape(-1), -1)

    return output


# def widget_wrap(plot_func):
#     def w_plot():
#         @widgets.interact(t = )

# def widget(plot_func, *args, **kwargs, autoplay=False):
#     if autoplay:
#         @widgets.interact(t = widgets.Play(0, 0, sig[0,0].dur - 1, interval=300))
#         def widget_plot(t):
#             plot_func(
#     else:
#         @widgets.interact(t = widgets.IntSlider(0, 0, sig[0,0].dur - 1))
#         def widget_plot:


def widget_array(
    arr_list, plot_func, interval=300, s=4, title=None, autoplay=True, **kwargs
):
    """
    autoplay: True: player, False: slider
    sig: GridSignal | array(GridSignal) 1d or 2d
    plot_func: plot_func(GridSignal, t, ax, **kwargs) | array of plot_func 1d or 2d
    if both sig and plot_func are 2d arrays then their shape should be identical. Each GridSignal is plotted by the corresponding plot_func
    """
    # sig = squeeze2D(sig)
    # in the version where sig has the __array__ method this is problematic so we have to work with lists
    arr_list = list(arr_list) if hasattr(arr_list, "__iter__") else arr_list
    plot_func = squeeze2D(plot_func)

    sig_shape = (len(arr_list), len(arr_list[0]))
    sig_size = np.prod(sig_shape)
    h, w = max(sig_shape, plot_func.shape)

    if autoplay:
        slider = widgets.Play(
            min=0, max=arr_list[0][0].dur - 1, step=1, interval=interval
        )

    else:
        slider = widgets.IntSlider(0, 0, arr_list[0][0].dur - 1)

    plt.ioff()
    fig = plt.figure(figsize=(int(w * s), int(h * s)))
    ax = []
    for i_ax, (ih, iw) in enumerate(np.ndindex((h, w))):
        ax.append(fig.add_subplot(h, w, i_ax + 1))
        if title is not None:
            ax[i_ax].set(title=title[i_ax])
    plt.ion()

    @widgets.interact(t=slider)
    def plot_frame(t):
        for i_ax, (ih, iw) in enumerate(np.ndindex((h, w))):
            iterator_choices = [(ih, iw), (0, 0)]
            ih_f, iw_f = iterator_choices[plot_func.size == 1]
            ih_s, iw_s = iterator_choices[sig_size == 1]
            plot_func[ih_f, iw_f](arr_list[ih_s][iw_s], t, ax=ax[i_ax], **kwargs)
        # plt.show()


@ax_plot
def widget(arr, plot_func, ax=None, autoplay=True, fs=None, **kwargs):
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
        if fs is not None:
            ax.set_title(f"t={1000 * t/fs: .2f} ms")
        plt.draw()


def save_frames(dir, frames):
    for i, frame in enumerate(frames):
        plt.savefig()


def save_movie(dir="."):
    os.system(
        "ffmpeg -r 1 -i {dir}/img%01d.png -vcodec mpeg4 -y movie.mp4".format(dir=dir)
    )
    return


grid_button = widgets.ToggleButton(value=False, description="Grid", icon="check")

extrema_button = widgets.ToggleButton(value=True, description="ext", icon="check")
bad_chan_button = widgets.ToggleButton(
    value=False, description="bad channels", icon="check"
)
autoplay_button = widgets.ToggleButton(
    value=False, description="autoplay", icon="check"
)
contour_button = widgets.ToggleButton(value=True, description="contour", icon="check")


# from contextlib import contextmanager

# @contextmanager
# def widget_context(*args, **kwargs):
#     %matplotlib widget
#     with interactive_backend():
#         fig, ax = plt.subplots(*args, **kwargs)
#     try:
#         yield fig, ax
#     finally:
#         plt.ioff()
#         %matplotlib inline

# @contextmanager
# def interactive_backend():
#     plt.ion()
#     try:
#         yield
#     finally:
#         plt.ioff()

# %matplotlib inline
# with widget_context() as (fig, ax):
#     with sig.reshape_context(sig.plotting_shape_context) as sig:
#         sig.widget(fplt.plot_signal, ax=ax, autoplay=False)
