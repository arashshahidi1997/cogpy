import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
import pickle as pkl
import os
from itertools import zip_longest
# plot
import src.grid_signal as gs

class FramePlot:
    def plot_signal(self, t, ax=plt.gca(), **kwargs):
        plot_signal(self.A.select('t', t), ax=ax, **kwargs)
        # ax.set(title=self.title)

    def plot_extrema(self, t, ax=plt.gca(), **kwargs):
        plot_extrema(self.ext[t], ax=ax, **kwargs)
        # ax.set(title=self.title)

    def plot_contour(self, t, ax=plt.gca(), **kwargs):
        plot_contour(self.A.select('t', t), ax=ax, **kwargs)
        # ax.set(title=self.title)

def plot_signal(arr, ax=plt.gca(), **kwargs):
    ax.imshow(arr, **kwargs)

def plot_extrema(ext, ax=plt.gca(), cmin='r', cmax='b', **kwargs):
    dfl, dfm = ext
    ax.scatter(dfl.w, dfl.h, c=cmin, **kwargs)
    ax.scatter(dfm.w, dfm.h, c=cmax, **kwargs)

# def plot_extrema_old(ext, ax=plt.gca(), **kwargs):
#     ext_modes = ['CoM_thr', 'CoM_dist']
#     c_order = ['green', 'blue', 'orange', 'red']
#     i = 0
#     for s in ext:
#         for m in ext_modes:
#             ax.scatter(ext[s][m].w, ext[s][m].h, c=c_order[i], **kwargs)
#             i += 1   

def plot_contour(arr, ax=plt.gca(), **kwargs):
    ax.imshow(arr, **kwargs, alpha=0.5)
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

def widget(sig, plot_func, s=4, autoplay=True, **kwargs):
    """
    autoplay: True: player, False: slider
    sig: GridSignal | array(GridSignal) 1d or 2d
    plot_func: plot_func(GridSignal, t, ax, **kwargs) | array of plot_func 1d or 2d 
    if both sig and plot_func are 2d arrays then their shape should be identical. Each GridSignal is plotted by the corresponding plot_func
    """
    sig = squeeze2D(sig)
    plot_func = squeeze2D(plot_func)
    
    h, w = max(sig.shape, plot_func.shape)
    
    if autoplay:
        slider = widgets.Play(0, 0, sig[0,0].dur - 1, interval=300)

    else:
        slider = widgets.IntSlider(0, 0, sig[0,0].dur - 1)

    @widgets.interact(t=slider)
    def plot_frame(t):
        fig = plt.figure(figsize=(int(w*s), int(h*s)))
        i_ax = 1
        for ih in range(h):
            for iw in range(w):
                ax = fig.add_subplot(h, w, i_ax)
                i_ax += 1
                iterator_choices = [(ih, iw), (0,0)]
                ih_f, iw_f = iterator_choices[plot_func.size == 1]
                ih_s, iw_s = iterator_choices[sig.size == 1]
                plot_func[ih_f, iw_f](sig[ih_s, iw_s], t, ax=ax, **kwargs)
                
        plt.show()

def save_frames(dir, frames):
    for i, frame in enumerate(frames):
        plt.savefig()

def save_movie(dir='.'):
    os.system("ffmpeg -r 1 -i {dir}/img%01d.png -vcodec mpeg4 -y movie.mp4".format(dir=dir))
    return
