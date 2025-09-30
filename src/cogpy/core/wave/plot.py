from ..utils.wrappers import ax_plot
from typing import Optional
from matplotlib.axes import Axes
from .utils import wave_gen

@ax_plot
def shade_waves(x, wave_df, colors=['r', 'b'], alpha=0.3, ax: Optional[Axes] = None):
    # shade area under waves curves
    # alternate color of successive waves
    for i, wave in enumerate(wave_gen(x, wave_df)):
        if i % 2 == 0:
            color = colors[0]
        else:
            color = colors[1]
        
        ax.fill_between(wave.time, wave.data, color=color, alpha=alpha)

