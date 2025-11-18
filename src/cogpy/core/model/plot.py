import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def visualize_modes(modes, combinations, freq_to_view, time_to_view):
    """
    visualize_modes(gaussian_cover_instance.modes, gaussian_cover_instance.combinations, freq_to_view, time_to_view)
    """
    num_modes = len(modes)
    grid_dim = int(np.ceil(np.sqrt(num_modes)))

    canonical_min = modes[0].sel(frequency=freq_to_view, time=time_to_view).min().values
    canonical_max = modes[0].sel(frequency=freq_to_view, time=time_to_view).max().values

    fig, axes = plt.subplots(
        grid_dim, grid_dim, figsize=(15, 15), sharex=True, sharey=True
    )
    axes = axes.ravel()

    for idx, (ax, bump) in enumerate(zip(axes, modes)):
        loc = combinations[idx]
        bump.sel(frequency=freq_to_view, time=time_to_view).plot(
            ax=ax,
            cmap="viridis",
            vmin=canonical_min,
            vmax=canonical_max,
            add_colorbar=False,
        )
        ax.set_title(f"Bump {idx} at {loc}")

    norm = Normalize(vmin=canonical_min, vmax=canonical_max)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
    mappable.set_array([])

    cax = fig.add_axes([0.4, 0.92, 0.2, 0.02])
    fig.colorbar(mappable, cax=cax, orientation="horizontal")

    coord_order = "(Row, Column, Frequency, Time)"
    fig.suptitle(f"Coordinates order: {coord_order}", fontsize=16, y=1.005)

    plt.show()
