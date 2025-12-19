import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d


def scree_plot(pca: PCA, cutoff=0.95, ax=plt.gca()):
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
    """
    Returns:
    dim: int
        dimensionality that the cumulative variance captures more than cutoff of variance

    cutoff: float [0,1]
        0.95 for 95 percent of explained variance
    """

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    dim = np.argmax((cumvar - cutoff) > 0)
    return dim, cumvar


def plot_view_grid(x, dim, title="", text="PC", half_view=True, **scatter_kwargs):
    fig, ax = plt.subplots(dim, dim)

    text_list = []
    if isinstance(text, str):
        text_list = [f"{text}{i+1}" for i in range(dim)]

    else:
        text_list = text

    for (ix, iy), iax in np.ndenumerate(np.arange(dim * dim).reshape(dim, dim)):
        _ax = ax.flatten()[iax]
        if ix >= iy or not half_view:
            _ax.scatter(x[:, iy], x[:, ix], **scatter_kwargs)
            _ax.set(xticks=[], yticks=[])
            if iy == 0:
                _ax.set_ylabel(f"{text_list[ix]}"),

            if ix == dim - 1:
                _ax.set_xlabel(f"{text_list[iy]}"),

        else:
            _ax.axis("off")


def plot_view_3D(x, y, z, **scatter3D_kwargs):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.scatter3D(x, y, z, **scatter3D_kwargs)
    ax.set(xlabel="x", ylabel="y", zlabel="z", xticks=[], yticks=[], zticks=[])
