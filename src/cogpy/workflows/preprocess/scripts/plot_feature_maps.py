import numpy as np
import pandas as pd
import holoviews as hv
from holoviews import opts
import xarray as xr
from holoviews.operation import gridmatrix

hv.extension("matplotlib")
from scipy.spatial.distance import cdist
from cogpy.core.utils.stats import robust_zscore


def compute_quantile_features(features, qmin=0.75, qmax=0.95, nq=5):
    quantiles = np.linspace(qmin, qmax, nq)
    qfeat = features.quantile(quantiles, dim=["time"]).stack(ch=["AP", "ML"])
    return qfeat.transpose("ch", "quantile")


def deviation_zscore(qfeat: np.ndarray) -> np.ndarray:
    """
    qfeat: (ch, qfeat)
    """
    qfeat_dist = cdist(qfeat, qfeat)  # (ch, ch)
    ch_deviation_score = np.nanmedian(qfeat_dist, axis=1)
    ch_deviation_zscore = robust_zscore(
        ch_deviation_score, scale="normal", nan_policy="omit"
    )
    return ch_deviation_zscore


def feature_df(feature_ds, outlier_map):
    AP = feature_ds.sizes["AP"]
    ML = feature_ds.sizes["ML"]

    qfeat = feature_ds.quantile(0.9, dim=["time"])
    df_bad = xr.DataArray(outlier_map.reshape(AP, ML), dims=["AP", "ML"]).to_dataframe(
        name="is_bad"
    )
    df_feat = qfeat.to_dataframe().drop(columns="quantile")
    df = pd.merge(df_feat, df_bad, left_index=True, right_index=True)
    return df


def get_bad_chs_df(outlier_map, AP, ML):
    bad_chs = np.argwhere(outlier_map.reshape(AP, ML))
    bad_chs_df = pd.DataFrame(bad_chs, columns=["AP", "ML"])
    return bad_chs_df


def bad_score_map(feature_ds):
    AP = feature_ds.sizes["AP"]
    ML = feature_ds.sizes["ML"]

    qfeat = feature_ds.apply(compute_quantile_features, qmin=0.8, qmax=0.95, nq=5)
    qfeat_arr = qfeat.to_array("feature")
    qfeat_stacked = qfeat_arr.stack(qfeat=["feature", "quantile"]).transpose(
        "ch", "qfeat"
    )
    devz = deviation_zscore(qfeat_stacked.data)
    return xr.DataArray(devz.reshape(AP, ML), dims=["AP", "ML"], name="bad_score")


def rotate_labels(plot, element):
    ax = plot.handles["axis"]
    ax.set_xlabel(ax.get_xlabel(), rotation=45, labelpad=20)
    ax.set_ylabel(ax.get_ylabel(), rotation=45, labelpad=20)


def multiline_labels(plot, element):
    ax = plot.handles["axis"]
    ax.set_xlabel(ax.get_xlabel().replace("_", "\n"))
    ax.set_ylabel(ax.get_ylabel().replace("_", "\n"))


def pairplot(feature_ds, outlier_map):
    df = feature_df(feature_ds, outlier_map)
    hv_ds = hv.Dataset(df)
    grouped_by_bad = hv_ds.groupby("is_bad", container_type=hv.NdOverlay)
    density_grid = gridmatrix(
        grouped_by_bad, diagonal_type=hv.Distribution, chart_type=hv.Scatter
    ).opts(
        opts.Scatter(hooks=[rotate_labels, multiline_labels]),  # axis title rotation
        opts.Distribution(
            hooks=[rotate_labels, multiline_labels]  # axis title rotation
        ),
    )
    return density_grid.opts(title="Feature Pairplot Colored by Bad Channels")


def plot_feature_maps(feature_ds, outlier_map):
    AP = feature_ds.sizes["AP"]
    ML = feature_ds.sizes["ML"]
    bad_chs_df = get_bad_chs_df(outlier_map, AP, ML)
    feature_maps = feature_ds.quantile(0.9, dim=["time"])
    badxmap = bad_score_map(feature_ds)

    ds = hv.Dataset(feature_maps)
    opts.defaults(opts.Image(cmap="viridis", colorbar=True, aspect=1))
    hmap = ds.to(hv.Image, ["ML", "AP"], groupby="feature")
    points_bad_chs = hv.Points(bad_chs_df, ["ML", "AP"]).opts(
        opts.Points(color="red", s=12)
    )
    bad_image = hv.Image(badxmap, ["ML", "AP"]).opts(
        title="Aggregate: Bad Score", fontsize={"title": 16}
    )

    all_images = [
        hv.Image(ds, ["ML", "AP"], vdims=feature_name).opts(
            title=feature_name.replace("_", " ")
        )
        for feature_name in feature_maps.data_vars
    ] + [bad_image]

    layout = hv.Layout(all_images) * points_bad_chs
    layout.opts(title="Feature Maps with Detected Bad Channels")
    return layout


def main(input_feature, input_badlabel, output_featuremap, output_pairplot):
    # load feature_ds
    feature_ds = xr.open_zarr(input_feature)
    # outlier_map
    outlier_map = np.load(input_badlabel)
    layout = plot_feature_maps(feature_ds, outlier_map)
    hv.save(layout, output_featuremap, fmt="png", dpi=150)
    density_grid = pairplot(feature_ds, outlier_map)
    hv.save(density_grid, output_pairplot, fmt="png", dpi=150)
    print(f"plots saved:\n\t-{output_featuremap}\n\t-{output_pairplot}")


if __name__ == "__main__":
    # snakemake
    if "snakemake" in globals():
        snakemake = globals()["snakemake"]

        # io
        input_feature = snakemake.input.feature
        input_badlabel = snakemake.input.badlabel
        output_pairplot = snakemake.output.pairplot
        output_featuremap = snakemake.output.featuremap

        # main
        main(input_feature, input_badlabel, output_featuremap, output_pairplot)

    else:
        raise RuntimeError("This script is intended to be run via Snakemake.")
