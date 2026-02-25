import numpy as np
import pytest


def test_example_ieeg_grid_schema_and_determinism():
    from cogpy.datasets.entities import example_ieeg_grid
    from cogpy.datasets.schemas import validate_ieeg_grid

    a = example_ieeg_grid(mode="small", seed=0)
    b = example_ieeg_grid(mode="small", seed=0)

    validate_ieeg_grid(a)
    assert tuple(a.dims) == ("time", "ML", "AP")
    assert a.sizes["time"] == b.sizes["time"]
    np.testing.assert_allclose(a.values, b.values, rtol=0, atol=0)


def test_ieeg_grid_bundle_indexing_coherence():
    from cogpy.datasets.gui_bundles import ieeg_grid_bundle
    from cogpy.datasets.schemas import validate_ieeg_time_channel

    bundle = ieeg_grid_bundle(mode="small", seed=0)

    assert tuple(bundle.sig_grid.dims) == ("time", "ML", "AP")
    assert tuple(bundle.sig_apml.dims) == ("time", "AP", "ML")
    assert tuple(bundle.sig_tc.dims) == ("time", "channel")
    assert bundle.rms_apml.shape == (bundle.n_ap, bundle.n_ml)
    validate_ieeg_time_channel(bundle.sig_tc)

    # Check that a few (ap,ml) positions map to the same time series in stacked view.
    pairs = [(0, 0), (1, 2), (bundle.n_ap - 1, bundle.n_ml - 1)]
    for ap, ml in pairs:
        flat = ap * bundle.n_ml + ml
        y_grid = bundle.sig_apml.isel(AP=ap, ML=ml).values
        y_flat = bundle.sig_tc.isel(channel=flat).values
        np.testing.assert_allclose(y_grid, y_flat)


@pytest.mark.parametrize("kind", ["toy"])
def test_spectrogram_bursts_bundle_schema(kind):
    from cogpy.datasets.gui_bundles import spectrogram_bursts_bundle

    bundle = spectrogram_bursts_bundle(mode="small", seed=0, kind=kind)
    assert set(["burst_id", "x", "y", "t", "z", "value"]).issubset(set(bundle.bursts.columns))
    assert set(["ml", "ap", "time", "freq"]).issubset(set(bundle.spec.dims))
