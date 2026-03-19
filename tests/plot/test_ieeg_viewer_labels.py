import warnings


def test_make_channel_labels_apml_when_ints_and_nml():
    from cogpy.plot.hv.ieeg_viewer import _make_channel_labels

    labels = _make_channel_labels(list(range(8)), n_ml=4)
    assert labels[0] == "(0,0)"
    assert labels[3] == "(3,0)"
    assert labels[4] == "(0,1)"


def test_make_channel_labels_fallback_warns_on_non_int_values():
    from cogpy.plot.hv.ieeg_viewer import _make_channel_labels

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        labels = _make_channel_labels(["a", "b", "c"], n_ml=4)
        assert labels == ["a", "b", "c"]
        assert any("falling back" in str(wi.message).lower() for wi in w)
