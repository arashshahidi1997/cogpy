"""Tests for resolve_channel_count — BIDS-iEEG standard + legacy fallback."""

import pytest

from cogpy.io.sidecars import resolve_channel_count


def test_bids_standard_ecog_only():
    meta = {"ECOGChannelCount": 64}
    assert resolve_channel_count(meta) == 64


def test_bids_standard_summed_modalities():
    meta = {
        "ECOGChannelCount": 128,
        "SEEGChannelCount": 16,
        "TriggerChannelCount": 2,
        "MiscChannelCount": 4,
    }
    assert resolve_channel_count(meta) == 150


def test_legacy_channel_count_fallback():
    meta = {"ChannelCount": 32}
    assert resolve_channel_count(meta) == 32


def test_bids_standard_takes_precedence_over_legacy():
    meta = {"ECOGChannelCount": 64, "ChannelCount": 999}
    assert resolve_channel_count(meta) == 64


def test_missing_raises_keyerror():
    with pytest.raises(KeyError):
        resolve_channel_count({"SamplingFrequency": 1000.0})
