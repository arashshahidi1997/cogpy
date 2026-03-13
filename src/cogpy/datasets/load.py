"""Load bundled sample ECoG recordings for quick testing and demos."""
from pathlib import Path
from ..io import ecog_io

CACHED_DATA_DIR = Path(__file__).parent / "cached_data"


def load_sample():
    """Load the preprocessed sample signal as an ``xarray.DataArray``.

    Returns
    -------
    xarray.DataArray
        Grid ECoG signal with dims ``("AP", "ML", "time")``.
    """
    load_sample_file = CACHED_DATA_DIR / "sample_signal.dat"
    sigx = ecog_io.from_file(load_sample_file)
    # swap AP and ML dimensions names
    # sigx = sigx.rename({'AP': 'ML', 'ML': 'AP'})
    # transpose to (AP, ML, time)
    sigx = sigx.transpose("AP", "ML", "time")
    sigx.name = "signal"
    return sigx


def load_raw_sample():
    """Load the raw (unprocessed) sample signal as an ``xarray.DataArray``."""
    load_raw_sample_file = CACHED_DATA_DIR / "sample_raw_signal.dat"
    sigx = ecog_io.from_file(load_raw_sample_file)
    sigx.name = "raw_signal"
    return sigx
