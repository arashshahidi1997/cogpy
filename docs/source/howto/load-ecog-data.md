# How to load ECoG data

## From binary LFP files

```python
from cogpy.io.ecog_io import from_file

sig = from_file("/path/to/recording.lfp")
# Returns xarray.DataArray with dims (time, AP, ML) and fs coordinate
```

## From BIDS-iEEG format

```python
from cogpy.io.ieeg_io import from_file

sig = from_file("/path/to/sub-01/ieeg/sub-01_task-rest_ieeg.edf")
```

## From Zarr archives

```python
import xarray as xr

sig = xr.open_dataarray("/path/to/signal.zarr", engine="zarr")
```

## Sample data (for testing)

```python
import cogpy

# Grid ECoG sample
sig = cogpy.datasets.load_sample()

# Raw sample (before preprocessing)
raw = cogpy.datasets.load_raw_sample()

# Synthetic bundles for GUI development
from cogpy.datasets import ieeg_grid_bundle, spectrogram_bursts_bundle
bundle = ieeg_grid_bundle(mode="small", seed=42)
```

## Verifying the schema

After loading, verify the signal has the expected schema:

```python
from cogpy.base import ensure_fs, SCHEMA

# Ensure fs is set (raises if missing)
sig = ensure_fs(sig, fs=1000.0)

# Check dimensions
assert set(sig.dims) >= {SCHEMA.time, SCHEMA.ap, SCHEMA.ml}
```
