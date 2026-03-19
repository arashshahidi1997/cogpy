# Detection and Events

This tutorial walks through the full event lifecycle in cogpy: detecting
events, exploring event catalogs, matching events across signals, and
building custom detection pipelines.

For quick reference on available detectors and their parameters, see the
{doc}`/howto/event-detection` guide.

## Load sample data

```python
from cogpy.datasets.entities import example_ieeg_grid

data = example_ieeg_grid(mode="small")
print(f"dims={data.dims}, shape={data.shape}")

# Pick a single electrode trace for 1D detection
trace = data.isel(AP=data.sizes["AP"] // 2, ML=data.sizes["ML"] // 2)
trace = trace.assign_attrs(fs=float(data.attrs.get("fs", 1000.0)))
```

## 1. Threshold detection

The simplest detector: find where a signal crosses a threshold.

```python
from cogpy.detect import ThresholdDetector

detector = ThresholdDetector(
    threshold=2.5,
    direction="both",       # "positive", "negative", or "both"
    min_duration=0.01,       # reject events shorter than 10 ms
    merge_gap=0.005,         # merge events less than 5 ms apart
)
catalog = detector.detect(trace)
print(f"Found {len(catalog)} events")
print(catalog.df.head())
```

The result is an `EventCatalog` — a thin pandas DataFrame wrapper with
standardized columns (`event_id`, `t`, `t0`, `t1`, `duration`).

## 2. Specialized detectors

### Ripple detector

Bandpass → Hilbert envelope → dual threshold. Events must cross the high
threshold and extend to the low threshold boundary.

```python
from cogpy.detect import RippleDetector

ripple_det = RippleDetector(
    freq_range=(100, 250),
    threshold_low=2.0,
    threshold_high=3.0,
    min_duration=0.02,
    max_duration=0.2,
)
ripple_cat = ripple_det.detect(trace)
print(f"Ripples: {len(ripple_cat)}")
if len(ripple_cat):
    print(f"Mean duration: {ripple_cat.df['duration'].mean():.3f} s")
```

### Burst detector

Detects spectral peaks via morphological h-maxima transform. Works on
spectrograms (explicit mode) or raw signals (implicit mode, computes
spectrogram internally).

```python
from cogpy.detect import BurstDetector
from cogpy.spectral.specx import spectrogramx

# Implicit mode: detector computes spectrogram
det_implicit = BurstDetector(h_quantile=0.9, nperseg=256, noverlap=128, bandwidth=4.0)
cat_implicit = det_implicit.detect(data)
print(f"Implicit mode: {len(cat_implicit)} events")

# Explicit mode: pass precomputed spectrogram
spec = spectrogramx(data, nperseg=256, noverlap=128, bandwidth=4.0, axis="time")
det_explicit = BurstDetector(h_quantile=0.9)
cat_explicit = det_explicit.detect(spec)
print(f"Explicit mode: {len(cat_explicit)} events")
```

## 3. EventCatalog

All detectors return an `EventCatalog`. You can also construct one directly.

### Point events

```python
import numpy as np
import pandas as pd
from cogpy.events import EventCatalog

peaks_df = pd.DataFrame({
    "t": [1.2, 2.5, 3.8, 5.1, 6.7],
    "AP": [2, 3, 4, 3, 2],
    "ML": [5, 6, 5, 4, 5],
    "freq": [40.0, 45.0, 38.5, 42.0, 41.5],
    "amp": [12.3, 15.1, 10.2, 14.5, 13.8],
})
catalog = EventCatalog.from_hmaxima(peaks_df, label="burst_peak")
print(f"Point events: {len(catalog)}")
print(f"Columns: {list(catalog.df.columns)}")
```

### Interval events

```python
blob_dict = {
    "t_peak_s": np.array([1.5, 3.2, 5.8]),
    "t0_s": np.array([1.2, 2.9, 5.5]),
    "t1_s": np.array([1.8, 3.5, 6.1]),
    "f_peak_hz": np.array([40.0, 45.0, 42.5]),
    "f0_hz": np.array([35.0, 40.0, 38.0]),
    "f1_hz": np.array([45.0, 50.0, 47.0]),
}
interval_cat = EventCatalog.from_blob_candidates(blob_dict)
print(f"Interval events: {len(interval_cat)}")
print(f"Mean duration: {interval_cat.df['duration'].mean():.2f} s")
```

### Filtering

```python
# Filter by time range
subset = catalog.filter_by_time(2.0, 5.5)
print(f"Events in [2, 5.5]: {len(subset)}")

# Filter by spatial location
nearby = catalog.filter_by_spatial(AP=3, ML=5, radius=1.5)
print(f"Events near (3, 5): {len(nearby)}")
```

### Format conversion

```python
# Convert to schema objects for downstream use
events = catalog.to_events()           # Events schema
intervals = catalog.to_point_intervals(half_window=0.05)  # Intervals around points
```

## 4. Event matching

Match events detected in two different signals (e.g. TTL pulses on two
recording systems).

```python
from cogpy.events.match import (
    match_nearest_symmetric,
    estimate_lag,
    event_lag_histogram,
)

# Simulate two event trains with a small lag
rng = np.random.default_rng(42)
times_a = np.sort(rng.uniform(0, 100, 50))
times_b = times_a + 0.003 + rng.normal(0, 0.001, 50)  # 3 ms lag + jitter

# One-to-one matching
idx_a, idx_b, lags = match_nearest_symmetric(times_a, times_b, max_lag=0.05)
print(f"Matched {len(idx_a)} / {len(times_a)} events")
print(f"Mean lag: {np.mean(lags)*1000:.1f} ms")

# Global lag estimate
lag = estimate_lag(times_a, times_b, max_lag=0.05, method="median")
print(f"Estimated lag: {lag*1000:.1f} ms")

# Cross-correlogram
counts, bin_edges = event_lag_histogram(times_a, times_b, max_lag=0.02, bin_width=0.001)
```

## 5. Detection pipelines

Chain transforms and a detector into a reproducible, serializable pipeline.

### Pre-built pipelines

```python
from cogpy.detect.pipelines import GAMMA_BURST_PIPELINE

cat = GAMMA_BURST_PIPELINE.run(trace)
print(f"Gamma bursts: {len(cat)}")
print(f"Pipeline config: {cat.metadata.get('pipeline')}")
```

### Custom pipelines

```python
from cogpy.detect import DetectionPipeline, ThresholdDetector
from cogpy.detect.transforms import BandpassTransform, HilbertTransform, ZScoreTransform

custom = DetectionPipeline(
    name="custom_beta",
    transforms=[
        BandpassTransform(low=13, high=30),
        HilbertTransform(),
        ZScoreTransform(),
    ],
    detector=ThresholdDetector(threshold=2.5, direction="positive", min_duration=0.05),
)
cat = custom.run(trace)
print(f"Custom pipeline: {len(cat)} events")
```

### Serialization

Pipelines can be saved and loaded as JSON for reproducibility:

```python
import json

config = custom.to_dict()
print(json.dumps(config, indent=2)[:300], "...")

# Reconstruct from config
loaded = DetectionPipeline.from_dict(config)
cat2 = loaded.run(trace)
print(f"Reloaded pipeline: {len(cat2)} events")
```

## Next steps

- {doc}`/howto/event-detection` — quick reference for all detectors and pipelines
- {doc}`/howto/compose-artifact-analysis` — using events with triggered analysis
  and template subtraction
- {doc}`/explanation/primitives` — full catalog of available primitives
