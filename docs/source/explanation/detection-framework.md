# Detection Framework

This page explains the design of cogpy's event detection system: why it uses
an abstract detector interface, how pipelines compose transforms and detectors,
and what trade-offs shaped the architecture.

## Design goals

ECoG analysis involves detecting many kinds of transient events — ripples,
spindles, gamma bursts, epileptiform discharges. Each has different signal
characteristics, but the workflow is always the same:

1. **Transform** the signal (filter, envelope, spectrogram)
2. **Detect** events in the transformed signal
3. **Collect** results into a uniform structure

The detection framework codifies this pattern so that detectors are
interchangeable and pipelines are reproducible.

## The EventDetector interface

All detectors inherit from `EventDetector` (in `cogpy.core.detect.base`):

```python
class EventDetector(ABC):
    @abstractmethod
    def detect(self, data: xr.DataArray, **kwargs) -> EventCatalog: ...

    @abstractmethod
    def get_event_dims(self) -> list[str]: ...
```

**`detect()`** accepts any `xr.DataArray` and returns an `EventCatalog`.
The detector declares which dimensions it operates over via
**`get_event_dims()`** — for example, `["time"]` for point events or
`["time", "freq", "AP", "ML"]` for spatiotemporal bursts.

Additional hooks:

| Method | Purpose |
|--------|---------|
| `can_accept(data)` | Input validation (default: `True`) |
| `needs_transform(data)` | Whether the detector requires preprocessing |
| `to_dict()` / `from_dict()` | Serialization for reproducibility |

**Why an ABC?** Detectors have fundamentally different algorithms (threshold
crossing vs. spectrogram peak finding vs. template matching). An ABC enforces
a common interface without constraining implementation.

## Built-in detectors

| Detector | Strategy | Event type | Key parameters |
|----------|----------|------------|----------------|
| `BurstDetector` | h-maxima on spectrogram | Point events | `h_quantile`, `bandwidth`, `footprint` |
| `ThresholdDetector` | Amplitude threshold crossing | Interval events | `threshold`, `direction`, `min_duration` |
| `RippleDetector` | Bandpass → envelope → dual threshold | Interval events | `freq_range`, `threshold_low/high` |
| `SpindleDetector` | Same as ripple, different defaults | Interval events | `freq_range=(11, 16)` |

**Point events** have a single time `t`. **Interval events** have `t0`, `t`,
`t1` (onset, peak, offset) plus `duration`.

## Transforms

Transforms are composable preprocessing steps that sit between raw signal
and detector:

```python
class Transform(ABC):
    @abstractmethod
    def compute(self, data: xr.DataArray) -> xr.DataArray: ...
```

Built-in transforms:

- `BandpassTransform`, `HighpassTransform`, `LowpassTransform` — frequency filtering
- `HilbertTransform` — analytic signal envelope
- `ZScoreTransform` — per-channel z-score normalization
- `SpectrogramTransform` — multitaper spectrogram

Transforms are deliberately simple — each wraps a single `cogpy.core` function.
This keeps them testable independently and avoids coupling detection logic to
specific filtering implementations.

## DetectionPipeline

`DetectionPipeline` chains transforms and a detector into a reproducible unit:

```python
pipeline = DetectionPipeline(
    transforms=[BandpassTransform(100, 250), HilbertTransform(), ZScoreTransform()],
    detector=ThresholdDetector(threshold=3.0, min_duration=0.02),
    name="ripple_pipeline",
)
catalog = pipeline.run(raw_signal)
```

The pipeline:
1. Applies each transform in sequence
2. Passes the result to the detector
3. Returns an `EventCatalog` with pipeline metadata attached

**Serialization:** `pipeline.to_dict()` captures the full configuration
(transform params, detector params) so that detection can be reproduced
from a config file.

## Pre-built pipelines

cogpy ships four ready-to-use pipelines:

| Pipeline | Target event | Frequency band | Threshold |
|----------|-------------|----------------|-----------|
| `BURST_PIPELINE` | Broadband bursts | Full spectrogram | h-maxima (90th percentile) |
| `RIPPLE_PIPELINE` | Sharp-wave ripples | 100–250 Hz | 3σ, ≥20 ms |
| `FAST_RIPPLE_PIPELINE` | Fast ripples | 250–500 Hz | 3σ, ≥10 ms |
| `GAMMA_BURST_PIPELINE` | Gamma bursts | 30–80 Hz | 2.5σ, ≥50 ms |

These are importable directly:

```python
from cogpy.detect import RIPPLE_PIPELINE
catalog = RIPPLE_PIPELINE.run(signal)
```

## EventCatalog

All detectors return `EventCatalog`, a thin pandas DataFrame wrapper
(see {doc}`/explanation/data-model` for column definitions).

Key design choices:

- **Minimal required schema:** Only `event_id` and `t` are mandatory.
  Everything else (intervals, spatial coords, labels) is optional. This
  keeps the catalog flexible across event types.

- **Validation on construction:** The catalog checks required columns,
  interval constraints (`t1 > t0`), and sorts by time.

- **Factory methods:** `from_hmaxima()`, `from_blob_candidates()`,
  `from_burst_dict()`, `from_spwr_mat()` convert legacy formats.

- **Query interface:** `filter_by_time()`, `filter_by_channel()`,
  `filter_by_spatial()` for subsetting.

## Why separate transforms from detectors?

An alternative design would have each detector handle its own preprocessing
internally (and `BurstDetector` does support this "implicit" mode). The
explicit transform pipeline is preferred because:

1. **Reuse.** The same bandpass + envelope chain serves ripple, spindle,
   and gamma detection — only parameters differ.
2. **Inspection.** Intermediate signals (filtered, enveloped) can be
   examined for debugging without re-running detection.
3. **Composition.** Users can insert custom transforms (e.g., spatial
   whitening) without modifying detector code.

The trade-off is slightly more verbose setup for simple cases. The pre-built
pipelines mitigate this.
