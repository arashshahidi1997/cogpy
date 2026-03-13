# How to detect events

## Using pre-built pipelines

cogpy ships with pre-configured detection pipelines:

```python
from cogpy.detect import (
    BURST_PIPELINE,
    RIPPLE_PIPELINE,
    FAST_RIPPLE_PIPELINE,
    GAMMA_BURST_PIPELINE,
)

events = RIPPLE_PIPELINE.run(sig)
# Returns EventCatalog (pandas DataFrame wrapper)
```

Each pipeline chains transforms (bandpass, Hilbert envelope, z-score) with a
detector and records full provenance.

## Using individual detectors

```python
from cogpy.detect import BurstDetector, ThresholdDetector, RippleDetector

# Burst detection on spectrograms
detector = BurstDetector(h=0.5, min_area=4)
events = detector.detect(spectrogram)

# Threshold crossing on 1-D signal
detector = ThresholdDetector(threshold=3.0, min_duration=0.01)
events = detector.detect(envelope)

# Ripple detection (bandpass + envelope + dual threshold)
detector = RippleDetector(band=(80, 250), z_threshold=3.0)
events = detector.detect(sig)
```

## Building custom pipelines

```python
from cogpy.detect import DetectionPipeline
from cogpy.detect.transforms import (
    BandpassTransform,
    HilbertTransform,
    ZScoreTransform,
)

pipeline = DetectionPipeline(
    transforms=[
        BandpassTransform(l_freq=30.0, h_freq=100.0),
        HilbertTransform(),
        ZScoreTransform(),
    ],
    detector=ThresholdDetector(threshold=2.5, min_duration=0.02),
    name="custom_gamma",
)

events = pipeline.run(sig)
```

## Serializing pipelines

Pipelines are fully serializable for reproducibility:

```python
# Save
config = pipeline.to_dict()

# Load
pipeline2 = DetectionPipeline.from_dict(config)
```

## Working with EventCatalog

```python
# Filter events by time
events_subset = events.query("t > 1.0 and t < 5.0")

# Convert to intervals
intervals = events.to_intervals()  # DataFrame with t0, t1, duration

# Access metadata
print(events.detector)   # which detector produced these events
print(events.pipeline)   # full pipeline provenance
```
