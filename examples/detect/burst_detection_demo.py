"""
BurstDetector demo (v2.6.1).

Run with:
    conda run -n cogpy python code/lib/cogpy/examples/detect/burst_detection_demo.py
"""

from __future__ import annotations

import json
from pathlib import Path

from cogpy.detect import BurstDetector
from cogpy.spectral.specx import spectrogramx
from cogpy.datasets.entities import example_ieeg_grid

print("=" * 60)
print("BurstDetector Demo")
print("=" * 60)

data = example_ieeg_grid(mode="small")
print("\nLoaded data:")
print("  shape:", data.shape)
print("  dims :", data.dims)

print("\nExample 1: implicit mode (detector computes spectrogram)")
det1 = BurstDetector(h_quantile=0.9, nperseg=256, noverlap=128, bandwidth=4.0)
cat1 = det1.detect(data)
print("  n events:", len(cat1))
print("  computed_spectrogram:", cat1.metadata.get("computed_spectrogram"))

print("\nExample 2: explicit mode (precomputed spectrogram)")
spec = spectrogramx(data, nperseg=256, noverlap=128, bandwidth=4.0, axis="time")
det2 = BurstDetector(h_quantile=0.9)
cat2 = det2.detect(spec)
print("  n events:", len(cat2))
print("  computed_spectrogram:", cat2.metadata.get("computed_spectrogram"))

print("\nExample 3: save/load detector config")
cfg_path = Path("burst_detector.json")
cfg_path.write_text(json.dumps(det2.to_dict(), indent=2), encoding="utf-8")
det3 = BurstDetector.from_dict(json.loads(cfg_path.read_text(encoding="utf-8")))
cfg_path.unlink()
print("  roundtrip:", det3.to_dict())

