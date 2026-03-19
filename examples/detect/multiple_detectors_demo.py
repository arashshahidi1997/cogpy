"""
Multiple Detectors Demo (v2.6.4).

Runs:
- ThresholdDetector on a single trace
- RippleDetector on a single trace
- BurstDetector optionally (requires spectral dependencies)
"""

from __future__ import annotations

import numpy as np

from cogpy.detect import RippleDetector, ThresholdDetector
from cogpy.datasets.entities import example_ieeg_grid


def main():
    data = example_ieeg_grid(mode="small")
    print(f"Data dims={data.dims}, shape={data.shape}")

    # Pick a single electrode trace.
    trace = data.isel(AP=int(data.sizes["AP"]) // 2, ML=int(data.sizes["ML"]) // 2)
    trace = trace.assign_attrs(fs=float(data.attrs.get("fs", 1000.0)))

    print("\nThresholdDetector:")
    thr = ThresholdDetector(threshold=2.5, direction="both", min_duration=0.01)
    thr_cat = thr.detect(trace)
    print(f"  n_events={len(thr_cat)}")

    print("\nRippleDetector:")
    rip = RippleDetector(freq_range=(100, 250), threshold_low=2.0, threshold_high=3.0)
    rip_cat = rip.detect(trace)
    print(f"  n_events={len(rip_cat)}")
    if len(rip_cat):
        print(f"  duration mean={rip_cat.df['duration'].mean():.3f}s")

    print("\nBurstDetector (optional):")
    try:
        from cogpy.detect import BurstDetector

        det = BurstDetector(h_quantile=0.9, nperseg=256, noverlap=128)
        cat = det.detect(data)
        print(f"  n_events={len(cat)}")
    except Exception as e:  # noqa: BLE001
        print(f"  skipped: {e}")


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    main()
