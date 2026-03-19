"""
Detection Pipeline Demo (v2.6.5).

Demonstrates:
- using a pre-built pipeline
- defining a custom pipeline
- serializing and re-loading a pipeline config
"""

from __future__ import annotations

import json
import tempfile

from cogpy.detect import DetectionPipeline, ThresholdDetector
from cogpy.detect.pipelines import GAMMA_BURST_PIPELINE
from cogpy.detect.transforms import BandpassTransform, HilbertTransform, ZScoreTransform
from cogpy.datasets.entities import example_ieeg_grid


def main():
    data = example_ieeg_grid(mode="small")
    trace = data.isel(AP=int(data.sizes["AP"]) // 2, ML=int(data.sizes["ML"]) // 2)
    trace = trace.assign_attrs(fs=float(data.attrs.get("fs", 1000.0)))

    print("Pre-built pipeline:")
    print(" ", GAMMA_BURST_PIPELINE)
    cat0 = GAMMA_BURST_PIPELINE.run(trace)
    print(f"  events={len(cat0)} pipeline={cat0.metadata.get('pipeline')}")

    print("\nCustom pipeline:")
    custom = DetectionPipeline(
        name="custom_gamma_like",
        transforms=[BandpassTransform(low=30, high=80), HilbertTransform(), ZScoreTransform()],
        detector=ThresholdDetector(threshold=2.5, direction="positive", min_duration=0.05),
    )
    cat1 = custom.run(trace)
    print(f"  events={len(cat1)}")

    print("\nSerialize / load:")
    cfg = custom.to_dict()
    print(json.dumps(cfg, indent=2)[:400] + " ...")
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=True) as f:
        json.dump(cfg, f, indent=2)
        f.flush()
        f.seek(0)
        loaded = json.load(f)
    custom2 = DetectionPipeline.from_dict(loaded)
    cat2 = custom2.run(trace)
    print(f"  loaded events={len(cat2)}")


if __name__ == "__main__":
    main()

