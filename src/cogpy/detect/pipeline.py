"""DetectionPipeline: compose transforms + detector (v2.6.5)."""

from __future__ import annotations

from typing import Any

import xarray as xr

__all__ = ["DetectionPipeline"]


class DetectionPipeline:
    """Chain transforms and an EventDetector into a reproducible workflow."""

    def __init__(
        self,
        *,
        transforms: list[Any] | None = None,
        detector: Any | None = None,
        name: str | None = None,
    ) -> None:
        self.transforms = list(transforms or [])
        self.detector = detector
        self.name = str(name or (f"{getattr(detector, 'name', 'detector')}_pipeline"))

        if not self.transforms and self.detector is None:
            raise ValueError("Pipeline must have transforms and/or a detector")

    def run(self, data: xr.DataArray):
        """Run transforms then detector. Returns EventCatalog (if detector set) else transformed data."""
        x = data
        for tr in self.transforms:
            x = tr.compute(x)

        if self.detector is None:
            return x

        catalog = self.detector.detect(x)
        # Attach pipeline metadata (non-breaking additive).
        try:
            md = getattr(catalog, "metadata", None)
            if isinstance(md, dict):
                md["pipeline"] = self.name
                md["transforms"] = [t.to_dict() for t in self.transforms]
                md["detector"] = getattr(
                    self.detector, "name", self.detector.__class__.__name__
                )
        except Exception:  # noqa: BLE001
            pass
        return catalog

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "transforms": [t.to_dict() for t in self.transforms],
        }
        if self.detector is not None:
            d["detector"] = self.detector.to_dict()
        return d

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "DetectionPipeline":
        from cogpy.detect import get_detector_class
        from cogpy.detect.transforms import get_transform_class

        transforms_cfg = (config or {}).get("transforms") or []
        transforms = []
        for td in transforms_cfg:
            if not isinstance(td, dict):
                continue
            tname = td.get("transform")
            if not tname:
                continue
            tcls = get_transform_class(str(tname))
            transforms.append(tcls.from_dict(td))

        detector = None
        det_cfg = (config or {}).get("detector")
        if isinstance(det_cfg, dict):
            det_name = det_cfg.get("detector")
            if det_name:
                dcls = get_detector_class(str(det_name))
                detector = dcls.from_dict(det_cfg)

        return cls(
            transforms=transforms, detector=detector, name=(config or {}).get("name")
        )

    def __repr__(self) -> str:
        parts = [str(t) for t in self.transforms]
        det = str(self.detector) if self.detector is not None else "None"
        chain = " -> ".join(parts) if parts else "(no transforms)"
        return f"DetectionPipeline(name={self.name!r}, chain={chain} -> {det})"
