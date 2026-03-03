"""
TensorScope events model.

EventStream is the unified representation for time-indexed events displayed as:
- tables (Tabulator)
- timeline overlays
- navigation targets (jump-to next/prev)

Phase 0: Stub only.
Phase 4+: Full event model + registry + overlays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class EventStream:
    """
    Container for a stream of events.

    Phase 0 is intentionally minimal: it captures a name and an opaque payload.
    Phase 4 will define a schema (point events vs intervals, required columns,
    label fields, optional spatial metadata).
    """

    name: str
    payload: Any

    def to_dict(self) -> dict[str, Any]:
        """Serialize this event stream (Phase 0: best-effort)."""

        return {"name": self.name}

