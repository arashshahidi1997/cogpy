# TensorScope v2.2 Specification: View Composer Architecture

## Version
- Version: 2.2.0
- Date: 2026-03-04
- Status: Implementation

## Overview

TensorScope v2.2 introduces a declarative view composition system that allows users to build custom visualizations from simple specifications. The system targets HoloViews as the rendering backend and builds reactive plots from lightweight, serializable `ViewSpec` objects.

This is an additive layer on top of the existing v2.1 controllers:
- `TensorScopeState` (authoritative state + controllers)
- `SignalRegistry` / `SignalObject` (signal management + duplication)
- `CoordinateSpace` (linked spatial selection)
- `TimeHair` (cursor time)

## Core Concepts

### 1. `ViewSpec` (Declarative Specification)

A `ViewSpec` describes *what* to display (dims), *what* is controlled by widgets/streams, and optional styling/transform settings:

```python
from dataclasses import dataclass, field
from typing import Callable

@dataclass
class ViewSpec:
    # What to display
    kdims: list[str]                 # e.g. ['ML', 'AP'] or ['time']

    # What to control
    controls: list[str]              # e.g. ['time'] or ['AP', 'ML']

    # Optional iteration (for layouts/animations)
    iterate: list[str] = field(default_factory=list)

    # Which signal to visualize (None = active signal)
    signal_id: str | None = None

    # Display settings
    view_type: str = "auto"          # 'auto', 'Image', 'Curve', ...
    colormap: str = "viridis"
    title: str | None = None

    # Advanced
    operation: Callable | None = None
    clim: tuple[float, float] | None = None
    symmetric_clim: bool = False
```

**Examples**

```python
# Spatial snapshot with time slider
ViewSpec(kdims=["ML", "AP"], controls=["time"], view_type="auto")

# Timeseries with spatial selector
ViewSpec(kdims=["time"], controls=["AP", "ML"], view_type="auto")

# Multi-signal comparison (layout iteration)
ViewSpec(kdims=["ML", "AP"], controls=["time"], iterate=["signal"], view_type="Image")
```

### 2. `ViewFactory` (Spec → HoloViews)

`ViewFactory` converts `ViewSpec` into HoloViews objects (typically `hv.DynamicMap`) by:
1. Resolving the `SignalObject` to render (explicit `signal_id` or active signal)
2. Inferring the view type if `view_type='auto'`
3. Creating HoloViews streams for `controls` and wiring them to `TensorScopeState`
4. Rendering an element (`hv.Image`, `hv.Curve`, …) in a DynamicMap callback

```python
class ViewFactory:
    @staticmethod
    def create(spec: ViewSpec, state: TensorScopeState) -> hv.DynamicMap:
        ...
```

**View Type Inference**

```python
def infer_view_type(kdims: list[str]) -> str:
    # Minimal first-pass rules (v2.2.0):
    # - 2 kdims -> Image
    # - 1 kdim  -> Curve
    # - else    -> generic placeholder
    if len(kdims) == 2:
        return "Image"
    if len(kdims) == 1:
        return "Curve"
    return "Generic"
```

### 3. `ViewPresetModule` (Collections)

Modules are named collections of `ViewSpec`s with a layout policy.

```python
from dataclasses import dataclass

@dataclass
class ViewPresetModule:
    name: str
    description: str
    specs: list[ViewSpec]
    layout: str = "grid"  # 'grid', 'stack', 'tabs' (backend-dependent)

    def activate(self, state: TensorScopeState):
        views = [ViewFactory.create(spec, state) for spec in self.specs]
        return arrange(views, layout=self.layout)
```

### 4. `ModuleRegistry`

Central registry of built-in modules (and user-registered custom ones):

- `basic`: spatial + temporal (linked by `TensorScopeState`)
- `comparison`: multiple spatial views side-by-side (different `signal_id`s)

## Architecture Diagrams

### Data Flow

```text
User Action (click, slider)
    ↓
HoloViews Stream (Time, Spatial, Tap, ...)
    ↓
TensorScopeState (TimeHair / CoordinateSpace)
    ↓
DynamicMap callback triggered
    ↓
SignalObject.data slicing / transform (operation, clim)
    ↓
HoloViews Element (Image, Curve, ...)
    ↓
Display updates
```

### Component Relationships

```text
ViewSpec (declarative)
    ↓
ViewFactory.create()
    ↓
HoloViews DynamicMap (reactive)
    ↓
Streams (Time, Spatial) + Tap callbacks
    ↓
TensorScopeState / CoordinateSpace / SignalRegistry
```

## API Examples

### Creating Views Programmatically

```python
from cogpy.core.tensorscope import TensorScopeState
from cogpy.core.tensorscope.view_spec import ViewSpec
from cogpy.core.tensorscope.view_factory import ViewFactory

state = TensorScopeState(data)

spatial = ViewFactory.create(ViewSpec(kdims=["ML", "AP"], controls=["time"]), state)
temporal = ViewFactory.create(ViewSpec(kdims=["time"], controls=["AP", "ML"]), state)

layout = (spatial + temporal).cols(1)
```

### Using Preset Modules

```python
from cogpy.core.tensorscope.modules import ModuleRegistry

registry = ModuleRegistry()
basic = registry.get("basic")
layout = basic.activate(state)
```

## File Structure (v2.2.0)

```text
cogpy/core/plot/tensorscope/
├── view_spec.py
├── view_factory.py
├── modules/
│   ├── __init__.py
│   ├── base.py
│   ├── basic.py
│   └── comparison.py
```

## Success Criteria

v2.2.0 is complete when:
- `ViewSpec` can represent spatial and temporal views
- `ViewFactory` creates working `hv.DynamicMap` for Image + Curve cases
- Built-in modules exist and activate correctly (`basic`, `comparison`)
- Tests cover `ViewSpec`, `ViewFactory`, and module registry/activation
- Example demonstrates ViewSpec + modules working together

