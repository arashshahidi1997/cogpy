# TensorScope Examples

This directory contains examples demonstrating TensorScope features by phase.

## Phase 0: Foundation

**hello_tensorscope.py** - Minimal Panel demo with placeholders
```bash
panel serve code/lib/cogpy/examples/tensorscope/hello_tensorscope.py --show
```

Shows the UI structure with placeholder content.

---

## Phase 1: State Architecture

### Quick Tests

**test_phase1_state.py** - Comprehensive state feature test
```bash
conda run -n cogpy python code/lib/cogpy/examples/tensorscope/test_phase1_state.py
```

Tests all Phase 1 features:
- State creation
- Controller ownership and delegation
- Time window management
- Channel selection
- Serialization/restoration
- Data registry

**test_session_persistence.py** - Session save/load demo
```bash
conda run -n cogpy python code/lib/cogpy/examples/tensorscope/test_session_persistence.py
```

Demonstrates saving state to JSON and restoring it.

### Interactive

**phase1_interactive.py** - Notebook-style Panel script
```bash
conda run -n cogpy panel serve code/lib/cogpy/examples/tensorscope/phase1_interactive.py --show
```

Interactive exploration of Phase 1 state with widgets.

**hello_tensorscope.py** (updated) - Panel demo with live state
```bash
panel serve code/lib/cogpy/examples/tensorscope/hello_tensorscope.py --show
```

Updated to show live Phase 1 state in sidebar.

---

## Phase 2+: Coming Soon

Check back after Phase 2 implementation for layer examples!

---

## Phase 2: Core Layers

### Quick Tests

**test_phase2_layers.py** - Layer creation and lifecycle test
```bash
conda run -n cogpy python code/lib/cogpy/examples/tensorscope/test_phase2_layers.py
```

Tests:
- Creating all 5 core layers
- Layer panel rendering
- State changes triggering updates
- Layer disposal and cleanup

### Interactive Demos

**phase2_interactive.py** - Full interactive visualization
```bash
conda run -n cogpy panel serve code/lib/cogpy/examples/tensorscope/phase2_interactive.py --show
```

**Features:**
- Real timeseries with MultichannelViewer
- Real spatial map with GridFrameElement
- Interactive channel selection updates traces
- Processing controls update both views
- Time navigator with play/pause

**hello_tensorscope.py** (updated) - Now with real data
```bash
conda run -n cogpy panel serve code/lib/cogpy/examples/tensorscope/hello_tensorscope.py --show
```

Placeholder cards replaced with actual visualization layers.

---

## Phase 3: Application Shell

### Quick Tests

**test_phase3_app.py** - Application shell test
```bash
conda run -n cogpy python code/lib/cogpy/examples/tensorscope/test_phase3_app.py
```

Tests:
- TensorScopeApp creation
- Builder API
- Layer management
- Layout presets
- Session save/load
- App shutdown

### Interactive Apps

**phase3_app.py** - Complete TensorScope application
```bash
conda run -n cogpy panel serve code/lib/cogpy/examples/tensorscope/phase3_app.py --show
```

**Features:**
- Full application shell with all layers
- Layout presets (default, spatial_focus, timeseries_focus)
- Integrated state management
- Session persistence

**hello_tensorscope.py** (updated) - Now uses TensorScopeApp
```bash
conda run -n cogpy panel serve code/lib/cogpy/examples/tensorscope/hello_tensorscope.py --show
```

Simplified to use TensorScopeApp builder API.

---

## Phase 4: Events System

### Quick Tests

**test_phase4_events.py** - Event system test
```bash
conda run -n cogpy python code/lib/cogpy/examples/tensorscope/test_phase4_events.py
```

Tests:
- Creating EventStream from DataFrame
- Event queries (window, next, prev)
- Event registration
- EventTableLayer
- Navigation (jump to event, next/prev)
- Serialization

### Interactive Demos

**phase4_events.py** - Interactive event browser
```bash
conda run -n cogpy panel serve code/lib/cogpy/examples/tensorscope/phase4_events.py --show
```

**Features:**
- Event table with 30 synthetic events
- Click row → jump to event time
- Prev/Next navigation buttons
- Real-time cursor sync with events
- Event metadata display

---

## Phase 5: Multi-Modal Support

### Quick Tests

**test_phase5_multimodal.py** - Multi-modal system test
```bash
conda run -n cogpy python code/lib/cogpy/examples/tensorscope/test_phase5_multimodal.py
```

Tests:
- Creating multiple modalities (LFP, spectrogram)
- Registering with data registry
- Switching active modality
- Time alignment utilities
- Windowing across different sampling rates
- Modality conversion (grid→flat)

### Interactive Demos

**phase5_multimodal.py** - Multi-modal browser
```bash
conda run -n cogpy panel serve code/lib/cogpy/examples/tensorscope/phase5_multimodal.py --show
```

---

## Phase 6: Polish & Optimization

### Performance Benchmarks

**benchmarks.py** - performance regression tests (run explicitly)
```bash
conda run -n cogpy python -m pytest code/lib/cogpy/tests/core/plot/tensorscope/benchmarks.py -m benchmark -v -s
```

### Complete System Test

**test_phase6_complete.py** - end-to-end integration smoke test
```bash
conda run -n cogpy python code/lib/cogpy/examples/tensorscope/test_phase6_complete.py
```

### CLI

After installing `cogpy`, use:
```bash
tensorscope presets
tensorscope serve data.nc --layout default --port 5008 --show
```

---

## Requirements

All examples require:
```bash
conda activate cogpy
```

Panel examples additionally require:
```bash
pip install panel>=1.8.8
```

---

## Development

When adding new examples:
1. Follow existing naming: `test_phaseN_feature.py` for tests
2. Add to this README with description and run command
3. Include expected output in docstring
4. Test in clean environment before committing
