# TensorScope Design Principles (Revised)

**Version 2.0 - Production Ready**

A comprehensive set of architectural principles to guide all implementation decisions, incorporating lessons from production data-intensive applications.

---

## 0. Preamble: Philosophy

TensorScope is built on these foundational beliefs:

1. **Clarity over cleverness** - Code should be obvious, not terse
2. **Composition over inheritance** - Small pieces, loosely joined
3. **Explicit over implicit** - No magic, no hidden dependencies
4. **Performance through architecture** - Fast by design, not by optimization
5. **Fail gracefully** - Errors happen; handle them with dignity

These principles are **guidelines, not laws**. Use judgment. Document exceptions.

---

## 1. Core Architectural Principles

### 1.1 Separation of Concerns

**State, Logic, and Presentation are Distinct**

```python
# ❌ BAD: State mixed with rendering
class BadViewer:
    def __init__(self):
        self.current_time = 0  # State
        self.plot = make_plot()  # Presentation
    
    def update(self):
        self.plot.data = compute(self.current_time)  # Logic + presentation

# ✅ GOOD: Clear separation
class TensorScopeState(param.Parameterized):
    current_time = param.Number()  # State only

class DataController:
    @staticmethod
    def get_window(state):  # Logic only
        return data[state.current_time - window:state.current_time + window]

class TimeseriesLayer(TensorLayer):
    def panel(self):  # Presentation only
        return pn.pane.HoloViews(self.plot)
```

**Three-layer architecture:**
1. **Model (State)**: What data exists and its current values
2. **Controller (Logic)**: How to transform/retrieve/compute data
3. **View (Presentation)**: How to display data

**Responsibilities:**
- **State**: Knows current values, emits change events, validates constraints
- **Controller**: Pure functions or stateless classes, no UI dependencies
- **View**: Renders state, emits user actions, no business logic

---

### 1.2 Composition Over Inheritance

**Prefer small, composable pieces over deep inheritance hierarchies**

```python
# ❌ BAD: Deep inheritance pyramid
class OrthoSlicer(BaseViewer):
    pass

class OrthoSlicerZoom(OrthoSlicer):
    pass

class OrthoSlicerRanger(OrthoSlicerZoom):
    pass

class OrthoSlicerBursts(OrthoSlicerRanger):
    pass
# Now have to understand 4 classes to know what this does!

# ✅ GOOD: Composition with clear intent
class TensorScopeApp:
    def __init__(self, state):
        self.state = state
        self.layers = []
    
    def add_layer(self, layer: TensorLayer):
        layer.bind(self.state)
        self.layers.append(layer)
        return self  # Allow chaining

app = (TensorScopeApp(state)
       .add_layer(SpatialMapLayer(state))
       .add_layer(EventOverlayLayer(state, events))
       .add_layer(TimeseriesLayer(state)))
```

**Why composition wins:**
- Each piece is independently testable
- Runtime configuration (add/remove layers dynamically)
- No "god class" problem
- Easier to understand (flat, not hierarchical)
- Avoids method resolution order (MRO) hell

**When to use inheritance:**
- Defining interfaces (ABC for TensorLayer)
- True "is-a" relationships (SpatialMapLayer *is-a* TensorLayer)
- Never more than 2 levels deep

---

### 1.3 Dependency Injection

**Components receive dependencies explicitly, not via globals or imports**

```python
# ❌ BAD: Hidden dependencies
_GLOBAL_STATE = None

class BadLayer:
    def __init__(self):
        self.state = _GLOBAL_STATE  # Where did this come from?
        self.config = load_config()  # Hidden file I/O!

# ✅ GOOD: Explicit dependencies
class GoodLayer:
    def __init__(self, state: TensorScopeState, config: dict):
        self.state = state  # Clear where state comes from
        self.config = config  # Clear where config comes from

# Easy to test with mocks
test_layer = GoodLayer(
    state=MockState(),
    config={'colormap': 'viridis'}
)
```

**Benefits:**
- **Testability**: Inject mocks/fakes
- **Clarity**: All dependencies visible in `__init__`
- **Flexibility**: Swap implementations easily
- **No hidden coupling**: Can't accidentally depend on global state

**Type hints make dependencies even clearer:**
```python
def __init__(
    self, 
    state: TensorScopeState,
    data_source: DataSource,
    theme: ThemeConfig = DEFAULT_THEME
):
```

---

## 2. State Management Principles

### 2.1 Single Source of Truth

**Every piece of state has exactly one owner**

```python
# ❌ BAD: Duplicate state → synchronization nightmare
class BadApp:
    def __init__(self):
        self.current_time = 0  # State in app
        self.layer1.current_time = 0  # Duplicate in layer1
        self.layer2.current_time = 0  # Duplicate in layer2
        # If one updates, others are stale!

# ✅ GOOD: Single source, shared references
class GoodApp:
    def __init__(self):
        self.state = TensorScopeState()  # One owner
        self.layer1 = Layer(self.state)  # References state
        self.layer2 = Layer(self.state)  # References state
```

**Ownership rules:**

| State | Owner | Consumers |
|-------|-------|-----------|
| Global app state (time, layout) | `TensorScopeState` | All layers |
| Time cursor value | `TimeHair` controller | `TensorScopeState` delegates to it |
| Channel selection | `ChannelGrid` controller | `TensorScopeState` delegates to it |
| Transform parameters | `ProcessingChain` | `TensorScopeState` delegates to it |
| Event data | `EventStream` | Registered in `TensorScopeState` |

**TensorScopeState delegates to controllers, doesn't duplicate:**
```python
class TensorScopeState(param.Parameterized):
    # Don't do this:
    # current_time = param.Number()  # Duplicate of TimeHair!
    
    # Do this:
    time_hair = param.Parameter()  # Owns the controller
    
    @property
    def current_time(self):
        return self.time_hair.t  # Delegates to controller
    
    @current_time.setter
    def current_time(self, value):
        self.time_hair.t = value  # Updates controller
```

---

### 2.2 Unidirectional Data Flow

**State changes flow in one direction: User Action → State → Controller → View**

```
┌─────────────┐
│ User Input  │ (Click, drag, keyboard)
└──────┬──────┘
       ↓
┌──────────────┐
│ State Update │ state.current_time = 5.3
└──────┬───────┘
       ↓
┌────────────────┐
│ Event Emission │ param.watch triggers
└──────┬─────────┘
       ↓
┌──────────────┐
│ Listeners    │ Layers react
│ React        │
└──────┬───────┘
       ↓
┌─────────────┐
│ View Update │ UI re-renders
└─────────────┘
```

**Never:**
- ❌ Views modify state directly (bypass state setters)
- ❌ Controllers call views (creates circular dependencies)
- ❌ Layers talk to each other (should only talk to state)

**Communication pattern:**
```python
# User clicks button
def on_button_click():
    state.current_time = 5.3  # Update state

# State emits event (automatic via param)
# ↓

# Layers react (declared once at init)
@param.depends('state.current_time', watch=True)
def on_time_change(self):
    self.update_view()
```

**Why this matters:**
- Predictable behavior (no hidden update chains)
- Easy to debug (follow the arrow)
- Testable (trigger state change, check view)

---

### 2.3 Reactive, Not Imperative

**Use declarative bindings, not manual update chains**

```python
# ❌ BAD: Manual updates (fragile)
def on_slider_change(value):
    state.current_time = value
    layer1.update_time(value)
    layer2.update_time(value)
    layer3.update_time(value)
    # Forgot layer4!
    # Added layer5 but forgot to update this!

# ✅ GOOD: Reactive binding (automatic)
state.current_time = param.Number()

class Layer:
    @param.depends('state.current_time', watch=True)
    def _on_time_change(self):
        self.update()
    # Automatically called when state.current_time changes
    # New layers automatically work
```

**Use reactive primitives:**
- `param.watch()` for side effects
- `pn.bind()` for Panel reactivity
- `hv.DynamicMap()` for HoloViews updates
- `streams.Params()` for cross-component linking

**Benefits:**
- Add layer → automatically reactive (no manual wiring)
- Remove layer → automatically stops reacting
- Can't forget to update a component

---

### 2.4 State is Serializable

**All state can be saved/loaded as JSON/dict**

```python
# ✅ State should support serialization
class TensorScopeState:
    def to_dict(self) -> dict:
        return {
            'current_time': self.current_time,
            'time_window': self.time_window,
            'selected_channels': list(self.selected_channels),
            'processing': self.processing.to_dict(),
            'layout': self.layout_preset,
            'active_layers': [l.name for l in self.layers],
            'events': {name: es.to_dict() for name, es in self.events.items()}
        }
    
    @classmethod
    def from_dict(cls, state_dict: dict, data) -> 'TensorScopeState':
        state = cls(data)
        state.current_time = state_dict['current_time']
        state.time_window = tuple(state_dict['time_window'])
        # ... restore all state
        return state
```

**Use cases:**
- **Session persistence**: Save work, restore later
- **Undo/redo**: Keep history of state dicts
- **Collaboration**: Share state with colleagues
- **Testing**: Load known-good states as fixtures
- **Reproducibility**: Exact state for figures/papers

**Serialization requirements:**
- Use JSON-compatible types (int, float, str, list, dict)
- Store references, not actual data (e.g., file paths, not arrays)
- Include version number for migrations

```python
{
    'version': '1.0',
    'data_path': '/path/to/data.nc',
    'timestamp': '2026-03-03T12:00:00Z',
    'state': { ... }
}
```

---

### 2.5 Derived vs Authoritative State ⭐ NEW

**Distinguish between state TensorScope owns vs. state third-party widgets own**

**Problem:** When wrapping existing UI components (Bokeh plots, Panel widgets), they often have their own internal state (zoom level, scroll position). This creates potential conflicts.

**Solution: Clear ownership model**

```text
┌─────────────────────────────┐
│ Authoritative State         │  TensorScope owns, single source of truth
│ (TensorScopeState)          │
├─────────────────────────────┤
│ - current_time              │
│ - selected_channels         │
│ - processing_params         │
│ - layout_preset             │
└─────────────────────────────┘

┌─────────────────────────────┐
│ Derived State               │  View widgets own, ephemeral
│ (UI Component Internal)     │
├─────────────────────────────┤
│ - zoom_level (Bokeh)        │
│ - scroll_position (Panel)   │
│ - hover_tooltip (HoloViews) │
└─────────────────────────────┘
```

**Synchronization pattern:**

```python
# Bokeh plot owns its zoom state (don't fight this)
bokeh_plot = figure(x_range=(0, 10), y_range=(0, 100))

# ✅ View → State (User pans plot)
range_stream = RangeXY(source=bokeh_plot)
range_stream.subscribe(lambda x_range, y_range: 
    state.visible_window = (x_range, y_range)  # Update authoritative state
)

# ✅ State → View (Programmatic jump)
def jump_to_event(event_time):
    state.current_time = event_time  # Update authoritative state
    # Command the view to update its derived state
    bokeh_plot.x_range.start = event_time - 1
    bokeh_plot.x_range.end = event_time + 1
```

**Rules:**
1. **Authoritative state** lives in `TensorScopeState` (or its controllers)
2. **Derived state** lives in view components
3. **Views emit events** when users change derived state → update authoritative state
4. **State can command views** to update derived state (bi-directional)
5. **Conflicts**: Authoritative state wins (state can override view)

**Example: Persistent zoom**
```python
# Save zoom as authoritative state if persistence needed
state.xy_zoom_range = param.Range()  # Authoritative

# Sync to Bokeh's derived state
@param.depends('state.xy_zoom_range', watch=True)
def sync_zoom(self):
    if self.state.xy_zoom_range:
        self.bokeh_plot.x_range.start, self.bokeh_plot.x_range.end = self.state.xy_zoom_range
```

---

## 3. Component Design Principles

### 3.1 Components Are Independent

**Every component should work standalone, outside TensorScope**

```python
# ✅ Component works independently
viewer = MultichannelViewer(
    sig_z=data,
    t_vals=time,
    ch_labels=labels
)
viewer.panel()  # Works in any Panel app, Jupyter, etc.

# ✅ Also works within TensorScope
layer = TimeseriesLayer(state)  # Wraps viewer
app.add_layer(layer)

# ❌ BAD: Component requires TensorScope context
class BadComponent:
    def __init__(self):
        # Tightly coupled to TensorScope!
        self.app = get_tensorscope_app()  
        self.state = self.app.state
        self.theme = self.app.theme
```

**Benefits:**
- **Reusable** in other projects/contexts
- **Easier to test** in isolation (no app mocking)
- **Lower barrier** to adoption
- **Gradual migration** path (use components before full TensorScope)

**Design for reuse:**
- Minimal required dependencies
- Configuration via constructor args, not globals
- No assumptions about surrounding app

---

### 3.2 Thin Wrappers, Not Reimplementations

**TensorScope layers wrap existing components, don't replace them**

```python
# ✅ GOOD: Thin wrapper (~10 lines)
class TimeseriesLayer(TensorLayer):
    def __init__(self, state: TensorScopeState):
        super().__init__(state)
        # Use existing, battle-tested component
        self._viewer = MultichannelViewer(
            sig_z=self._get_data(),
            t_vals=state.data.time.values,
            ch_labels=self._get_labels()
        )
        # Wire up state bindings
        state.channel_grid.param.watch(self._on_selection_change, 'selected')
    
    def panel(self):
        return self._viewer.panel()  # Delegate
    
    def _on_selection_change(self, event):
        self._viewer.show_channels(event.new)  # Delegate

# ❌ BAD: Reimplementation (hundreds of lines)
class TimeseriesLayer(TensorLayer):
    def __init__(self, state):
        # Rewrite MultichannelViewer from scratch!
        self.figure = bokeh.figure(...)
        self.datasource = ColumnDataSource(...)
        # Now we maintain a second timeseries viewer implementation!
```

**Wrapper responsibilities (and ONLY these):**
- Adapt `TensorScopeState` to component's API
- Handle component lifecycle (init, dispose)
- Provide layer metadata (name, type, description)
- Wire up reactive bindings

**Do NOT:**
- Reimplement the component's logic
- Add significant new functionality (extend component instead)
- Break the wrapped component's API

---

### 3.3 Public APIs, No Private Coupling

**Components communicate via documented, public APIs**

```python
# ❌ BAD: Private coupling (fragile)
def link_views(viewer, topomap):
    viewer._range_stream.add_subscriber(topomap._update)  # Private!
    # Breaks if MultichannelViewer refactors internals

# ✅ GOOD: Public API (stable)
def link_views(viewer, topomap):
    viewer.time_stream.subscribe(topomap.on_time_change)  # Public
    # Documented contract, won't break
```

**Rules:**
- **Never access `._private_attributes`** of other components
- **All integration points must be documented** in docstrings
- **Use public streams/events** for communication
- **Deprecate before removing** public APIs

**Defining public APIs:**
```python
class MultichannelViewer:
    """
    High-performance stacked trace viewer.
    
    Public API
    ----------
    time_stream : Stream
        Emits (t0, t1) when user changes visible time window
    
    Methods
    -------
    show_channels(indices)
        Update which channels are displayed
    add_time_hair(hair)
        Link a TimeHair controller for cursor display
    """
    
    @property
    def time_stream(self) -> Stream:
        """Public stream for time window changes."""
        return self._range_stream  # OK to expose private via property
```

---

### 3.4 Fail Gracefully

**Errors should not crash the entire app**

```python
# ✅ Layer handles its own errors
class RobustLayer(TensorLayer):
    def update(self):
        try:
            self._compute_and_render()
        except DataError as e:
            logger.error(f"Layer {self.name} data error: {e}")
            self._show_error_state(
                "Data Error",
                f"Could not load data: {e}",
                recovery_action="Reload Data"
            )
        except Exception as e:
            logger.exception(f"Layer {self.name} unexpected error")
            self._show_error_state(
                "Unexpected Error",
                "Something went wrong. Please report this bug.",
                recovery_action="Reset Layer"
            )
        # App continues working!
```

**Error handling strategy:**

1. **Catch at layer boundaries** (don't let exceptions escape)
2. **Display errors in-place** (not global modals that block UI)
3. **Log for debugging** (include context: layer name, state, data shape)
4. **Provide recovery** actions when possible (reload, reset, skip)

**Error UI examples:**
```python
def _show_error_state(self, title, message, recovery_action=None):
    """Replace layer panel with error display."""
    error_pane = pn.pane.Alert(
        f"**{title}**\n\n{message}",
        alert_type='danger'
    )
    if recovery_action:
        button = pn.widgets.Button(name=recovery_action)
        button.on_click(self._recover)
        error_pane = pn.Column(error_pane, button)
    
    self._panel = error_pane  # Replace normal view
```

**When to let exceptions propagate:**
- Initialization errors (bad config, missing data) → fail fast
- Unrecoverable errors (out of memory) → let app handle globally

---

## 4. Data Flow Principles

### 4.1 Lazy Evaluation

**Don't compute what you don't need**

```python
# ✅ GOOD: Windowed processing
class ProcessingChain:
    def get_window(self, t0, t1, channels=None):
        """Only process visible time window + selected channels."""
        subset = self.data.sel(
            time=slice(t0, t1),
            channel=channels if channels else slice(None)
        )
        # Apply transforms only to this small subset
        return self._apply_transforms(subset).compute()

# ❌ BAD: Eager computation
class BadProcessing:
    def __init__(self, data):
        # Processes entire 10-minute dataset on init!
        self.bandpassed = bandpass(data, 0.1, 300).compute()
        self.zscored = zscore(self.bandpassed).compute()
        # Most of this computation is wasted
```

**Lazy principles:**
- **Compute on demand** (when view requests it)
- **Cache intelligently** (memoize expensive operations)
- **Process only visible data** (time window + selected channels)
- **Defer until needed** (don't process at initialization)

**Smart caching:**
```python
from functools import lru_cache

@lru_cache(maxsize=32)
def get_processed_window(t0, t1, channels_tuple, transform_hash):
    """Cache recent windows to avoid recomputation."""
    # Recompute only if time window or transforms changed
    return process(data, t0, t1, list(channels_tuple))
```

---

### 4.2 Immutability for State, Performance for Data ⭐ REVISED

**Different rules for UI state vs. tensor data**

**Original principle was too dogmatic.** In high-performance scientific computing, strict immutability can destroy performance.

**New principle: Context-dependent mutability**

#### **UI State: ALWAYS Immutable**

```python
# ✅ UI state is always immutable
class TensorScopeState(param.Parameterized):
    current_time = param.Number()  # Param enforces immutability
    selected_channels = param.List()  # New list on update
    
# Never mutate param values
state.selected_channels.append(5)  # ❌ WRONG
state.selected_channels = state.selected_channels + [5]  # ✅ CORRECT
```

**Why:** Reactive systems (param) need to detect changes. Mutation breaks change detection.

#### **Tensor Data: Mutation Allowed in Pipelines**

```python
# ✅ In-place operations OK for performance-critical paths
def zscore_inplace(data: np.ndarray) -> None:
    """
    Z-score normalization IN-PLACE.
    
    ⚠️  MUTATES INPUT ARRAY - do not use if original data needed.
    """
    data -= data.mean(axis=0, keepdims=True)
    data /= data.std(axis=0, keepdims=True)

# Use in isolated processing pipeline
def get_processed_window(t0, t1):
    window = data.sel(time=slice(t0, t1)).values.copy()  # Explicit copy
    zscore_inplace(window)  # Mutate the copy
    return window
```

**When mutation is acceptable:**
- ✅ Inside processing pipelines (isolated, documented)
- ✅ Pre-allocated buffers for real-time operations
- ✅ Explicit `.copy()` before mutation
- ✅ Clearly documented in function signature

**When mutation is forbidden:**
- ❌ Param attributes (breaks reactivity)
- ❌ Shared state objects
- ❌ Function arguments unless clearly documented

**Documentation standard:**
```python
def process(data: np.ndarray, inplace: bool = False) -> np.ndarray:
    """
    Process data array.
    
    Parameters
    ----------
    data : np.ndarray
        Input data
    inplace : bool
        If True, mutates `data` in-place for performance.
        If False, returns a new array.
        
    Returns
    -------
    np.ndarray
        Processed data (same object if inplace=True)
    """
```

**Performance vs. immutability trade-off:**
```python
# Immutable (safe, slower)
def transform(data):
    return (data - data.mean()) / data.std()

# Mutable (fast, requires care)
def transform_fast(data):
    data = data.copy()  # Explicit copy
    data -= data.mean()
    data /= data.std()
    return data

# Best: Let caller choose
def transform(data, copy=True):
    if copy:
        data = data.copy()
    data -= data.mean()
    data /= data.std()
    return data
```

---

### 4.3 Schema Validation at Boundaries

**Validate data shape/dimensions at entry points**

```python
# ✅ Validate on data load
class TensorScopeState:
    def __init__(self, data: xr.DataArray):
        validated = validate_grid_schema(data)  # Raises if invalid
        self.data = to_canonical_grid(validated)  # Normalize to standard

def validate_grid_schema(data: xr.DataArray) -> xr.DataArray:
    """
    Ensure data conforms to (time, AP, ML) schema.
    
    Raises
    ------
    ValueError
        If dimensions are wrong or missing required coordinates
    """
    required_dims = {'time', 'AP', 'ML'}
    if set(data.dims) != required_dims:
        raise ValueError(
            f"Expected dims {required_dims}, got {set(data.dims)}"
        )
    
    # Check coordinate types, monotonicity, etc.
    ...
    
    return data
```

**Canonical schemas:**

**Grid data:**
```python
(time, AP, ML)
- AP: rows (anterior-posterior), integer indices
- ML: columns (medial-lateral), integer indices
- Flatten: channel = ap * n_ml + ml  (row-major)
```

**Flat data:**
```python
(time, channel)
- channel: has coords AP(channel), ML(channel) or MultiIndex (AP, ML)
```

**Validation rules:**
- **At app boundaries**: Loading files, API calls, user uploads
- **Convert immediately**: To canonical form as soon as data enters
- **Internal code assumes**: Canonical schema (no defensive checks everywhere)
- **Document schema**: In every function that accepts xarray

**Testing schema compliance:**
```python
def test_schema_validation():
    # Valid data
    valid = xr.DataArray(
        np.random.randn(100, 8, 8),
        dims=('time', 'AP', 'ML'),
        coords={'time': np.arange(100), 'AP': np.arange(8), 'ML': np.arange(8)}
    )
    assert validate_grid_schema(valid) is valid
    
    # Invalid dims
    invalid = xr.DataArray(
        np.random.randn(100, 8, 8),
        dims=('time', 'ML', 'AP')  # Wrong order!
    )
    with pytest.raises(ValueError, match="Expected dims"):
        validate_grid_schema(invalid)
```

---

### 4.4 Coordinate Conventions Are Explicit

**Never assume coordinate order; always document and verify**

```python
# ❌ BAD: Implicit assumptions
def flatten(data):
    return data.reshape(-1)  # Which dimension order? Unknown!

# ✅ GOOD: Explicit and verified
def flatten_grid(data: xr.DataArray) -> xr.DataArray:
    """
    Flatten (time, AP, ML) → (time, channel) using row-major order.
    
    Flattening convention: channel = ap * n_ml + ml
    
    Parameters
    ----------
    data : xr.DataArray
        Grid data with dims (time, AP, ML)
        
    Returns
    -------
    xr.DataArray
        Flat data with dims (time, channel)
        Includes coords: AP(channel), ML(channel)
    """
    assert data.dims == ('time', 'AP', 'ML'), \
        f"Expected (time, AP, ML), got {data.dims}"
    
    n_ap, n_ml = data.sizes['AP'], data.sizes['ML']
    
    # Row-major: channel = ap * n_ml + ml
    flat = data.stack(channel=('AP', 'ML'))
    
    # Verify flattening
    assert flat.dims == ('time', 'channel')
    assert len(flat.channel) == n_ap * n_ml
    
    return flat
```

**Document everywhere:**
```python
def process_spatial(
    data: xr.DataArray  # Must be (time, AP, ML)
) -> xr.DataArray:  # Returns (time, AP, ML)
    """
    Process spatial data.
    
    Coordinate conventions
    ----------------------
    - AP (rows): anterior-posterior, 0 = anterior
    - ML (columns): medial-lateral, 0 = medial (left hemisphere)
    - Row-major flattening: channel = ap * n_ml + ml
    """
```

**Test coordinate conventions:**
```python
def test_flattening_is_row_major():
    """Verify channel = ap * n_ml + ml convention."""
    data = xr.DataArray(
        np.arange(24).reshape(2, 3, 4),  # 2 time, 3 AP, 4 ML
        dims=('time', 'AP', 'ML')
    )
    
    flat = flatten_grid(data)
    
    # Channel 0 should be (AP=0, ML=0)
    assert flat.AP.values[0] == 0
    assert flat.ML.values[0] == 0
    
    # Channel 5 should be (AP=1, ML=1)  [1 * 4 + 1 = 5]
    assert flat.AP.values[5] == 1
    assert flat.ML.values[1] == 1
```

---

## 5. UI/UX Principles

### 5.1 Progressive Disclosure

**Show simple by default, reveal complexity on demand**

```python
# ✅ Simple interface first (80% use case)
app = TensorScopeApp.from_file('data.nc')
app.serve()  # Just works

# Power users can customize (20% use case)
app = (TensorScopeApp(data)
       .with_layout('custom')
       .add_layer(CustomLayer(...))
       .with_processing({'bandpass': [0.1, 300]})
       .with_events('bursts', burst_df))
```

**UI pattern:**
- **Collapsible cards** for advanced controls
- **Sensible defaults** (works out of the box)
- **Progressive enhancement** (simple → advanced)

**Example:**
```python
# Default: Simple controls
pn.Card(
    pn.widgets.Select(name='Filter', options=['Raw', 'Bandpass', 'Notch']),
    title="Processing",
    collapsible=True,
    collapsed=False  # Open by default
)

# Advanced: Detailed controls (hidden by default)
pn.Card(
    pn.Column(
        pn.widgets.FloatSlider(name='Low cutoff (Hz)', start=0.1, end=10),
        pn.widgets.FloatSlider(name='High cutoff (Hz)', start=10, end=500),
        pn.widgets.IntSlider(name='Filter order', start=1, end=10),
    ),
    title="Advanced Filter Settings",
    collapsible=True,
    collapsed=True  # Closed by default
)
```

---

### 5.2 Immediate Visual Feedback

**User actions should produce visible results <100ms**

```python
# ✅ Responsive
state.current_time = 5.3  
# → Time hair moves immediately (<10ms)
# → Spatial view updates within frame (<50ms)
# → Events table highlights current event (<50ms)
# Total: <100ms feels instant

# ❌ BAD: Laggy
state.current_time = 5.3
# → 2 second delay while entire dataset reprocesses
# User thinks app is broken
```

**Performance budget:**

| Action | Target | Maximum | Strategy |
|--------|--------|---------|----------|
| User input → state update | <10ms | <50ms | Direct assignment |
| State update → view redraw | <50ms | <100ms | Incremental updates |
| Complex operation | <500ms | <2s | Show progress indicator |
| Initial load | <2s | <5s | Lazy load, show skeleton |

**Achieving responsiveness:**
- **Incremental updates** (don't rebuild entire plot)
- **Windowed processing** (only process visible data)
- **Debouncing** (batch rapid updates)
- **Loading indicators** for unavoidable delays

---

### 5.3 Consistent Theming

**All components use the same color palette and styling**

```python
# ✅ Centralized theme
from cogpy.plot.theme import BG, BG_PANEL, BLUE, TEAL, PALETTE, COLORMAPS

# All components use theme constants
card = pn.Card(
    ...,
    header_background=BLUE,
    styles={'background': BG_PANEL}
)

plot = hv.Curve(...).opts(
    bgcolor=BG,
    color=PALETTE[0]
)

# ❌ BAD: Inline colors (inconsistent)
card = pn.Card(..., header_background='#4a90d9')  # Magic number
plot = hv.Curve(...).opts(bgcolor='black', color='blue')  # Doesn't match theme
```

**Theme system:**
```python
# theme.py
BG = "#181825"          # Figure background
BG_PANEL = "#1e1e2e"    # Panel/widget background
BORDER = "#3a3a5c"      # Axis lines, borders
TEXT = "#cdd6f4"        # Text color
BLUE = "#4a90d9"        # Primary accent
TEAL = "#4fc3f7"        # Secondary accent

PALETTE = [             # 20-color palette for traces
    "#4fc3f7", "#81c784", "#ffb74d", ...
]

COLORMAPS = {
    'viridis': [...],
    'plasma': [...],
    'rdbu': [...]
}
```

**Future: Theme switching**
```python
# Design for future theme support
def get_theme() -> ThemeConfig:
    """Get current theme (future: user selectable)."""
    return DARK_THEME  # Future: could be LIGHT_THEME

# Use theme values, not constants
bg = get_theme().background
```

---

### 5.4 Keyboard Navigation

**Power users should be able to work without mouse**

**Keyboard shortcuts:**
```
Navigation:
  Space       Play/pause
  ←/→         Step time backward/forward (1 sample)
  Shift+←/→   Step by window size
  Home/End    Jump to start/end
  
Events:
  N           Next event
  P           Previous event
  J/K         Next/prev (vim-style)
  
View:
  T           Jump to time (opens dialog)
  [/]         Decrease/increase time window
  1-9         Toggle panel 1-9 visibility
  Z           Reset zoom
  C           Center on cursor
  
Editing:
  A           Add annotation
  D           Delete selected
  Esc         Cancel/deselect
```

**Implementation:**
```python
# Centralized keybinding registry
class KeyBindings:
    _bindings = {}
    
    @classmethod
    def register(cls, key: str, callback: Callable, context='global'):
        """Register keyboard shortcut."""
        cls._bindings[(key, context)] = callback
    
    @classmethod
    def handle(cls, event, context='global'):
        """Dispatch keyboard event."""
        key = event.key
        if (key, context) in cls._bindings:
            cls._bindings[(key, context)]()

# Usage
KeyBindings.register('n', lambda: state.next_event())
KeyBindings.register('Space', lambda: state.toggle_play())
```

**Accessibility:**
- All interactive elements keyboard-accessible
- Visible focus indicators
- Logical tab order
- Screen reader support (aria labels)

---

## 6. Testing Principles

### 6.1 Test Behavior, Not Implementation

```python
# ✅ GOOD: Test observable behavior
def test_time_navigation():
    state = TensorScopeState(data)
    state.current_time = 5.3
    
    # Test what user can observe
    assert layer.displayed_time == 5.3
    assert layer.cursor_visible

# ❌ BAD: Test internals
def test_time_navigation():
    state = TensorScopeState(data)
    state.current_time = 5.3
    
    # Testing implementation details
    assert state._internal_cache_invalidated
    assert layer._redraw_count == 1
```

**Why:**
- Implementation can change without breaking tests
- Tests document actual behavior
- Catches real bugs (not just code structure changes)

---

### 6.2 Test Pyramid Balance ⭐ REVISED

**Original principle overstated "integration over unit". Correct balance:**

```
        /\
       /E2E\      5% - Full UI, realistic data (slow, few scenarios)
      /------\
     /  Integ \   25% - State bindings, component interaction
    /----------\
   /Unit Tests  \  70% - Pure logic, coordinate transforms (fast, many cases)
  /--------------\
```

**Test distribution:**

| Level | % of Tests | Speed | What to Test | Example |
|-------|-----------|-------|--------------|---------|
| **Unit** | 70% | <1ms | Pure functions, algorithms | Coordinate flattening, downsampling, event detection |
| **Integration** | 25% | <100ms | State synchronization, bindings | Param.watch triggers, stream subscriptions |
| **E2E** | 5% | <10s | Full workflows | Load data → navigate → save session |

**Unit tests: Pure logic**
```python
def test_flatten_grid_row_major():
    """Test coordinate flattening convention."""
    data = make_grid_data(n_ap=3, n_ml=4)
    flat = flatten_grid(data)
    
    # Channel 0 should be (AP=0, ML=0)
    assert flat.AP.values[0] == 0
    assert flat.ML.values[0] == 0
    
    # Channel 5 should be (AP=1, ML=1)  [1*4 + 1 = 5]
    assert flat.AP.values[5] == 1
    assert flat.ML.values[5] == 1
```

**Integration tests: Component interaction**
```python
def test_time_hair_updates_layer():
    """Test that TimeHair triggers layer updates."""
    state = TensorScopeState(data)
    layer = SpatialMapLayer(state)
    
    update_count = 0
    def count_updates():
        nonlocal update_count
        update_count += 1
    
    layer.on_update = count_updates
    
    state.time_hair.t = 5.3
    assert update_count == 1  # Layer reacted to state change
```

**E2E tests: Critical workflows**
```python
@pytest.mark.slow
def test_full_event_navigation_workflow():
    """Test complete event analysis workflow."""
    # Load realistic 10-minute dataset
    data = load_ieeg_fixture('10min_8x8_1khz.nc')
    events = load_event_fixture('bursts.csv')
    
    # Create app
    app = TensorScopeApp(data, events=events)
    
    # Navigate to first event
    app.state.jump_to_event(0)
    assert app.state.current_time == events.iloc[0]['time']
    
    # Check all views updated
    assert app.timeseries_layer.cursor_at(events.iloc[0]['time'])
    assert app.events_table.selected_row == 0
    
    # Save session
    session = app.save_session('/tmp/test.json')
    
    # Restore and verify
    app2 = TensorScopeApp.from_session('/tmp/test.json')
    assert app2.state.current_time == events.iloc[0]['time']
```

**When to write each type:**

**Unit tests:**
- ✅ Coordinate transformations
- ✅ Downsampling algorithms
- ✅ Event detection logic
- ✅ Schema validation
- ✅ Data processing functions

**Integration tests:**
- ✅ State synchronization (param.watch works)
- ✅ Stream subscriptions (events fire correctly)
- ✅ Layer bindings (state → view updates)
- ❌ Don't test Panel/HoloViews internals (already tested)

**E2E tests:**
- ✅ Load data → navigate → save session
- ✅ Detect events → navigate → export
- ✅ Multi-modal workflow (LFP + spikes)
- ❌ Don't test every feature combination (combinatorial explosion)

---

### 6.3 Test with Real Data

**Use realistic fixtures, not toy data**

```python
# ✅ GOOD: Realistic test data
@pytest.fixture
def ieeg_10min_8x8_1khz():
    """
    10 minutes of 8×8 grid iEEG at 1kHz.
    
    Includes:
    - Realistic noise characteristics
    - Known bursts at t=[30s, 120s, 240s]
    - Missing data at t=[180s-185s]
    - Bad channels: (AP=2, ML=3), (AP=5, ML=7)
    """
    return generate_realistic_ieeg(
        duration=600,
        fs=1000,
        grid=(8, 8),
        bursts=[30, 120, 240],
        gaps=[(180, 185)],
        bad_channels=[(2, 3), (5, 7)]
    )

# ❌ BAD: Toy data that doesn't test real scenarios
@pytest.fixture
def tiny_data():
    return np.random.randn(10, 2)  # Doesn't stress anything
```

**Why realistic data matters:**
- Catches edge cases (missing data, bad channels)
- Tests performance (10min dataset reveals bottlenecks)
- Validates assumptions (noise statistics, burst characteristics)

**Test data library:**
```python
# tests/fixtures/data.py
def generate_realistic_ieeg(
    duration: float,
    fs: float,
    grid: tuple[int, int],
    bursts: list[float] = None,
    gaps: list[tuple[float, float]] = None,
    bad_channels: list[tuple[int, int]] = None,
    seed: int = 42
) -> xr.DataArray:
    """
    Generate realistic synthetic iEEG data for testing.
    
    Returns data with known ground truth for validation.
    """
```

---

## 7. Performance Principles

### 7.1 Measure Before Optimizing

**Profile first, optimize second**

```python
# Don't optimize based on hunches
# Measure actual bottlenecks:

from line_profiler import LineProfiler

profiler = LineProfiler()
profiler.add_function(layer.update)

with profiler:
    layer.update()

profiler.print_stats()
# Output shows actual slow lines:
#   Line 45: 1.2s (bottleneck!)
#   Line 67: 0.1s
```

**Profiling tools:**
- `line_profiler` for line-by-line timing
- `cProfile` for function call statistics
- `memory_profiler` for memory usage
- Browser dev tools for UI rendering

**Optimization priority:**
1. Profile to find bottleneck
2. Fix bottleneck
3. Measure improvement
4. Repeat if still slow

**Never optimize without measuring.**

---

### 7.2 Bounded Compute

**All operations have maximum time/memory bounds**

```python
# ✅ Processing is windowed
def get_window(t0, t1, channels=None):
    """
    Retrieve processed data for time window.
    
    Complexity: O(window_size × n_channels)
    Max memory: window_size × n_channels × 8 bytes
    
    For window=2s, fs=1kHz, channels=64:
      Memory: 2000 × 64 × 8 = 1MB (bounded)
    """
    # Only process visible window, not entire 10min dataset
    return data.sel(time=slice(t0, t1), channel=channels).compute()

# ✅ Downsampling has budget
viewer = MultichannelViewer(data, max_points=10000)
# Will decimate to stay under 10k points per channel
# Guarantees fast rendering regardless of data length
```

**Performance budgets:**

| Operation | Target | Budget | Strategy |
|-----------|--------|--------|----------|
| Time window processing | <50ms | 2s × 1kHz × 64ch | Windowed compute |
| Timeseries rendering | <100ms | 10k points/channel | Downsample |
| Spatial map update | <50ms | 64×64 pixels | Direct array update |
| Event detection | <500ms | Process in chunks | Background task |

**Enforce budgets:**
```python
def process_window(t0, t1, max_samples=20000):
    """Process window with sample budget."""
    window = data.sel(time=slice(t0, t1))
    
    if len(window.time) * len(window.channel) > max_samples:
        # Decimate to stay under budget
        step = len(window.time) // (max_samples // len(window.channel))
        window = window.isel(time=slice(None, None, step))
    
    return window.compute()
```

---

### 7.3 Incremental Updates

**Update only what changed**

```python
# ✅ GOOD: Patch data (fast)
topomap.update(new_values)  
# Uses ColumnDataSource.patch() internally
# Only updates changed pixels
# ~1ms

# ❌ BAD: Rebuild everything (slow)
topomap = TopoMap(new_values)  
# Recreates entire Bokeh figure
# ~100ms
```

**Incremental update strategies:**

**For Bokeh plots:**
```python
# Use ColumnDataSource.patch()
self.source.patch({'y': [(0, new_value)]})  # Update index 0 only
```

**For HoloViews:**
```python
# Use DynamicMap with streams
def update(x_range):
    # Only recompute visible range
    return hv.Curve(data.sel(time=slice(*x_range)))

dmap = hv.DynamicMap(update, streams=[RangeX()])
```

**For Panel:**
```python
# Use pn.bind for reactive updates
@pn.depends(state.param.current_time)
def update_text(current_time):
    return f"Time: {current_time:.2f}s"  # Only text changes

pn.panel(update_text)  # Panel handles minimal DOM updates
```

---

## 8. Documentation Principles

### 8.1 Code is Documentation

**Write self-documenting code**

```python
# ✅ GOOD: Clear names explain intent
def get_channels_with_activity_in_time_window(
    t0: float, 
    t1: float,
    threshold: float = 3.0
) -> list[int]:
    """
    Return channel indices with RMS amplitude > threshold in [t0, t1].
    
    Parameters
    ----------
    t0, t1 : float
        Time window in seconds
    threshold : float
        RMS threshold in units of std dev
        
    Returns
    -------
    list[int]
        Channel indices (flat, row-major order)
    """

# ❌ BAD: Cryptic names require comments to explain
def gcitw(t0, t1, th=3.0):
    """Get channels in time window."""  # Still unclear what this does!
    # Get the active channels
    # Using RMS
    # Threshold is in std dev units
    ...
```

**Self-documenting principles:**
- **Descriptive names** (not abbreviations)
- **Type hints** (document expected types)
- **Docstrings** (explain purpose and contracts)
- **Clear structure** (function does one thing)

---

### 8.2 Document Why, Not What

```python
# ❌ BAD: Obvious comment (documents WHAT)
x = x + 1  # Increment x

# ✅ GOOD: Explains WHY
# Use row-major flattening to match Neuralynx acquisition system convention
channel_idx = ap * n_ml + ml

# ✅ GOOD: Explains non-obvious choice
# Use 98th percentile instead of max to avoid outlier saturation
clim_max = np.percentile(data, 98)
```

**What to document:**
- ❌ What the code does (code already says this)
- ✅ Why you made this choice
- ✅ Alternative approaches considered
- ✅ Known limitations or assumptions
- ✅ References to papers/specs

**Example:**
```python
def bandpass(data, low, high, order=4):
    """
    Butterworth bandpass filter.
    
    We use Butterworth (not Chebyshev) because:
    - Maximally flat passband (no ripple)
    - Phase response is better for visual inspection
    - Performance difference is negligible for our window sizes
    
    Order=4 is a compromise:
    - Higher order: sharper cutoff but more ringing artifacts
    - Lower order: gradual cutoff but preserves transients
    See Smith 1997, "The Scientist and Engineer's Guide to DSP", Ch. 20
    """
```

---

### 8.3 Examples in Docstrings

```python
def link_time_cursor(source_stream: Stream, target_callback: Callable):
    """
    Link time cursor between components.
    
    Parameters
    ----------
    source_stream : Stream
        Stream that emits time updates
    target_callback : Callable[[float], None]
        Function to call with new time value
        
    Examples
    --------
    Link viewer's time stream to topomap:
    
    >>> link_time_cursor(
    ...     viewer.time_stream, 
    ...     topomap.on_time_change
    ... )
    >>> state.current_time = 5.3
    >>> # Both viewer and topomap update
    
    Link multiple targets:
    
    >>> for layer in layers:
    ...     link_time_cursor(viewer.time_stream, layer.update_time)
    """
```

**Good examples:**
- Show actual usage
- Cover common cases
- Demonstrate patterns
- Executable (can be tested with doctest)

---

## 9. Concurrency and Long-Running Operations ⭐ NEW

Modern UIs must handle long-running operations without blocking.

### 9.1 Async Operations Must Not Block UI

**Problem:** Heavy computation blocks Panel's server thread, freezing the UI.

**Solution:** Run compute in background, update UI when done.

```python
import threading
from concurrent.futures import ThreadPoolExecutor

# ✅ GOOD: Background computation
class ProcessingLayer:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def apply_filter(self, filter_type):
        """Apply filter without blocking UI."""
        state.is_processing = True
        state.processing_message = f"Applying {filter_type} filter..."
        
        # Submit to background thread
        future = self.executor.submit(self._compute_filter, filter_type)
        future.add_done_callback(self._on_filter_complete)
    
    def _compute_filter(self, filter_type):
        """Heavy computation (runs in background thread)."""
        return expensive_bandpass(self.data, filter_type)
    
    def _on_filter_complete(self, future):
        """Update UI when done (runs in main thread)."""
        try:
            result = future.result()
            self.data = result
            state.is_processing = False
            state.processing_message = ""
        except Exception as e:
            state.is_processing = False
            state.processing_message = f"Error: {e}"

# ❌ BAD: Blocking computation
def apply_filter(filter_type):
    # This blocks the entire UI for 5+ seconds!
    result = expensive_bandpass(data, filter_type)
    state.data = result
```

**For Panel apps:**
```python
# Use panel.io.state for thread-safe updates
from panel.io import state as pn_state

def _on_filter_complete(self, future):
    result = future.result()
    
    # Schedule UI update on main thread
    pn_state.add_next_tick_callback(
        lambda: self._update_ui(result)
    )
```

---

### 9.2 Loading State is Part of State

**Loading/processing is state, not just a side effect**

```python
class TensorScopeState(param.Parameterized):
    # Processing state
    is_processing = param.Boolean(default=False)
    processing_message = param.String(default="")
    processing_progress = param.Number(default=0.0, bounds=(0, 1))
    
    # Cancellation flag
    _cancel_flag = threading.Event()
    
    def cancel_processing(self):
        """Request cancellation of current operation."""
        self._cancel_flag.set()
```

**Views react to loading state:**
```python
class ProcessingIndicator:
    @param.depends('state.is_processing', 'state.processing_message')
    def panel(self):
        if not self.state.is_processing:
            return pn.pane.Markdown("")  # Hidden
        
        return pn.Column(
            pn.indicators.LoadingSpinner(
                value=True,
                size=25,
                name=self.state.processing_message
            ),
            pn.widgets.Progress(
                name='Progress',
                value=int(self.state.processing_progress * 100),
                max=100
            ),
            pn.widgets.Button(
                name='Cancel',
                button_type='danger',
                on_click=lambda e: self.state.cancel_processing()
            )
        )
```

---

### 9.3 Operations Are Cancellable

**Long operations must check cancellation flag**

```python
def expensive_bandpass(data, cancel_flag: threading.Event):
    """
    Apply bandpass filter with cancellation support.
    
    Parameters
    ----------
    data : xr.DataArray
        Input data
    cancel_flag : threading.Event
        Set this to cancel operation
        
    Returns
    -------
    xr.DataArray | None
        Filtered data, or None if cancelled
    """
    n_channels = len(data.channel)
    
    for i, ch in enumerate(data.channel):
        # Check cancellation every channel
        if cancel_flag.is_set():
            logger.info(f"Bandpass cancelled at channel {i}/{n_channels}")
            return None
        
        # Process one channel
        data.loc[dict(channel=ch)] = bandpass_channel(data.sel(channel=ch))
        
        # Update progress
        progress = i / n_channels
        pn_state.add_next_tick_callback(
            lambda: setattr(state, 'processing_progress', progress)
        )
    
    return data
```

---

### 9.4 UI Never Blocks

**Golden rule: The UI thread only does UI work**

**Fast operations** (<100ms): Execute synchronously
```python
state.current_time = 5.3  # Fast, direct assignment
```

**Medium operations** (100ms-500ms): Debounce and execute
```python
from panel.io import state as pn_state

# Debounce rapid slider changes
@debounce(0.1)  # Wait 100ms after last change
def on_slider_change(value):
    state.current_time = value
```

**Slow operations** (>500ms): Background thread + progress
```python
def on_apply_filter():
    state.is_processing = True
    executor.submit(expensive_filter).add_done_callback(on_complete)
```

**Architecture:**
```
UI Thread                   Background Thread
    │                              │
    │─── User clicks button        │
    │─── Set is_processing=True    │
    │─── Submit task ────────────>│
    │                              │─── Heavy computation
    │                              │─── (bandpass 64 channels)
    │                              │─── (5 seconds)
    │                              │
    │<─── Callback ────────────────│
    │─── Update UI                 │
    │─── Set is_processing=False   │
    │                              │
```

---

## 10. Component Lifecycle ⭐ NEW

Components must clean up after themselves to avoid memory leaks.

### 10.1 Every Component Has dispose()

**All components must implement cleanup**

```python
class TensorLayer(ABC):
    """Base class for all TensorScope layers."""
    
    def __init__(self, state: TensorScopeState):
        self.state = state
        self._watchers = []  # Track all watchers
        self._streams = []  # Track all streams
        self._data_refs = []  # Track large data
    
    @abstractmethod
    def panel(self) -> pn.viewable.Viewable:
        """Return Panel viewable."""
        pass
    
    def dispose(self):
        """
        Clean up resources.
        
        MUST be called when layer is removed!
        """
        # Unregister all watchers
        for watcher in self._watchers:
            self.state.param.unwatch(watcher)
        
        # Dispose all streams
        for stream in self._streams:
            stream.clear()
        
        # Release large data references
        for ref in self._data_refs:
            del ref
        
        # Clear tracking lists
        self._watchers.clear()
        self._streams.clear()
        self._data_refs.clear()
```

**Concrete example:**
```python
class TimeseriesLayer(TensorLayer):
    def __init__(self, state):
        super().__init__(state)
        
        # Create viewer
        self.viewer = MultichannelViewer(...)
        self._data_refs.append(self.viewer)
        
        # Set up watcher
        watcher = state.param.watch(
            self._on_time_change, 
            'current_time'
        )
        self._watchers.append(watcher)  # Track for cleanup
        
        # Set up stream
        stream = RangeX(source=self.viewer.plot)
        stream.subscribe(self._on_range_change)
        self._streams.append(stream)  # Track for cleanup
    
    def panel(self):
        return self.viewer.panel()
    
    # dispose() inherited from TensorLayer
```

---

### 10.2 App Manages Lifecycle

**Application is responsible for calling dispose()**

```python
class TensorScopeApp:
    def __init__(self, state):
        self.state = state
        self.layers = []
    
    def add_layer(self, layer: TensorLayer):
        """Add layer to app."""
        layer.bind(self.state)  # Future: explicit binding step
        self.layers.append(layer)
        return self
    
    def remove_layer(self, layer: TensorLayer):
        """Remove layer and clean up."""
        # CRITICAL: Must call dispose!
        layer.dispose()
        
        # Remove from list
        self.layers.remove(layer)
    
    def shutdown(self):
        """Shut down app and clean up all resources."""
        for layer in self.layers:
            layer.dispose()
        self.layers.clear()
```

---

### 10.3 Use Context Managers When Possible

**For temporary operations, use context managers**

```python
class TensorLayer:
    @contextmanager
    def active(self):
        """Context manager for temporary layer activation."""
        try:
            self.bind(state)
            yield self
        finally:
            self.dispose()

# Usage
with SpatialMapLayer(state).active() as layer:
    layer.update()
    result = layer.compute()
# Automatically disposed on exit
```

---

### 10.4 Memory Leak Prevention Checklist

**Before merging code, verify:**

- [ ] Every `param.watch()` has corresponding `param.unwatch()` in `dispose()`
- [ ] Every stream created has `.clear()` called in `dispose()`
- [ ] Large data arrays (>1MB) are explicitly deleted in `dispose()`
- [ ] Circular references are broken (e.g., `layer.state` = None)
- [ ] App calls `layer.dispose()` when removing layers
- [ ] Long-running apps (>1hr) don't show memory growth

**Testing for leaks:**
```python
import gc
import psutil

def test_no_memory_leak():
    """Verify layers don't leak memory."""
    process = psutil.Process()
    initial_memory = process.memory_info().rss
    
    # Create and destroy 100 layers
    for _ in range(100):
        layer = SpatialMapLayer(state)
        _ = layer.panel()  # Trigger rendering
        layer.dispose()  # Clean up
        gc.collect()  # Force garbage collection
    
    final_memory = process.memory_info().rss
    memory_growth = final_memory - initial_memory
    
    # Should not grow more than 10MB (some overhead expected)
    assert memory_growth < 10 * 1024 * 1024, \
        f"Memory leak detected: {memory_growth / 1024 / 1024:.1f}MB growth"
```

---

## Summary: The Golden Rules (Revised)

1. **Separate state from logic from presentation**
2. **Compose, don't inherit**
3. **One owner per piece of state**
4. **Data flows in one direction: State → Controller → View**
5. **Components are independent and reusable**
6. **Wrap, don't rewrite**
7. **Public APIs only, no private coupling**
8. **Validate data at boundaries, assume canonical schema internally**
9. **Fail gracefully with in-place error UI**
10. **Test behavior with realistic data**
11. **Immutable UI state, mutable tensor data (with care)** ⭐
12. **Unit tests for logic, integration for bindings, E2E for workflows** ⭐
13. **Distinguish authoritative state from derived state** ⭐
14. **Long operations run in background with cancellation** ⭐
15. **Every component has dispose(), app calls it** ⭐

---

## Appendix: Common Anti-Patterns to Avoid

### ❌ The God Class
```python
class MassiveApp:
    """Tries to do everything."""
    def __init__(self):
        # 50 attributes
        # 100 methods
        # 2000 lines
```
**Fix:** Decompose into State + Controllers + Layers

### ❌ Hidden Globals
```python
_CURRENT_APP = None  # Global state

def get_app():
    return _CURRENT_APP
```
**Fix:** Dependency injection

### ❌ Circular Dependencies
```python
layer1.other_layer = layer2
layer2.other_layer = layer1
```
**Fix:** Both reference shared state

### ❌ Premature Optimization
```python
# Optimizing before profiling
def ultra_fast_but_unreadable_code():
    return (x:=a@b.T)@(y:=c.T@d)@x  # What?
```
**Fix:** Measure first, optimize second

### ❌ Testing Implementation
```python
assert obj._internal_count == 5  # Fragile!
```
**Fix:** Test observable behavior

### ❌ Forgetting Cleanup
```python
def create_layer():
    layer = Layer(state)
    state.param.watch(layer.update, 'time')
    return layer  # Watcher never cleaned up!
```
**Fix:** Implement dispose()

---

**This document is a living guide. When adding features, ask: "Does this align with our principles?"**

**End of Design Principles v2.0**