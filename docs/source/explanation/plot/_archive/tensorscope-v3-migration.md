# Migration Guide: TensorScope v2.8 → v3.0

TensorScope v3.0 introduces **breaking changes** to refactor from tool-centric to tensor-centric architecture.

## Breaking Changes

### 1. State Structure

v2.8 (old pattern):

```python
state.time_hair.t = 5.0
state.spatial_space.selection = (8, 6)
```

v3.0 (new):

```python
state.selection.time = 5.0
state.selection.ap = 8
state.selection.ml = 6
```

### 2. Module System

v2.8 (old):

```python
app = TensorScopeApp(data)
app.with_module("psd_explorer")
```

v3.0 (new):

```python
app = TensorScopeApp()
app.add_tensor("signal", data)
app.add_psd_tensor("psd", source="signal")
```

### 3. View Discovery

Views are now discovered from tensor dimensions:

```python
from cogpy.core.tensorscope.views import get_available_views

views = get_available_views(tensor_node)
```

## Migration Checklist

- [ ] Replace controller-based state access with `SelectionState`
- [ ] Replace module calls with explicit tensor creation
- [ ] Ensure PSD tensors have a `freq` dimension
- [ ] Ensure views are pure projections (no mutations)

