**Brilliant!** You've just articulated a much cleaner, more object-oriented architecture. Let me redesign TensorScope around this paradigm.

---

# TensorScope Redesign: Signal-as-Object Architecture

## Core Concept

```
Signal Object = Data + Processing Pipeline + Metadata
    ↓
Views (spatial, timeseries, spectrogram, PSD) render the signal
    ↓
User can duplicate → tweak processing → compare side-by-side
```

---

## New Architecture

### 1. **Signal Object**

```python
class SignalObject:
    """
    A signal with its processing pipeline and metadata.
    
    Represents one "version" of the data - could be raw, filtered,
    high-gamma extracted, etc.
    
    Users can:
    - View in different modes (spatial, timeseries, TF)
    - Duplicate and modify processing
    - Compare multiple versions side-by-side
    """
    
    def __init__(
        self,
        data: xr.DataArray,
        name: str,
        processing: ProcessingChain | None = None,
        metadata: dict | None = None
    ):
        self.data = data  # Raw data reference (shared)
        self.name = name
        self.processing = processing or ProcessingChain(data)
        self.metadata = metadata or {}
        self.id = str(uuid.uuid4())[:8]
    
    def get_window(self, t0: float, t1: float, **kwargs) -> xr.DataArray:
        """Get processed data window."""
        return self.processing.get_window(t0, t1, **kwargs)
    
    def compute_psd(self, t0: float, t1: float, **kwargs) -> xr.DataArray:
        """Compute PSD over window."""
        return self.processing.compute_psd(t0, t1, **kwargs)
    
    def duplicate(self, name: str | None = None) -> "SignalObject":
        """Create a copy with independent processing chain."""
        new_name = name or f"{self.name} (copy)"
        
        # New processing chain with same settings
        new_processing = ProcessingChain(self.data)
        new_processing.param.update(self.processing.param.values())
        
        return SignalObject(
            data=self.data,  # Shared data reference
            name=new_name,
            processing=new_processing,
            metadata=dict(self.metadata)
        )
    
    def to_dict(self) -> dict:
        """Serialize (for session save)."""
        return {
            'id': self.id,
            'name': self.name,
            'processing': self.processing.to_dict(),
            'metadata': self.metadata
        }
```

---

### 2. **SignalRegistry (replaces DataRegistry)**

```python
class SignalRegistry(param.Parameterized):
    """
    Manages multiple signal objects.
    
    Signals can be:
    - Base signals (raw LFP, raw spectrogram)
    - Derived signals (filtered versions, spectral decompositions)
    
    Each signal has independent processing pipeline.
    """
    
    signals = param.Dict(default={})  # id → SignalObject
    active_signal_id = param.String(default=None)
    
    def register(self, signal: SignalObject) -> str:
        """Register a signal, return its ID."""
        self.signals = {**self.signals, signal.id: signal}
        
        # Set as active if first signal
        if self.active_signal_id is None:
            self.active_signal_id = signal.id
        
        return signal.id
    
    def get(self, signal_id: str) -> SignalObject | None:
        """Get signal by ID."""
        return self.signals.get(signal_id)
    
    def get_active(self) -> SignalObject | None:
        """Get currently active signal."""
        if self.active_signal_id:
            return self.signals.get(self.active_signal_id)
        return None
    
    def list(self) -> list[str]:
        """List signal IDs."""
        return list(self.signals.keys())
    
    def list_names(self) -> list[tuple[str, str]]:
        """List (id, name) pairs."""
        return [(sid, sig.name) for sid, sig in self.signals.items()]
    
    def duplicate(self, signal_id: str, new_name: str | None = None) -> str:
        """Duplicate a signal with independent processing."""
        source = self.get(signal_id)
        if not source:
            raise ValueError(f"Signal {signal_id} not found")
        
        new_signal = source.duplicate(new_name)
        return self.register(new_signal)
    
    def remove(self, signal_id: str):
        """Remove a signal."""
        if signal_id in self.signals:
            signals = dict(self.signals)
            del signals[signal_id]
            self.signals = signals
            
            # Update active if needed
            if self.active_signal_id == signal_id:
                self.active_signal_id = list(self.signals.keys())[0] if self.signals else None
```

---

### 3. **View Objects (instead of Layers)**

Views render a signal in different ways. Multiple views can show the same signal, or different signals.

```python
class SignalView(ABC):
    """
    Base class for signal visualization.
    
    A view renders a SignalObject in a specific way:
    - SpatialView: electrode grid heatmap
    - TimeseriesView: stacked line plots
    - SpectrogramView: time-frequency with orthoslicer
    - PSDView: frequency spectrum
    
    Each view:
    - References a signal_id (can be changed)
    - Has view-specific parameters (colormap, window_s, etc.)
    - Updates when signal or view params change
    """
    
    def __init__(
        self,
        state: TensorScopeState,
        signal_id: str,
        view_id: str | None = None
    ):
        self.state = state
        self.signal_id = signal_id
        self.view_id = view_id or str(uuid.uuid4())[:8]
        
        # Watch for signal changes
        self._watchers = []
        self._setup_watchers()
    
    def _setup_watchers(self):
        """Setup reactive updates."""
        # Watch signal registry for changes to our signal
        w = self.state.signal_registry.param.watch(
            self._on_signal_update, 'signals'
        )
        self._watchers.append(w)
    
    def _on_signal_update(self, event):
        """React to signal changes."""
        self._update()
    
    def get_signal(self) -> SignalObject | None:
        """Get the signal this view is rendering."""
        return self.state.signal_registry.get(self.signal_id)
    
    def set_signal(self, signal_id: str):
        """Switch to different signal."""
        self.signal_id = signal_id
        self._update()
    
    @abstractmethod
    def _update(self):
        """Update view rendering."""
        pass
    
    @abstractmethod
    def panel(self) -> pn.viewable.Viewable:
        """Return Panel widget."""
        pass
```

---

### 4. **Concrete View Implementations**

#### SpatialView

```python
class SpatialView(SignalView):
    """
    Spatial map view of a signal.
    
    Parameters
    ----------
    mode : str
        'rms', 'mean', 'max', 'instant'
    window_s : float
        Time window for reduction (ignored for 'instant')
    render_mode : str
        'image' or 'electrode'
    """
    
    def __init__(
        self,
        state,
        signal_id: str,
        mode: str = 'rms',
        window_s: float = 0.1,
        render_mode: str = 'image',
        colormap: str = 'viridis'
    ):
        self.mode = mode
        self.window_s = window_s
        self.render_mode = render_mode
        self.colormap = colormap
        
        super().__init__(state, signal_id)
        
        # Create GridFrameElement
        self._build_element()
        
        # Watch time changes
        w = state.param.watch(self._update, 'selected_time')
        self._watchers.append(w)
    
    def _build_element(self):
        """Build GridFrameElement for this signal."""
        signal = self.get_signal()
        if not signal:
            return
        
        # For instant mode, use selected_time; otherwise use time_hair
        time_source = self.state if self.mode == 'instant' else self.state.time_hair
        
        self._element = GridFrameElement(
            sig_grid=signal.data,
            time_hair=time_source if self.mode != 'instant' else None,
            mode=self.mode,
            window_s=self.window_s,
            chain=signal.processing,
            colormap=self.colormap,
            title=f"{signal.name} ({self.mode})",
            render_mode=self.render_mode
        )
    
    def _update(self, *args):
        """Rebuild element when signal changes."""
        self._build_element()
    
    def panel(self):
        if hasattr(self, '_element'):
            return self._element.panel()
        return pn.pane.Markdown(f"**Signal not found:** {self.signal_id}")
```

#### TimeseriesView

```python
class TimeseriesView(SignalView):
    """
    Timeseries stacked line plot view.
    
    Wraps MultichannelViewer.
    """
    
    def __init__(
        self,
        state,
        signal_id: str,
        show_hair: bool = True
    ):
        self.show_hair = show_hair
        super().__init__(state, signal_id)
        
        self._build_viewer()
        
        # Watch selection
        w = state.channel_grid.param.watch(self._on_selection, 'selected')
        self._watchers.append(w)
    
    def _build_viewer(self):
        """Build MultichannelViewer for this signal."""
        signal = self.get_signal()
        if not signal:
            return
        
        # Get processed data for viewer
        from cogpy.tensorscope.schema import flatten_grid_to_channels
        flat = flatten_grid_to_channels(signal.data)
        
        # Create viewer
        self._viewer = MultichannelViewer(
            sig_z=flat.values.T.copy(),
            t_vals=flat.time.values.copy(),
            ch_labels=[f"Ch{i}" for i in range(flat.sizes['channel'])]
        )
        
        if self.show_hair:
            self._viewer.add_time_hair(self.state.time_hair)
    
    def _on_selection(self, event):
        """Update displayed channels."""
        selected = self.state.selected_channels_flat
        if len(selected) > 0:
            self._viewer.show_channels(selected)
    
    def _update(self, *args):
        """Rebuild viewer when signal changes."""
        self._build_viewer()
    
    def panel(self):
        if hasattr(self, '_viewer'):
            return self._viewer.panel()
        return pn.pane.Markdown(f"**Signal not found:** {self.signal_id}")
```

#### PSDView

```python
class PSDView(SignalView):
    """
    PSD view of a signal at selected time.
    
    Shows:
    - Mean PSD curve
    - Optional per-channel heatmap
    """
    
    def __init__(
        self,
        state,
        signal_id: str,
        window_s: float = 2.0,
        method: str = 'multitaper',
        bandwidth: float = 4.0
    ):
        self.window_s = window_s
        self.method = method
        self.bandwidth = bandwidth
        
        super().__init__(state, signal_id)
        
        self._container = pn.Column(sizing_mode='stretch_both')
        
        # Watch selected_time
        w = state.param.watch(self._update, 'selected_time')
        self._watchers.append(w)
        
        # Initial update
        self._update()
    
    def _update(self, *args):
        """Recompute PSD."""
        signal = self.get_signal()
        if not signal:
            self._container.objects = [
                pn.pane.Markdown(f"**Signal not found:** {self.signal_id}")
            ]
            return
        
        if self.state.selected_time is None:
            self._container.objects = [
                pn.pane.Markdown(f"**{signal.name}**\n\nNo time selected.")
            ]
            return
        
        try:
            # Compute PSD
            t_center = self.state.selected_time
            half_w = self.window_s / 2.0
            t0 = max(0, t_center - half_w)
            t1 = t_center + half_w
            
            psd = signal.compute_psd(
                t0, t1,
                method=self.method,
                bandwidth=self.bandwidth
            )
            
            # Create plot
            plot = self._make_plot(psd, signal.name)
            
            self._container.objects = [
                pn.pane.HoloViews(plot, sizing_mode='stretch_both')
            ]
            
        except Exception as e:
            self._container.objects = [
                pn.pane.Markdown(f"**Error:**\n\n```\n{str(e)}\n```")
            ]
    
    def _make_plot(self, psd, signal_name):
        """Create HoloViews plot."""
        # Average across spatial dims
        if 'AP' in psd.dims and 'ML' in psd.dims:
            psd_mean = psd.mean(dim=['AP', 'ML'])
        elif 'channel' in psd.dims:
            psd_mean = psd.mean(dim='channel')
        else:
            psd_mean = psd
        
        curve = hv.Curve(
            (psd_mean.freq.values, psd_mean.values),
            kdims=['Frequency (Hz)'],
            vdims=['Power/Hz']
        ).opts(
            width=600,
            height=300,
            logy=True,
            tools=['hover'],
            title=f"PSD: {signal_name}"
        )
        
        return curve
    
    def panel(self):
        return self._container
```

---

### 5. **Updated TensorScopeState**

```python
class TensorScopeState(param.Parameterized):
    """
    Central state for TensorScope.
    
    Signal-centric design:
    - Manages SignalObjects (data + processing)
    - Manages Views (visualizations of signals)
    - Global navigation state (cursor, selected_time)
    - Global UI state (selection, etc.)
    """
    
    # Navigation
    time_hair = param.Parameter(default=None)  # TimeHair instance
    selected_time = param.Number(default=None)  # Analysis anchor
    time_window = param.Parameter(default=None)  # TimeWindowCtrl
    
    # Signal management (replaces data_registry + processing)
    signal_registry = param.Parameter(default=None)  # SignalRegistry
    
    # Events (unchanged)
    event_registry = param.Parameter(default=None)
    
    # UI state
    channel_grid = param.Parameter(default=None)  # ChannelGrid
    
    def __init__(self, data: xr.DataArray, **params):
        super().__init__(**params)
        
        # Initialize controllers
        self.time_hair = TimeHair()
        self.time_window = TimeWindowCtrl(
            bounds=(float(data.time.values[0]), float(data.time.values[-1]))
        )
        self.channel_grid = ChannelGrid.from_grid(data)
        
        # Initialize registries
        self.signal_registry = SignalRegistry()
        self.event_registry = EventRegistry()
        
        # Create base signal from input data
        base_signal = SignalObject(
            data=data,
            name="Raw LFP",
            processing=None,  # Will create default ProcessingChain
            metadata={'type': 'grid_lfp', 'is_base': True}
        )
        
        self.signal_registry.register(base_signal)
        
        # Set initial selected_time
        self.selected_time = float(data.time.values[0])
    
    def create_derived_signal(
        self,
        source_id: str,
        name: str,
        processing_config: dict | None = None
    ) -> str:
        """
        Create a new signal derived from existing signal.
        
        Returns new signal ID.
        """
        # Duplicate source
        new_id = self.signal_registry.duplicate(source_id, name)
        
        # Configure processing if provided
        if processing_config:
            signal = self.signal_registry.get(new_id)
            for key, value in processing_config.items():
                setattr(signal.processing, key, value)
        
        return new_id
    
    # Legacy compatibility
    @property
    def processing(self) -> ProcessingChain:
        """Get processing chain of active signal (legacy compat)."""
        signal = self.signal_registry.get_active()
        return signal.processing if signal else None
    
    @property
    def selected_channels_flat(self) -> list[int]:
        """Flat channel indices for selection."""
        from cogpy.tensorscope.schema import flatten_grid_to_channels
        selected = self.channel_grid.selected
        if not selected:
            return []
        
        # Get grid dimensions from active signal
        signal = self.signal_registry.get_active()
        if not signal:
            return []
        
        data = signal.data
        n_ml = data.sizes.get('ML', 0)
        
        # Convert (AP, ML) to flat index
        return [ap * n_ml + ml for ap, ml in selected]
```

---

### 6. **SignalManagerLayer - UI for Managing Signals**

```python
class SignalManagerLayer(TensorLayer):
    """
    UI for managing signal objects.
    
    Allows users to:
    - View all signals
    - Duplicate signals
    - Configure processing per signal
    - Delete derived signals
    - Switch active signal
    """
    
    def __init__(self, state):
        super().__init__(state)
        
        self.layer_id = "signal_manager"
        self.title = "Signal Manager"
        
        self._build_ui()
    
    def _build_ui(self):
        """Build signal management UI."""
        # Signal list
        self._signal_list = pn.widgets.Select(
            name='Signals',
            options={sig.name: sid for sid, sig in self.state.signal_registry.signals.items()},
            size=6
        )
        
        # Buttons
        self._duplicate_btn = pn.widgets.Button(
            name='📋 Duplicate',
            button_type='primary'
        )
        self._duplicate_btn.on_click(self._duplicate_signal)
        
        self._delete_btn = pn.widgets.Button(
            name='🗑️ Delete',
            button_type='danger'
        )
        self._delete_btn.on_click(self._delete_signal)
        
        # Processing controls for selected signal
        self._processing_pane = pn.Column()
        
        self._signal_list.param.watch(self._update_processing_controls, 'value')
        self._update_processing_controls()
        
        self._ui = pn.Column(
            pn.pane.Markdown("## Signals"),
            self._signal_list,
            pn.Row(self._duplicate_btn, self._delete_btn),
            pn.layout.Divider(),
            self._processing_pane,
            sizing_mode='stretch_width'
        )
    
    def _duplicate_signal(self, event):
        """Duplicate selected signal."""
        if not self._signal_list.value:
            return
        
        signal_id = self._signal_list.value
        signal = self.state.signal_registry.get(signal_id)
        
        # Generate new name
        import datetime
        new_name = f"{signal.name} ({datetime.datetime.now().strftime('%H:%M:%S')})"
        
        # Duplicate
        new_id = self.state.signal_registry.duplicate(signal_id, new_name)
        
        # Update UI
        self._refresh_signal_list()
        self._signal_list.value = new_id
    
    def _delete_signal(self, event):
        """Delete selected signal (if not base)."""
        if not self._signal_list.value:
            return
        
        signal_id = self._signal_list.value
        signal = self.state.signal_registry.get(signal_id)
        
        # Don't delete base signals
        if signal.metadata.get('is_base'):
            return
        
        self.state.signal_registry.remove(signal_id)
        self._refresh_signal_list()
    
    def _refresh_signal_list(self):
        """Refresh signal list options."""
        self._signal_list.options = {
            sig.name: sid 
            for sid, sig in self.state.signal_registry.signals.items()
        }
    
    def _update_processing_controls(self, *args):
        """Show processing controls for selected signal."""
        if not self._signal_list.value:
            return
        
        signal_id = self._signal_list.value
        signal = self.state.signal_registry.get(signal_id)
        
        if signal:
            self._processing_pane.objects = [
                pn.pane.Markdown(f"### {signal.name}"),
                signal.processing.controls()
            ]
    
    def panel(self):
        return self._ui
```

---

## Example Usage

```python
# Create app with base signal
data = example_ieeg_grid(mode="small")
app = TensorScopeApp(data, title="Signal-Centric TensorScope")

# Get base signal ID
base_id = list(app.state.signal_registry.signals.keys())[0]

# Create derived signals programmatically
high_gamma_id = app.state.create_derived_signal(
    base_id,
    "High Gamma (70-150Hz)",
    processing_config={
        'bandpass_on': True,
        'bandpass_lo': 70.0,
        'bandpass_hi': 150.0,
        'zscore_on': True
    }
)

# Create views
spatial_raw = SpatialView(app.state, base_id, mode='rms', render_mode='image')
spatial_hg = SpatialView(app.state, high_gamma_id, mode='rms', render_mode='image')
psd_raw = PSDView(app.state, base_id)
psd_hg = PSDView(app.state, high_gamma_id)

# Build layout
template = pn.template.FastGridTemplate(title="Multi-Signal View")
template.main[0:5, 0:6] = spatial_raw.panel()
template.main[0:5, 6:12] = spatial_hg.panel()
template.main[5:10, 0:6] = psd_raw.panel()
template.main[5:10, 6:12] = psd_hg.panel()
template.sidebar.append(SignalManagerLayer(app.state).panel())
```

---

## Summary

**This architecture:**
✅ Treats signals as first-class objects  
✅ Each signal = data + processing + metadata  
✅ Views render signals (spatial, timeseries, PSD, spectrogram)  
✅ Users duplicate signals → modify processing → compare  
✅ Clean separation: Signal (what) vs View (how)  
✅ Multi-modal ready: register different data types as signals  
✅ Extensible: new view types are just new classes  

**Does this match your vision? Should I proceed with the implementation prompt?**