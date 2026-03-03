import panel as pn
import pandas as pd
import numpy as np
import holoviews as hv

# Added a fallback for the custom theme imports so the script runs standalone
try:
    from cogpy.core.plot.theme import (
        BG, BG_PANEL, BORDER, TEXT, BLUE, TEAL, PALETTE, COLORMAPS
    )
except ImportError:
    # Default fallback dark theme colors
    BG = "#121212"
    BG_PANEL = "#1e1e1e"
    BORDER = "#333333"
    TEXT = "#ffffff"
    BLUE = "#1f77b4"
    TEAL = "#008080"
    PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    COLORMAPS = {'viridis': 'viridis'}

pn.extension('tabulator')
hv.extension('bokeh')

class TensorScopeApp:
    def __init__(self, data=None):
        # Mock data
        self.fs = 2000
        self.duration = 10
        self.time = np.linspace(0, self.duration, self.fs * self.duration)
        self.n_channels = 16
        
        self.traces = np.random.randn(self.n_channels, len(self.time)).cumsum(axis=1)
        self.spatial_data = np.random.randn(4, 4)
        
        self.selected_channels = [0, 1, 2, 3]
            
    def build(self):
        # ==========================================
        # SIDEBAR (Recreated to prevent NameError)
        # ==========================================
        sidebar = pn.Column(
            pn.pane.Markdown("### ⚙️ Settings", styles={'color': TEXT}),
            pn.widgets.Select(name='Dataset', options=['Session_A', 'Session_B']),
            pn.widgets.MultiChoice(
                name='Active Channels', 
                value=self.selected_channels, 
                options=list(range(self.n_channels))
            ),
            pn.widgets.Toggle(name='Sync Plots', button_type='primary', value=True),
            sizing_mode='stretch_width'
        )
        
        # ==========================================
        # PANELS (FIXED SIZING)
        # ==========================================
        
        # Spatial
        spatial_plot = hv.Image(self.spatial_data).opts(
            cmap=COLORMAPS.get('viridis', 'viridis'),
            colorbar=True,
            tools=['hover', 'tap'],
            bgcolor=BG,
            responsive=True 
        )
        
        spatial_card = pn.Card(
            pn.pane.HoloViews(spatial_plot, sizing_mode='stretch_both'),
            title="🗺️ Spatial Map",
            header_background=BLUE,
            styles={'background': BG_PANEL},
            sizing_mode='stretch_both' 
        )
        
        # Events (Tabulator)
        events_df = pd.DataFrame({
            'Time (s)': [1.25, 3.50, 5.82, 7.21, 8.45],
            'Type': ['Burst', 'Ripple', 'Burst', 'Spindle', 'Ripple'],
            'Ch': [2, 5, 3, 1, 7],
            'Amp': [3.2, 2.1, 4.5, 1.8, 2.9],
        })
        
        events_card = pn.Card(
            pn.Column(
                pn.Row(
                    pn.widgets.Button(name='◀ Prev', width=70, button_type='primary'),
                    pn.widgets.Button(name='Next ▶', width=70, button_type='primary'),
                    pn.Spacer(),
                    sizing_mode='stretch_width'
                ),
                pn.widgets.Tabulator(
                    events_df,
                    show_index=False,
                    theme='midnight',
                    sizing_mode='stretch_both' 
                ),
                pn.widgets.Checkbox(name='Link to time', value=True),
                sizing_mode='stretch_both'
            ),
            title="📋 Events",
            header_background=TEAL,
            styles={'background': BG_PANEL},
            sizing_mode='stretch_both' 
        )
        
        # Timeseries 
        curves = [
            hv.Curve((self.time, self.traces[i]), label=f'Ch {i}').opts(
                color=PALETTE[i % len(PALETTE)] if PALETTE else 'blue',
                line_width=1.5
            )
            for i in self.selected_channels
        ]
        
        ts_plot = hv.Overlay(curves).opts(
            legend_position='right',
            bgcolor=BG,
            show_grid=True,
            responsive=True 
        )
        
        ts_card = pn.Card(
            pn.Column(
                pn.Row(
                    pn.widgets.Player(start=0, end=100, width=150), 
                    pn.widgets.FloatSlider(
                        name='Time (s)', start=0, end=self.duration, value=0,
                        sizing_mode='stretch_width' 
                    ),
                    sizing_mode='stretch_width'
                ),
                pn.Row(
                    pn.pane.Markdown("**Window:**", width=80),
                    *[pn.widgets.Button(name=label, width=65) for label in ['10ms', '100ms', '1s', 'Full']],
                    sizing_mode='stretch_width'
                ),
                pn.pane.HoloViews(ts_plot, sizing_mode='stretch_both'),
                sizing_mode='stretch_both'
            ),
            title="📈 Timeseries",
            header_background=BLUE,
            styles={'background': BG_PANEL},
            sizing_mode='stretch_both' 
        )
        
        # ==========================================
        # TEMPLATE
        # ==========================================
        
        tmpl = pn.template.FastGridTemplate(
            title="🧠 TensorScope",
            sidebar=[sidebar],
            sidebar_width=280,
            theme='dark',
            prevent_collision=True,
            row_height=60 
        )
        
        tmpl.main[0:6, 0:6] = spatial_card
        tmpl.main[0:6, 6:12] = events_card
        tmpl.main[6:12, 0:12] = ts_card
        
        return tmpl

# # Serve
# app = TensorScopeApp()
# app.build().servable()


if __name__ == "__main__":
    app = TensorScopeApp()
    pn.serve(app.build, show=True)