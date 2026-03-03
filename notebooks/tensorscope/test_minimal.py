import panel as pn
import pandas as pd
import numpy as np
import holoviews as hv

pn.extension('tabulator')
hv.extension('bokeh')

# Simple test without themes first
print("Starting app...")

# Create simple data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create simple plot
curve = hv.Curve((x, y)).opts(width=600, height=300)

# Create simple table
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
table = pn.widgets.Tabulator(df, show_index=False)

# Create template
tmpl = pn.template.FastGridTemplate(
    title="Debug Test",
    theme='dark'
)

# Add components
tmpl.sidebar.append(pn.pane.Markdown("## Sidebar Test"))
tmpl.main[0:4, 0:6] = pn.Card(pn.pane.HoloViews(curve), title="Plot")
tmpl.main[0:4, 6:12] = pn.Card(table, title="Table")

print("Template created, calling servable()...")

tmpl.servable()

print("App should be running!")
