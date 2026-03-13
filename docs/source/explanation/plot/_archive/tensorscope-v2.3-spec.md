# TensorScope v2.3 Specification: View Builder UI & Advanced Modules

## Version
- Version: 2.3.0
- Date: 2026-03-04
- Status: Implementation
- Builds on: v2.2 (ViewSpec & ViewFactory)

## Overview

TensorScope v2.3 adds an interactive UI for building custom views and switching between preset modules. Users can compose views without writing code, save custom configurations, and explore advanced visualization patterns like montages and electrode panels.

v2.3 is additive and keeps v2.2 programmatic APIs stable:
- `ViewSpec` / `ViewFactory`
- `ViewPresetModule` / `ModuleRegistry`

## Core Features

### 1. View Builder UI

An interactive Panel UI for creating `ViewSpec` objects:
- Choose a signal (active or specific)
- Choose display dims (`kdims`) and control dims (`controls`)
- Select or override view type
- Configure display settings (colormap, symmetric clim, title)
- Preview the resulting `HoloViews` object

### 2. Module Selector UI

A small UI for switching between registered preset modules:
- Dropdown of modules (name + description)
- Load button that activates the selected module and renders its layout

### 3. Advanced Modules

Two additional built-in modules:
- **Montage**: multiple spatial Image views in a grid (currently share the cursor time control)
- **Electrode panel**: `hv.GridSpace` of timeseries for all (AP, ML) positions

### 4. Layout Persistence

Save/load a module (name/description/specs/layout) to JSON:

```json
{
  "name": "my_analysis",
  "description": "Custom analysis layout",
  "specs": [
    {
      "kdims": ["AP", "ML"],
      "controls": ["time"],
      "colormap": "RdBu_r",
      "symmetric_clim": true
    }
  ],
  "layout": "grid",
  "version": "2.3.0"
}
```

## File Structure (v2.3.0)

```text
cogpy/core/plot/tensorscope/
├── ui/
│   ├── __init__.py
│   ├── view_builder.py
│   └── module_selector.py
├── modules/
│   ├── montage.py
│   └── electrode_panel.py
└── layout_persistence.py
```

## Success Criteria

v2.3.0 is complete when:
- `ViewBuilderLayer` creates ViewSpecs interactively and previews the view
- `ModuleSelectorLayer` loads/activates modules
- Montage + electrode panel modules are available in `ModuleRegistry`
- `save_layout` / `load_layout` work for `ViewPresetModule`
- Tests cover UI construction + persistence
- Example demonstrates module selector + view builder

