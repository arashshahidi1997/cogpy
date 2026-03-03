# TensorScope Known Issues and Workarounds

This document tracks known issues, Panel/HoloViews quirks, and their solutions.

> For environment/runtime setup commands, see [tensorscope-runtime.md](tensorscope-runtime.md).

---

## Issue 1: FastGridTemplate `.append()` Not Supported (Panel 1.8.8)

**Symptoms:**
```python
tmpl = pn.template.FastGridTemplate(...)
tmpl.main.append(widget)  # AttributeError: 'GridSpec' object has no attribute 'append'
```

**Root cause:**  
In Panel 1.8.8, `FastGridTemplate.main` is a `GridSpec` object. Unlike `FastListTemplate.main` (which is a list-like container), `GridSpec` requires dictionary-style grid assignment.

**Workaround: Use grid assignment with FastGridTemplate**
```python
# For complex multi-panel layouts
tmpl = pn.template.FastGridTemplate(...)

# Dictionary-style assignment (row:row, col:col)
tmpl.main[0:4, 0:12] = widget1  # ✅ Works
tmpl.main[4:8, 0:12] = widget2
```

**Also: Pass sidebar at init (not append)**
```python
# WRONG
tmpl = pn.template.FastGridTemplate(...)
tmpl.sidebar.append(widget)  # May fail in some Panel versions

# RIGHT
tmpl = pn.template.FastGridTemplate(
    sidebar=[widget],  # Pass as list
    ...
)
```

**Decision for TensorScope:**
- **Phase 0–3:** Use `FastGridTemplate` with grid assignment (full layout control)
- Avoid `FastListTemplate` in core demos (less layout control)

**References:**
- Panel docs: https://panel.holoviz.org/reference/templates/FastGridTemplate.html
- Working example: `notebooks/tensorscope/tensorscope_app.py`

---

## Issue 2: [Placeholder for future issues]

(To be added as encountered)

---

## Issue Reporting

When adding new issues to this document:

1. **Title:** Clear, searchable description
2. **Symptoms:** Error message or unexpected behavior
3. **Root cause:** Technical explanation
4. **Workarounds:** 2-3 solutions with code examples
5. **Decision:** Which approach TensorScope uses
6. **References:** Links to docs, issues, working examples

Keep entries concise. Link to relevant sections of tensorscope-spec.md or tensorscope-plan.md where applicable.
