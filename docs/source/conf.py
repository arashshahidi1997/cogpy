"""Sphinx configuration for the cogpy package."""

from __future__ import annotations

import os
import sys
from datetime import datetime

# Ensure the package is importable when autodoc runs.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

project = "cogpy"
author = "Arash Shahidi"
copyright = f"{datetime.now():%Y}, cogpy"
release = "0.1.0"

extensions = [
    "myst_nb",
    "sphinx.ext.mathjax",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_design",
]

autodoc_typehints = "description"

autosummary_generate = True
autosummary_generate_overwrite = True
napoleon_google_docstring = False
napoleon_numpy_docstring = True

templates_path = ["_templates"]
exclude_patterns: list[str] = []

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = [
    "custom.css",
    "chatbot.css",
    "https://cdn.bokeh.org/bokeh/release/bokeh-3.3.2.min.css",
]
html_js_files = [
    "https://cdn.jsdelivr.net/npm/marked/marked.min.js",  # markdown renderer
    "chatbot.js",
    "https://cdn.bokeh.org/bokeh/release/bokeh-3.3.2.min.js",
    "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-3.3.2.min.js",
    "https://cdn.bokeh.org/bokeh/release/bokeh-tables-3.3.2.min.js",
]
nb_execution_mode = "cache"
myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
]

# Ensure code inputs/outputs are not globally removed
nb_remove_code_source = False
nb_remove_code_outputs = False

# Better markdown rendering for cell outputs that contain markdown
nb_render_markdown_format = "myst"
