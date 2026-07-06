import os
import sys
from importlib.util import find_spec

sys.path.insert(0, os.path.abspath("../.."))

project = "ARTist"
author = "Pankaj Kumar"
copyright = "2026, Pankaj Kumar"
release = "0.0.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

if find_spec("sphinx_copybutton"):
    extensions.append("sphinx_copybutton")

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
napoleon_google_docstring = False
napoleon_numpy_docstring = True

autodoc_mock_imports = [
    "cartopy",
    "cartopy.crs",
    "cartopy.feature",
    "cftime",
    "matplotlib",
    "matplotlib.pyplot",
    "matplotlib.cm",
    "matplotlib.collections",
    "numexpr",
    "scipy",
    "scipy.interpolate",
    "shapely",
]

templates_path = ["_templates"]
exclude_patterns = []

has_pydata_theme = find_spec("pydata_sphinx_theme") is not None
html_theme = "pydata_sphinx_theme" if has_pydata_theme else "alabaster"
html_static_path = ["_static"]
html_css_files = ["artist.css"]
html_logo = None
html_title = "ARTist"

html_theme_options = {}

if has_pydata_theme:
    html_theme_options.update(
        {
            "github_url": "https://github.com/pankajkarman/ARTist",
            "show_toc_level": 2,
            "navbar_align": "left",
            "navigation_with_keys": True,
        }
    )
