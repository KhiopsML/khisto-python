# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import re
import sys
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent
ROOT_DIR = DOCS_DIR.parent

sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "src"))


def _read_release() -> str:
    init_file = ROOT_DIR / "src" / "khisto" / "__init__.py"
    match = re.search(
        r'^__version__\s*=\s*"(?P<version>[^"]+)"',
        init_file.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    if match is None:
        raise RuntimeError(f"Could not determine khisto version from {init_file}")
    return match.group("version")

project = 'khisto-python'
copyright = '2026, The Khiops Team'
author = 'The Khiops Team'
release = _read_release()

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
# Do not be strict about any broken references, because of Sphinx limitations in
# getting the Numpy scalar type references:
# see https://github.com/sphinx-doc/sphinx/issues/10974
nitpicky = False

# To avoid using qualifiers like :class: to reference objects within the same context
default_role = "obj"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "numpydoc",
    "sphinx_copybutton",
    "nbsphinx",
]

## Numpydoc extension config
numpydoc_show_class_members = False

## Autodoc extension config
autodoc_default_options = {
    "members": True,
    "inherited-members": False,
    "private-members": False,
    "show-inheritance": True,
    "special-members": False,
}

## Intersphinx extension config
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}

templates_path = ['_templates']
exclude_patterns = ['_templates', '_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#FF7900",
        "color-brand-content": "#F16E00",
        "color-brand-visited": "#FF7900",
        "color-sidebar-background": "#FFFFFF",
        "color-highlighted-background": "#FFD200",
        "color-admonition-title--note": "#FF7900",
        "color-admonition-title-background--note": "#FFF0E2",
        "font-stack": "Helvetica Neue, Helvetica, sans-serif",
    },
    "dark_css_variables": {
        "color-brand-primary": "#FF7900",
        "color-brand-content": "#F16E00",
        "color-brand-visited": "#FF7900",
        "color-sidebar-background": "#000000",
        "color-highlighted-background": "#FFD200",
        "color-admonition-title--note": "#FF7900",
        "color-admonition-title-background--note": "#CC6100",
        "font-stack": "Helvetica Neue, Helvetica, sans-serif",
    },
    # Sets the Github Icon (the SVG is embedded, copied from furo's repo)
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/khiopsml/khisto-python",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
}
html_title = f"<h6><center>{project} {release}</center></h6>"
html_static_path = ['_static']
html_css_files = ["css/custom.css"]
