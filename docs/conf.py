# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from pathlib import Path

# -- Project information -----------------------------------------------------

project = "Invariant Estimation"
author = "Lucas Libshutz, Beomyeong Park"
copyright = "2026, Lucas Libshutz, Beomyeong Park"

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",        # write pages in Markdown with LaTeX-style math
    "sphinx.ext.mathjax",  # render math (HTML) with MathJax
    "sphinx.ext.todo",
    "sphinx_copybutton",   # copy button on code blocks
    "sphinx_design",       # cards, grids, tabs, dropdowns
]

source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}

master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README.md"]

# -- MyST (Markdown) ---------------------------------------------------------
# Enables LaTeX-style math so you can write theory naturally:
#   inline:  $\rotmat{I}{B}\vbf{v}$
#   display: $$ \dfv{I}{p} = \angvel{I}{B}\times \fv{I}{p} $$
#   and AMS environments via the `amsmath` extension:
#       \begin{align} ... \end{align}
myst_enable_extensions = [
    "amsmath",        # \begin{align}, \begin{equation}, etc.
    "dollarmath",     # $...$ and $$...$$
    "colon_fence",    # ::: fenced directives
    "deflist",
    "smartquotes",
    "substitution",
]
myst_dmath_double_inline = True
myst_heading_anchors = 3

# -- LaTeX macros ------------------------------------------------------------
# Translated from preamble.sty. These are shared between the HTML (MathJax)
# and PDF (LaTeX) builds so the same source renders identically in both.
#
# MathJax format: no-arg macro -> "name": r"replacement"
#                 n-arg macro  -> "name": [r"replacement", n]
#
# Use MathJax 4. Its default font is New Computer Modern, which renders
# calligraphic letters (e.g. \mathcal{I}) like real LaTeX. MathJax 3's default
# font draws \mathcal{I} so it looks like a "J".
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@4/tex-mml-chtml.js"
mathjax3_config = {
    "tex": {
        "macros": {
            "vbs": [r"\vec{\boldsymbol{#1}}", 1],
            "vbf": [r"\vec{\mathbf{#1}}", 1],
            "hbs": [r"\hat{\boldsymbol{#1}}", 1],
            "hbf": [r"\hat{\mathbf{#1}}", 1],
            "dbf": [r"\dot{\mathbf{#1}}", 1],
            "tbf": [r"\tilde{\mathbf{#1}}", 1],
            "bs": [r"\boldsymbol{#1}", 1],
            "uv": [r"\hat{\mathbf{#1}}", 1],
            "wbs": [r"\widehat{\boldsymbol{#1}}", 1],
            "cbs": [r"\boldsymbol{\mathcal{#1}}", 1],
            "iv": [r"{}^\mathcal{I}\mathbf{#1}", 1],
            "id": r"{}^\mathcal{I}\frac{d}{dt}",
            "dcm": [r"{}^\mathcal{#1}C^{#2}", 2],
            "angvel": [r"{}^\mathcal{#1}\boldsymbol{\omega}^\mathcal{#2}", 2],
            "pd": r"\partial",
            "mag": [r"\| #1 \|", 1],
            "rotmat": [r"{}^\mathcal{#1}R_\mathcal{#2}", 2],
            "fv": [r"{}^\mathcal{#1}\mathbf{#2}", 2],
            "dfv": [r"{}^\mathcal{#1}\dot{\mathbf{#2}}", 2],
        }
    },
    # Use the original "MathJax TeX" font (what Obsidian renders math with).
    # Its calligraphic \mathcal{I} reads as an I; newcm/CM draws it J-like.
    "output": {"font": "mathjax-tex"},
}

# -- HTML output (Furo, dark-friendly) ---------------------------------------

html_theme = "furo"
html_title = "Invariant Estimation"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# Furo follows the OS light/dark preference and shows a toggle in the sidebar.
# These tweaks make the dark palette look good for math-heavy theory pages.
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#0b6bcb",
        "color-brand-content": "#0b6bcb",
    },
    "dark_css_variables": {
        "color-brand-primary": "#6cb6ff",
        "color-brand-content": "#6cb6ff",
        "color-background-primary": "#14171c",
        "color-background-secondary": "#1b1f26",
    },
}

# -- LaTeX / PDF output ------------------------------------------------------
# Reuse the real preamble.sty so PDF builds stay LaTeX-compatible.
_preamble = (Path(__file__).parent / "preamble.sty").read_text()
latex_elements = {
    "preamble": _preamble,
}

# -- todo --------------------------------------------------------------------
todo_include_todos = True
