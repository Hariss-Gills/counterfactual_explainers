# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

# Adjust the path to point to your project's root directory (one level up from docs)
sys.path.insert(0, os.path.abspath(".."))

project = "Counterfactual Explainers"
copyright = "2025, Hariss Ali Gills"
author = "Hariss Ali Gills"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # Core library for extracting docs from docstrings
    "sphinx.ext.napoleon",  # Support for Google and NumPy style docstrings
    "sphinx.ext.viewcode",  # Add links to highlighted source code
    "sphinx.ext.intersphinx",  # Link to other projects' documentation
    # ... other extensions
]

napoleon_google_docstring = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    "papersize": "a4paper",
    # The font size ('10pt', '11pt' or '12pt').
    "pointsize": "10pt",
    # Additional stuff for the LaTeX preamble.
    # 'preamble': '',
    # Latex figure (float) alignment
    # 'figure_align': 'htbp',
    # Example: Add packages or custom commands
    # 'preamble': r'''
    # \usepackage{amsmath}
    # \usepackage{amsfonts}
    # \usepackage{amssymb}
    # ''',
    # Set a specific Sphinx theme for LaTeX
    # 'sphinxsetup': 'verbatimwithframe=true, VerbatimColor={rgb}{0.95,0.95,0.95}',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual]).
# Make sure 'index' matches your master_doc (usually index.rst)
latex_documents = [
    (
        "index",
        "MyProject.tex",
        "My Project Documentation",
        "Your Name",
        "manual",
    ),  # 'manual' or 'howto'
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
# latex_logo = '_static/logo.png'

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
# latex_use_parts = False

# If true, show page references after internal links.
# latex_show_pagerefs = False

# If true, show URL addresses after external links.
# latex_show_urls = 'footnote' # or 'inline' or 'no'

# Documents to append as an appendix to all manuals.
# latex_appendices = []

# Control whether to display URL addresses.
# latex_domain_indices = True
