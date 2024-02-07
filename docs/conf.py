# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import pathlib
import sys
import os
sys.path.insert(0, os.path.abspath('../'))

project = 'anomaly_lib'
copyright = '2024, -'
author = '-'


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.duration', 'sphinx.ext.doctest', 'sphinx.ext.autodoc',]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']

html_baseurl = ''
html_logo = '_static/img/logo.png'
html_favicon = '_static/img/logo.png'

html_css_files = [
    'css/custom.css',
]

html_theme_options = {
    "sidebar_hide_name": True,
    "light_css_variables": {
        "color-brand-primary": "#2e5c7d",
        "color-brand-content": "#2e5c7d",
        "codebgcolor": "red",
        "codetextcolor": "red",
    },
    "dark_css_variables": {
        "color-brand-primary": "#6998b4",
        "color-brand-content": "#6998b4",
        "codebgcolor": "green",
        "codetextcolor": "green",
    }

}