# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(1, os.path.abspath('../../federated_learning/client'))
sys.path.insert(2, os.path.abspath('../../dicom_pseudonymizer/'))

# autodoc_mock_imports = ["pytorch" , "opacus", "flwr", "torchvision", "fastai", "fastcore"]

# -- Project information -----------------------------------------------------

project = 'D-SAIL'
copyright = '2021, D-SAIL'
author = 'D-SAIL'

# The full version, including alpha/beta/rc tags
release = '0.1'


# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'numpydoc'
]

numpydoc_show_class_members = False
autosummary_generate = True

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'
html_static_path = [] # = ['_static']

# -- Options for EPUB output
epub_show_urls = 'footnote'

html_logo = '../img/d-sail_square.png'
