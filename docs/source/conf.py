"""Configure sphinx documentation builds."""

# -- Path setup --------------------------------------------------------------
import os
from os.path import abspath
import sys

sys.path.append(os.path.abspath("../src/arpes"))
sys.path.append(os.path.abspath(".."))

import warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

import datetime

import arpes
import arpes.config
from arpes.io import example_data

# -- Project information -----------------------------------------------------
project = "arpes"
CURRENT_YEAR = datetime.datetime.now().year
copyright = f"2018-2020, Conrad Stansbury, 2023-{CURRENT_YEAR}, Ryuichi Arafune"
author = "Conrad Stansbury"
maintainer = "Ryuichi Arafune"

# The short X.Y version
version = ".".join(arpes.__version__.split(".")[:2])
# The full version, including alpha/beta/rc tags
release = arpes.__version__

# suppress some output information for nbconvert, don't open tools

nbsphinx_allow_errors = True


# -- Options for rst extension -----------------------------------------------

rst_file_suffix = ".rst"
rst_link_suffix = ""  # we will generate a webpage with docsify so leave this blank


def transform_rst_link(docname):
    """Make sure links to docstrings are rendered at the correct relative path."""
    return "api/rst/" + docname + rst_link_suffix


rst_link_transform = transform_rst_link

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinxcontrib.restbuilder",
    # "sphinxcontrib.katex",
    "sphinx_rtd_theme",
    "nbsphinx",
    "sphinx_copybutton",
    "sphinxnotes.strike",
]

suppress_warnings = [
    "nbsphinx",
]


apidoc_separate_modules = True

katex_version = "0.13.13"
katex_css_path = f"https://cdn.jsdelivr.net/npm/katex@{katex_version}/dist/katex.min.css"
katex_js_path = f"https://cdn.jsdelivr.net/npm/katex@{katex_version}/dist/katex.min.js"
katex_inline = [r"\(", r"\)"]
katex_display = [r"\[", r"\]"]
katex_prerender = False
katex_options = ""


# autodoc settings
def autodoc_skip_member(app, what, name, obj, skip, options):
    """Don't include parts of code which require optional dependencies for now."""
    # This is a noop for now
    return skip


def setup(app):
    app.connect("autodoc-skip-member", autodoc_skip_member)


autodoc_mock_imports = ["torch", "pytorch_lightning"]


# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# nbsphinx settings
nbsphinx_timeout = 600

nbsphinx_execute = "never" if os.getenv("READTHEDOCS") else "always"


autosummary_generate = True


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

language = "en"

exclude_patterns = []

pygments_style = "sphinx"


# HTML Configuration
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["style.css"]
html_logo = "_static/PyARPES-Logo.svg"
html_theme_options = {
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["navbar-icon-links"],
    "navbar_persistent": ["search-button"],
}


# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "arpesdoc"


# -- Options for LaTeX output ------------------------------------------------

latex_engine = "xelatex"
latex_elements = {
    'papersize': 'a4paper',
    'pointsize': '12pt',
    'fontpkg': r'''
        \usepackage{fontspec}
        \setmainfont{Times New Roman}
    ''',
}
latex_documents = [
    (
        master_doc,
        "arpes.tex",
        "arpes Documentation",
        "Conrad Stansbury/Ryuichi Arafune (>= V4)",
        "manual",
    ),
]

# -- Options for manual page output ------------------------------------------
man_pages = [(master_doc, "arpes", "arpes Documentation", [author], 1)]


# -- Options for Texinfo output ----------------------------------------------
texinfo_documents = [
    (
        master_doc,
        "arpes",
        "arpes Documentation",
        author,
        "arpes",
        "One line description of project.",
        "Miscellaneous",
    ),
]

# -- Options for Epub output -------------------------------------------------
epub_title = project
epub_exclude_files = ["search.html"]

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True
