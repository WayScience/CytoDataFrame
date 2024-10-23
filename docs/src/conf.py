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
import pathlib
import sys

from dunamai import Version

basedir = str(pathlib.Path(__file__).parent.parent.parent.resolve())

sys.path.insert(0, basedir)

# -- Project information -----------------------------------------------------

project = "CytoDataFrame"
# is used here due to sphinx decision-making: https://github.com/sphinx-doc/sphinx/issues/8132
copyright = "2024, WayScience Community"  # noqa: A001
author = "WayScience Community"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_multiversion",
    "sphinx_wagtail_theme",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

html_sidebars = {
    "**": [
        "searchbox.html",
        "globaltoc.html",
        "relations.html",
        "versioning.html",
    ],
}

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []  # type: ignore

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_wagtail_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# set theme options
html_theme_options = {
    "project_name": "coSMicQC",
    "logo": "just-icon.png",
    "header_links": "<span class='fab fa-github'></span> GitHub Repository|https://github.com/WayScience/coSMicQC",
    "logo_alt": "coSMicQC logo",
    "logo_height": 59,
    "logo_url": "https://wayscience.github.io/coSMicQC",
    "logo_width": 45,
    "github_url": "https://github.com/WayScience/coSMicQC/tree/main/docs/src",
    "footer_links": ",".join(
        [
            "Way Lab|https://www.waysciencelab.com/",
            "coSMicQC GitHub Repository|https://github.com/WayScience/coSMicQC",
        ]
    ),
}

# Options for myst-nb
# turn off notebook execution for docs builds
# (we rely on the notebook already being processed
# prior to the publish to help navigate compute needs)
nb_execution_mode = "off"

# set option to avoid rendering default variables
autodoc_preserve_defaults = True

# enable anchor creation
myst_heading_anchors = 3

# sphinx-multiversion configuration
# latest version reference
smv_latest_version = f"v{Version.from_git().base}"

# only include tags which match semver standards (e.g. v0.0.0)
smv_tag_whitelist = r"^v\d+\.\d+\.\d+$"

# Pattern for released versions
smv_released_pattern = r"^tags/v\d+\.\d+\.\d+$"

# Format for versioned output directories inside the build directory
smv_outputdir_format = "{ref.name}"

# Whitelist pattern for branches (set to None to ignore all branches)
smv_branch_whitelist = r"main"

# Whitelist pattern for remotes (set to None to use local branches only)
smv_remote_whitelist = None
