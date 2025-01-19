# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


import os, subprocess
from pathlib import Path

def configureDoxyfile(base_dir):
    """
    Updates input and output directories in Doxyfile.
    """
    docs_dir = os.path.join(base_dir,"docs")
    src_dir = os.path.join(base_dir,"src")
    output_dir = os.path.join(base_dir, "build","docs","doxygen")
    template_doxyfile = os.path.join(docs_dir,"Doxyfile")
    output_doxyfile = os.path.join(output_dir,"Doxyfile")
    with open(template_doxyfile, 'r') as file :
        filedata = file.read()

    filedata = filedata.replace('@DOXYGEN_INPUT_DIR@', src_dir)
    filedata = filedata.replace('@DOXYGEN_OUTPUT_DIR@', output_dir)

    subprocess.call(f"mkdir -p {output_dir}", shell=True)
    subprocess.call(f"touch {output_doxyfile}", shell=True)
    with open(output_doxyfile, 'w') as file:
        file.write(filedata)

# Check if we're running on Read the Docs' servers
read_the_docs_build = os.environ.get('READTHEDOCS', None) == 'True'

breathe_projects = {}
base_dir = Path(__file__).parent.parent.parent
breathe_projects['MachineLearning'] = os.path.join(base_dir,"build", "docs","doxygen","xml")
if read_the_docs_build:
    configureDoxyfile(base_dir)
    subprocess.call('cd ..; doxygen', shell=True)


html_theme = "sphinx_rtd_theme"

def setup(app):
    app.add_css_file("main_stylesheet.css")

extensions = ['breathe', 'sphinx_rtd_theme','myst_parser']
breathe_default_project = "MachineLearning"
templates_path = ['_templates']
html_static_path = ['_static']
source_suffix = '.md'
master_doc = 'index'
project = 'Machine Learning'
copyright = '2025, Nagarajan Karthik'
author = 'Nagarajan Karthik'
release = '1.0'

exclude_patterns = []
highlight_language = 'c++'
pygments_style = 'sphinx'