# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


import os, subprocess
import sphinx_rtd_theme

def configureDoxyfile():
    """
    Updates input and output directories in Doxyfile.
    """
    work_dir = os.getcwd()
    docs_dir = os.path.join(work_dir,"docs")
    src_dir = os.path.join(work_dir,"src")
    output_dir = os.path.join(work_dir, "build","docs")
    template_doxyfile = os.path.join(docs_dir,"Doxyfile")
    output_doxyfile = os.path.join(output_dir,"Doxyfile")
    with open(template_doxyfile, 'r') as file :
        filedata = file.read()

    filedata = filedata.replace('@DOXYGEN_INPUT_DIR@', src_dir)
    filedata = filedata.replace('@DOXYGEN_OUTPUT_DIR@', output_dir)

    with open(output_doxyfile, 'w') as file:
        file.write(filedata)

# Check if we're running on Read the Docs' servers
read_the_docs_build = os.environ.get('READTHEDOCS', None) == 'True'

# breathe_projects = {"Machine Learning":'build/docs/doxygen/xml'}
breathe_projects = {}
if read_the_docs_build:
    input_dir = '../src'
    output_dir = 'build'
    configureDoxyfile(input_dir, output_dir)
    subprocess.call('doxygen', shell=True)
    breathe_projects['MachineLearning'] = output_dir + '/docs/doxygen/xml'


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