# Documentation for alibi

This directory contains the sources (`.md` and `.rst` files) for the
documentation. The main index page is defined in `source/index.rst`.
The Sphinx options and plugins are found in the `source/conf.py` file.
The documentation is generated in full by calling `make html` which
also automatically generates the Python API documentation from
docstrings.

## Building documentation locally
To build the documentation, first we need to install Python requirements:

`pip install -r ../requirements/requirements_ci.txt -r ../requirements/requirements.txt`

We also need `pandoc` for parsing Jupyter notebooks, the easiest way
to install this is using conda:

`conda install -c conda-forge pandoc=1.19`

Finally install the `alibi` package:

`make -C .. install`

We are now ready to build the docs:

`make html`

Note this can take some time as some of the notebooks may be executed
during the build process.

## Sphinx extensions and plugins
We use various Sphinx extensions and plugins to build the documentation:
 * `recommonmark` - to handle both `.rst` and `.md`
 * `sphinx.ext.napoleon` - support extracting Numpy style doctrings for API doc generation
 * `sphinx_autodoc_typehints` - support parsing of typehints for API doc generation
 * `sphinxcontrib.apidoc` - automatic running of `sphinx-apidoc` during the build to document API
 * `nbsphinx` - parsing Jupyter notebooks to generate static documentation
 * `nbsphinx_link` - support linking to notebooks outside of Sphinx source directory via `.nblink` files

The full list of plugins and their options can be found in `source/conf.py`.