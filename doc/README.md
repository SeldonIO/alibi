# Documentation for alibi

This directory contains the sources (`.md`, `.rst` and `.ipynb` files) for the
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
during the build process. The resulting documentation is located in the
`_build` directory with `_build/html/index.html` marking the homepage.

## Sphinx extensions and plugins
We use various Sphinx extensions and plugins to build the documentation:
 * [m2r](https://github.com/miyakogi/m2r) - to handle both `.rst` and `.md`
 * [sphinx.ext.napoleon](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html) - support extracting Numpy style doctrings for API doc generation
 * [sphinx_autodoc_typehints](https://github.com/agronholm/sphinx-autodoc-typehints) - support parsing of typehints for API doc generation
 * [sphinxcontrib.apidoc](https://github.com/sphinx-contrib/apidoc) - automatic running of [sphinx-apidoc](https://www.sphinx-doc.org/en/master/man/sphinx-apidoc.html) during the build to document API
 * [nbsphinx](https://nbsphinx.readthedocs.io) - parsing Jupyter notebooks to generate static documentation
 * [nbsphinx_link](https://nbsphinx-link.readthedocs.io) - support linking to notebooks outside of Sphinx source directory via `.nblink` files

The full list of plugins and their options can be found in `source/conf.py`.

## Adding new examples
All examples are Jupyter notebooks and live in the top level `examples` directory. To make them available as documentation, create an `.nblink` file under `doc/source/examples` which is a piece of json pointing to the `.ipynb` example notebook. E.g. if there is a notebook called `examples/notebook.ipynb`, then create a file `doc/source/examples/notebook.nblink` with the following contents:
```json
{
  "path": "../../../examples/notebook.ipynb"
}
```
From here on you can link and refer to the `notebook.ipynb` elsewhere in the documentation as if it lived under `doc/source/examples`.

## Gotchas when writing notebooks as examples
We use Jupyter notebooks for examples and method descriptions and invoke the [nbsphinx](https://nbsphinx.readthedocs.io) plugin for rendering the notebooks as static documentation. Generally, the Jupyter notebook is more permissive for what it can render correctly than the static documentation, so it is important to check that the content is rendered correctly in the static docs as well. Here is a list of common formatting gotchas and how to fix them:
* When using a bullet-point list, leave a blank line before the preceding paragraph, otherwise it will fail to render
* Always use `$$ $$` or `\begin{equation}\end{equation}`to delimit display math
* For references and footnotes, the tag indicating the section needs to start with an uppercase letter, e.g. `[[1]](#References)` linking to a section `<a id='References'></a>
[1](#f_1) reference here`
* Whilst superscript (for e.g. footnotes) can be rendered in Jupyter using `<sup></sup>` tags, this won't work in the static docs. To avoid jarring appearence of footnote numbers in the text, wrap them in parentheses, e.g. <sup>`(1)`</sup> will be rendered inline as `(1)`.
