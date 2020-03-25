# Contributing Code

We welcome PRs from the community. This document outlines the standard
practices and development tools we use.

## Getting started
The easiest way to get started is to install all the development dependencies
in a separate virtual environment:
```
pip install -r requirements/requirements_all.txt
```
This will install everything needed to run alibi and all the dev tools
(docs builder, testing, linting etc.)

## Git pre-commit hooks
To make it easier to format code locally before submitting a PR, we provide
integration with `pre-commit` to run `flake8` and `mypy` hooks before every commit.
After installing the development requirements and cloning the package, run `pre-commit install`
from the project root to install the hooks locally. Now before every `git commit ...`
these hooks will be run to verify that the linting and type checking is correct. If there are
errors, the commit will fail and you will see the changes that need to be made.

## Testing
We use `pytest` to run tests. To run all tests just call `pytest` from the root of the project.
Test files live together with the library files under `tests` folders.

## Linter
We use `flake8` for linting adhering to PEP8 with exceptions defined in `setup.cfg`.

## Type checking
We use type hints to develop the libary and `mypy` to for static type checking. Some
options are defined in `setup.cfg`.

## Docstrings
We adhere to the `numpy` style docstrings (https://numpydoc.readthedocs.io/en/latest/format.html)
with the exception of ommiting argument types in docstrings in favour of type hints in function
and class signatures. If you're using a `PyCharm`, you can configure this under
`File -> Settings -> Tools -> Python Integrated Tools -> Docstrings`.

## Building documentation
We use `sphinx` for building documentation. You can call `make build_docs` from the project root,
the docs will be built under `doc/_build/html`. Detail information about documentation can be found [here](doc/README.md).

## CI
All PRs triger a TravisCI build to run linting, type checking, tests, and build docs.
