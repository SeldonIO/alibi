# Contributing Code

## Style
We follow PEP8 with the exception of maximum line length set to 120. We
use `numpy` style docstrings with the exception of ommitting argument
types in the docstrings in favour of type hint in function and class
signatures. We use `mypy` to typecheck code with exceptions for
libraries with missing stubs specified in `setup.cfg`.

## Pull requests
Before a PR is accepted, CircleCI builds must pass. We use `flake8` for
linting (with exceptions to PEP8 described in `setup.cfg` and `pytest`
for testing. We use `sphinx` for building documentation.