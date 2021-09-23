"""
This script is an example of using `jupytext` to execute notebooks for testing instead of relying on `nbmake`
plugin. This approach may be more flexible if our requirements change in the future.
"""

import argparse
import glob
from pathlib import Path
import pytest
from jupytext.cli import jupytext

# Set of all example notebooks
# NOTE: we specifically get only the name of the notebook not the full path as we want to
# use these as variables on the command line for `pytest` for the workflow executing only
# changed notebooks. `pytest` does not allow `/` as part of the test name for the -k argument.
# This also means that the approach is limited to all notebooks being in the `NOTEBOOK_DIR`
# top-level path.
NOTEBOOK_DIR = 'examples'
ALL_NOTEBOOKS = {Path(x).name for x in glob.glob('examples/*.ipynb')}
# notebooks = glob.glob('examples/*.ipynb')

# The following set includes notebooks which are not to be executed during notebook tests.
# These are typically those that would take too long to run in a CI environment or impractical
# due to other dependencies (e.g. downloading large datasets
EXCLUDE_NOTEBOOKS = {
    'test-exclude.ipynb'
}

EXECUTE_NOTEBOOKS = ALL_NOTEBOOKS - EXCLUDE_NOTEBOOKS


@pytest.mark.timeout(300)
@pytest.mark.parametrize("notebook", EXECUTE_NOTEBOOKS)
def test_notebook_execution(notebook):
    notebook = Path(NOTEBOOK_DIR, notebook)
    jupytext(args=[notebook, "--execute"])


if __name__ == '__main__':
    # When run as a script, execute the notebook passed on the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('notebook', type=str)
    args = parser.parse_args()

    test_notebook_execution(args.notebook)
