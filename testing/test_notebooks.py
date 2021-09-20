"""
This script is an example of using `jupytext` to execute notebooks for testing instead of relying on `nbmake`
plugin. This approach may be more flexible if our requirements change in the future.
"""

import argparse
import glob
import pytest
from jupytext.cli import jupytext

# list of all example notebooks
notebooks = glob.glob('examples/*.ipynb')


@pytest.mark.timeout(300)
@pytest.mark.parametrize("notebook", notebooks)
def test_notebook_execution(notebook):
    jupytext(args=[notebook, "--execute"])


if __name__ == '__main__':
    # When run as a script, execute the notebook passed on the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('notebook', type=str)
    args = parser.parse_args()

    test_notebook_execution(args.notebook)
