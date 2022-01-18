"""
This script is an example of using `jupytext` to execute notebooks for testing instead of relying on `nbmake`
plugin. This approach may be more flexible if our requirements change in the future.
"""

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
NOTEBOOK_DIR = 'doc/source/examples'
ALL_NOTEBOOKS = {Path(x).name for x in glob.glob(str(Path(NOTEBOOK_DIR).joinpath('*.ipynb')))}

# The following set includes notebooks which are not to be executed during notebook tests.
# These are typically those that would take too long to run in a CI environment or impractical
# due to other dependencies (e.g. downloading large datasets
EXCLUDE_NOTEBOOKS = {
    # the following are all long-running
    'anchor_text_movie.ipynb',  # `autoregressive` filling example
    'cem_mnist.ipynb',  # black-box example
    'cfrl_mnist.ipynb',
    'cfrl_adult.ipynb',
    'cfproto_mnist.ipynb',  # black-box example
    'distributed_kernel_shap_adult_lr.ipynb',  # sequential explainer cell. (Also needs excluding from Windows)
    'interventional_tree_shap_adult_xgb.ipynb',  # comparison with KernelShap
    'kernel_shap_adult_lr.ipynb',  # slow to explain 128 instances
    'xgboost_model_fitting_adult.ipynb',  # very expensive hyperparameter tuning
    'integrated_gradients_transformers.ipynb',  # forward pass through BERT to get embeddings is very slow
}
EXECUTE_NOTEBOOKS = ALL_NOTEBOOKS - EXCLUDE_NOTEBOOKS


@pytest.mark.timeout(600)
@pytest.mark.parametrize("notebook", EXECUTE_NOTEBOOKS)
def test_notebook_execution(notebook):
    notebook = Path(NOTEBOOK_DIR, notebook)
    jupytext(args=[str(notebook), "--execute"])
