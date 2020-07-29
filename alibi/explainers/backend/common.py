# flake8: noqa: F401
import inspect
import importlib

from typing_extensions import Literal
from alibi.explainers.backend import backends_registry


def load_backend(class_name: str,
                 framework: Literal['pytorch', 'tensorflow'],
                 predictor_type: Literal['blackbox', 'whitebox']):
    """
    Update the backend registry returns the backend for the class specified by `class_name`. _It assumes that the name 
    of the module where the backend is implemented is the same as the name of the calling context._

    Parameters
    ----------
    class_name
        The name of the class whose backend will be loaded.
    framework: {'pytorch', 'tensorflow'}
        Indicates which backend should be loaded for `class_name`.
    predictor_type: {'blackbox', 'whitebox'}
        'whitebox' indicates that the registered backend has access to the predictor parameters (e.g., it can use the
        framework autograd functions for differentiation). `blackbox` indicates that the predictor should be treated as
        a black-box (e.g., numerical differentiation is needed to differentiate the predictor output wrt its input).
    """  # noqa W605

    # find out where the loader is called from
    caller_file = inspect.stack()[1].filename
    # assume a backend module with the same name as the caller exists
    module_name = caller_file.split("/")[-1][:-3]  # exclude .py
    backend_module = ".".join([__package__, framework, module_name])
    # update the backend registry
    importlib.import_module(backend_module)

    return backends_registry[framework][class_name][predictor_type]
