from alibi.imports import _BACKENDS_PRESENT
from alibi.explainers import DISPATCH_DICT
from alibi.explainers.base import Explanation
import logging

# from alibi.explainers.anchor_tabular import _show_anchortabular

logger = logging.getLogger(__name__)

_DEFAULT_BACKEND = 'altair'
_CURRENT_BACKEND = _DEFAULT_BACKEND
_BACKENDS = ['altair', 'matplotlib', 'plaintext']

if not any(_BACKENDS_PRESENT):
    logger.warning("No visualization backend found, will use `plaintext` to display explanations.")
    _CURRENT_BACKEND = 'plaintext'
else:
    _CURRENT_BACKEND = _BACKENDS[next(i for i, v in enumerate(_BACKENDS_PRESENT) if v)]
    logger.info("Visualization backend set to {}".format(_CURRENT_BACKEND))


def set_backend(backend: str) -> None:
    """
    Set the visualization backend.

    Parameters
    ----------
    backend
        String representing the chosen backend in _BACKENDS
    """
    if backend not in _BACKENDS:
        raise ValueError("Backend not supported. Supported backends: {}".format(_BACKENDS))
    else:
        global _CURRENT_BACKEND
        _CURRENT_BACKEND = backend


def show(exp: Explanation, **kwargs) -> None:
    """
    Main visualization function. Will dispatch based on the `name` attribute of the Explanation
    instance to specific implementations.

    Parameters
    ----------
    exp
        Instance of an Explanation object
    kwargs
        Additional keyword arguments passed to specific implementations
    """
    try:
        DISPATCH_DICT[exp.name](exp, **kwargs)  # type: ignore
    except KeyError:
        raise NotImplementedError("Visualization for `{}` not implemented".format(exp.name))  # type: ignore
