from alibi.imports import _BACKENDS_PRESENT
import logging

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


def set_backend(backend: str):
    if backend not in _BACKENDS:
        raise ValueError("Backend not supported. Supported backends: {}".format(_BACKENDS))
    else:
        global _CURRENT_BACKEND
        _CURRENT_BACKEND = backend
