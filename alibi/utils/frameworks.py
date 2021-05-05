# flake8: noqa: F401
import logging
from typing import Optional

FRAMEWORKS = ['pytorch', 'tensorflow']

try:
    import tensorflow as tf

    has_tensorflow = True
except ImportError:
    has_tensorflow = False

try:
    import torch

    has_pytorch = True
except ImportError:
    has_pytorch = False


def infer_device(predictor, predictor_type: str, framework: str) -> Optional[str]:
    """
    A function that returns the device on which a predictor.

    Parameters
    ----------
    predictor
        A predictor, implemented in PyTorch or Tensorflow.
    predictor_type: {'blackbox', 'whitebox}
        Indicates whether the caller has access to the predictor parameters of not. Returns `None` for blackbox
        predictors since the device cannot be inferred in this case.
    framework: {'pytorch', 'tensorflow}
        The framework in which the predictor is implemented. `None` is returned for `'tensorflow'` since this framework
        handles device automatically.
    """

    if framework == 'tensorflow':
        return  # type: ignore
    if predictor_type == 'blackbox':
        return  # type: ignore

    default_model_device = next(predictor.parameters()).device
    logging.warning(f"No device specified for the predictor. Inferred {default_model_device}")
    return default_model_device


def _validate_framework(framework: str) -> bool:
    """
    Checks if PyTorch or TensorFlow is installed.

    Parameters
    ---------
    framework: {'pytorch', 'tensorflow'}

    Raises
    ------
    ImportError
        If the specified framework is not installed.
    NotImplementedError
        If the value of `framework` is not 'pytorch' or 'tensorflow'.
    """

    framework = framework.lower()
    if framework == 'tensorflow' and not has_tensorflow or framework == 'pytorch' and not has_pytorch:
        raise ImportError(f'{framework} not installed. ')
    elif framework not in FRAMEWORKS:
        raise NotImplementedError(f'{framework} not implemented. Use tensorflow or pytorch instead.')
