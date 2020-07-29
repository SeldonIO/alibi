# flake8: noqa: F401
import logging
import warnings
from typing import Union


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

tf_required = "tensorflow<2.0.0"
tf_upgrade = "tensorflow>2.0.0"
tf_version: str = tf.__version__


def tensorflow_installed() -> bool:
    """
    Raise an ImportError if TensorFlow is not installed.
    If TensorFlow>=2.0.0 is installed, issue a warning that some functionality may not work.
    If TensorFlow<2.0.0 is installed, issue a warning that in the future some functionality will require an upgrade.
    """

    template = "Module requires {pkg}: pip install alibi[tensorflow]"
    # TODO: ALEX: TBD: This should return False, and the error should be raised in files that req. TF.
    if not has_tensorflow:
        raise ImportError(template.format(pkg=tf_required))
    if int(tf_version[0]) > 1:
        template = "Detected tensorflow={pkg1} in the environment. Some functionality requires {pkg2}."
        warnings.warn(template.format(pkg1=tf_version, pkg2=tf_required))
    if int(tf_version[0]) < 2:
        template = "Detected tensorflow={pkg1} in the environment." \
                   "In the near future some functionality will require {pkg2}"
        warnings.warn(template.format(pkg1=tf_version, pkg2=tf_upgrade))
    return True


def pytorch_installed() -> bool:
    """
    Returns `True` if PyTorch is installed, false otherwise.
    """
    return has_pytorch


def infer_device(predictor, predictor_type: str, framework: str) -> Union[None, str]:
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
        return

    default_model_device = next(predictor.parameters()).device
    logging.warning(f"No device specified for the predictor. Inferred {default_model_device}")
    return default_model_device


def _check_tf_or_pytorch(framework: str) -> bool:
    """
    Checks if PyTorch or TensorFlow is installed.

    Parameters
    ---------
    framework: {'pytorch', 'tensorflow'}

    Raises
    ------
    ValueError
        If the value of `package` is not 'pytorch' or 'tensorflow'.
    """

    if framework == 'tensorflow':
        return has_tensorflow
    elif framework == 'pytorch':
        return has_pytorch
    else:
        raise ValueError(
                "Unknown framework specified or framework not installed. Please check spelling and/or install the "
                "framework in order to run this explainer."
            )
