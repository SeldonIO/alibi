from typing import Union, Type

from alibi.utils.frameworks import has_pytorch, has_tensorflow, Framework

if has_pytorch:
    # import pytorch backend
    from alibi.explainers.similarity.backends.pytorch.base import _PytorchBackend

if has_tensorflow:
    # import tensorflow backend
    from alibi.explainers.similarity.backends.tensorflow.base import _TensorFlowBackend


def _select_backend(backend: Framework = Framework.TENSORFLOW) \
        -> Union[Type['_TensorFlowBackend'], Type['_PytorchBackend']]:
    """
    Selects the backend according to the `backend` flag.

    Parameters
    ---------
    backend
        Deep learning backend.
    """
    # Check if pytorch/tensorflow backend supported.
    if (backend == Framework.PYTORCH and not has_pytorch) or \
            (backend == Framework.TENSORFLOW and not has_tensorflow):
        raise ImportError(f'{backend} not installed. Cannot initialize and run the GradientExplainer'
                          f' with {backend} backend.')

    # Allow only pytorch and tensorflow.
    elif backend not in [Framework.PYTORCH, Framework.TENSORFLOW]:
        raise NotImplementedError(f'{backend} not implemented. Use `tensorflow` or `pytorch` instead.')

    return _TensorFlowBackend if backend == Framework.TENSORFLOW else _PytorchBackend
