from alibi.utils.frameworks import has_pytorch, has_tensorflow

if has_pytorch:
    # import pytorch backend
    from alibi.explainers.similarity.backends.pytorch.base import _TorchBackend

if has_tensorflow:
    # import tensorflow backend
    from alibi.explainers.similarity.backends.tensorflow.base import _TensorFlowBackend


def _select_backend(backend, **kwargs):
    """
    Selects the backend according to the `backend` flag.

    Parameters
    ---------
    backend
        Deep learning backend: `tensorflow` | `torch`. Default `tensorflow`.
    """
    return _TensorFlowBackend if backend == "tensorflow" else _TorchBackend
