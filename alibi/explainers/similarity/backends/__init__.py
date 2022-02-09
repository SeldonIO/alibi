from alibi.utils.frameworks import has_pytorch, has_tensorflow

if has_pytorch:
    # import pytorch backend
    from alibi.explainers.similarity.backends.pytorch.base import TorchBackend

if has_tensorflow:
    # import tensorflow backend
    from alibi.explainers.similarity.backends.tensorflow.base import TensorFlowBackend


def select_backend(backend, **kwargs):
    """
    Selects the backend according to the `backend` flag.

    Parameters
    ---------
    backend
        Deep learning backend: `tensorflow` | `torch`. Default `tensorflow`.
    """
    return TensorFlowBackend if backend == "tensorflow" else TorchBackend
