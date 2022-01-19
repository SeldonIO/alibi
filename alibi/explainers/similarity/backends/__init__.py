from alibi.utils.frameworks import Framework, has_pytorch, has_tensorflow

if has_pytorch:
    # import pytorch backend
    from alibi.explainers.similarity.backends.pytorch import base as pytorch_base_backend

if has_tensorflow:
    # import tensorflow backend
    from alibi.explainers.similarity.backends.tensorflow import base as tensorflow_base_backend


def select_backend(backend, **kwargs):
    """
    Selects the backend according to the `backend` flag.

    Parameters
    ---------
    backend
        Deep learning backend: `tensorflow` | `pytorch`. Default `tensorflow`.
    """
    return tensorflow_base_backend if backend == "tensorflow" else pytorch_base_backend
