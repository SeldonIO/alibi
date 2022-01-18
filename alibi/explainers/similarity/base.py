import numpy as np
# import tensorflow as tf
# import tensorflow.keras as keras

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Any, Callable, Optional
from tqdm import tqdm


from alibi.utils.frameworks import Framework, has_pytorch, has_tensorflow

if TYPE_CHECKING:
    import tensorflow
    import torch

if has_pytorch:
    # import pytorch backend
    from alibi.explainers.backends.pytorch import cfrl_base as pytorch_base_backend

if has_tensorflow:
    # import tensorflow backend
    from alibi.explainers.backends.tensorflow import cfrl_base as tensorflow_base_backend


class Explainer(ABC):
    def __init__(self,
                 # model: keras.Model,
                 # loss_fn: Callable,
                 # sim_fn: Callable,
                 # store_grads: bool = False,
                 seed: int = 0,
                 backend: str = "tensorflow",
                 **kwargs
                 ):
        """
        Constructor

        Parameters
        ----------
        model:
            Model to be explained.
        loss_fn:
            Loss function.
        sim_fn:
            Similarity function. This can be either: `dot` | `cos`.
        store_grads:
            Whether to precompute and store the gradients when fitting.
        seed:
            Seed to be used,
        backend:
            Deep learning backend: `tensorflow` | `pytorch`. Default `tensorflow`.
        """
        # Select backend.
        self.backend = self._select_backend(backend, **kwargs)

        # Set seed for reproducibility.
        self.backend.set_seed(seed)

        # self.params: Dict[Any, Any] = {
        #     "model": model,
        #     "loss_fn": loss_fn,
        #     "sim_fn": sim_fn,
        #     "store_grads": store_grads,
        #     "seed": seed,
        #     "backend": backend
        # }

    def _select_backend(self, backend, **kwargs):
        """
        Selects the backend according to the `backend` flag.

        Parameters
        ---------
        backend
            Deep learning backend: `tensorflow` | `pytorch`. Default `tensorflow`.
        """
        return tensorflow_base_backend if backend == "tensorflow" else pytorch_base_backend

    def fit(self):
        pass

    @abstractmethod
    def explain(self, X: np.ndarray):
        pass

    def compute_adhoc_similarity(self, grad_X: np.ndarray) -> np.ndarray:
        pass
