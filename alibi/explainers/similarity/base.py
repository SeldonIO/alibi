from abc import ABC
import numpy as np
from typing import TYPE_CHECKING, Any, Callable, Optional, Union
from tqdm import tqdm
from alibi.api.interfaces import Explainer
from alibi.explainers.similarity.backends import select_backend


if TYPE_CHECKING:
    import tensorflow as tf
    import torch


class BaseSimilarityExplainer(Explainer, ABC):
    def __init__(self,
                 model: 'Union[tf.keras.Model, torch.nn.Module]',
                 loss_fn: '''Union[Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
                                   Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]''',
                 sim_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 store_grads: bool = False,
                 seed: int = 0,
                 backend: str = "tensorflow",
                 device: str = "cpu",
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
        # TODO: test this
        if backend not in ['torch', 'tensorflow']:
            raise ValueError(f'Unknown backend {backend}. Consider using: `torch` | `tensorflow` .')


        # Select backend.
        self.backend = select_backend(backend, **kwargs)
        self.backend.set_device(device)

        # Set seed for reproducibility.
        self.backend.set_seed(seed)

        self.model = model
        self.loss_fn = loss_fn
        self.sim_fn = sim_fn
        self.store_grads = store_grads
        self.seed = seed

        self.x_train = None
        self.y_train = None
        self.x_train_full = None
        self.y_train_full = None
        self.grad_x_train = None

        super().__init__(**kwargs)

    def fit(self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            x_train_full: Optional[np.ndarray] = None,
            y_train_full: Optional[np.ndarray] = None) -> "Explainer":

        self.x_train = x_train
        self.y_train = y_train
        self.x_train_full = x_train if x_train_full is None else x_train_full
        self.y_train_full = y_train if y_train_full is None else y_train_full

        # compute and store gradients
        if self.store_grads:
            self.grad_x_train = []
            for i in tqdm(range(self.x_train.shape[0])):
                x = self.backend.to_tensor(self.x_train[i:i + 1])
                y = self.backend.to_tensor(self.y_train[i:i + 1])
                grad_x_train = self.backend.get_grads(self.model, x, y, self.loss_fn)
                self.grad_x_train.append(grad_x_train[None])
            self.grad_x_train = np.concatenate(self.grad_x_train, axis=0)
        return self

    def compute_adhoc_similarity(self, grad_x: np.ndarray) -> np.ndarray:
        scores = np.zeros(self.x_train.shape[0])
        for i in tqdm(range(self.x_train.shape[0])):
            x = self.backend.to_tensor(self.x_train[i:i + 1])
            y = self.backend.to_tensor(self.y_train[i:i + 1])
            grad_x_train = self.backend.get_grads(self.model, x, y, self.loss_fn)
            scores[i] = self.sim_fn(grad_x_train, grad_x)
        return scores

    def reset_predictor(self, predictor: Any) -> None:
        pass
