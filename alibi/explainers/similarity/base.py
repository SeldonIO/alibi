from abc import ABC
import numpy as np
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

from tqdm import tqdm
from alibi.api.interfaces import Explainer
from alibi.explainers.similarity.backends import _select_backend


if TYPE_CHECKING:
    import tensorflow as tf
    import torch


class BaseSimilarityExplainer(Explainer, ABC):
    """Base class for similarity explainers.

    Exposes fit method and _compute_adhoc_similarity method. Methods that interact with the backends should only touch
    this class. Here we use
    """

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
            Deep learning backend: `tensorflow` | `torch`. Default `tensorflow`.
        """

        if backend not in ['torch', 'tensorflow']:
            raise ValueError(f'Unknown backend {backend}. Consider using: `torch` | `tensorflow` .')

        # Select backend.
        self.backend = _select_backend(backend, **kwargs)
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
        self.x_dims = None
        self.y_dims = None

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

        self.x_dims = self.x_train.shape[1:]
        self.y_dims = self.y_train.shape[1:]

        # compute and store gradients
        if self.store_grads:
            self.grad_x_train = []
            for x, y in tqdm(zip(self.x_train, self.y_train)):
                grad_x_train = self._compute_grad(x, y)
                self.grad_x_train.append(grad_x_train[None])
            self.grad_x_train = np.concatenate(self.grad_x_train, axis=0)
        return self

    def _match_shape_to_data(self, data: 'Union[np.ndarray, tf.Tensor, torch.Tensor]', type: str) -> np.ndarray:
        """
        Verify the shape of x against the shape of the training data. If x is not a batch, reshape to be a single batch
        element. i.e. if training data shape is (3, 28, 28) and data shape is (3, 28, 28) we reshape data to
        (1, 3, 28, 28).

        Parameters
        ----------
        data:
            Data to be matched shape-wise against the training data.
        type:
            Type of data: `x` | `y`. Used to determine if data should take the shape of model input or model output.
        """
        target_shape = getattr(self, f'{type}_dims')
        if data.shape == target_shape:
            data = data[None]
        if data.shape[1:] != target_shape:
            raise ValueError(f'Input x has shape {data.shape[1:]} but training data has shape {target_shape}')

        return data

    def _compute_adhoc_similarity(self, grad_x: np.ndarray) -> np.ndarray:
        scores = np.zeros(self.x_train.shape[0])
        for i, (x, y) in tqdm(enumerate(zip(self.x_train, self.y_train))):
            grad_x_train = self._compute_grad(x, y)
            scores[i] = self.sim_fn(grad_x_train, grad_x)
        return scores

    def _compute_grad(self, x, y) -> np.ndarray:
        x = self.backend.to_tensor(x[None])
        y = self.backend.to_tensor(y[None])
        return self.backend.get_grads(self.model, x, y, self.loss_fn)

    def reset_predictor(self, predictor: Any) -> None:
        pass
