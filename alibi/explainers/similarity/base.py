from abc import ABC
from typing import TYPE_CHECKING, Any, Callable, Union

import numpy as np
from tqdm import tqdm

from alibi.api.interfaces import Explainer
from alibi.explainers.similarity.backends import _select_backend


if TYPE_CHECKING:
    import tensorflow
    import torch


class BaseSimilarityExplainer(Explainer, ABC):
    """Base class for similarity explainers."""

    def __init__(self,
                 model: 'Union[tensorflow.keras.Model, torch.nn.Module]',
                 loss_fn: '''Union[Callable[[tensorflow.Tensor, tensorflow.Tensor], tensorflow.Tensor],
                                   Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]''',
                 sim_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 store_grads: bool = False,
                 seed: int = 0,
                 backend: str = "tensorflow",
                 device: str = "cpu",
                 **kwargs
                 ):
        """Constructor

        Parameters
        ----------
        model:
            Model to be explained.
        loss_fn:
            Loss function.
        sim_fn:
            Similarity function. Takes two inputs and returns a similarity value.
        store_grads:
            Whether to precompute and store the gradients when fitting.
        seed:
            Seed to be used,
        backend:
            Deep learning backend: ``'tensorflow'`` | ``'torch'``. Default ``'tensorflow'``.
        device:
            Device to be used. Default `cpu`.
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
            y_train: np.ndarray) -> "Explainer":
        """Fit the explainer. If ``self.store_grads == True`` then the gradients are precomputed and stored.

        Parameters
        ----------
        x_train:
            Training data.
        y_train:
            Training labels.

        Returns
        -------
        self:
            Returns self.
        """

        self.x_train = x_train
        self.y_train = y_train
        self.x_dims = self.x_train.shape[1:]
        self.y_dims = self.y_train.shape[1:]

        # compute and store gradients
        if self.store_grads:
            self.grad_x_train = []
            for x, y in tqdm(zip(self.x_train, self.y_train)):
                grad_x_train = self._compute_grad(x[None], y[None])
                self.grad_x_train.append(grad_x_train[None])
            self.grad_x_train = np.concatenate(self.grad_x_train, axis=0)
        return self

    def _match_shape_to_data(self,
                             data: 'Union[np.ndarray, tensorflow.Tensor, torch.Tensor]',
                             target_type: str) -> np.ndarray:
        """Verify the shape of `x` against the shape of the training data. If `x` is not a batch, reshape to be a single
        batch element. i.e. if training data shape is `(3, 28, 28)` and data shape is `(3, 28, 28)` we reshape data to
        `(1, 3, 28, 28)`.

        Parameters
        ----------
        data:
            Data to be matched shape-wise against the training data.
        target_type:
            Type of data: ``'x'`` | ``'y'``. Used to determine if data should take the shape of model input or model \
            output.
        """
        target_shape = getattr(self, f'{target_type}_dims')
        if data.shape == target_shape:
            data = data[None]
        if data.shape[1:] != target_shape:
            raise ValueError(f'Input x has shape {data.shape[1:]} but training data has shape {target_shape}')
        return data

    def _compute_adhoc_similarity(self, grad_x: np.ndarray) -> np.ndarray:
        """
        Computes the similarity between the gradients/matrix `x` gradients of the test instances and all the training
        instances. The method performs the computation of the gradients of the training instance on the flight without
        storing them.

        parameters
        ----------
        grad_x:
            Gradients of the test instances.
        """
        scores = np.zeros(self.x_train.shape[0])
        for i, (x, y) in tqdm(enumerate(zip(self.x_train, self.y_train))):
            grad_x_train = self._compute_grad(x[None], y[None])
            scores[i] = self.sim_fn(grad_x_train, grad_x)
        return scores

    def _compute_grad(self, x, y) -> np.ndarray:
        """Computes model parameter gradients and returns a flattened `numpy` array."""
        x = self.backend.to_tensor(x)
        y = self.backend.to_tensor(y)
        return self.backend.get_grads(self.model, x, y, self.loss_fn)
