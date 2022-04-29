from abc import ABC
from typing import TYPE_CHECKING, Callable, Union, Tuple, Optional
from typing_extensions import Literal

import numpy as np
from tqdm import tqdm

from alibi.api.interfaces import Explainer
from alibi.explainers.similarity.backends import _select_backend
from alibi.utils.frameworks import Framework

if TYPE_CHECKING:
    import tensorflow
    import torch


class BaseSimilarityExplainer(Explainer, ABC):
    """Base class for similarity explainers."""

    def __init__(self,
                 predictor: 'Union[tensorflow.keras.Model, torch.nn.Module]',
                 loss_fn: '''Union[Callable[[tensorflow.Tensor, tensorflow.Tensor], tensorflow.Tensor],
                                   Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]''',
                 sim_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 precompute_grads: bool = False,
                 backend: Framework = Framework.TENSORFLOW,
                 device: 'Union[int, str, torch.device, None]' = None,
                 meta: Optional[dict] = None,
                 verbose: bool = False,
                 ):
        """Constructor

        Parameters
        ----------
        predictor
            Model to be explained.
        loss_fn
            Loss function.
        sim_fn
            Similarity function. Takes two inputs and returns a similarity value.
        precompute_grads
            Whether to precompute and store the gradients when fitting.
        backend
            Deep learning backend.
        device
            Device to be used. Will default to the same device the backend defaults to.
        meta
            Metadata specific to explainers that inherit from this class. Should be initialized in the child class and
            passed in here. Is used in the `__init__` of the base Explainer class.
        """

        # Select backend.
        self.backend = _select_backend(backend)
        self.backend.set_device(device)  # type: ignore

        self.predictor = predictor
        self.loss_fn = loss_fn
        self.sim_fn = sim_fn
        self.precompute_grads = precompute_grads
        self.verbose = verbose

        meta = {} if meta is None else meta
        super().__init__(meta=meta)

    def fit(self,
            X_train: np.ndarray,
            Y_train: np.ndarray) -> "Explainer":
        """Fit the explainer. If ``self.precompute_grads == True`` then the gradients are precomputed and stored.

        Parameters
        ----------
        X_train
            Training data.
        Y_train
            Training labels.

        Returns
        -------
        self
            Returns self.
        """
        self.X_train: np.ndarray = X_train
        self.Y_train: np.ndarray = Y_train
        self.X_dims: Tuple = self.X_train.shape[1:]
        self.Y_dims: Tuple = self.Y_train.shape[1:]
        self.grad_X_train: np.ndarray = np.array([])

        # compute and store gradients
        if self.precompute_grads:
            grads = []
            for X, Y in tqdm(zip(self.X_train, self.Y_train), disable=not self.verbose):
                grad_X_train = self._compute_grad(X[None], Y[None])
                grads.append(grad_X_train[None])
            self.grad_X_train = np.concatenate(grads, axis=0)
        return self

    def _verify_fit(self) -> None:
        """Verify that the explainer has been fitted.

        Raises
        ------
        ValueError
            If the explainer has not been fitted.
        """

        if not hasattr(self, 'X_train') or not hasattr(self, 'Y_train'):
            raise ValueError('Training data not set. Call `fit` and pass training data first.')

    def _match_shape_to_data(self,
                             data: 'Union[np.ndarray, tensorflow.Tensor, torch.Tensor]',
                             target_type: Literal['X', 'Y']) -> 'Union[np.ndarray, tensorflow.Tensor, torch.Tensor]':
        """Verify the shape of `data` against the shape of the training data.

        Used to ensure input is correct shape for gradient methods implemented in the backends. `data` will be the
        features or label of the instance being explained. If the `data` is not a batch, reshape to be a single batch
        element. i.e. if training data shape is `(3, 28, 28)` and data shape is `(3, 28, 28)` we reshape data to
        `(1, 3, 28, 28)`.

        Parameters
        ----------
        data
            Data to be matched shape-wise against the training data.
        target_type
            Type of data: ``'X'`` | ``'Y'``. Used to determine if data should take the shape of predictor input or
            predictor output. ``'X'`` will utilize the `X_dims` attribute which stores the shape of the training data.
            ``'Y'`` will match the shape of `Y_dims` which is the shape of the target data.

        Raises
        ------
        ValueError
            If the shape of `data` does not match the shape of the training data, or fit has not been called prior to
            calling this method.
        """
        target_shape = getattr(self, f'{target_type}_dims')
        if data.shape == target_shape:
            data = data[None]
        if data.shape[1:] != target_shape:
            raise ValueError((f'Input `{target_type}` has shape {data.shape[1:]}'
                              f' but training data has shape {target_shape}'))
        return data

    def _compute_adhoc_similarity(self, grad_X: np.ndarray) -> np.ndarray:
        """
        Computes the similarity between the gradients of the test instances and all the training instances. The method
        performs the computation of the gradients of the training instance on the fly without storing them in memory.

        parameters
        ----------
        grad_X
            Gradients of the test instances.
        """
        scores = np.zeros((grad_X.shape[0], self.X_train.shape[0]))
        for i, (X, Y) in tqdm(enumerate(zip(self.X_train, self.Y_train)), disable=not self.verbose):
            grad_X_train = self._compute_grad(X[None], Y[None])
            scores[:, i] = self.sim_fn(grad_X, grad_X_train[None])[:, 0]
        return scores

    def _compute_grad(self,
                      X: 'Union[np.ndarray, tensorflow.Tensor, torch.Tensor]',
                      Y: 'Union[np.ndarray, tensorflow.Tensor, torch.Tensor]') \
            -> np.ndarray:
        """Computes predictor parameter gradients and returns a flattened `numpy` array."""

        X = self.backend.to_tensor(X) if isinstance(X, np.ndarray) else X
        Y = self.backend.to_tensor(Y) if isinstance(Y, np.ndarray) else Y
        return self.backend.get_grads(self.predictor, X, Y, self.loss_fn)

    def reset_predictor(self, predictor: 'Union[tensorflow.keras.Model, torch.nn.Module]') -> None:
        """Resets the predictor to the given predictor.

        Parameters
        ----------
        predictor
            The new predictor to use.
        """
        self.predictor = predictor
