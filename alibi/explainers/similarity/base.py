from abc import ABC
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Union

import numpy as np
from alibi.api.interfaces import Explainer
from alibi.explainers.similarity.backends import _select_backend
from alibi.utils.frameworks import Framework, has_pytorch, has_tensorflow
from alibi.utils.missing_optional_dependency import import_optional
from tqdm import tqdm
from typing_extensions import Literal

_TfTensor = import_optional('tensorflow', ['Tensor'])
_PtTensor = import_optional('torch', ['Tensor'])

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
            X_train: Union[np.ndarray, List[Any]],
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
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_dims = self.X_train.shape[1:] if isinstance(self.X_train, np.ndarray) else None
        self.Y_dims = self.Y_train.shape[1:]
        self.grad_X_train = np.array([])

        # compute and store gradients
        if self.precompute_grads:
            grads = []
            X: Union[np.ndarray, List[Any]]
            for X, Y in tqdm(zip(self.X_train, self.Y_train), disable=not self.verbose):
                grad_X_train = self._compute_grad(self._format(X), Y[None])
                grads.append(grad_X_train[None])

            self.grad_X_train = np.concatenate(grads, axis=0)
        return self

    @staticmethod
    def _is_tensor(x: Any) -> bool:
        """Checks if an obejct is a tensor."""
        if has_tensorflow and isinstance(x, _TfTensor):
            return True
        if has_pytorch and isinstance(x, _PtTensor):
            return True
        if isinstance(x, np.ndarray):
            return True
        return False

    @staticmethod
    def _format(x: 'Union[np.ndarray, tensorflow.Tensor, torch.Tensor, Any]'
                ) -> 'Union[np.ndarray, tensorflow.Tensor, torch.Tensor, List[Any]]':
        """Adds batch dimension."""
        if BaseSimilarityExplainer._is_tensor(x):
            return x[None]
        return [x]

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
                             data: 'Union[np.ndarray, tensorflow.Tensor, torch.Tensor, Any, List[Any]]',
                             target_type: Literal['X', 'Y']
                             ) -> 'Union[np.ndarray, tensorflow.Tensor, torch.Tensor, List[Any]]':
        """
        Verify the shape of `data` against the shape of the training data.

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
        if self._is_tensor(data):
            return self._match_shape_to_data_tensor(data, target_type)
        return self._match_shape_to_data_any(data)

    def _match_shape_to_data_tensor(self,
                                    data: 'Union[np.ndarray, tensorflow.Tensor, torch.Tensor]',
                                    target_type: Literal['X', 'Y']
                                    ) -> 'Union[np.ndarray, tensorflow.Tensor, torch.Tensor]':
        """ Verify the shape of `data` against the shape of the training data for tensor like data."""
        target_shape = getattr(self, f'{target_type}_dims')
        if data.shape == target_shape:
            data = data[None]
        if data.shape[1:] != target_shape:
            raise ValueError((f'Input `{target_type}` has shape {data.shape[1:]}'
                              f' but training data has shape {target_shape}'))
        return data

    @staticmethod
    def _match_shape_to_data_any(data: Union[Any, List[Any]]) -> list:
        """ Ensures that any other data type is a list."""
        if isinstance(data, list):
            return data
        return [data]

    def _compute_adhoc_similarity(self, grad_X: np.ndarray) -> np.ndarray:
        """
        Computes the similarity between the gradients of the test instances and all the training instances. The method
        performs the computation of the gradients of the training instance on the fly without storing them in memory.

        parameters
        ----------
        grad_X
            Gradients of the test instances.
        """
        scores = np.zeros((len(grad_X), len(self.X_train)))
        X: Union[np.ndarray, List[Any]]
        for i, (X, Y) in tqdm(enumerate(zip(self.X_train, self.Y_train)), disable=not self.verbose):
            grad_X_train = self._compute_grad(self._format(X), Y[None])
            scores[:, i] = self.sim_fn(grad_X, grad_X_train[None])[:, 0]
        return scores

    def _compute_grad(self,
                      X: 'Union[np.ndarray, tensorflow.Tensor, torch.Tensor, List[Any]]',
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
