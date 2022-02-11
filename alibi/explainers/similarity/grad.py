"""Gradient-based explainer.

This module implements the gradient-based explainers grad-dot and grad-cos.
"""

import copy
from typing import TYPE_CHECKING, Callable, Optional, Union, Dict, Tuple

import numpy as np

from alibi.api.interfaces import Explanation
from alibi.explainers.similarity.base import BaseSimilarityExplainer
from alibi.explainers.similarity.metrics import dot, cos, asym_dot
from alibi.api.defaults import DEFAULT_META_SIM, DEFAULT_DATA_SIM

if TYPE_CHECKING:
    import tensorflow
    import torch


class SimilarityExplainer(BaseSimilarityExplainer):
    def __init__(self,
                 model: 'Union[tensorflow.keras.Model, torch.nn.Module]',
                 loss_fn: '''Callable[[Union[tensorflow.Tensor, torch.Tensor],
                                    Union[tensorflow.Tensor, torch.Tensor]],
                                   Union[tensorflow.Tensor, torch.Tensor]]''',
                 sim_fn: 'Union[Callable[[np.array, np.array], np.array], str]' = 'grad_dot',
                 task: str = "classification",
                 store_grads: bool = False,
                 seed: int = 0,
                 backend: str = "tensorflow",
                 **kwargs
                 ):
        """Constructor

        Parameters
        ----------
        model:
            Model to explain.
        loss_fn:
            Loss function used.
        sim_fn:
            Similarity function to use. ``'grad_dot'`` | ``'grad_cos'``. Default: ``'grad_dot'``.
        task:
            Task to perform. ``'classification'`` | ``'regression'``. Default: ``'classification'``.
        store_grads:
            Whether to store gradients. Default ``False``. If ``False``, gradients are computed on the fly otherwise we
            store them which can be faster when it comes to explaining and instances.
        seed:
            Random seed. Default: 0.
        backend:
            Backend to use. ``'tensorflow'`` | ``'torch'``. Default: ``'tensorflow'``.
        kwargs:
            Additional arguments to pass to the similarity function.
        """

        self.meta = copy.deepcopy(DEFAULT_META_SIM)
        self.meta['params'].update(
            sim_fn_name=sim_fn,
            store_grads=store_grads,
            seed=seed,
            backend_name=backend,
            task_name=task
        )

        sim_fn_opts: Dict[str, Callable] = {
            'grad_dot': dot,
            'grad_cos': cos,
            'grad_asym_dot': asym_dot
        }

        if isinstance(sim_fn, str):
            if sim_fn not in sim_fn_opts.keys():
                raise ValueError(f'Unknown method {sim_fn}. Consider using: `{"` | `".join(sim_fn_opts.keys())}`.')

            sim_fn = sim_fn_opts[sim_fn]

        if task not in ['classification', 'regression']:
            raise ValueError(f'Unknown task {task}. Consider using: `classification` | `regression`.')

        self.task = task

        super().__init__(model, loss_fn, sim_fn, store_grads, seed, backend, meta=self.meta, **kwargs)

    def _preprocess_args(
            self,
            x: 'Union[np.ndarray, tensorflow.Tensor, torch.Tensor]',
            y: 'Optional[Union[np.ndarray, tensorflow.Tensor, torch.Tensor, Callable]]' = None) \
            -> 'Union[Tuple[torch.Tensor, torch.Tensor], Tuple[tensorflow.Tensor, tensorflow.Tensor]]':
        """Formats `x`, `y` for explain method.

        Parameters
        ----------
        x:
            Input data requiring formatting.
        y:
            target data or function requiring formatting.

        Returns
        -------
        x:
            Input data formatted for explain method.
        y:
            Target data formatted for explain method.

        """
        x = self._match_shape_to_data(x, 'x')
        if isinstance(x, np.ndarray):
            x = self.backend.to_tensor(x)

        if self.task == 'regression' and y is None:
            err_msg = 'Regression task requires a target value. y must be provided, either as a value or a function.'
            raise ValueError(err_msg)

        if y is None:
            y = self.model(x)
            y = self.backend.argmax(y)
        elif callable(y):
            y = y(x)

        y = self._match_shape_to_data(y, 'y')
        if isinstance(y, np.ndarray):
            y = self.backend.to_tensor(y)

        return x, y

    def explain(
            self,
            x: 'Union[np.ndarray, tensorflow.Tensor, torch.Tensor]',
            y: 'Optional[Union[np.ndarray, tensorflow.Tensor, torch.Tensor, Callable]]' = None) -> "Explanation":

        """Explain the model's predictions for a given input.

        Computes the similarity score between the input and the training set. Reorders the training set according to the
        score in descending order. Returns an explainer object containing the scores and the corresponding training set
        instances as well as the most and least similar instances of the data set.


        Parameters
        ----------
        x:
            `x` can be a `numpy` array, `tensorflow` tensor, or `torch` tensor of same shape as the training data with
            or without the batch dimension. If the batch dimension is missing it's added.
        y:
            `y` can be a `numpy` array, `tensorflow` tensor, `torch` tensor or a function that returns one of these. It
            must either be or return a value of the same shape as `x`. If the batch dimension is missing it's added. In
            the case of a regression task the `y` argument must be present. If the task is classification then `y`
            defaults to the model prediction.

        Returns
        -------
        `Explanation` object containing the ordered similarity scores for the instance with additional metadata as \
        attributes. Contains the following data-related attributes
            -  `scores`: ``np.array`` - similarity scores for each instance in the training set.
            -  `x_train`: ``np.array`` - training set instances in the order of descending similarity scores.
            -  `y_train`: ``np.array`` - training set labels in the order of descending similarity scores.
            -  `most_similar`: ``np.array`` - most similar instances to the input.
            -  `least_similar`: ``np.array`` - least similar instances to the input.

        Raises
        -------
        ValueError:
            If `y` is `None` and the `task` is `regression`.
        """
        self._verify_fit()
        x, y = self._preprocess_args(x, y)
        grad_x_test = self._compute_grad(x, y)
        if not self.store_grads:
            scores = self._compute_adhoc_similarity(grad_x_test)
        else:
            scores = self.sim_fn(self.grad_x_train, grad_x_test)
        return self._build_explanation(scores)

    def _build_explanation(self, scores: np.ndarray) -> "Explanation":
        """Builds an explanation object.

        Parameters
        ----------
        scores:
            The scores for each of the instances in the data set computed by the similarity method.

        Returns
        -------
        `Explanation`: object containing the ordered similarity scores for the instance with additional metadata as \
        attributes. Contains the following data-related attributes
            -  `scores`: ``np.array`` - similarity scores for each instance in the training set.
            -  `x_train`: ``np.array`` - training set instances in the order of descending similarity scores.
            -  `y_train`: ``np.array`` - training set labels in the order of descending similarity scores.
            -  `most_similar`: ``np.array`` - most similar instances to the input.
            -  `least_similar`: ``np.array`` - least similar instances to the input.
        """
        if self.x_train is None or self.y_train is None:
            raise ValueError('Training data is not available. Please call `fit` before calling `explain`.')

        data = copy.deepcopy(DEFAULT_DATA_SIM)
        sorted_score_indices = np.argsort(scores)[::-1]
        data.update(
            scores=scores[sorted_score_indices],
            x_train=self.x_train[sorted_score_indices],
            y_train=self.y_train[sorted_score_indices],
            most_similar=self.x_train[sorted_score_indices[0]],
            least_similar=self.x_train[sorted_score_indices[-1]],
        )
        return Explanation(meta=self.meta, data=data)
