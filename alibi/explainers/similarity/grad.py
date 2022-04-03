"""Gradient-based explainer.

This module implements the gradient-based explainers grad-dot and grad-cos.
"""

import copy
from typing import TYPE_CHECKING, Callable, Optional, Union, Dict, Tuple, Literal
from enum import Enum

import numpy as np

from alibi.api.interfaces import Explanation
from alibi.explainers.similarity.base import BaseSimilarityExplainer
from alibi.explainers.similarity.metrics import dot, cos, asym_dot
from alibi.api.defaults import DEFAULT_META_SIM, DEFAULT_DATA_SIM
from alibi.utils.frameworks import Framework
from alibi.api.interfaces import Explainer

if TYPE_CHECKING:
    import tensorflow
    import torch


class Task(Enum):
    """
    Enum of supported tasks.
    """
    CLASSIFICATION = "classification"
    REGRESSION = "regression"

    @staticmethod
    def from_str(name: str):
        return {
            'classification': Task.CLASSIFICATION,
            'regression': Task.REGRESSION
        }[name]

    @staticmethod
    def values():
        return {item.value for item in Task.__members__.values()}

    @staticmethod
    def options_string():
        return f"""'{"' | '".join(Task.values())}'"""


class GradientSimilarity(BaseSimilarityExplainer):

    def __init__(self,
                 predictor: 'Union[tensorflow.keras.Model, torch.nn.Module]',
                 loss_fn: '''Union[Callable[[tensorflow.Tensor, tensorflow.Tensor], tensorflow.Tensor],
                                   Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]''',
                 sim_fn: Literal['grad_dot', 'grad_cos', 'grad_asym_dot'] = 'grad_dot',
                 task: Literal["classification", 'regression'] = 'classification',
                 store_grads: bool = False,
                 backend: Literal['tensorflow', 'pytorch'] = "tensorflow",
                 device: 'Union[int, str, torch.device, None]' = None,
                 ):
        """Constructor for GradientSimilarity explainer.

        Parameters
        ----------
        predictor
            Model to explain.
        loss_fn
            Loss function used. The gradient of the loss function is used to compute the similarity between the test
            instances and the training set. This should be the same loss used to train the model.
        sim_fn
            Similarity function to use. ``'grad_dot'`` | ``'grad_cos'``. Default: ``'grad_dot'``.
        task
            Task performed by the model. ``'classification'`` | ``'regression'``.
        store_grads
            Whether to store gradients. If ``False``, gradients are computed on the fly otherwise we store them which
            can be faster when it comes to computing explanations. Storing gradients may be memory intensive if the
            model is large.
        backend
            Backend to use. ``'tensorflow'`` | ``'pytorch'``.
        device
            Device to use. If ``None``, the default device for the backend is used. If using `pytorch` backend see
            `pytorch device docs <https://pytorch.org/docs/stable/tensor_attributes.html#torch-device>`_ for correct
            options. Note that in the `pytorch` backend case this parameter can be a ``torch.device``. If using
            `tensorflow` backend see `tensorflow docs <https://www.tensorflow.org/api_docs/python/tf/device>`_ for
            correct options.
        """
        # TODO: add link to docs page for GradientSimilarity explainer in the docstring

        sim_fn_opts: Dict[str, Callable] = {
            'grad_dot': dot,
            'grad_cos': cos,
            'grad_asym_dot': asym_dot
        }

        if sim_fn not in sim_fn_opts.keys():
            raise ValueError(f"""Unknown method {sim_fn}. Consider using: '{"' | '".join(sim_fn_opts.keys())}'.""")

        resolved_sim_fn = sim_fn_opts[sim_fn]

        if task not in Task.values():
            raise ValueError(f"Unknown task {task}. Consider using: {Task.options_string()}.")

        self.task: Task = Task.from_str(task)

        if backend not in Framework.values():
            raise ValueError(f"Unknown backend {backend}. Consider using: {Framework.options_string()}.")

        super().__init__(predictor, loss_fn, resolved_sim_fn, store_grads, Framework.from_str(backend),
                         device=device, meta=copy.deepcopy(DEFAULT_META_SIM))

        self.meta['params'].update(
            sim_fn_name=sim_fn,
            store_grads=store_grads,
            backend_name=backend,
            task_name=task
        )

    def fit(self,
            X_train: np.ndarray,
            Y_train: np.ndarray) -> "Explainer":
        """Fit the explainer. If ``store_grads`` was set to ``True`` on initialization then the gradients are
        precomputed and stored.

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
        return super().fit(X_train, Y_train)

    def _preprocess_args(
            self,
            X: 'Union[np.ndarray, tensorflow.Tensor, torch.Tensor]',
            Y: 'Optional[Union[np.ndarray, tensorflow.Tensor, torch.Tensor, Callable]]' = None) \
            -> 'Union[Tuple[torch.Tensor, torch.Tensor], Tuple[tensorflow.Tensor, tensorflow.Tensor]]':
        """Formats `X`, `Y` for explain method.

        Parameters
        ----------
        X
            Input data requiring formatting.
        Y
            Target data or function requiring formatting.

        Returns
        -------
        X
            Input data formatted for explain method.
        Y
            Target data formatted for explain method.

        """
        X = self._match_shape_to_data(X, 'X')
        if isinstance(X, np.ndarray):
            X = self.backend.to_tensor(X)

        if self.task == Task.REGRESSION and Y is None:
            err_msg = "Regression task requires a target value. 'Y' must be provided, either as a value or a function."
            raise ValueError(err_msg)

        if Y is None:
            Y = self.predictor(X)
            Y = self.backend.argmax(Y)
        elif callable(Y):
            Y = Y(X)

        Y = self._match_shape_to_data(Y, 'Y')
        if isinstance(Y, np.ndarray):
            Y = self.backend.to_tensor(Y)

        return X, Y

    def explain(
            self,
            X: 'Union[np.ndarray, tensorflow.Tensor, torch.Tensor]',
            Y: 'Optional[Union[np.ndarray, tensorflow.Tensor, torch.Tensor, Callable]]' = None) -> "Explanation":
        """Explain the predictor's predictions for a given input.

        Computes the similarity score between the input and the training set. Reorders the training set according to the
        score in descending order. Returns an explainer object containing the scores and the corresponding training set
        instances as well as the most and least similar instances of the data set.


        Parameters
        ----------
        X
            `X` can be a `numpy` array, `tensorflow` tensor, or `pytorch` tensor of the same shape as the training data
            with or without the batch dimension. If the batch dimension is missing it's added.
        Y
            `Y` can be a `numpy` array, `tensorflow` tensor, `pytorch` tensor or a function that returns one of these.
            It must either be or return a value of the same shape as `X`. If the batch dimension is missing it's added.
            In the case of a regression task, the `Y` argument must be present. If the task is classification then `Y`
            defaults to the model prediction.

        Returns
        -------
        `Explanation` object containing the ordered similarity scores for the instance with additional metadata as \
        attributes. Contains the following data-related attributes
            -  `scores`: ``np.array`` - similarity scores for each instance in the training set.
            -  `X_train`: ``np.array`` - training set instances in the order of descending similarity scores.
            -  `Y_train`: ``np.array`` - training set labels in the order of descending similarity scores.
            -  `most_similar`: ``np.array`` - most similar instances to the input.
            -  `least_similar`: ``np.array`` - least similar instances to the input.

        Raises
        -------
        ValueError
            If `Y` is ``None`` and the `task` is ``'regression'``.
        """
        self._verify_fit()
        X, Y = self._preprocess_args(X, Y)
        grad_X_test = self._compute_grad(X, Y)
        if not self.store_grads:
            scores = self._compute_adhoc_similarity(grad_X_test)
        else:
            scores = self.sim_fn(self.grad_X_train, grad_X_test)
        return self._build_explanation(scores)

    def _build_explanation(self, scores: np.ndarray) -> "Explanation":
        """Builds an explanation object.

        Parameters
        ----------
        scores
            The scores for each of the instances in the data set computed by the similarity method.
        """
        if self.X_train is None or self.Y_train is None:
            raise ValueError("Training data is not available. Please call 'fit' before calling 'explain'.")

        data = copy.deepcopy(DEFAULT_DATA_SIM)
        sorted_score_indices = np.argsort(scores)[::-1]
        data.update(
            scores=scores[sorted_score_indices],
            ordered_X_train=self.X_train[sorted_score_indices],
            ordered_Y_train=self.Y_train[sorted_score_indices],
            most_similar=self.X_train[sorted_score_indices[0]],
            least_similar=self.X_train[sorted_score_indices[-1]],
        )
        return Explanation(meta=self.meta, data=data)
