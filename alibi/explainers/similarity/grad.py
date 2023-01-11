"""Gradient-based explainer.

This module implements the gradient-based explainers grad-dot and grad-cos.
"""

import copy
from typing import TYPE_CHECKING, Callable, Optional, Union, Dict, Tuple
from typing_extensions import Literal
from enum import Enum
import warnings

import numpy as np

from alibi.api.interfaces import Explanation
from alibi.explainers.similarity.base import BaseSimilarityExplainer
from alibi.explainers.similarity.metrics import dot, cos, asym_dot
from alibi.api.defaults import DEFAULT_META_SIM, DEFAULT_DATA_SIM
from alibi.utils import _get_options_string
from alibi.utils.frameworks import Framework
from alibi.api.interfaces import Explainer

if TYPE_CHECKING:
    import tensorflow
    import torch


class Task(str, Enum):
    """
    Enum of supported tasks.
    """
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class GradientSimilarity(BaseSimilarityExplainer):

    def __init__(self,
                 predictor: 'Union[tensorflow.keras.Model, torch.nn.Module]',
                 loss_fn: '''Union[Callable[[tensorflow.Tensor, tensorflow.Tensor], tensorflow.Tensor],
                                   Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]''',
                 sim_fn: Literal['grad_dot', 'grad_cos', 'grad_asym_dot'] = 'grad_dot',
                 task: Literal['classification', 'regression'] = 'classification',
                 precompute_grads: bool = False,
                 backend: Literal['tensorflow', 'pytorch'] = 'tensorflow',
                 device: 'Union[int, str, torch.device, None]' = None,
                 verbose: bool = False,
                 ):
        """`GradientSimilarity` explainer.

        The gradient similarity explainer is used to find examples in the training data that the predictor considers
        similar to test instances the user wants to explain. It uses the gradients of the loss between the model output
        and the training data labels. These are compared using the similarity function specified by ``sim_fn``. The
        `GradientSimilarity` explainer can be applied to models trained for both classification and regression tasks.


        Parameters
        ----------
        predictor
            Model to explain.
        loss_fn
            Loss function used. The gradient of the loss function is used to compute the similarity between the test
            instances and the training set.
        sim_fn
            Similarity function to use. The ``'grad_dot'`` similarity function computes the dot product of the
            gradients, see :py:func:`alibi.explainers.similarity.metrics.dot`. The ``'grad_cos'`` similarity function
            computes the cosine similarity between the gradients, see
            :py:func:`alibi.explainers.similarity.metrics.cos`. The ``'grad_asym_dot'`` similarity function is similar
            to ``'grad_dot'`` but is asymmetric, see :py:func:`alibi.explainers.similarity.metrics.asym_dot`.
        task
            Type of task performed by the model. If the task is ``'classification'``, the target value passed to the
            explain method of the test instance can be specified either directly or left  as ``None``, if left ``None``
            we use the model's maximum prediction. If the task is ``'regression'``, the target value of the test
            instance must be specified directly.
        precompute_grads
            Whether to precompute the gradients. If ``False``, gradients are computed on the fly otherwise we
            precompute them which can be faster when it comes to computing explanations. Note this option may be memory
            intensive if the model is large.
        backend
            Backend to use.
        device
            Device to use. If ``None``, the default device for the backend is used. If using `pytorch` backend see
            `pytorch device docs <https://pytorch.org/docs/stable/tensor_attributes.html#torch-device>`_ for correct
            options. Note that in the `pytorch` backend case this parameter can be a ``torch.device``. If using
            `tensorflow` backend see `tensorflow docs <https://www.tensorflow.org/api_docs/python/tf/device>`_ for
            correct options.
        verbose
            Whether to print the progress of the explainer.

        Raises
        ------
        ValueError
            If the ``task`` is not ``'classification'`` or ``'regression'``.
        ValueError
            If the ``sim_fn`` is not ``'grad_dot'``, ``'grad_cos'`` or ``'grad_asym_dot'``.
        ValueError
            If the ``backend`` is not ``'tensorflow'`` or ``'pytorch'``.
        TypeError
            If the device is not an ``int``, ``str``, ``torch.device`` or ``None`` for the torch backend option or if
            the device is not ``str`` or ``None`` for the tensorflow backend option.
        """
        # TODO: add link to docs page for GradientSimilarity explainer in the docstring once written

        sim_fn_opts: Dict[str, Callable] = {
            'grad_dot': dot,
            'grad_cos': cos,
            'grad_asym_dot': asym_dot
        }

        if sim_fn not in sim_fn_opts.keys():
            raise ValueError(f"""Unknown method {sim_fn}. Consider using: '{"' | '".join(sim_fn_opts.keys())}'.""")

        resolved_sim_fn = sim_fn_opts[sim_fn]

        if task not in Task.__members__.values():
            raise ValueError(f"Unknown task {task}. Consider using: {_get_options_string(Task)}.")

        self.task = task

        if backend not in Framework.__members__.values():
            raise ValueError(f"Unknown backend {backend}. Consider using: {_get_options_string(Framework)}.")

        super().__init__(predictor, loss_fn, resolved_sim_fn, precompute_grads, Framework(backend), device=device,
                         meta=copy.deepcopy(DEFAULT_META_SIM), verbose=verbose)

        self.meta['params'].update(
            sim_fn_name=sim_fn,
            store_grads=precompute_grads,
            backend_name=backend,
            task_name=task
        )

        num_non_trainable = self.backend._count_non_trainable(self.predictor)
        if num_non_trainable:
            warning_msg = (f"Found {num_non_trainable} non-trainable parameters in the model. These parameters "
                           "don't have gradients and will not be included in the computation of gradient similarity."
                           " This might be because your model has layers that track statistics using non-trainable "
                           "parameters such as batch normalization layers. In this case, you don't need to worry. "
                           "Otherwise it's because you have set some parameters to be non-trainable and alibi is "
                           "letting you know.")
            warnings.warn(warning_msg)

    def fit(self,
            X_train: np.ndarray,
            Y_train: np.ndarray) -> "Explainer":
        """Fit the explainer.

        The `GradientSimilarity` explainer requires the model gradients over the training data. In the explain method
        it compares them to the model gradients for the test instance(s). If ``precompute_grads=True`` on
        initialization then the gradients are precomputed here and stored. This will speed up the explain method call
        but storing the gradients may not be feasible for large models.

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
            Y: 'Optional[Union[np.ndarray, tensorflow.Tensor, torch.Tensor]]' = None) \
            -> 'Union[Tuple[torch.Tensor, torch.Tensor], Tuple[tensorflow.Tensor, tensorflow.Tensor]]':
        """Formats `X`, `Y` for explain method.

        Parameters
        ----------
        X
            Input data requiring formatting.
        Y
            Target data requiring formatting.

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
            err_msg = "Regression task requires a target value. 'Y' must be provided."
            raise ValueError(err_msg)

        if Y is None:
            Y = self.predictor(X)
            Y = self.backend.argmax(Y)  # type: ignore

        Y = self._match_shape_to_data(Y, 'Y')
        if isinstance(Y, np.ndarray):
            Y = self.backend.to_tensor(Y)

        return X, Y

    def explain(
            self,
            X: 'Union[np.ndarray, tensorflow.Tensor, torch.Tensor]',
            Y: 'Optional[Union[np.ndarray, tensorflow.Tensor, torch.Tensor]]' = None) -> "Explanation":
        """Explain the predictor's predictions for a given input.

        Computes the similarity score between the inputs and the training set. Returns an explainer object
        containing the scores, the indices of the training set instances sorted by descending similarity and the
        most similar and least similar instances of the data set for the input. Note that the input may be a single
        instance or a batch of instances.

        Parameters
        ----------
        X
            `X` can be a `numpy` array, `tensorflow` tensor, or `pytorch` tensor of the same shape as the training data
            with or without a leading batch dimension. If the batch dimension is missing it's added.
        Y
            `Y` can be a `numpy` array, `tensorflow` tensor or a `pytorch` tensor. In the case of a regression task, the
            `Y` argument must be present. If the task is classification then `Y` defaults to the model prediction.

        Returns
        -------
        `Explanation` object containing the ordered similarity scores for the test instance(s) with additional \
        metadata as attributes. Contains the following data-related attributes
            - `scores`: ``np.ndarray`` - similarity scores for each pair of instances in the training and test set \
            sorted in descending order.
            - `ordered_indices`: ``np.ndarray`` - indices of the paired training and test set instances sorted by the \
            similarity score in descending order.
            - `most_similar`: ``np.ndarray`` - 5 most similar instances in the training set for each test instance \
            The first element is the most similar instance.
            -  `least_similar`: ``np.ndarray`` - 5 least similar instances in the training set for each test instance. \
            The first element is the least similar instance.

        Raises
        ------
        ValueError
            If `Y` is ``None`` and the `task` is ``'regression'``.
        ValueError
            If the shape of `X` or `Y` does not match the shape of the training or target data.
        ValueError
            If the fit method has not been called prior to calling this method.
        """
        self._verify_fit()
        X, Y = self._preprocess_args(X, Y)
        test_grads = []
        for x, y in zip(X, Y):
            test_grads.append(self._compute_grad(x[None], y[None])[None])
        grads_X_test = np.concatenate(np.array(test_grads), axis=0)
        if not self.precompute_grads:
            scores = self._compute_adhoc_similarity(grads_X_test)
        else:
            scores = self.sim_fn(grads_X_test, self.grad_X_train)
        return self._build_explanation(scores)

    def _build_explanation(self, scores: np.ndarray) -> "Explanation":
        """Builds an explanation object.

        Parameters
        ----------
        scores
            The scores for each of the instances in the data set computed by the similarity method.
        """
        data = copy.deepcopy(DEFAULT_DATA_SIM)
        sorted_score_indices = np.argsort(scores)[:, ::-1]
        broadcast_indices = np.expand_dims(sorted_score_indices, axis=tuple(range(2, len(self.X_train[None].shape))))
        data.update(
            scores=np.take_along_axis(scores, sorted_score_indices, axis=1),
            ordered_indices=sorted_score_indices,
            most_similar=np.take_along_axis(self.X_train[None], broadcast_indices[:, :5], axis=1),
            least_similar=np.take_along_axis(self.X_train[None], broadcast_indices[:, -1:-6:-1], axis=1),
        )
        return Explanation(meta=self.meta, data=data)
