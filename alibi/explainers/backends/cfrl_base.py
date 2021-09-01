"""
This module contains utility functions for the Counterfactual with Reinforcement Learning base class,
:py:class:`alibi.explainers.cfrl_base`, that are common for both Tensorflow and Pytorch backends.
"""

import numpy as np
from typing import Callable, Any, Optional
from abc import ABC, abstractmethod


def identity_function(X: Any) -> Any:
    """
    Identity function.

    Parameters
    ----------
    X
        Input instance.

    Returns
    -------
    X
        The input instance.
    """
    return X


def generate_empty_condition(X: Any) -> None:
    """
    Empty conditioning.

    Parameters
    ----------
    X
        Input instance.

    Returns
    --------
        None
    """
    return None


def get_classification_reward(Y_pred: np.ndarray, Y_true: np.ndarray):
    """
    Computes classification reward per instance given the prediction output and the true label. The classification
    reward is a sparse/binary reward: 1 if the most likely classes from the prediction output and the label match,
    0 otherwise.

    Parameters
    ----------
    Y_pred
        Prediction output as a distribution over the possible classes.
    Y_true
        True label as a distribution over the possible classes.

    Returns
    -------
        Classification reward per instance. 1 if the most likely classes match, 0 otherwise.
    """
    if len(Y_pred.shape) != 2:
        raise ValueError("Prediction labels should be a 2D array for classification task.")

    if len(Y_true.shape) != 2:
        raise ValueError("Target labels should be a 2D array for classification task.")

    Y_pred = np.argmax(Y_pred, axis=1)
    Y_true = np.argmax(Y_true, axis=1)
    return Y_pred == Y_true


def get_hard_distribution(Y: np.ndarray, num_classes: Optional[int] = None) -> np.ndarray:
    """
    Constructs the hard label distribution (one-hot encoding).

    Parameters
    ----------
    Y
        Prediction array. Can be soft or hard label distribution, or a label.
    num_classes
        Number of classes to be considered.

    Returns
    -------
        Hard label distribution (one-hot encoding).
    """
    if len(Y.shape) == 1 or (len(Y.shape) == 2 and Y.shape[1] == 1):
        if num_classes is None:
            raise ValueError("Number of classes has to be specified to transform the labels into one-hot encoding.")

        Y = Y.reshape(-1).astype(np.int32)
        Y_ohe = np.zeros((Y.shape[0], num_classes))
        Y_ohe[np.arange(Y.shape[0]), Y] = 1
        return Y_ohe

    if len(Y.shape) != 2:
        raise ValueError(f"Expected a 2D array, but the input array has a dimension of {len(Y.shape)}")

    Y_ohe = np.zeros_like(Y)
    Y_ohe[np.arange(Y.shape[0]), np.argmax(Y, axis=1)] = 1
    return Y_ohe


class CounterfactualRLDataset(ABC):
    @staticmethod
    def predict_batches(X: np.ndarray, predictor: Callable, batch_size: int) -> np.ndarray:
        """
        Predict the classification labels of the input dataset. This is performed in batches.

        Parameters
        ----------
        X
            Input to be classified.
        predictor
            Prediction function.
        batch_size
            Maximum batch size to be used during each inference step.

        Returns
        -------
            Classification labels.
        """
        n_minibatch = int(np.ceil(X.shape[0] / batch_size))
        Y_m = []

        for i in range(n_minibatch):
            istart, istop = i * batch_size, min((i + 1) * batch_size, X.shape[0])
            preds = predictor(X[istart:istop])

            # Check if the prediction task is classification. We do this by checking if the last dimension of
            # the prediction is grater than 1. Note that this makes the assumption that the regression task has only
            # a single output (multi-output regression exists too). To be addressed in the future.
            if len(preds.shape) == 2 and preds.shape[-1] > 1:
                preds = get_hard_distribution(preds)

            # Add predictions to the model predictions buffer.
            Y_m.append(preds)

        return np.concatenate(Y_m, axis=0)

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, item):
        pass
