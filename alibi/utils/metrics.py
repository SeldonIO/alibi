import numpy as np
import sklearn.metrics as metrics
from typing import Optional


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: Optional[np.ndarray] = None, **kwargs):
    """
    Computes the accuracy score.

    Parameters
    ----------
    y_true
        1D array of size `N` (i.e., `(N, )`) containing the ground truth labels, where `N` is the number of samples.
    y_pred
        1D array of size `N` (i.e., `(N, )`) or 2D array of size `N x C` containing the predictions as returned by
        the classifier, where `N` is the number of samples and `C` is the number of classes. If the array is 2D, then
        the `argmax` operation is performed along the last axis.
    sample_weight
        Optional 1D array of size `N` (i.e., `(N, )`) containing the sample weight.
    **kwargs
        See `sklearn.metrics.accuracy_score`_ documentation

        .. _sklearn.metrics.accuracy_score:
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score

    Returns
    -------
    Accuracy score.
    """
    if len(y_pred.shape) == 2:
        y_pred = np.argmax(y_pred, axis=-1)

    return metrics.accuracy_score(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight, **kwargs)


def f1_score(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: Optional[np.ndarray] = None, **kwargs):
    """
    Computes the F1 score.

    Parameters
    ----------
    y_true
        1D array of size `N` (i.e., `(N, )`) containing the ground truth labels, where `N` is the number of samples.
    y_pred
        1D array of size `N` (i.e., `(N, )`) or 2D array of size `N x C` containing the predictions as returned by
        the classifier, where `N` is the number of samples and `C` is the number of classes. If the array is 2D, then
        the `argmax` operation is performed along the last axis.
    sample_weight
        Optional 1D array of size `N` (i.e., `(N, )`) containing the sample weight.
    **kwargs
        See `sklearn.metrics.f1_score`_ documentation

        .. _sklearn.metrics.f1_score:
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score

    Returns
    -------
    F1 score.
    """
    if len(y_pred.shape) == 2:
        y_pred = np.argmax(y_pred, axis=-1)

    return metrics.f1_score(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight, **kwargs)


def roc_auc_score(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: Optional[np.ndarray] = None, **kwargs):
    """
    Computes the ROC AUC score.

    Parameters
    ----------
    y_true, y_pred, sample_weight, **kwargs
        See `sklearn.metrics.roc_auc_score`_ documentation, where `y_pred` corresponds to `y_score`.

        .. _sklearn.metrics.roc_auc_score:
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score

    Returns
    -------
    ROC AUC score.
    """
    return metrics.roc_auc_score(y_true=y_true, y_score=y_pred, sample_weight=sample_weight, **kwargs)
