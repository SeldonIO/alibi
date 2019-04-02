import numpy as np
from functools import reduce
from typing import Callable, Dict, Union
from scipy.spatial.distance import cityblock
import logging

logger = logging.getLogger(__name__)


def _mad_distance(x0: np.ndarray, x1: np.ndarray, mads: np.ndarray) -> float:
    """Calculate l1 distance scaled by MAD (Median Absolute Deviation) for each features,

    Parameters
    ----------
    x0
        feature vector
    x1
        feature vectors
    mads
        median absolute deviations for each feature

    Returns
    -------
    distance: float
    """
    return (np.abs(x0 - x1) / mads).sum()


_metric_dict = {'l1_distance': cityblock,
                'mad_distance': _mad_distance}  # type: Dict[str, Callable]


def _metric_distance_func(metric: Union[Callable, str]) -> Callable:
    """metric function wrapper

    Parameters
    ----------
    x0
        feature vector
    x1
        feature vector

    Returns
    -------
    A metric distance function
    """
    if isinstance(metric, str):
        try:
            return _metric_dict[metric]
        except KeyError:
            logger.exception('Metric {} not implemented. For custom metrics, pass a callable function'.format(metric))
            raise

    else:
        return metric


def _reshape_batch_inverse(batch: np.ndarray, X: np.ndarray) -> np.ndarray:
    return batch.reshape((batch.shape[0],) + X.shape[1:])


def _reshape_X(X: np.ndarray) -> np.ndarray:
    """reshape batch flattening features dimensions.

    Parameters
    ----------
    X

    Returns
    -------
    flatten_batch
    """
    if len(X.shape) > 1:
        nb_features = reduce((lambda x, y: x * y), X.shape[1:])
        return X.reshape(X.shape[0], nb_features)
    else:
        return X


def _calculate_franges(X_train: np.ndarray) -> list:
    """Calculates features ranges from train data

    Parameters
    ----------
    X_train
        training feature vectors

    Returns
    -------
    f_ranges
        Min ad Max values in dataset for each feature
    """
    X_train = _reshape_X(X_train)
    f_ranges = []
    for i in range(X_train.shape[1]):
        mi, ma = X_train[:, i].min(), X_train[:, i].max()
        f_ranges.append((mi, ma))
    return f_ranges


def _calculate_radius(f_ranges: list, epsilon: float = 1) -> list:
    """Scales the feature range h-l by parameter epsilon

    Parameters
    ----------
    f_ranges
        Min ad Max values in dataset for each feature
    epsilon
        scaling factor, default=1

    Returns
    -------
    rs
        scaled ranges for each feature
    """
    rs = []
    for l, h in f_ranges:
        r = epsilon * (h - l)
        rs.append(r)
    return rs


def _generate_rnd_samples(X: np.ndarray, rs: list, nb_samples: int, all_positive: bool = True) -> np.ndarray:
    """Samples points from a uniform distribution around instance X

    Parameters
    ----------
    X
        Central instance
    rs
        scaled ranges for each feature
    nb_samples
        Number of points to sample
    all_positive
        if True, will only sample positive values, default=True

    Returns
    ------
    samples_in
        Sampled points
    """
    X_flatten = _reshape_X(X).flatten()
    lower_bounds, upper_bounds = X_flatten - rs, X_flatten + rs

    if all_positive:
        lower_bounds[lower_bounds < 0] = 0

    samples_in = np.asarray([np.random.uniform(low=lower_bounds[i], high=upper_bounds[i], size=nb_samples)
                             for i in range(X_flatten.shape[0])]).T
    return samples_in


def _generate_poisson_samples(X: np.ndarray, nb_samples: int, all_positive: bool = True) -> np.ndarray:
    """Samples points from a Poisson distribution around instance X

    Parameters
    ----------
    X
        Central instance
    nb_samples
        Number of points to sample
    all_positive
        if True, will only sample positive values, default=True

    Returns
    ------
    samples_in
        Sampled points
    """
    X_flatten = X.flatten()
    samples_in = np.asarray([np.random.poisson(lam=X_flatten[i], size=nb_samples) for i in range(len(X_flatten))]).T

    return samples_in


def _generate_gaussian_samples(X: np.ndarray, rs: list, nb_samples: int, all_positive: bool = True) -> np.ndarray:
    """Samples points from a Gaussian distribution around instance X

    Parameters
    ----------
    X
        Central instance
    rs
        scaled standard deviations for each feature
    nb_samples
        Number of points to sample
    all_positive
        if True, will only sample positive values, default=True

    Return
    ------
    samples_in
        Sampled points
    """

    X_flatten = X.flatten()
    samples_in = np.asarray([np.random.normal(loc=X_flatten[i], scale=rs[i], size=nb_samples)
                             for i in range(len(X_flatten))]).T
    if all_positive:
        samples_in[samples_in < 0] = 0

    return samples_in


def _calculate_confidence_threshold(X: np.ndarray, predict_fn: Callable, y_train: np.ndarray) -> float:
    """Unused
    """
    preds = predict_fn(X)
    assert isinstance(preds, np.ndarray), 'predictions not in a np.ndarray format. ' \
                                          'Prediction format: {}'.format(type(preds))
    pred_class = np.argmax(preds)
    p_class = len(y_train[np.where(y_train == pred_class)]) / len(y_train)
    return 1 - p_class


def _has_predict_proba(model: object) -> bool:
    """Check if model has method 'predict_proba'

    Parameters
    ----------
    model
        model instance

    Returns
    -------
    has_predict_proba
        returns True if the model instance has a 'predict_proba' method, False otherwise
    """
    if hasattr(model, 'predict_proba'):
        return True
    else:
        return False


def _has_predict(model: object) -> bool:
    """Check if model has method 'predict_proba'

    Parameters
    ----------
    model
        model instance

    Returns
    -------
    has_predict_proba
        returns True if the model instance has a 'predict_proba' method, False otherwise
    """
    if hasattr(model, 'predict'):
        return True
    else:
        return False

# def _predict(model: object, X: np.ndarray) -> np.ndarray:
#     """Model prediction function wrapper.
#
#     Parameters
#     ----------
#     model
#         model's instance
#
#     Returns
#     -------
#     predictions
#         Predictions array
#     """
#     if _has_predict_proba(model):
#         return model.predict_proba(X)
#     elif _has_predict(model):
#         return model.predict(X)
#     else:
#         return None
