import logging
import numpy as np
from sklearn.neighbors import NearestNeighbors
from functools import reduce
from typing import Any, Tuple, Callable, Union, List

logger = logging.getLogger(__name__)


def _flatten_features(X_train: np.ndarray) -> Tuple:
    """Flatten training set

    Parameters
    ----------
    X_train
        Training set

    Returns
    -------
    Flatten training set, original features shape

    """
    X_train_reshaped = X_train.reshape((X_train.shape[0], reduce(lambda x, y: x * y, X_train.shape[1:])))
    original_shape = X_train.shape[1:]
    return X_train_reshaped, original_shape


def _reshape_features(X: np.ndarray, features_shape: Tuple) -> np.ndarray:
    """Reshape training set

    Parameters
    ----------
    X
        Training set
    features_shape
        Original features shape

    Returns
    -------
    Reshaped training set

    """
    return X.reshape((X.shape[0],) + features_shape)


def _calculate_linearity_regression(predict_fn: Callable, samples: Union[List, np.ndarray],
                                    alphas: List, verbose: bool = True):
    """Calculates the similarity between a regressor's output of a linear superposition of features vectors and
    the linear superposition of the regressor's output for each of the components of the superposition.

    Parameters
    ----------
    predict_fn
        Predict function
    samples
        List of features vectors in the linear superposition
    alphas
        List of coefficients in the linear superposition
    verbose
        Prints logs if true

    Returns
    -------
    Output of the superpositon, superposition of the outpu, linearity score

    """
    outs = []
    for s in samples:
        try:
            outs.append(predict_fn(s))
        except ValueError:
            outs.append(predict_fn(s.reshape((1,) + s.shape)))
    summ = reduce(lambda x, y: x + y, [alphas[i] * samples[i] for i in range(len(alphas))])

    try:
        out_sum = predict_fn(summ)
    except ValueError:
        summ = summ.reshape((1,) + summ.shape)
        out_sum = predict_fn(summ)

    sum_out = reduce(lambda x, y: x + y, [alphas[i] * outs[i] for i in range(len(alphas))])

    if verbose:
        logger.debug(out_sum.shape)
        logger.debug(sum_out.shape)

    linearity_score = ((out_sum - sum_out) ** 2).sum()

    return out_sum, sum_out, linearity_score


def _calculate_linearity_measure(predict_fn: Callable, samples: Union[List, np.ndarray],
                                 alphas: List, verbose: bool = False) -> Tuple:
    """Calculates the similarity between a classifier's output of a linear superposition of features vectors and
    the linear superposition of the classifier's output for each of the components of the superposition.

    Parameters
    ----------
    predict_fn
        Predict function
    samples
        List of features vectors in the linear superposition
    alphas
        List of coefficients in the linear superposition
    verbose
        Prints logs if true

    Returns
    -------
    Output of the superpositon, superposition of the outpu, linearity score

    """

    assert len(samples) == len(alphas), 'The number of elements in samples and alphas must be the same; ' \
                                        'len(samples)={}, len(alphas)={}'.format(len(samples), len(alphas))

    ps = []
    for s in samples:
        # print(s.shape)
        try:
            ps.append(predict_fn(s))
        except ValueError:
            ps.append(predict_fn(s.reshape((1,) + s.shape)))
    # ps = [predict_fn(samples[i]) for i in range(len(alphas))]

    outs = [np.log(p + 1e-10) for p in ps]
    summ = reduce(lambda x, y: x + y, [alphas[i] * samples[i] for i in range(len(alphas))])
    # print(summ.shape)

    try:
        out_sum = np.log(predict_fn(summ) + 1e-10)
    except ValueError:
        summ = summ.reshape((1,) + summ.shape)
        out_sum = np.log(predict_fn(summ) + 1e-10)

    sum_out = reduce(lambda x, y: x + y, [alphas[i] * outs[i] for i in range(len(alphas))])

    if verbose:
        logger.debug(out_sum.shape)
        logger.debug(sum_out.shape)

    linearity_score = ((out_sum - sum_out) ** 2).sum() / out_sum.shape[1]  # normalize or not normalize ...

    return out_sum, sum_out, linearity_score



def _sample_train(x: np.ndarray, X_train: np.ndarray, nb_samples: int = 10) -> np.ndarray:
    """Samples data points from training set around instance x

    Parameters
    ----------
    x
        Centre instance for sampling
    X_train
        Training set
    nb_samples
        Number of samples to generate

    Returns
    -------
    Sampled vectors

    """
    X_train, _ = _flatten_features(X_train)
    X_stack = np.stack([x for i in range(X_train.shape[0])], axis=0)

    X_stack, features_shape = _flatten_features(X_stack)
    nbrs = NearestNeighbors(n_neighbors=nb_samples, algorithm='ball_tree').fit(X_train)
    distances, indices = nbrs.kneighbors(X_stack)
    distances, indices = distances[0], indices[0]

    X_sampled = X_train[indices]

    X_sampled = _reshape_features(X_sampled, features_shape)

    return X_sampled


def _sample_sphere(x: np.ndarray, epsilon: float = 0.5, nb_samples: int = 10) -> np.ndarray:
    """Samples datapoints from a gaussian distribution centered at x and with standard deviation epsilon.

    Parameters
    ----------
    x
        Centre of the Gaussian
    epsilon
        Standard deviation of the Gaussian
    nb_samples
        Number of samples to generate

    Returns
    -------
    Sampled vectors

    """
    features_shape = x.shape
    x = x.flatten()
    dim = len(x)
    assert dim > 0, 'Dimension of the sphere must be bigger than 0'

    u = np.random.normal(scale=epsilon, size=(nb_samples, dim))
    u /= u.max()
    # u /= np.linalg.norm(u, axis=1).reshape(-1, 1)  # uniform distribution on the unit dim-sphere

    X_sampled = x + u
    X_sampled = _reshape_features(X_sampled, features_shape)

    return X_sampled


def _generate_pairs(x: np.ndarray, X_train: np.ndarray = None, epsilon: float = 0.5, nb_samples: int = 10,
                    order: int = 2, superposition: str = 'uniform', verbose: bool = False) -> Tuple:
    """Generates the components of the linear superposition and their coefficients.

    Parameters
    ----------
    x
        Central instance
    X_train
        Training set
    epsilon
        Standard deviation of Gaussian for sampling
    nb_samples
        Number of samples to genarate
    order
        Number of components in the linear superposition
    superposition
        Defines the way the vectors are combined in the superposition.
    verbose
        Prints logs if true

    Returns
    -------
    Vectors in the linear superposition, coefficients.

    """
    if X_train is not None:
        X_sampled = _sample_train(x, X_train, nb_samples=nb_samples)

    else:
        X_sampled = _sample_sphere(x, epsilon=epsilon, nb_samples=nb_samples)

    x = x.reshape((1,) + x.shape)

    if verbose:
        logger.debug(x.shape)
        logger.debug(X_sampled.shape)

    X_pairs = [np.vstack((x, X_sampled[i:i + order - 1])) for i in range(X_sampled.shape[0])]

    if superposition == 'uniform':
        alphas = [1 / float(order) for j in range(order)]
    else:
        logs = np.asarray([np.random.rand() + np.random.randint(1) for i in range(order)])
        alphas = np.exp(logs) / np.exp(logs).sum()

    if verbose:
        logger.debug([X_tmp.shape for X_tmp in X_pairs])
        logger.debug(len(alphas))

    return X_pairs, alphas


def linearity_measure(predict_fn: Callable, x: np.ndarray, X_train: np.ndarray = None,
                      epsilon: float = 0.5, nb_samples: int = 10, order: int = 2,
                      superposition: str = 'uniform', model_type: str = 'classifier', verbose: bool = False) -> float:
    """Calculate the linearity measure of the model around a certain instance.

    Parameters
    ----------
    predict_fn
        Predict function
    x
        Central instance
    X_train
        Training set
    epsilon
        Standard deviation of the Gaussian for sampling
    nb_samples
        Number of samples to generate
    order
        Number of component in the linear superposition
    superposition
        Defines the way the vectors are combined in the superposition
    verbose
        Prints logs if true

    Returns
    -------
    Linearity measure

    """

    X_pairs, alphas = _generate_pairs(x, X_train=X_train, epsilon=epsilon, nb_samples=nb_samples, order=order,
                                      superposition=superposition, verbose=verbose)
    scores = []
    for pair in X_pairs:

        if model_type == 'classifier':
            out_sum, sum_out, score = _calculate_linearity_measure(predict_fn, pair, alphas, verbose=verbose)
        elif model_type == 'regressor':
            out_sum, sum_out, score = _calculate_linearity_regression(predict_fn, pair, alphas, verbose=verbose)
        else:
            raise NameError('model_type not supported. Supported model types: classifier, regressor')

        scores.append(score)

    return np.asarray(scores).mean()
