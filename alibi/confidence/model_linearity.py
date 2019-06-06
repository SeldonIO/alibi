import logging
import numpy as np
from sklearn.neighbors import NearestNeighbors
from functools import reduce
from typing import Any, Tuple, Callable, Union, List
from time import time

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
    original_shape = X_train.shape[1:]
    X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
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


def _calculate_linearity_regression(predict_fn: Callable, input_shape: Tuple, samples: Union[List, np.ndarray],
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

    print('v2reg')
    ss = samples.shape[:2]
    samples = samples.reshape((samples.shape[0] * samples.shape[1],) + input_shape)
    outs = predict_fn(samples)
    samples = samples.reshape(ss + input_shape)
    outs = outs.reshape(ss)
    sum_out = reduce(lambda x, y: x + y, [alphas[i] * outs[:, i] for i in range(len(alphas))])

    summ = reduce(lambda x, y: x + y, [alphas[i] * samples[:, i] for i in range(len(alphas))])
    try:
        out_sum = predict_fn(summ)
    except ValueError:
        summ = summ.reshape((1,) + summ.shape)
        out_sum = predict_fn(summ)


    if verbose:
        logger.debug(out_sum.shape)
        logger.debug(sum_out.shape)

    linearity_score = ((out_sum - sum_out) ** 2).mean()

    return out_sum, sum_out, linearity_score


def _calculate_linearity_measure(predict_fn: Callable, input_shape: Tuple, samples: Union[List, np.ndarray],
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
    print('v2class')
    ss = samples.shape[:2]
    samples = samples.reshape((samples.shape[0] * samples.shape[1],) + input_shape)
    t_0 = time()
    outs = np.log(predict_fn(samples) + 1e-10)
    t_f = time() - t_0
    print('predict time', t_f)
    samples = samples.reshape(ss + input_shape)
    outs = outs.reshape(ss + outs.shape[-1:])
    sum_out = reduce(lambda x, y: x + y, [alphas[i] * outs[:, i] for i in range(len(alphas))])

    summ = reduce(lambda x, y: x + y, [alphas[i] * samples[:, i] for i in range(len(alphas))])
    try:
        out_sum = np.log(predict_fn(summ) + 1e-10)
    except ValueError:
        summ = summ.reshape((1,) + input_shape)
        out_sum = np.log(predict_fn(summ) + 1e-10)

    if verbose:
        logger.debug(out_sum.shape)
        logger.debug(sum_out.shape)

    linearity_score = ((out_sum - sum_out) ** 2).sum(axis=1).mean() / out_sum.shape[-1]  # normalize or not normalize ...

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
    X_sampled = X_sampled.reshape(X_sampled.shape[0], -1)

    return X_sampled


def _sample_sphere(x: np.ndarray, features_range: List = None, epsilon: float = 0.04,
                   nb_samples: int = 10) -> np.ndarray:
    """Samples datapoints from a gaussian distribution centered at x and with standard deviation epsilon.

    Parameters
    ----------
    x
        Centre of the Gaussian
    features_range
        Array with min and max values for each feature
    epsilon
        Size of the sampling region around central instance as percentage of features range
    nb_samples
        Number of samples to generate

    Returns
    -------
    Sampled vectors

    """
    x = x.flatten()
    dim = len(x)
    assert dim > 0, 'Dimension of the sphere must be bigger than 0'
    assert features_range is not None, 'Features range can not be None'
    size = np.round(epsilon * 100).astype(int)
    if size <= 2:
        size = 2

    features_range = np.asarray(features_range)
    deltas = (np.abs(features_range[:, 1] - features_range[:, 0]) * 0.01)

    rnd_sign = 2 * (np.random.randint(2, size=(nb_samples, dim))) - 1
    rnd = np.random.randint(size, size=(nb_samples, dim)) + 1
    rnd = rnd_sign * rnd

    vprime = rnd * deltas
    X_sampled = x + vprime

    return X_sampled


def _generate_pairs(x: np.ndarray, X_train: np.ndarray = None, features_range: List = None,
                    epsilon: float = 0.5, nb_samples: int = 10, order: int = 2, superposition: str = 'uniform',
                    verbose: bool = False) -> Tuple:
    """Generates the components of the linear superposition and their coefficients.

    Parameters
    ----------
    x
        Central instance
    X_train
        Training set
    features_range
        Array with min and max values for each feature
    epsilon
        Size of the sampling region around central instance as percentage of features range
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
        X_sampled = _sample_sphere(x, features_range=features_range, epsilon=epsilon, nb_samples=nb_samples)

    if verbose:
        logger.debug(x.shape)
        logger.debug(X_sampled.shape)

    t_0 = time()
    X_pairs = np.asarray([np.vstack((x.flatten(), X_sampled[i: i + order - 1])) for i in range(X_sampled.shape[0])])
    t_f = time() - t_0
    print('time stacking', t_f)
    if superposition == 'uniform':
        alphas = [1 / float(order) for j in range(order)]
    else:
        logs = np.asarray([np.random.rand() + np.random.randint(1) for i in range(order)])
        alphas = np.exp(logs) / np.exp(logs).sum()

    if verbose:
        logger.debug([X_tmp.shape for X_tmp in X_pairs])
        logger.debug(len(alphas))

    return X_pairs, alphas


def _linearity_measure(predict_fn: Callable, x: np.ndarray, X_train: np.ndarray = None,
                      features_range: Union[List, np.ndarray, str] = None, epsilon: float = 0.04, nb_samples: int = 10,
                      order: int = 2, superposition: str = 'uniform', model_type: str = 'classifier',
                      verbose: bool = False) -> float:
    """Calculate the linearity measure of the model around a certain instance.

    Parameters
    ----------
    predict_fn
        Predict function
    x
        Central instance
    X_train
        Training set
    features_range
        Array with min and max values for each feature
    epsilon
        Size of the sampling region around central instance as percentage of features range
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
    print('v2800A')
    input_shape = x.shape[1:]
    X_pairs, alphas = _generate_pairs(x, X_train=X_train, features_range=features_range, epsilon=epsilon,
                                      nb_samples=nb_samples, order=order, superposition=superposition, verbose=verbose)
    #print(X_pairs.shape)
    if model_type == 'classifier':
        out_sum, sum_out, score = _calculate_linearity_measure(predict_fn, input_shape,
                                                               X_pairs, alphas, verbose=verbose)
    elif model_type == 'regressor':
        out_sum, sum_out, score = _calculate_linearity_regression(predict_fn, input_shape,
                                                                  X_pairs, alphas, verbose=verbose)
    else:
        raise NameError('model_type not supported. Supported model types: classifier, regressor')

    return score


def _infer_features_range(X_train: np.ndarray) -> np.ndarray:
    X_train = X_train.reshape(X_train.shape[0], -1)
    return np.vstack((X_train.min(axis=0), X_train.max(axis=0))).T


class LinearityMeasure(object):

    def __init__(self, method: str = 'gridSampling', epsilon: float = 0.04, nb_samples: int = 10, order: int = 2,
                 superposition: str = 'uniform', model_type: str = 'classifier', verbose: bool = False) -> None:
        """

        Parameters
        ----------
        method
            Method for sampling. Supported methods 'knn' or 'gridSampling'
        epsilon
            Size of the sampling region around central instance as percentage of features range
        nb_samples
            Number of samples to generate
        order
            Number of component in the linear superposition
        superposition
            Defines the way the vectors are combined in the superposition
        model_type
            'classifier' or 'regressor'
        verbose
            Prints logs if true
        """
        self.method = method
        self.epsilon = epsilon
        self.nb_samples = nb_samples
        self.order = order
        self.superposition = superposition
        self.model_type = model_type
        self.verbose = verbose
        self.is_fit = False

    def fit(self, X_train: np.ndarray) -> None:
        """

        Parameters
        ----------
        X_train
            Features vectors of the training set

        Returns
        -------
        None
        """
        self.X_train = X_train
        self.features_range = _infer_features_range(X_train)
        self.input_shape = X_train.shape[1:]
        self.is_fit = True

    def linearity_measure(self, predict_fn: Callable, x: np.ndarray) -> float:
        """

        Parameters
        ----------
        predict_fn
            Predict function
        x
            Central instance

        Returns
        -------
        Linearity measure

        """
        input_shape = x.shape[1:]
        if self.is_fit:
            assert input_shape == self.input_shape

        if self.method == 'knn':
            lin = _linearity_measure(predict_fn, x, X_train=self.X_train, features_range=None,
                                     nb_samples=self.nb_samples, epsilon=self.epsilon, order=self.order,
                                     superposition=self.superposition, model_type=self.model_type, verbose=self.verbose)

        elif self.method == 'gridSampling':
            lin = _linearity_measure(predict_fn, x, X_train=None, features_range=self.features_range,
                                     nb_samples=self.nb_samples, epsilon=self.epsilon, order=self.order,
                                     superposition=self.superposition, model_type=self.model_type, verbose=self.verbose)

        else:
            raise NameError('method not understood. Supported methods: "knn", "gridSampling"')

        return lin


def linearity_measure(predict_fn: Callable, x: np.ndarray, features_range: Union[List, str] = None,
                      X_train: np.ndarray = None, epsilon: float = 0.5, nb_samples: int = 10,
                      order: int = 2, superposition: str = 'uniform', model_type: str = 'classifier',
                      verbose: bool = False) -> float:
    """Calculate the linearity measure of the model around a certain instance.

    Parameters
    ----------
    predict_fn
        Predict function
    x
        Central instance
    features_range
        Array with min and max values for each feature
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
    input_shape = x.shape[1:]

    if features_range is None and X_train is not None:
        features_range = _infer_features_range(X_train)  # infer from dataset
    elif features_range is None and X_train is None:
        features_range = [[0, 1] for _ in x.shape[1]]  # hardcoded (e.g. from 0 to 1)
    elif features_range == 'infer' and X_train is not None:
        features_range = _infer_features_range(X_train)
        X_train = None

    X_pairs, alphas = _generate_pairs(x, X_train=X_train, features_range=features_range, epsilon=epsilon,
                                      nb_samples=nb_samples, order=order, superposition=superposition, verbose=verbose)

    if model_type == 'classifier':
        out_sum, sum_out, score = _calculate_linearity_measure(predict_fn, input_shape, X_pairs,
                                                               alphas, verbose=verbose)
    elif model_type == 'regressor':
        out_sum, sum_out, score = _calculate_linearity_regression(predict_fn, input_shape, X_pairs,
                                                                  alphas, verbose=verbose)
    else:
        raise NameError('model_type not supported. Supported model types: classifier, regressor')

    return score
