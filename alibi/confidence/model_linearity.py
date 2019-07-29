import logging
from time import time
from typing import Tuple, Callable, Union, List
from numpy.linalg import norm
import numpy as np
from sklearn.neighbors import NearestNeighbors
import string

logger = logging.getLogger(__name__)


def _linear_superposition(alphas, vecs, shape):
    input_str = string.ascii_lowercase[2: 2 + len(shape)]
    einstr = 'a,ba{}->b{}'.format(input_str, input_str)
    return np.einsum(einstr, alphas, vecs)


def _calculate_global_linearity(predict_fn: Callable, input_shape: Tuple, X_samples: np.ndarray,
                                model_type: str, alphas: np.ndarray) -> np.ndarray:
    """Calculates the norm of the difference between the output of a linear superposition of vectors and the
    linear superposition of the outputs for each individual vector.

    Parameters
    ----------
    predict_fn
        Model prediction function
    input_shape
        Shape of the input
    X_samples
        Array of feature vectors in the linear superposition
    model_type
        'classifier' or 'regressor'
    alphas
        Array of coefficients in the linear superposition

    Returns
    -------
    linearity score

    """
    ss = X_samples.shape[:2]  # X_samples shape=(nb_instances, nb_samples, nb_features)
    X_samples = X_samples.reshape((X_samples.shape[0] * X_samples.shape[1],) + input_shape)

    t_0 = time()
    if model_type == 'classifier':
        outs = np.log(predict_fn(X_samples) + 1e-10)
        outs_shape = outs.shape[1:]
        outs = outs.reshape(ss + outs_shape)  # shape=(nb_instances, nb_samples, nb_classes)
    elif model_type == 'regressor':
        outs = predict_fn(X_samples)
        outs_shape = outs.shape[1:]
        if len(outs.shape) == 1:
            outs = outs.reshape(ss + (1,))  # shape=(nb_instances, nb_samples, 1)
        else:  # if regression on multiple targets
            outs = outs.reshape(ss + outs_shape)  # shape=(nb_instances, nb_samples, nb_targets)
    else:
        raise ValueError("Passed 'model_type' not supported. Supported model types: 'classifier', 'regressor'")
    t_f = time() - t_0
    logger.debug('predict time {}'.format(t_f))

    if len(outs_shape) == 0:
        sum_out = np.matmul(alphas, outs)
    else:
        sum_out = _linear_superposition(alphas, outs, outs_shape)

    X_samples = X_samples.reshape(ss + input_shape)
    summ = _linear_superposition(alphas, X_samples, input_shape)

    if model_type == 'classifier':
        out_sum = np.log(predict_fn(summ) + 1e-10)
        out_sum_shape = out_sum.shape[1:]
    elif model_type == 'regressor':
        out_sum = predict_fn(summ)
        out_sum_shape = out_sum.shape[1:]
        if len(out_sum.shape) == 1:
            out_sum = out_sum.reshape((ss[0],) + (1,))
        else:
            out_sum = out_sum.reshape((ss[0],) + out_sum_shape)
    else:
        raise ValueError("Passed 'model_type' not supported. Supported model types: 'classifier', 'regressor'")
    logger.debug(out_sum.shape)
    logger.debug(sum_out.shape)

    diff = out_sum - sum_out
    linearity_score = norm(diff.reshape(diff.shape[0], -1), axis=1)

    return linearity_score


def _calculate_pairwise_linearity(predict_fn: Callable, x: np.ndarray, input_shape: Tuple, X_samples: np.ndarray,
                                  model_type: str, alphas: np.ndarray) -> np.ndarray:
    """Calculates the norm of the difference between the output of a linear superposition of a test vector x and
    vectors in X_samples and the linear superposition of the outputs, averaged over all the vectors in X_samples.

    Parameters
    ----------
    predict_fn
        Model prediction function
    x
        Test instance for which to calculate the linearity measure
    input_shape
        Shape of the input
    X_samples
        Array of feature vectors in the linear superposition
    model_type
        'classifier' or 'regressor'
    alphas
        Array of coefficients in the linear superposition

    Returns
    -------
    linearity score

    """
    ss = X_samples.shape[:2]  # X_samples shape=(nb_instances, nb_samples, nb_features)
    X_samples = X_samples.reshape((X_samples.shape[0] * X_samples.shape[1],) + input_shape)

    t_0 = time()
    if model_type == 'classifier':
        outs = np.log(predict_fn(X_samples) + 1e-10)
        outs_shape = outs.shape[1:]
        x_out = np.log(predict_fn(x) + 1e-10)  # shape=(nb_instances, nb_classes)
        outs = outs.reshape(ss + outs_shape)  # shape=(nb_instances, nb_samples, nb_classes)
    elif model_type == 'regressor':
        outs = predict_fn(X_samples)
        outs_shape = outs.shape[1:]
        x_out = predict_fn(x)
        if len(outs.shape) == 1:
            outs = outs.reshape(ss + (1,))  # shape=(nb_instances, nb_samples, 1)
            x_out = x_out.reshape(x_out.shape + (1,))  # shape=(nb_instances, 1)
        else:  # if regression on multiple targets
            outs = outs.reshape(ss + outs_shape)  # shape=(nb_instances, nb_samples, nb_targets)
    else:
        raise ValueError("Passed 'model_type' not supported. Supported model types: 'classifier', 'regressor'")
    t_f = time() - t_0
    logger.debug('predict time', t_f)

    x_out_stack = np.repeat(x_out.reshape((x_out.shape[0], 1,) + (x_out.shape[1:])), outs.shape[1], axis=1)

    # linear superposition of the outputs
    sum_out = np.matmul(np.array([x_out_stack, outs]).T, alphas).T  # shape=(nb_instances,nb_samples,nb_targets)

    X_samples = X_samples.reshape(ss + input_shape)
    x_stack = np.repeat(x.reshape((x.shape[0], 1,) + (x.shape[1:])), X_samples.shape[1], axis=1)

    # linear superposition of the inputs
    summ = np.matmul(np.array([x_stack, X_samples]).T, alphas).T  # shape=(nb_instances,nb_samples,input_shape)
    if model_type == 'classifier':
        # output of the linear superposition of inputs
        out_sum = np.log(predict_fn(summ.reshape((summ.shape[0] * summ.shape[1],) + summ.shape[2:])) + 1e-10)
        out_sum_shape = out_sum.shape[1:]
        out_sum = out_sum.reshape(ss + out_sum_shape)
    elif model_type == 'regressor':
        out_sum = predict_fn(summ.reshape((summ.shape[0] * summ.shape[1],) + summ.shape[2:]))
        out_sum_shape = out_sum.shape[1:]
        if len(out_sum.shape) == 1:
            out_sum = out_sum.reshape(ss + (1,))
        else:
            out_sum = out_sum.reshape(ss + out_sum_shape)
    else:
        raise ValueError("Passed 'model_type' not supported. Supported model types: 'classifier', 'regressor'")
    logger.debug(out_sum.shape)
    logger.debug(sum_out.shape)

    diff = out_sum - sum_out
    linearity_score = norm(diff.reshape(diff.shape[0], diff.shape[1], -1), axis=2).mean(axis=1)

    return linearity_score


def _sample_knn(x: np.ndarray, X_train: np.ndarray, nb_samples: int = 10) -> np.ndarray:
    """Samples data points from a training set around instance x using k-nearest neighbours.

    Parameters
    ----------
    x
        Central instance for sampling
    X_train
        Training set
    nb_samples
        Number of samples to generate

    Returns
    -------
    Sampled vectors

    """
    x = x.reshape(x.shape[0], -1)
    nb_instances = x.shape[0]

    X_sampled = []
    for i in range(nb_instances):
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_stack = np.stack([x[i] for _ in range(X_train.shape[0])], axis=0)

        X_stack = X_stack.reshape(X_stack.shape[0], -1)
        nbrs = NearestNeighbors(n_neighbors=nb_samples, algorithm='ball_tree').fit(X_train)
        distances, indices = nbrs.kneighbors(X_stack)
        distances, indices = distances[0], indices[0]

        X_sampled_tmp = X_train[indices]
        X_sampled.append(X_sampled_tmp)

    X_sampled = np.array(X_sampled)  # shape=(nb_instances, nb_samples, nb_features)

    return X_sampled


def _sample_gridSampling(x: np.ndarray, features_range: np.ndarray = None, epsilon: float = 0.04,
                         nb_samples: int = 10, res: int = 100) -> np.ndarray:
    """Samples data points from a gaussian distribution centered at x and with standard deviation epsilon.

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
    nb_instances = x.shape[0]
    x = x.reshape(x.shape[0], -1)
    dim = x.shape[1]  # number of features

    size = np.round(epsilon * res).astype(int)
    if size <= 2:
        size = 2

    deltas = (np.abs(features_range[:, 1] - features_range[:, 0]) / float(res))  # shape=(nb_features)

    rnd_sign = 2 * (np.random.randint(2, size=(nb_instances, nb_samples, dim))) - 1
    rnd = np.random.randint(size, size=(nb_instances, nb_samples, dim)) + 1
    rnd = rnd_sign * rnd  # shape=(nb_instances, nb_samples, nb_features)

    vprime = rnd * deltas
    X_sampled = x.reshape(x.shape[0], 1, x.shape[1]) + vprime  # shape=(nb_instances, nb_samples, nb_features)

    return X_sampled


def _linearity_measure(predict_fn: Callable, x: np.ndarray, X_train: np.ndarray = None,
                       feature_range: Union[List, np.ndarray] = None, method: str = None,
                       epsilon: float = 0.04, nb_samples: int = 10, res: int = 100,
                       alphas: np.ndarray = None, model_type: str = 'classifier', agg: str = 'global') -> np.ndarray:
    """Calculate the linearity measure of the model around a test instance.

    Parameters
    ----------
    predict_fn
        Model prediction function
    x
        Central instance
    X_train
        Training set
    feature_range
        Array with min and max values for each feature
    method
        Method for sampling. Supported methods: 'knn' or 'gridSampling'
    epsilon
        Size of the sampling region around central instance as percentage of feature range
    nb_samples
        Number of samples to generate
    res
        Resolution of the grind. Number of intervals in which the feature range is discretized
    alphas
        Array of coefficients in the superposition
    agg
        Aggregation method. Supported methods: 'global' or 'pairwise'
    verbose
        Prints logs if true

    Returns
    -------
    Linearity measure

    """
    input_shape = x.shape[1:]

    assert method == 'knn' or method == 'gridSampling', "sampling method not supported. " \
                                                        "Supported methods 'knn' or 'gridSampling'. "

    if method == 'knn':
        assert X_train is not None, "The 'knn' method requires X_train != None"
        X_sampled = _sample_knn(x, X_train, nb_samples=nb_samples)
    elif method == 'gridSampling':
        assert feature_range is not None, "The 'gridSampling' method requires features_range != None."
        if isinstance(feature_range, list):
            feature_range = np.asarray(feature_range)
        X_sampled = _sample_gridSampling(x, features_range=feature_range, epsilon=epsilon,
                                         nb_samples=nb_samples, res=res)
    else:
        raise ValueError('method not understood. Supported methods: "knn", "gridSampling"')
    logger.debug(x.shape)
    logger.debug(X_sampled.shape)

    if agg == 'pairwise':
        if alphas is None:
            alphas = np.array([0.5, 0.5])
        score = _calculate_pairwise_linearity(predict_fn, x, input_shape, X_sampled, model_type, alphas)
    elif agg == 'global':
        if alphas is None:
            alphas = np.array([1 / float(nb_samples) for _ in range(nb_samples)])
        score = _calculate_global_linearity(predict_fn, input_shape, X_sampled, model_type, alphas)
    else:
        raise ValueError('Aggregation argument agg allowed values: "global" or "pairwise "')

    return score


def _infer_feature_range(X_train: np.ndarray) -> np.ndarray:
    X_train = X_train.reshape(X_train.shape[0], -1)
    return np.vstack((X_train.min(axis=0), X_train.max(axis=0))).T


class LinearityMeasure(object):

    def __init__(self, method: str = 'gridSampling', epsilon: float = 0.04, nb_samples: int = 10, res: int = 100,
                 alphas: np.ndarray = None, model_type: str = 'classifier', agg: str = 'pairwise',
                 verbose: bool = False) -> None:
        """

        Parameters
        ----------
        method
            Method for sampling. Supported methods 'knn' or 'gridSampling'
        epsilon
            Size of the sampling region around central instance as percentage of features range
        nb_samples
            Number of samples to generate
        res
            Resolution of the grind. Number of interval in which the features range is discretized
        alphas
            Array of coefficients in the superposition
        model_type
            'classifier' or 'regressor'
        verbose
            Prints logs if true
        """
        self.method = method
        self.epsilon = epsilon
        self.nb_samples = nb_samples
        self.res = res
        self.alphas = alphas
        self.model_type = model_type
        self.agg = agg
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
        self.features_range = _infer_feature_range(X_train)
        self.input_shape = X_train.shape[1:]
        self.is_fit = True

    def score(self, predict_fn: Callable, x: np.ndarray) -> np.ndarray:
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
            assert self.is_fit, "Method 'knn' cannot be use without calling fit(). "
            lin = _linearity_measure(predict_fn, x, X_train=self.X_train, feature_range=None, method=self.method,
                                     nb_samples=self.nb_samples, res=self.res, epsilon=self.epsilon, alphas=self.alphas,
                                     model_type=self.model_type, agg=self.agg)
        elif self.method == 'gridSampling':
            if not self.is_fit:
                self.features_range = [[0, 1] for _ in x.shape[1]]  # hardcoded (e.g. from 0 to 1)

            lin = _linearity_measure(predict_fn, x, X_train=None, feature_range=self.features_range,
                                     method=self.method, nb_samples=self.nb_samples, res=self.res, epsilon=self.epsilon,
                                     alphas=self.alphas, model_type=self.model_type, agg=self.agg)
        else:
            raise ValueError('method not understood. Supported methods: "knn", "gridSampling"')

        return lin


def linearity_measure(predict_fn: Callable, x: np.ndarray, features_range: Union[List, np.ndarray] = None,
                      method: str = 'gridSampling', X_train: np.ndarray = None, epsilon: float = 0.04,
                      nb_samples: int = 10, res: int = 100, alphas: np.ndarray = None, agg: str = 'global',
                      model_type: str = 'classifier', verbose: bool = False) -> np.ndarray:
    """Calculate the linearity measure of the model around a test instance.

    Parameters
    ----------
    predict_fn
        Predict function
    x
        Central instance
    features_range
        Array with min and max values for each feature
    method
        Method for sampling. Supported methods 'knn' or 'gridSampling'
    X_train
        Training set
    epsilon
        Standard deviation of the Gaussian for sampling
    nb_samples
        Number of samples to generate
    res
        Resolution of the grind. Number of interval in which the features range is discretized
    alphas
        Coefficients in the superposition
    agg
        Aggragation method
    model_type
        Type of task: 'regressor' or 'classifier'
    verbose
        Prints logs if true

    Returns
    -------
    Linearity measure

    """
    if method == 'knn':
        assert X_train is not None, " Method 'knn' requires X_train != None"
        lin = _linearity_measure(predict_fn, x, X_train=X_train, feature_range=None, method=method,
                                 nb_samples=nb_samples, res=res, epsilon=epsilon, alphas=alphas,
                                 model_type=model_type, agg=agg)
    elif method == 'gridSampling':
        assert features_range is not None or X_train is not None, "Method 'gridSampling' requires " \
                                                                  "features_range != None or X_train != None"
        if X_train is not None and features_range is None:
            features_range = _infer_feature_range(X_train)  # infer from dataset
        elif features_range is not None:
            features_range = np.asarray(features_range)

        lin = _linearity_measure(predict_fn, x, X_train=None, feature_range=features_range, method=method,
                                 nb_samples=nb_samples, res=res, epsilon=epsilon, alphas=alphas,
                                 model_type=model_type, agg=agg)
    else:
        raise ValueError('method not understood. Supported methods: "knn", "gridSampling"')

    return lin
