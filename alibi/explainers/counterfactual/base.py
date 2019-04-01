from scipy.spatial.distance import cityblock
import numpy as np
from abc import abstractmethod
from typing import Union, Any, Callable


def _mad_distance(x0: np.array, x1: np.array, mads: np.array) -> float:
    """Calculate l1 distance scaled by MAD (Median Absolute Deviation) for each features,

    Parameters
    ----------
    x0
        features vector
    x1
        features vectors
    mads
        median absolute deviations for each feature

    Returns
    -------
    distance: float
    """
    return (np.abs(x0-x1)/mads).sum()


class BaseCounterFactual(object):

    @abstractmethod
    def __init__(self, predict_fn: Callable, sampling_method: Union[str, None], method: Union[str, None],
                 target_probability: Union[float, None], epsilon: Union[float, None], epsilon_step: Union[float, None],
                 max_epsilon: Union[float, None], nb_samples: Union[int, None], optimizer: Union[str, None],
                 metric: Any, flip_treshold: Union[float, None],
                 aggregate_by: Union[str, None], tollerance: Union[float, None], maxiter: Union[int, None],
                 initial_lam: Union[float, None], lam_step: Union[float, None], max_lam: Union[float, None],
                 verbose: bool) -> None:
        """

        Parameters
        ----------
        predict_fn
            model predict function instance
        sampling_method
            sampling distributions; 'uniform', 'poisson' or 'gaussian'
        method
            algorithm used; 'Watcher' or ...
        epsilon
            scale parameter for neighbourhoods radius
        epsilon_step
            epsilon incremental step
        max_epsilon
            max value for epsilon at which the search is stopped
        nb_samples
            number of samples at each iteration
        metric
            distance metric between features vectors
        flip_treshold
            probability treshold at which the predictions is considered flipped
        aggregate_by
            not used
        tollerance
            allowed tollerance in reaching target probability
        maxiter
            max number of iteration at which minimization is stopped
        initial_lam
            initial weight of first loss term
        lam_step
            incremental step for lam
        max_lam
            max value for lam at which the minimization is stopped

        Return
        ------
        None
        """

        self.predict_fn = predict_fn
        self.target_probability = target_probability
        self.sampling_method = sampling_method
        self.epsilon = epsilon
        self.epsilon_step = epsilon_step
        self.max_epsilon = max_epsilon
        self.nb_samples = nb_samples
        self.optimizer = optimizer
        self.callable_distance = metric
        self.flip_treshold = flip_treshold
        self.aggregate_by = aggregate_by
        self.method = method
        self.tollerance = tollerance
        self._maxiter = maxiter
        self.lam = initial_lam
        self.lam_step = lam_step
        self.max_lam = max_lam
        self.callable_distance = metric
        self.explaning_instance = None
        self.verbose = verbose
        self.mads = None

    @abstractmethod
    def fit(self, X_train: np.array, y_train: np.array) -> None:
        """Abtract fit method

        Parameters
        ----------
        X_train
            features vectors
        y_train
            targets

        Return
        ------
        None
        """

    def _metric_distance(self, x0: np.array, x1: np.array) -> float:
        """metric function wrapper

        Parameters
        ----------
        x0
            features vector
        x1
            features vector

        Returns
        -------
        distance: float
        """
        if isinstance(self.callable_distance, str):
            if self.callable_distance == 'l1_distance':
                self.callable_distance = cityblock
            elif self.callable_distance == 'mad_distance':
                self.callable_distance = _mad_distance
            else:
                raise NameError('Metric {} not implemented. '
                                'For custom metrics, pass a callable function'.format(self.callable_distance))
        try:
            return self.callable_distance(x0, x1)
        except TypeError:
            return self.callable_distance(x0, x1, self.mads)
