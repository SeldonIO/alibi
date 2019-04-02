from scipy.spatial.distance import cityblock
import numpy as np
from abc import abstractmethod, ABC
from typing import Union, Callable, Optional


def _mad_distance(x0: np.ndarray, x1: np.ndarray, mads: np.ndarray) -> float:
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
    return (np.abs(x0 - x1) / mads).sum()


class BaseCounterFactual(ABC):

    @abstractmethod
    def __init__(self, predict_fn: Callable,
                 target_probability: float,
                 metric: Union[Callable, str],
                 tolerance: float,
                 maxiter: int,
                 sampling_method: Optional[str],
                 method: Optional[str],
                 epsilon: Optional[float],
                 epsilon_step: Optional[float],
                 max_epsilon: Optional[float],
                 nb_samples: Optional[int],
                 optimizer: Optional[str],
                 flip_threshold: Optional[float],
                 aggregate_by: Optional[str],
                 initial_lam: Optional[float],
                 lam_step: Optional[float],
                 max_lam: Optional[float],
                 verbose: bool) -> None:
        """

        Parameters
        ----------
        predict_fn
            model predict function instance
        target_probability
            TODO
        metric
            distance metric between features vectors
        tolerance
            allowed tolerance in reaching target probability
        maxiter
            max number of iteration at which minimization is stopped
        sampling_method
            sampling distributions; 'uniform', 'poisson' or 'gaussian'
        method
            algorithm used; 'Wachter' or ...  TODO
        epsilon
            scale parameter for neighbourhoods radius
        epsilon_step
            epsilon incremental step
        max_epsilon
            max value for epsilon at which the search is stopped
        nb_samples
            number of samples at each iteration
        optimizer
            TODO
        flip_threshold
            probability threshold at which the predictions is considered flipped
        aggregate_by
            not used
        initial_lam
            initial weight of first loss term
        lam_step
            incremental step for lam
        max_lam
            max value for lam at which the minimization is stopped
        verbose
            flag to set verbosity

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
        self.flip_threshold = flip_threshold
        self.aggregate_by = aggregate_by
        self.method = method
        self.tolerance = tolerance
        self._maxiter = maxiter
        self.lam = initial_lam
        self.lam_step = lam_step
        self.max_lam = max_lam
        self.explaning_instance = None
        self.verbose = verbose
        self.mads = None

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Abtract fit method

        Parameters
        ----------
        X_train
            feature vectors
        y_train
            targets

        """

    def _metric_distance(self, x0: np.ndarray, x1: np.ndarray) -> float:
        """metric function wrapper

        Parameters
        ----------
        x0
            features vector
        x1
            features vector

        Returns
        -------
        distance
        """
        if isinstance(self.callable_distance, str):
            if self.callable_distance == 'l1_distance':
                self.callable_distance = cityblock
            elif self.callable_distance == 'mad_distance':
                self.callable_distance = _mad_distance
            else:
                raise NameError('Metric {} not implemented. '
                                'For custom metrics, pass a callable function'.format(self.callable_distance))

        assert callable(self.callable_distance)

        try:
            return self.callable_distance(x0, x1)
        except TypeError:
            return self.callable_distance(x0, x1, self.mads)
