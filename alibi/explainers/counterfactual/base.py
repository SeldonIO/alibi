from scipy.spatial.distance import cityblock
import numpy as np
from abc import abstractmethod
from typing import Union


def _mad_distance(x0: np.array, x1: np.array, mads: np.array) -> float:
    """Calculate l1 distance scaled by MAD (Median Absolute Deviation) for each features,

    Parameters
    ----------
    x0: np.array
    x1: np.array
    mads: np.array

    Returns
    -------
    distance: float
    """
    return (np.abs(x0-x1)/mads).sum()


class BaseCounterFactual(object):

    @abstractmethod
    def __init__(self, model: object, sampling_method: Union[str, None], method: Union[str, None],
                 target_probability: Union[float, None], epsilon: Union[float, None], epsilon_step: Union[float, None],
                 max_epsilon: Union[float, None], nb_samples: Union[int, None], optimizer: Union[str, None],
                 metric: Union[str, callable], flip_treshold: [float, None],
                 aggregate_by: Union[str, None], tollerance: Union[float, None], maxiter: Union[int, None],
                 initial_lam: Union[float, None], lam_step: Union[float, None], max_lam: Union[float, None],
                 verbose: bool) -> None:
        """

        Parameters
        ----------
        model
        sampling_method
        method
        epsilon
        epsilon_step
        max_epsilon
        nb_samples
        metric
        flip_treshold
        aggregate_by
        tollerance
        maxiter
        initial_lam
        lam_step
        max_lam
        """

        self.model = model
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

    @abstractmethod
    def fit(self, X_train: np.array, y_train: np.array) -> None:
        """Abtract fit method

        Parameters
        ----------
        X_train: np.array; features vectors
        y_train: np.array; targets

        Return
        ------
        None
        """

    def _metric_distance(self, x0: np.array, x1: np.array) -> float:
        """metric function wrapper

        Parameters
        ----------
        x0: np.array
        x1: np.array


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

