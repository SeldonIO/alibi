import numpy as np
from time import time
from typing import Callable, Dict, Union
from .util import _metric_distance_func, _calculate_franges, _calculate_radius, _reshape_batch_inverse, \
    _generate_rnd_samples, _generate_gaussian_samples, _generate_poisson_samples
import logging

logger = logging.getLogger(__name__)


class CounterFactualRandomSearch:
    """
    """

    def __init__(self, predict_fn: Callable,
                 target_probability: float = 0.1,
                 metric: Union[Callable, str] = 'l1_distance',
                 tolerance: float = 0.1,
                 maxiter: int = 100,
                 sampling_method: str = 'uniform',
                 epsilon: float = 0.1,
                 epsilon_step: float = 0.1,
                 max_epsilon: float = 5,
                 nb_samples: int = 100,
                 aggregate_by: str = 'closest') -> None:
        """

        Parameters
        ----------
        predict_fn
            model predict function
        target_probability
            TODO
        metric
            distance metric between features vectors. Can be 'l1_distance', 'mad_distance' or a callable function
            taking 2 vectors as input and returning a float
        tolerance
            allowed tolerance in reaching target probability
        maxiter
            max number of iteration at which minimization is stopped
        sampling_method
            probability distribution for sampling; Poisson, Uniform or Gaussian.
        epsilon
            scale parameter to calculate sampling region. Determines the size of the neighbourhood around the
            instance to explain in which the sampling is performed.
        epsilon_step
            incremental step for epsilon in the expanding spheres approach.
        max_epsilon
            maximum value of epsilon at which the search is stopped
        nb_samples
            Number of points to sample at every iteration
        aggregate_by
            method to choose the counterfactual instance; 'closest' or 'mean'
        """

        self.predict_fn = predict_fn
        self.target_probability = target_probability
        self.metric = metric
        self.tolerance = tolerance
        self.maxiter = maxiter
        self.sampling_method = sampling_method
        self.epsilon = epsilon
        self.epsilon_step = epsilon_step
        self.max_epsilon = max_epsilon
        self.nb_samples = nb_samples
        self.aggregate_by = aggregate_by

        # create the metric distance function
        self._metric_distance = _metric_distance_func(metric)

    def fit(self, X_train, y_train=None):
        """

        Parameters
        ----------
        X_train
            feature vectors
        y_train
            targets

        """
        self.f_ranges = _calculate_franges(X_train)

    def explain(self, X: np.ndarray, nb_instances: int = 10, return_as: str = 'all') -> dict:
        """Generate a counterfactual instance with respect to the input instance X with the
        expanding neighbourhoods method.

        Parameters
        ----------
        X
            reference instance for counterfactuals
        nb_instances
            nb of counterfactual instances to generate
        return_as
            controls which counterfactual instance will be returned by the model

        Returns
        -------
        explaining_instance
            np.array of same shape as X; counterfactual instance TODO

        """
        probas_x = self.predict_fn(X)
        #        probas_x = _predict(self.model, X)
        pred_class = np.argmax(probas_x, axis=1)[0]
        max_proba_x = probas_x[:, pred_class]

        cf_instances = {'idx': [], 'vector': [], 'distance_from_orig': []}  # type: Dict[str, list]
        for i in range(nb_instances):
            logger.debug('Searching instance nb {} of {}'.format(i, nb_instances))
            t_0 = time()
            cond = False
            centre = X
            _epsilon = self.epsilon

            def _contrains_diff(pred_tmp):
                return (abs(pred_tmp - self.target_probability)) - self.tolerance

            #            find counterfactual instance with random sampling method
            iter = 0
            while not cond:
                rs = _calculate_radius(f_ranges=self.f_ranges, epsilon=self.epsilon)

                if self.sampling_method == 'uniform':
                    samples_in = _reshape_batch_inverse(_generate_rnd_samples(centre, rs, self.nb_samples), X)
                elif self.sampling_method == 'poisson':
                    samples_in = _reshape_batch_inverse(_generate_poisson_samples(centre, self.nb_samples), X)
                elif self.sampling_method == 'gaussian':
                    samples_in = _reshape_batch_inverse(_generate_gaussian_samples(centre, rs, self.nb_samples), X)
                else:
                    raise NameError('method {} not implemented'.format(self.sampling_method))

                prob_diff = _contrains_diff(max_proba_x) + self.tolerance
                probas_si = self.predict_fn(samples_in)
                #                probas_si = _predict(self.model, samples_in)
                proba_class = probas_si[:, pred_class]
                diffs = [_contrains_diff(p) + self.tolerance for p in proba_class]
                min_diff_instance = samples_in[np.argmin(diffs)]
                min_diff_proba = proba_class[np.argmin(diffs)]
                diff = np.min(diffs)
                cond = _contrains_diff(min_diff_proba) <= 0

                if diff >= prob_diff:
                    logger.debug('Increasing epsilon from {} to {}'.format(_epsilon, _epsilon + self.epsilon_step))
                    _epsilon += self.epsilon_step
                else:
                    _epsilon = self.epsilon
                    centre = min_diff_instance

                iter += 1
                logger.debug('Diff: %s, min_diff_proba: %s, tolerance: %s', diff, min_diff_proba, self.tolerance)
                if iter >= self.maxiter:
                    cond = True

            logger.debug('Search time: ', time() - t_0)
            cf_instances['idx'].append(i)
            cf_instances['vector'].append(min_diff_instance.reshape(X.shape))
            cf_instances['distance_from_orig'].append(self._metric_distance(min_diff_instance.flatten(), X.flatten()))

        self.cf_instances = cf_instances

        if return_as == 'all':
            return self.cf_instances
        else:
            return {}
