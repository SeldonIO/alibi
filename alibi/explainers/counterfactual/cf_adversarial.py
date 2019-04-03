import numpy as np
from time import time
from typing import Callable, Dict, Optional, Union
from statsmodels import robust
from scipy.optimize import minimize
from .util import _metric_distance_func, _calculate_franges
import logging

logger = logging.getLogger(__name__)


class CounterFactualAdversarialSearch:
    """
    """

    def __init__(self,
                 predict_fn: Callable,
                 target_probability: float = 0.5,  # TODO what should be default?
                 metric: Union[Callable, str] = 'l1_distance',  # TODO should transition to mad_distance
                 tolerance: float = 0,
                 maxiter: int = 300,
                 method: str = 'OuterBoundary',
                 initial_lam: float = 0,
                 lam_step: float = 0.1,
                 max_lam: float = 1,
                 flip_threshold: float = 0.5,
                 optimizer: Optional[str] = None) -> None:
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
            minimum difference between predicted and predefined probabilities for the counterfactual instance
        maxiter
            maximum number of iterations of the optimizer
        method
            algorithm used to find a counterfactual instance; 'OuterBoundary', 'Wachter' or 'InnerBoundary'
        initial_lam
            initial value of lambda parameter. Higher value of lambda will give more weight to prediction accuracy
            respect to proximity of the counterfactual instance with the original instance
        lam_step
            incremental step for lambda
        max_lam
            maximum value for lambda at which the search is stopped
        flip_threshold
            probability at which the predicted class is considered flipped (e.g. 0.5 for binary classification)
        optimizer
            TODO
        """
        self.predict_fn = predict_fn
        self.target_probability = target_probability
        self.metric = metric
        self.tolerance = tolerance
        self.maxiter = maxiter
        self.method = method
        self.lam = initial_lam
        self.lam_step = lam_step
        self.max_lam = max_lam
        self.flip_threshold = flip_threshold
        self.optimizer = optimizer

        # create the metric distance function
        self._metric_distance = _metric_distance_func(metric)

    def fit(self, X_train, y_train=None, dataset_sample_size=5000):
        """

        Parameters
        ----------
        X_train
            features vectors
        y_train
            targets
        dataset_sample_size
            nb of data points to sample from training data

        """
        self._lam = self.lam
        self.f_ranges = _calculate_franges(X_train)
        self.mads = robust.mad(X_train, axis=0) + 10e-10
        if dataset_sample_size is not None:
            self._samples = np.random.permutation(X_train)[:dataset_sample_size]
        else:
            self._samples = X_train
        _distances = [self._metric_distance(self._samples[i], np.roll(self._samples, 1, axis=0)[i])
                      for i in range(self._samples.shape[0])]
        self._norm = 1.0 / max(_distances)

    def explain(self, X: np.ndarray, nb_instances: int = 2, return_as: str = 'all') -> dict:
        """

        Parameters
        ----------
        X
            instance to explain

        Returns
        -------
        explaining_instance
            counterfactual instance serving as an explanation

        """
        probas_x = self.predict_fn(X)
        #        probas_x = _predict(self.model, X)
        pred_class = np.argmax(probas_x, axis=1)[0]
        #        print(pred_class)
        max_proba_x = probas_x[:, pred_class]

        cf_instances = {'idx': [], 'vector': [], 'distance_from_orig': []}  # type: Dict[str, list]
        for i in range(nb_instances):
            if self.method == 'Wachter' or self.method == 'OuterBoundary':
                cond = False
                _maxiter = self.maxiter
                #                initial_instance = np.random.permutation(self._samples)[0]
                initial_instance = X

                def _countefactual_loss(x):
                    pred_tmp = self.predict_fn(x.reshape(X.shape))[:, pred_class]
                    #                    pred_tmp = _predict(self.model, x.reshape(X.shape))[:, pred_class]
                    #                    print(pred_class, pred_tmp)
                    loss_0 = self._lam * (pred_tmp - self.target_probability) ** 2
                    loss_1 = (1 - self._lam) * self._norm * self._metric_distance(x, X.flatten())
                    #                    print(loss_0,loss_1,self._lam)
                    return loss_0 + loss_1

                def _contrains_diff(x):
                    pred_tmp = self.predict_fn(x.reshape(X.shape))[:, pred_class]
                    #                    pred_tmp = _predict(self.model, x.reshape(X.shape))[:, pred_class]
                    return -(abs(pred_tmp - self.target_probability)) + self.tolerance

                t_0 = time()

                while not cond:
                    logger.debug('Starting minimization with Lambda = {}'.format(self._lam))
                    cons = ({'type': 'ineq', 'fun': _contrains_diff})

                    res = minimize(_countefactual_loss, initial_instance, constraints=cons,
                                   method=self.optimizer, options={'maxiter': _maxiter})
                    probas_exp = self.predict_fn(res.x.reshape(X.shape))
                    #                    probas_exp = _predict(self.model, res.x.reshape(X.shape))
                    pred_class_exp = np.argmax(probas_exp, axis=1)[0]
                    #                    print('++++++++++++++++++++++', pred_class_exp, probas_exp)
                    max_proba_exp = probas_exp[:, pred_class_exp]
                    probas_original = probas_exp[:, pred_class]
                    cond = _contrains_diff(res.x) >= 0

                    logger.debug('Loss:', res.fun)
                    logger.debug('Constraint fullfilled:', cond)

                    initial_instance = res.x
                    logger.debug('_maxiter: %s', _maxiter)

                    self._lam += self.lam_step
                    if _maxiter > self.maxiter or self._lam > self.max_lam:
                        logger.debug(self._lam, 'Stopping minimization')
                        self._lam = self.lam
                        cond = True
                    if self._lam > self.max_lam - self.lam_step:
                        _maxiter = 1 * self.maxiter

                logger.debug('Minimization time: ', time() - t_0)
                cf_instances['idx'].append(i)
                cf_instances['vector'].append(res.x.reshape(X.shape))
                cf_instances['distance_from_orig'].append(self._metric_distance(res.x, X.flatten()))

                logger.debug('Counterfactual instance {} of {} generated'.format(i, nb_instances - 1))
                logger.debug(
                    'Original instance predicted class: {} with probability {}:'.format(pred_class, max_proba_x))
                logger.debug('Countfact instance original class probability: {}'.format(probas_original))
                logger.debug('Countfact instance predicted class: '
                             '{} with probability {}:'.format(pred_class_exp, max_proba_exp))
                logger.debug('Original instance shape', X.shape)

        self.cf_instances = cf_instances

        if return_as == 'all':
            return self.cf_instances
        else:
            return {}
