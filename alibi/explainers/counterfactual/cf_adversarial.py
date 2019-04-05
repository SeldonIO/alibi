import numpy as np
from time import time
from typing import Callable, Dict, Optional, Union
from statsmodels import robust
from scipy.optimize import minimize
from .util import _metric_distance_func, _calculate_franges
import logging

logger = logging.getLogger(__name__)


def _define_func(predict_fn, x, target_class='same'):

    x_shape_batch_format = (1,) + x.shape

    if target_class == 'other':
        def func():
            pass
    else:
        if target_class == 'same':
            target_class = np.argmax(predict_fn(x), axis=1)[0]

        def func(x, target_class=target_class):
            return predict_fn(x)[:, target_class]

    return func, target_class


def _calculate_funcgradx(func, x, epsilon=1.0):

    x = x.reshape(x.shape[1:])
    x_shape = x.shape

    x = x.flatten()
    X_plus, X_minus = [], []
    for i in range(len(x)):
        x_plus, x_minus = np.copy(x), np.copy(x)
        x_plus[i] += epsilon
        x_minus[i] -= epsilon
        x_plus, x_minus = x_plus.reshape(x_shape), x_minus.reshape(x_shape)
        X_plus.append(x_plus)
        X_minus.append(x_minus)

    X_plus = np.asarray(X_plus)
    X_minus = np.asarray(X_minus)

    gradients = (func(X_plus) - func(X_minus)) / (2 * epsilon)

    return gradients


def _define_metric_loss(metric, x_0):

    def _metric_loss(x):
        batch_size = x.shape[0]
        distances = []
        for i in range(batch_size):
            distances.append(metric(x[i].flatten(), x_0.flatten()))
        return np.asarray(distances)

    return _metric_loss


def _calculate_watcher_grads(x, func, metric, x_0, target_probability, _lam, _norm,
                             epsilon_func=1.0, epsilon_metric=1e-10):

    x_shape_batch_format = (1,) + x.shape
    preds = func(x)
    metric_loss = _define_metric_loss(metric, x_0)

    funcgradx = _calculate_funcgradx(func, x, epsilon=epsilon_func)
    metricgradx = _calculate_funcgradx(metric_loss, x, epsilon=epsilon_metric)

    gradients_0 = _lam * 2 * (preds[0] - target_probability) * funcgradx
    gradients_1 = (1 - _lam) * _norm * metricgradx
    gradients = gradients_0 + gradients_1

    return gradients


def minimize_watcher(predict_fn, metric, x_i, x_0, target_class, target_probability,
                     epsilon_func=5.0, epsilon_metric=0.1, maxiter=50,
                     initial_lam=0.0, lam_step=0.001, final_lam=1.0, lam_how='adiabatic',
                     norm=1.0, lr=50.0):
    x_shape = x_i.shape
    x_shape_batch_format = (1,) + x_i.shape
    func, target_class = _define_func(predict_fn, x_0, target_class=target_class)
    for i in range(maxiter):
        if lam_how == 'fixed':
            _lam = initial_lam
        elif lam_how == 'adiabatic':
            _lam = (i / maxiter) * (final_lam - initial_lam)
        gradients = _calculate_watcher_grads(x_i, func, metric, x_0, target_probability,
                                             _lam, norm,
                                             epsilon_func=epsilon_func,
                                             epsilon_metric=epsilon_metric)

        x_i = (x_i.flatten() - lr * gradients).reshape(x_shape)
        if i % 50 == 0:
            print('Target class:', target_class)
            print('Proba:', predict_fn(x_i)[:, target_class])
            logger.debug('Target class:', target_class)
            logger.debug('Proba:', predict_fn(x_i)[:, target_class])
    return x_i


class CounterFactualAdversarialSearch:
    """
    """

    def __init__(self,
                 predict_fn: Callable,
                 metric: Union[Callable, str] = 'l1_distance',  # TODO should transition to mad_distance
                 target_class: Union[int, str] = 'same',
                 target_probability: float = 0.5,  # TODO what should be default?
                 tolerance: float = 0,
                 epsilon_func: float = 5,
                 epsilon_metric: float = 0.1,
                 maxiter: int = 300,
                 method: str = 'OuterBoundary',
                 initial_lam: float = 0,
                 lam_step: float = 0.1,
                 final_lam: float = 1,
                 lam_how: str = 'fixed', #
                 flip_threshold: float = 0.5,
                 lr: float = 50.0,
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
        self._metric_distance = _metric_distance_func(metric)
        self.target_class = target_class
        self.target_probability = target_probability
        self.epsilon_func = epsilon_func
        self.epsilon_metric = epsilon_metric
        self.tolerance = tolerance
        self.maxiter = maxiter
        self.method = method
        self.initial_lam = initial_lam
        self.lam_step = lam_step
        self.final_lam = final_lam
        self.lam_how = lam_how
        self.flip_threshold = flip_threshold
        self.lr = lr
        self.optimizer = optimizer
        self.metric = metric
        # create the metric distance function


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

    def explain(self, X: np.ndarray,  nb_instances: int = 2, return_as: str = 'all') -> dict:
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
        if X.shape[0] != 1:
            X = X.reshape((1,)+X.shape)
        probas = self.predict_fn(X)
        class_x = np.argmax(probas, axis=1)[0]
        proba_x = probas[:, class_x]

        cf_instances = {'idx': [], 'vector': [], 'distance_from_orig': []}  # type: Dict[str, list]
        for i in range(nb_instances):
            if self.method == 'Wachter' or self.method == 'OuterBoundary':
                #                initial_instance = np.random.permutation(self._samples)[0]
                initial_instance = X

                t_0 = time()

                logger.debug('Starting minimization')

                x_min = minimize_watcher(self.predict_fn, self._metric_distance , initial_instance, X,
                                         target_class=self.target_class, target_probability=self.target_probability,
                                         epsilon_func=self.epsilon_func,epsilon_metric=self.epsilon_metric,
                                         initial_lam=self.initial_lam, final_lam=self.final_lam,
                                         lam_how=self.lam_how, maxiter=self.maxiter, lr=self.lr)

                #res = minimize(_countefactual_loss, initial_instance, constraints=cons,
                #               method=self.optimizer, options={'maxiter': _maxiter})
                #x_min=res.x

                probas_cf = self.predict_fn(x_min.reshape(X.shape))
                class_cf = np.argmax(probas_cf, axis=1)[0]
                proba_cf = probas_cf[:, class_cf]

                proba_cf_class_x = probas_cf[:, class_x]

                #self._lam += self.lam_step
                #if _maxiter > self.maxiter or self._lam > self.max_lam:
                #    logger.debug(self._lam, 'Stopping minimization')
                #    self._lam = self.lam
                #    cond = True
                #if self._lam > self.max_lam - self.lam_step:
                #    _maxiter = 1 * self.maxiter

                logger.debug('Minimization time: ', time() - t_0)
                cf_instances['idx'].append(i)
                cf_instances['vector'].append(x_min.reshape(X.shape))
                cf_instances['distance_from_orig'].append(self._metric_distance(x_min.flatten(), X.flatten()))

                logger.debug('Counterfactual instance {} of {} generated'.format(i, nb_instances - 1))
                logger.debug(
                    'Original instance predicted class: {} with probability {}:'.format(class_x, proba_x))
                logger.debug('Countfact instance original class probability: {}'.format(proba_cf_class_x))
                logger.debug('Countfact instance predicted class: '
                             '{} with probability {}:'.format(class_cf, proba_cf))
                logger.debug('Original instance shape', X.shape)

        self.cf_instances = cf_instances

        if return_as == 'all':
            return self.cf_instances
        else:
            return {}
