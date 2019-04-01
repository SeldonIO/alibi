from scipy.optimize import minimize
from time import time
from .base import BaseCounterFactual
import numpy as np
from statsmodels import robust
from functools import reduce
from typing import Dict, Callable


def _reshape_batch_inverse(batch: np.array, X: np.array) -> np.array:
    return batch.reshape((batch.shape[0],)+X.shape[1:])


def _reshape_X(X: np.array) -> np.array:
    """reshape batch flattening features dimentions.

    Parameters
    ----------
    X: np.array

    Returns
    -------
    flatten_batch: np.array
    """
    if len(X.shape) > 1:
        nb_features = reduce((lambda x, y: x * y), X.shape[1:])
        return X.reshape(X.shape[0], nb_features)
    else:
        return X


def _calculate_franges(X_train: np.array) -> list:
    """Calculates features ranges from train data

    Parameters
    ----------
    X_train: np.array; training fuatures vectors

    Returns
    -------
    f_ranges: list; Min ad Max values in dataset for each feature
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
    f_ranges: list; Min ad Max values in dataset for each feature
    epsilon: float; scaling factor, default=1

    Returns
    -------
    rs: list; scaled ranges for each feature
    """
    rs = []
    for l, h in f_ranges:
        r = epsilon * (h - l)
        rs.append(r)
    return rs


def _generate_rnd_samples(X: np.array, rs: list, nb_samples: int, all_positive: bool = True) -> np.array:
    """Samples points from a uniform distribution around instance X

    Parameters
    ----------
    X: np.array; Central instance
    rs: list; scaled ranges for each feature
    nb_samples: int; NUmber of points to sample
    all_positive: bool; if True, will only sample positive values, default=True

    Return
    ------
    samples_in: np.array; Sampled points
    """
    X_flatten = _reshape_X(X).flatten()
    lower_bounds, upper_bounds = X_flatten - rs, X_flatten + rs

    if all_positive:
        lower_bounds[lower_bounds < 0] = 0

    samples_in = np.asarray([np.random.uniform(low=lower_bounds[i], high=upper_bounds[i], size=nb_samples)
                             for i in range(X_flatten.shape[0])]).T
    return samples_in


def _generate_poisson_samples(X: np.array, nb_samples: int, all_positive: bool = True) -> np.array:
    """Samples points from a Poisson distribution around instance X

    Parameters
    ----------
    X: np.array; Central instance
    nb_samples: int; NUmber of points to sample
    all_positive: bool; if True, will only sample positive values, default=True

    Return
    ------
    samples_in: np.array; Sampled points
    """
    X_flatten = X.flatten()
    samples_in = np.asarray([np.random.poisson(lam=X_flatten[i], size=nb_samples) for i in range(len(X_flatten))]).T

    return samples_in


def _generate_gaussian_samples(X: np.array, rs: list,  nb_samples: int, all_positive: bool = True) -> np.array:
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
        np.array; Sampled points
    """

    X_flatten = X.flatten()
    samples_in = np.asarray([np.random.normal(loc=X_flatten[i], scale=rs[i], size=nb_samples)
                             for i in range(len(X_flatten))]).T
    if all_positive:
        samples_in[samples_in < 0] = 0

    return samples_in


def _calculate_confidence_treshold(X: np.array, predict_fn: Callable, y_train: np.array) -> float:
    """Unused
    """
    preds = predict_fn(X)
    assert isinstance(preds, np.array), 'predictions not in a np.array format. ' \
                                        'Prediction format: {}'.format(type(preds))
    pred_class = np.argmax(preds)
    p_class = len(y_train[np.where(y_train == pred_class)]) / len(y_train)
    return 1 - p_class


def _has_predict_proba(model: object) -> bool:
    """Check if model has method 'predict_proba'

    Parameters
    ----------
    model
        model instace

    Returns
    -------
    has_predict_proba
        returns True if the model instance has a 'predict_proba' meethod, False otherwise
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
        model instace

    Returns
    -------
    has_predict_proba
        returns True if the model instance has a 'predict_proba' meethod, False otherwise
    """
    if hasattr(model, 'predict'):
        return True
    else:
        return False


# def _predict(model: object, X: np.array) -> np.array:
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


class CounterFactualRandomSearch(BaseCounterFactual):
    """
    """

    def __init__(self, predict_fn, sampling_method='uniform', epsilon=0.1, epsilon_step=0.1, max_epsilon=5, maxiter=100,
                 nb_samples=100, metric='l1_distance', aggregate_by='closest', verbose=False, target_probability=0.1,
                 tollerance=0.1):
        """

        Parameters
        ----------
        predict_fn
            model predict function
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
            Number os points to sample at every iteration
        metric
            distance metric between features vectors. Can be 'l1_distance', 'mad_distance' or a callable function
            taking 2 vectors as input and returning a float
        aggregate_by
            method to choose the countefactual instance; 'closest' or 'mean'
        """
        super().__init__(predict_fn=predict_fn, sampling_method=sampling_method, epsilon=epsilon,
                         epsilon_step=epsilon_step, max_epsilon=max_epsilon, nb_samples=nb_samples,
                         metric=metric, flip_treshold=None, aggregate_by=aggregate_by, method=None, maxiter=maxiter,
                         optimizer=None, target_probability=target_probability, tollerance=tollerance,
                         initial_lam=None, lam_step=None, max_lam=None, verbose=verbose)

    def fit(self, X_train, y_train=None):
        """

        Parameters
        ----------
        X_train
            features vectors
        y_train
            targets

        Returns
        -------
        None
        """
        self.f_ranges = _calculate_franges(X_train)

    def explain(self, X: np.array, nb_instances: int = 10, return_as: str = 'all') -> dict:
        """Generate a counterfactual instance respect to the input instance X with the expanding neighbourhoods method.

        Parameters
        ----------
        X
            reference instance for counterfactuals
        nb_instances
            nb of counterfactual instance to generate
        return_as
            controls which counterfactual instance will be returned by the model

        Returns
        -------
        explaining_instance
            np.array of same shape as X; counterfactual instance

        """
        probas_x = self.predict_fn(X)
#        probas_x = _predict(self.model, X)
        pred_class = np.argmax(probas_x, axis=1)[0]
        max_proba_x = probas_x[:, pred_class]

        cf_instances = {'idx': [], 'vector': [], 'distance_from_orig': []}  # type: Dict[str, list]
        for i in range(nb_instances):
            print('Instance nb {} of {}'.format(i, nb_instances))
            t_0 = time()
            cond = False
            centre = X
            _epsilon = self.epsilon

            def _contrains_diff(pred_tmp):
                return (abs(pred_tmp - self.target_probability)) - self.tollerance

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

                prob_diff = _contrains_diff(max_proba_x) + self.tollerance
                probas_si = self.predict_fn(samples_in)
#                probas_si = _predict(self.model, samples_in)
                proba_class = probas_si[:, pred_class]
                diffs = [_contrains_diff(p) + self.tollerance for p in proba_class]
                min_diff_instance = samples_in[np.argmin(diffs)]
                min_diff_proba = proba_class[np.argmin(diffs)]
                diff = np.min(diffs)
                cond = _contrains_diff(min_diff_proba) <= 0

                if diff >= prob_diff:
                    print('Increasing epsion from {} to {}'.format(_epsilon, _epsilon+self.epsilon_step))
                    _epsilon += self.epsilon_step
                else:
                    _epsilon = self.epsilon
                    centre = min_diff_instance

                iter += 1
                print(diff, min_diff_proba, self.tollerance)
                if iter >= self._maxiter:
                    cond = True

            print('Search time: ', time() - t_0)
            cf_instances['idx'].append(i)
            cf_instances['vector'].append(min_diff_instance.reshape(X.shape))
            cf_instances['distance_from_orig'].append(self._metric_distance(min_diff_instance.flatten(), X.flatten()))
            if self.verbose:
                print('Search time', time()-t_0)

        self.cf_instaces = cf_instances

        if return_as == 'all':
            return self.cf_instaces
        else:
            return {}


class CounterFactualAdversarialSearch(BaseCounterFactual):
    """
    """
    def __init__(self, predict_fn, method='OuterBoundary', target_probability=0.5, tollerance=0, maxiter=300,
                 initial_lam=0, lam_step=0.1, max_lam=1, metric='mad_distance', optimizer=None, flip_treshold=0.5,
                 verbose=False):
        """

        Parameters
        ----------
        predict_fn
            model predict function
        method
            algorithm used to find a counterfactual instance; 'OuterBoundary', 'Wachter' or 'InnerBoundary'
        tollerance
            minimum difference between predicted and predefined probabilities for the counterfactual instance
        maxiter
            maximum number of iterations of the optimizer
        initial_lam
            initial value of lambda parameter. Higher value of lambda will give more weight to prediction accuracy
            respect to proximity of the counterfactual instance with the original instance
        lam_step
            incremental step for lambda
        max_lam
            maximum value for lambda at which the search is stopped
        metric
            distance metric between features vectors. Can be 'l1_distance', 'mad_distance' or a callable function
            taking 2 vectors as input and returning a float
        flip_treshold
            probability at which the predicted class is considered flipped (e.g. 0.5 for binary classification)
        """
        super().__init__(predict_fn=predict_fn, target_probability=target_probability,
                         sampling_method=None, epsilon=None, epsilon_step=None, optimizer=optimizer,
                         max_epsilon=None, nb_samples=None, metric=metric, aggregate_by=None,
                         method=method, tollerance=tollerance, flip_treshold=flip_treshold,
                         initial_lam=initial_lam, lam_step=lam_step, max_lam=max_lam, maxiter=maxiter, verbose=verbose)

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

        Returns
        -------
        None
        """
        self._lam = self.lam
        self.f_ranges = _calculate_franges(X_train)
        self.mads = robust.mad(X_train, axis=0)+10e-10
        if dataset_sample_size is not None:
            self._samples = np.random.permutation(X_train)[:dataset_sample_size]
        else:
            self._samples = X_train
        _distances = [self._metric_distance(self._samples[i], np.roll(self._samples, 1, axis=0)[i])
                      for i in range(self._samples.shape[0])]
        self._norm = 1.0/max(_distances)

    def explain(self, X, nb_instances=2, return_as='all'):
        """

        Parameters
        ----------
        X
            instance to explain

        Returns
        -------
        explaning_instance
            counterfactual instance serving as an explanation

        """
        probas_x = self.predict_fn(X)
#        probas_x = _predict(self.model, X)
        pred_class = np.argmax(probas_x, axis=1)[0]
#        print(pred_class)
        max_proba_x = probas_x[:, pred_class]

        cf_instances = {'idx': [], 'vector': [], 'distance_from_orig': []}
        for i in range(nb_instances):
            if self.method == 'Wachter' or self.method == 'OuterBoundary':
                cond = False
                _maxiter = self._maxiter
#                initial_instance = np.random.permutation(self._samples)[0]
                initial_instance = X

                def _countefactual_loss(x):
                    pred_tmp = self.predict_fn(x.reshape(X.shape))[:, pred_class]
#                    pred_tmp = _predict(self.model, x.reshape(X.shape))[:, pred_class]
                    print(pred_class, pred_tmp)
                    loss_0 = self._lam*(pred_tmp - self.target_probability)**2
                    loss_1 = (1-self._lam)*self._norm*self._metric_distance(x, X.flatten())
#                    print(loss_0,loss_1,self._lam)
                    return loss_0+loss_1

                def _contrains_diff(x):
                    pred_tmp = self.predict_fn(x.reshape(X.shape))[:, pred_class]
#                    pred_tmp = _predict(self.model, x.reshape(X.shape))[:, pred_class]
                    return -(abs(pred_tmp - self.target_probability)) + self.tollerance

                t_0 = time()

                while not cond:
                    print('Starting minimization with Lambda = {}'.format(self._lam))
                    cons = ({'type': 'ineq', 'fun': _contrains_diff})

                    res = minimize(_countefactual_loss, initial_instance, constraints=cons,
                                   method=self.optimizer, options={'maxiter': _maxiter})
                    probas_exp = self.predict_fn(res.x.reshape(X.shape))
#                    probas_exp = _predict(self.model, res.x.reshape(X.shape))
                    pred_class_exp = np.argmax(probas_exp, axis=1)[0]
                    print('++++++++++++++++++++++', pred_class_exp, probas_exp)
                    max_proba_exp = probas_exp[:, pred_class_exp]
                    probas_original = probas_exp[:, pred_class]
                    cond = _contrains_diff(res.x) >= 0
                    if self.verbose:
                        print('Loss:', res.fun)
                        print('Constrain fullfilled:', cond)
                    initial_instance = res.x
                    print(_maxiter)

                    self._lam += self.lam_step
                    if _maxiter > self._maxiter or self._lam > self.max_lam:
                        print(self._lam, 'Stopping minimization')
                        self._lam = self.lam
                        cond = True
                    if self._lam > self.max_lam - self.lam_step:
                        _maxiter = 1*self._maxiter

                print('Minimization time: ', time() - t_0)
                cf_instances['idx'].append(i)
                cf_instances['vector'].append(res.x.reshape(X.shape))
                cf_instances['distance_from_orig'].append(self._metric_distance(res.x, X.flatten()))
                if self.verbose:
                    print('Counterfactual instance {} of {} generated'.format(i, nb_instances-1))
                    print('Original instance predicted class: {} with probability {}:'.format(pred_class, max_proba_x))
                    print('Countfact instance original class probability: {}'.format(probas_original))
                    print('Countfact instance predicted class: '
                          '{} with probability {}:'.format(pred_class_exp, max_proba_exp))
                    print('Original instance shape', X.shape)

        self.cf_instaces = cf_instances

        if return_as == 'all':
            return self.cf_instaces
