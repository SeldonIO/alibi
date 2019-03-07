from scipy.optimize import minimize
from time import time
#from base_cf import BaseCounterFactual
from scipy.spatial.distance import cityblock
import numpy as np
from abc import abstractmethod
from statsmodels import robust
from functools import reduce


def _flatten_X(X: np.array) -> np.array:
    if len(X.shape)>1:
        nb_features=reduce((lambda x, y: x * y), X.shape[1:])
        return X.reshape(X.shape[0], nb_features)
    else:
        return X


def _calculate_franges(X_train: np.array) -> list:
    X_train=_flatten_X(X_train)
    f_ranges = []
    print(X_train.shape)
    for i in range(X_train.shape[1]):
        mi, ma = X_train[:, i].min(), X_train[:, i].max()
        f_ranges.append((mi, ma))
    return f_ranges


def _calculate_radius(f_ranges: list, epsilon: float = 1) -> list:
    rs = []
    for l, h in f_ranges:
        r = epsilon * (h - l)
        rs.append(r)
    return rs


def _generate_rnd_samples(X: np.array, rs: list, nb_samples: int, all_positive: bool = True) -> np.array:

    X_flatten = X.flatten()
    lower_bounds, upper_bounds = X_flatten - rs, X_flatten + rs
    if all_positive:
        lower_bounds[lower_bounds < 0]=0
    samples_in = np.asarray([np.random.uniform(low=lower_bounds[i], high=upper_bounds[i], size=nb_samples)
                             for i in range(len(X_flatten))]).T
    return samples_in


def _generate_poisson_samples(X: np.array, nb_samples: int, all_positive: bool = True) -> np.array:

    X_flatten = X.flatten()
    samples_in = np.asarray([np.random.poisson(lam=X_flatten[i], size=nb_samples) for i in range(len(X_flatten))]).T

    return samples_in


def _generate_gaussian_samples(X: np.array, rs: list,  nb_samples: int, all_positive: bool = True) -> np.array:

    X_flatten = X.flatten()
    samples_in = np.asarray([np.random.normal(loc=X_flatten[i], scale=rs[i], size=nb_samples)
                             for i in range(len(X_flatten))]).T
    if all_positive:
        samples_in[samples_in<0]=0

    return samples_in


def _calculate_confidence_treshold(X: np.array, model: object, y_train: np.array) -> float:
    preds = model.predict(X)
    assert isinstance(preds, np.array), 'predictions not in a np.array format. ' \
                                        'Prediction format: {}'.format(type(preds))
    pred_class = np.argmax(preds)
    p_class = len(y_train[np.where(y_train == pred_class)]) / len(y_train)
    return 1 - p_class


def _has_predict_proba(model: object) -> bool:
    if hasattr(model, 'predict_proba'):
        return True
    else:
        return False


def _predict(model: object, X: np.array) -> np.array:
    if _has_predict_proba(model):
        return model.predict_proba(X)
    else :
        return model.predict(X)


def _mad_distance(x0: np.array, x1: np.array, mads: np.array) -> float:
    return (np.abs(x0-x1)/mads).sum()


class BaseCounterFactual(object):

    @abstractmethod
    def __init__(self, model, sampling_method, method,  epsilon, epsilon_step, max_epsilon, nb_samples,
                 metric, aggregate_by, tollerance, initial_lam):

        self.model = model
        self.sampling_method = sampling_method
        self.epsilon = epsilon
        self.epsilon_step = epsilon_step
        self.max_epsilon = max_epsilon
        self.nb_samples = nb_samples
        self.callable_distance = metric
        self.aggregate_by = aggregate_by
        self.method = method
        self.tollerance = tollerance
        self.lam = initial_lam
        self.callable_distance = metric

        self.explaning_instance = None

    @abstractmethod
    def fit(self, X_train, y_train):
        pass
        # self.f_ranges = _calculate_franges(X_train)
        # self.mads = robust.mad(X_train, axis=0)+10e-10

    def _metric_distance(self, x0, x1):
        if isinstance(self.callable_distance, str):
            if self.callable_distance=='l1_distance':
                self.callable_distance = cityblock
            elif self.callable_distance=='mad_distance':
                self.callable_distance = _mad_distance
            else:
                raise NameError('Metric {} not implemented. '
                                'For custom metrics, pass a callable function'.format(self.callable_distance))
        try:
            return self.callable_distance(x0, x1)
        except TypeError:
            return self.callable_distance(x0, x1, self.mads)


class ExpandingSpheresSearch(BaseCounterFactual):

    def __init__(self, model, sampling_method='poisson', epsilon=0.1, epsilon_step=0.1, max_epsilon=5,
                 nb_samples=100, metric='l1_distance', aggregate_by='closest'):

        super().__init__(model=model, sampling_method=sampling_method, epsilon=epsilon, epsilon_step=epsilon_step,
                         max_epsilon=max_epsilon, nb_samples=nb_samples, metric=metric, aggregate_by=aggregate_by,
                         method=None, tollerance=None, initial_lam=None)

    def fit(self, X_train, y_train):
        self.f_ranges = _calculate_franges(X_train)
        self.mads = robust.mad(X_train, axis=0)+10e-10

    def explain(self, X):

        probas_x = _predict(self.model, X)
        pred_class = np.argmax(probas_x, axis=1)[0]
        max_proba_x = probas_x[:, pred_class]
        cond = False

        # find counterfactual instance with random sampling method
        while not cond:
            rs = _calculate_radius(f_ranges=self.f_ranges , epsilon=self.epsilon)

            if self.sampling_method=='uniform':
                samples_in = _flatten_X(_generate_rnd_samples(X, rs, self.nb_samples))
            elif self.sampling_method=='poisson':
                samples_in = _flatten_X(_generate_poisson_samples(X, self.nb_samples))
            elif self.sampling_method=='gaussian':
                samples_in = _flatten_X(_generate_gaussian_samples(X, rs, self.nb_samples))
            else:
                raise NameError('method {} not implemented'.format(self.sampling_method))

            probas_si = _predict(self.model, samples_in.reshape((samples_in.shape[0],) + X.shape[1:]))
            pred_classes = np.argmax(probas_si, axis=1)
            unique, counts = np.unique(pred_classes, return_counts=True)
            majority_class = unique[np.argmax(counts)]
            print('Original predicted class: {}; Majority class in sampled data: {}'.format(pred_class, majority_class))

            if self.aggregate_by == 'closest':
                cond = (pred_classes != pred_class).any()
            elif self.aggregate_by == 'mean':
                cond = (majority_class != pred_class)
            if cond:
                samples_flip = samples_in[np.where(pred_classes != pred_class)]
                distances = [self._metric_distance(samples_flip[i], X.flatten()) for i in range(len(samples_flip))]

                if self.aggregate_by == 'closest':
                    cf_instance=samples_flip[np.argmin(distances)].reshape(X.shape)
                elif self.aggregate_by == 'mean':
                    cf_instance = samples_flip.mean(axis=0).reshape(X.shape)
                else:
                    cf_instance = None
                    raise NameError('Supported values for arg  "aggragate_by": {}, {}'.format('closest', 'mean'))

                print('Epsilon', self.epsilon)
                print('==========================')
                print('Number of samples:', len(samples_in))
                print('Original predicted class {} with probability {}: '.format(pred_class, max_proba_x))
                print('Majority class in sampled data points ', majority_class)
                print('Closest flipped class: ',
                      pred_classes[np.where(pred_classes != pred_class)][np.argmin(distances)])
                print('Original instance shape:', X.shape)
                print('Counfact instance shape:', cf_instance.shape)
                print('L1 distance from X ', self._metric_distance(cf_instance.flatten(), X.flatten()))

                self.explaning_instance = cf_instance
                self.samples_flip = samples_flip
                self.features_values_diff = cf_instance.flatten() - X.flatten()
                self.l1_distance = self._metric_distance(cf_instance.flatten(), X.flatten())

            self.epsilon += self.epsilon_step
            if self.epsilon >= self.max_epsilon:
                break

        if self.explaning_instance is None:
            raise NameError('Instance not found')

        return self.explaning_instance


class AdversarialSearch(BaseCounterFactual):

    def __init__(self, model, method='InnerBoundary', tollerance=0, initial_lam=1, metric='mad_distance'):

        super().__init__(model=model, sampling_method=None, epsilon=None, epsilon_step=None,
                         max_epsilon=None, nb_samples=None, metric=metric, aggregate_by=None,
                         method=method, tollerance=tollerance, initial_lam=initial_lam)

    def generate_explanation(self, X):

        probas_x = _predict(self.model, X)
        pred_class = np.argmax(probas_x, axis=1)[0]
        max_proba_x = probas_x[:, pred_class]

        if self.method == 'Wachter' or self.method == 'OuterBoundary':
            pass
        elif self.method == 'InnerBoundary':
            # find counterfactual instance with loss minimization method
            rs = _calculate_radius(f_ranges=self.f_ranges, epsilon=1)
            initial_instance = _generate_rnd_samples(X, rs, 1)
            initial_instance=_flatten_X(initial_instance)[0]

            def _countefactual_loss(x, XX=X.flatten(), lam=self.lam, yy=max_proba_x):
                return lam*(_predict(self.model, x.reshape(X.shape))[:, pred_class] - yy)**2 + \
                       self._metric_distance(x, XX)

            def _contrains_diff(x, yy=max_proba_x, tollerance=self.tollerance):
                return (_predict(self.model, x.reshape(X.shape))[:, pred_class] - yy) ** 2 <= tollerance

            cons=({'type': 'ineq', 'fun': _contrains_diff})
            t_0=time()
            res = minimize(_countefactual_loss, initial_instance, method='COBYLA', constraints=cons)
            print('Minimization time: ', time()-t_0)
            self.explaning_instance=res.x.reshape(X.shape)

            probas_exp = _predict(self.model, np.asarray(self.explaning_instance).reshape(X.shape))

            pred_class_exp = np.argmax(probas_exp, axis=1)[0]
            max_proba_exp = probas_exp[:, pred_class_exp]

            print('Original instance predicted class: {} with probability {}:'.format(pred_class, max_proba_x))
            print('Countfact instance predicted class: {} with probability {}:'.format(pred_class_exp, max_proba_exp))
            print('Original instance shape', X.shape)
            print('Countfact instance shape', self.explaning_instance.shape)
            print('L1 distance from X', self._metric_distance(self.explaning_instance.flatten(), X.flatten()))

        return self.explaning_instance










'''
class CounterFactual(object):

    def __init__(self, model, method='rndsample', tollerance=0, epsilon=None, max_epsilon=5, epsilon_step=0.1, lam=1,
                 nb_samples=100, epsilon_rnd=0.1, epsilon_minloss=1, metric=_mad_distance):

        self.model = model
        self.method = method

        if self.method=='rndsample' and epsilon is None:
            self.epsilon = epsilon_rnd
        elif self.method=='minloss' and epsilon is None:
            self.epsilon = epsilon_minloss
        else:
            self.epsilon = epsilon

        self.max_epsilon = max_epsilon
        self.tollerance = tollerance
        self.epsilon_step = epsilon_step
        self.lam = lam
        self.nb_samples = nb_samples
        self.callable_distance = metric

    def fit(self, X_train, y_train):

        self.f_ranges = _calculate_franges(X_train)
        self.mads = robust.mad(X_train,axis=0)+10e-10

    def _metric_distance(self, x0, x1):
        try:
            return self.callable_distance(x0, x1)
        except TypeError:
            return self.callable_distance(x0, x1, self.mads)

    def generate_explanation(self, X, aggregate_by='closest', sampling_method='poisson'):

        probas_x = _predict(self.model, X)

        pred_class = np.argmax(probas_x, axis=1)[0]
        max_proba_x = probas_x[:, pred_class]
        cond = False

        if self.method == 'rndsample':
            # find counterfactual instance with random sampling method
            while not cond:
                rs = _calculate_radius(f_ranges=self.f_ranges , epsilon=self.epsilon)

                if sampling_method=='uniform':
                    samples_in = _flatten_X(_generate_rnd_samples(X, rs, self.nb_samples))
                elif sampling_method=='poisson':
                    samples_in = _flatten_X(_generate_poisson_samples(X, self.nb_samples))
                elif sampling_method=='gaussian':
                    samples_in = _flatten_X(_generate_gaussian_samples(X, rs, self.nb_samples))
                else:
                    raise NameError('method {} not implemented'.format(sampling_method))

                probas_si = _predict(self.model, samples_in.reshape((samples_in.shape[0],) + X.shape[1:]))

                pred_classes = np.argmax(probas_si, axis=1)
                unique, counts = np.unique(pred_classes, return_counts=True)
                majority_class = unique[np.argmax(counts)]
                print('Original predicted class: {}; Majority class in sampled data: {}'.format(pred_class,
                                                                                                majority_class))

                self.explaning_instance='Instance not found'
                if aggregate_by == 'closest':
                    cond = (pred_classes != pred_class).any()
                elif aggregate_by == 'mean':
                    cond = (majority_class != pred_class)
                if cond:
                    samples_flip = samples_in[np.where(pred_classes != pred_class)]
                    distances = [self._metric_distance(samples_flip[i], X.flatten()) for i in range(len(samples_flip))]

                    if aggregate_by == 'closest':
                        cf_instance=samples_flip[np.argmin(distances)].reshape(X.shape)
                    elif aggregate_by == 'mean':
                        cf_instance = samples_flip.mean(axis=0).reshape(X.shape)

                    print('Epsilon', self.epsilon)
                    print('==========================')
                    print('Number of samples:', len(samples_in))
                    print('Original predicted class {} with probability {}: '.format(pred_class, max_proba_x))
                    print('Majority class in sampled data points ', majority_class)
                    print('Closest flipped class: ',
                          pred_classes[np.where(pred_classes != pred_class)][np.argmin(distances)])
                    print('Original instance shape:', X.shape)
                    print('Counfact instance shape:', cf_instance.shape)
                    print('L1 distance from X ', self._metric_distance(cf_instance.flatten(), X.flatten()))

                    self.explaning_instance = cf_instance
                    self.samples_flip = samples_flip
                    self.features_values_diff = cf_instance.flatten() - X.flatten()
                    self.l1_distance = self._metric_distance(cf_instance.flatten(), X.flatten())

                self.epsilon += self.epsilon_step
                if self.epsilon >= self.max_epsilon:
                    break

            return self.explaning_instance

        elif self.method == 'minloss':
            # find counterfactual instance with loss minimization method
            rs = _calculate_radius(f_ranges=self.f_ranges, epsilon=self.epsilon)
            initial_instance = _generate_rnd_samples(X, rs, self.nb_samples)
            initial_instance=_flatten_X(initial_instance)[0]

            def _countefactual_loss(x, XX=X.flatten(), lam=self.lam, yy=max_proba_x):
                return lam*(_predict(self.model, x.reshape(X.shape))[:, pred_class] - yy)**2 + \
                       self._metric_distance(x, XX)

            def _contrains_diff(x, yy=max_proba_x, tollerance=self.tollerance):
                return (_predict(self.model, x.reshape(X.shape))[:, pred_class] - yy) ** 2 <= tollerance

            def _contrains_dist(x, XX=X.flatten(), epsilon=self.epsilon):
                return self._metric_distance(x, XX) < epsilon * len(XX)

            #cons=({'type': 'ineq', 'fun': _contrains_dist},
            #      {'type': 'ineq', 'fun': _contrains_diff})
            cons=({'type': 'ineq', 'fun': _contrains_diff})
            t_0=time()
            res = minimize(_countefactual_loss, initial_instance, method='COBYLA', constraints=cons)
            print('Minimization time: ', time()-t_0)
            self.explaning_instance=res.x.reshape(X.shape)

            probas_exp = _predict(self.model, np.asarray(self.explaning_instance).reshape(X.shape))

            pred_class_exp = np.argmax(probas_exp, axis=1)[0]
            max_proba_exp = probas_exp[:, pred_class_exp]

            print('Original instance predicted class: {} with probability {}:'.format(pred_class, max_proba_x))
            print('Countfact instance predicted class: {} with probability {}:'.format(pred_class_exp, max_proba_exp))
            print('Original instance shape', X.shape)
            print('Countfact instance shape', self.explaning_instance.shape)
            print('L1 distance from X', self._metric_distance(self.explaning_instance.flatten(), X.flatten()))
            return self.explaning_instance
'''