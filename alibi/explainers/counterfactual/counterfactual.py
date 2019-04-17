import numpy as np
from typing import Callable, Dict, Optional, Tuple, Union
from statsmodels.tools.numdiff import approx_fprime
from scipy.spatial.distance import cityblock
import tensorflow as tf
import logging

logger = logging.getLogger(__name__)

_metric_dict = {'l1': cityblock}  # type: Dict[str, Callable]


def _define_func(predict_fn: Callable,
                 pred_class: int,
                 target_class: Union[str, int] = 'same') -> Tuple[Callable, Union[str, int]]:
    """
    Define the class-specific prediction function to be used in the optimization.

    Parameters
    ----------
    predict_fn
        Classifier prediction function
    pred_class
        Predicted class of the instance to be explained
    target_class
        Target class of the explanation, one of 'same', 'other' or an integer class

    Returns
    -------
        Class-specific prediction function and the target class used.

    """
    if target_class == 'other':

        def func(X):
            probas = predict_fn(X)
            sorted = np.argsort(-probas)  # class indices in decreasing order of probability

            # take highest probability class different from class predicted for X
            if sorted[0, 0] == pred_class:
                target_class = sorted[0, 1]
                # logger.debug('Target class equals predicted class')
            else:
                target_class = sorted[0, 0]

            # logger.debug('Current best target class: %s', target_class)
            return predict_fn(X)[:, target_class]

        return func, target_class

    elif target_class == 'same':
        target_class = pred_class

    def func(X):  # type: ignore
        return predict_fn(X)[:, target_class]

    return func, target_class


def num_grad(func: Callable, X: np.ndarray, args: Tuple = (), epsilon: float = 1e-08) -> np.ndarray:
    """
    Compute the numerical gradient using the symmetric difference. Currently wraps statsmodels implementation.
    Parameters
    ----------
    func
        Function to differentiate
    X
        Point at which to compute the gradient
    args
        Additional arguments to the function
    epsilon
        Step size for computing the gradient
    Returns
    -------
    Numerical gradient
    """
    gradient = approx_fprime(X, func, epsilon=epsilon, args=args, centered=True)
    return gradient


def get_wachter_grads(X_current: np.ndarray,
                      predict_class_fn: Callable,
                      distance_fn: Callable,
                      X_test: np.ndarray,
                      target_proba: float,
                      lam: float,
                      epsilons: Union[float, np.ndarray] = None,
                      method: str = 'wachter') -> Tuple[Union[float, np.ndarray], ...]:
    """
    Calculate the gradients of the loss function in Wachter et al. (2017)
    Parameters
    ----------
    X_current
        Candidate counterfactual wrt which the gradient is taken
    predict_class_fn
        Prediction function specific to the target class of the counterfactual
    distance_fn
        Distance function in feature space
    X_test
        Sample to be explained
    target_proba
        Target probability to for the counterfactual instance to satisfy
    lam
        Hyperparameter balancing the loss contribution of the distance in prediction (higher lam -> more weight)
    epsilons
        Steps sizes for computing the gradient passed to the num_grad function
    method
        Loss optimization method - one of 'wachter' or 'adiabatic'
    Returns
    -------
    Loss and gradient of the Wachter loss

    """
    if isinstance(epsilons, float):
        eps = epsilons
    else:
        eps = None

    pred = predict_class_fn(X_current)
    logger.debug('Current prediction: %s', pred)

    # numerical gradient of the black-box prediction function (specific to the target class)
    prediction_grad = num_grad(predict_class_fn, X_current.squeeze(), epsilon=eps)  # TODO feature-wise epsilons

    # numerical gradient of the distance function between the current point and the point to be explained
    distance_grad = num_grad(distance_fn, X_current.squeeze(), args=tuple([X_test.squeeze()]),
                             epsilon=eps)  # TODO epsilons

    logger.debug('Norm of prediction_grad: %s', np.linalg.norm(prediction_grad.flatten()))
    logger.debug('Norm of distance_grad: %s', np.linalg.norm(distance_grad.flatten()))
    logger.debug('pred - target_proba = %s', pred - target_proba)

    # gradient of the Wachter loss
    if method == 'wachter':
        loss = lam * (pred - target_proba) ** 2 + distance_fn(X_current, X_test)
        grad_loss = 2 * lam * (pred - target_proba) * prediction_grad + distance_grad  # TODO convex combination

    elif method == 'adiabatic':
        loss = lam * (pred - target_proba) ** 2 + (1 - lam) * distance_fn(X_current, X_test)
        grad_loss = 2 * lam * (pred - target_proba) * prediction_grad + (1 - lam) * distance_grad

    else:
        raise ValueError('Only loss optimization methods available are wachter and adiabatic')

    logger.debug('Method: %s', method)
    logger.debug('Loss: %s', loss)
    logger.debug('Norm of grad_loss: %s', np.linalg.norm(grad_loss.flatten()))

    return loss, grad_loss


class CounterFactual:

    def __init__(self,
                 sess: tf.Session,
                 predict_fn: Callable,
                 data_shape: Tuple[int, ...],
                 distance_fn: str = 'l1',
                 target_proba: float = 0.9,
                 target_class: Union[str, int] = 'other',
                 max_iter: int = 100,
                 lam_init: float = 0.01,
                 lam_step: float = 0.001,
                 tol: float = 0.05,
                 feature_range: Union[Tuple, str] = None,  # important for positive features
                 epsilons: Union[float, np.ndarray] = None,  # feature-wise epsilons
                 method: str = 'wachter'):
        logger.warning('Counterfactual explainer currently only supports numeric features')
        self.sess = sess
        self.data_shape = data_shape
        self.target_proba = target_proba
        self.target_class = target_class

        # options for the optimizer
        self.max_iter = max_iter
        self.lam_init = lam_init
        self.lam_step = lam_step
        self.tol = tol

        self.epsilons = epsilons
        self.method = method

        # TODO: support predict and predict_proba types for functions
        self.predict_fn = lambda x: predict_fn(x.reshape(1, -1))  # Is this safe?

        try:
            self.distance_fn = _metric_dict[distance_fn]
        except KeyError:
            logger.exception('Distance metrics %s not supported', distance_fn)
            raise

        if feature_range is not None:
            logger.warning('Feature range specified')

        # flag to keep track if explainer is fit or not
        self.fitted = False

        # set up graph session
        self.cf = tf.get_variable('counterfactual', shape=data_shape,
                                  dtype=tf.float32)  # TODO initialize when explain is called

        # optimizer
        opt = tf.train.AdamOptimizer()  # TODO optional argument to change type

        # training setup
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        # grads_and_vars = opt.compute_gradients(, vars = [cf]) TODO if differentiable distance
        self.grad_ph = tf.placeholder(shape=data_shape, dtype=tf.float32)
        grad_and_var = [(self.grad_ph, self.cf)]  # could be cf.name

        self.apply_grad = opt.apply_gradients(grad_and_var, global_step=self.global_step)  # TODO gradient clipping?

        # self.init = tf.variables_initializer(var_list=[self.global_step, self.cf])
        # self.sess.run(self.init)  # where to put this?
        self.sess.run(tf.global_variables_initializer())  # TODO replace with localized version?

        return

    def _initialize(self):
        # TODO initialization strategies ("same", "random", "from_train")

        pass

    def fit(self,
            X: np.ndarray,
            y: Optional[np.ndarray]) -> None:
        # TODO feature ranges, epsilons and MADs

        self.fitted = True

    def explain(self, X: np.ndarray) -> Dict:

        # make a prediction
        probas = self.predict_fn(X)
        pred_class = probas.argmax()

        # define the class-specific prediction function
        self.predict_class_fn, t_class = _define_func(self.predict_fn, pred_class, self.target_class)

        if not self.fitted:
            logger.warning('Explain called before fit, explainer will operate in unsupervised mode.')

        # initialize with an instance
        X_init = X  # TODO use _initialize

        # minimize loss iteratively
        exp_dict = self._minimize_wachter_loss(X, X_init)

        return exp_dict

    def _minimize_wachter_loss(self,
                               X: np.ndarray,
                               X_init: np.ndarray) -> Dict:
        # first minimization
        logger.debug('######################################################## ITERATION: 0')

        X_current = X_init
        lam = self.lam_init
        loss, gradients = get_wachter_grads(X_current=X_current, predict_class_fn=self.predict_class_fn,
                                            distance_fn=self.distance_fn, X_test=X, target_proba=self.target_proba,
                                            lam=lam, epsilons=self.epsilons, method=self.method)
        self.sess.run(self.apply_grad, feed_dict={self.grad_ph: gradients})
        X_current = self.sess.run(self.cf)
        probas = self.predict_fn(X_current)
        pred_class = probas.argmax()
        p = probas.max()
        logger.debug('Iteration: 0, cf pred_class: %s, cf proba: %s', pred_class, p)
        logger.debug('Iteration: 0, distance d(X_current, X): %s', self.distance_fn(X, X_current))

        Xs = []
        losses = []
        grads = []
        lambdas = []
        Xs.append(X_current)
        losses.append(loss)
        grads.append(gradients)
        lambdas.append(lam)

        num_iter = 0
        return_dict = {'X_cf': X_current,
                       'loss': losses,
                       'grads': grads,
                       'Xs': Xs,
                       'lambdas': lambdas,
                       'n_iter': num_iter}
        # main loop
        while np.abs(self.predict_class_fn(X_current) - self.target_proba) > self.tol:
            logger.debug('######################################################## ITERATION: %s', num_iter + 1)
            if num_iter == self.max_iter:
                logger.warning(
                    'Maximum number of iterations reached without finding a counterfactual.'
                    'Increase max_iter, tolerance or the lambda hyperparameter.')
                return return_dict

            # minimize the loss
            num_iter += 1
            lam += self.lam_step
            logger.debug('Increased lambda to %s', lam)
            loss, gradients = get_wachter_grads(X_current=X_current, predict_class_fn=self.predict_class_fn,
                                                distance_fn=self.distance_fn, X_test=X, target_proba=self.target_proba,
                                                lam=lam, epsilons=self.epsilons, method=self.method)
            self.sess.run(self.apply_grad, feed_dict={self.grad_ph: gradients})
            X_current = self.sess.run(self.cf)

            probas = self.predict_fn(X_current)
            pred_class = probas.argmax()
            p = probas.max()
            logger.debug('Iteration: %s, cf pred_class: %s, cf proba: %s', num_iter, pred_class, p)
            logger.debug('Iteration: %s, distance d(X_current, X): %s', num_iter, self.distance_fn(X, X_current))

            Xs.append(X_current)
            losses.append(loss)
            grads.append(gradients)
            lambdas.append(lam)

            return_dict['X_cf'] = X_current
            return_dict['n_iter'] = num_iter

        return return_dict
