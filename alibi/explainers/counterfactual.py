import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union
import tensorflow as tf
import logging

logger = logging.getLogger(__name__)


def cityblock_batch(X: np.ndarray,
                    y: np.ndarray) -> np.ndarray:
    """
    Calculate the L1 distances between a batch of arrays X and an array of the same shape y.

    Parameters
    ----------
    X
        Batch of arrays to calculate the distances from
    y
        Array to calculate the distance to

    Returns
    -------
    Array of distances from each array in X to y

    """
    X_dim = len(X.shape)
    y_dim = len(y.shape)

    if X_dim == y_dim:
        assert y.shape[0] == 1, 'y mush have batch size equal to 1'
    else:
        assert X.shape[1:] == y.shape, 'X and y must have matching shapes'

    return np.abs(X - y).sum(axis=tuple(np.arange(1, X_dim))).reshape(X.shape[0], -1)


_metric_dict = {'l1': cityblock_batch}  # type: Dict[str, Callable]


def _define_func(predict_fn: Callable,
                 pred_class: int,
                 target_class: Union[str, int] = 'same') -> Tuple[Callable, Union[str, int]]:
    # TODO: convert to batchwise function
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
        # TODO: need to optimize this

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
            return (predict_fn(X)[:, target_class]).reshape(-1, 1)

        return func, target_class

    elif target_class == 'same':
        target_class = pred_class

    def func(X):  # type: ignore
        return (predict_fn(X)[:, target_class]).reshape(-1, 1)

    return func, target_class


def _perturb(X: np.ndarray,
             eps: Union[float, np.ndarray] = 1e-08,
             proba: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply perturbation to instance or prediction probabilities. Used for numerical calculation of gradients.

    Parameters
    ----------
    X
        Array to be perturbed
    eps
        Size of perturbation
    proba
        If True, the net effect of the perturbation needs to be 0 to keep the sum of the probabilities equal to 1

    Returns
    -------
    Instances where a positive and negative perturbation is applied.
    """
    # N = batch size; F = nb of features in X
    shape = X.shape
    X = np.reshape(X, (shape[0], -1))  # NxF
    dim = X.shape[1]  # F
    pert = np.tile(np.eye(dim) * eps, (shape[0], 1))  # (N*F)xF
    if proba:
        eps_n = eps / (dim - 1)
        pert += np.tile((np.eye(dim) - np.ones((dim, dim))) * eps_n, (shape[0], 1))  # (N*F)xF
    X_rep = np.repeat(X, dim, axis=0)  # (N*F)xF
    X_pert_pos, X_pert_neg = X_rep + pert, X_rep - pert
    shape = (dim * shape[0],) + shape[1:]
    X_pert_pos = np.reshape(X_pert_pos, shape)  # (N*F)x(shape of X[0])
    X_pert_neg = np.reshape(X_pert_neg, shape)  # (N*F)x(shape of X[0])
    return X_pert_pos, X_pert_neg


def num_grad_batch(func: Callable,
                   X: np.ndarray,
                   args: Tuple = (),
                   eps: Union[float, np.ndarray] = 1e-08) -> np.ndarray:
    """
    Calculate the numerical gradients of a vector-valued function (typically a prediction function in classification)
    with respect to a batch of arrays X.

    Parameters
    ----------
    func
        Function to be differentiated
    X
        A batch of vectors at which to evaluate the gradient of the function
    args
        Any additional arguments to pass to the function
    eps
        Gradient step to use in the numerical calculation, can be a single float or one for each feature

    Returns
    -------
    An array of gradients at each point in the batch X

    """
    # N = gradient batch size; F = nb of features in X, P = nb of prediction classes, B = instance batch size
    batch_size = X.shape[0]
    data_shape = X[0].shape
    preds = func(X, *args)
    X_pert_pos, X_pert_neg = _perturb(X, eps)  # (N*F)x(shape of X[0])
    X_pert = np.concatenate([X_pert_pos, X_pert_neg], axis=0)
    preds_concat = func(X_pert, *args)  # make predictions
    n_pert = X_pert_pos.shape[0]

    grad_numerator = preds_concat[:n_pert] - preds_concat[n_pert:]  # (N*F)*P
    grad_numerator = np.reshape(np.reshape(grad_numerator, (batch_size, -1)),
                                (batch_size, preds.shape[1], -1), order='F')  # NxPxF

    grad = grad_numerator / (2 * eps)  # NxPxF
    grad = grad.reshape(preds.shape + data_shape)  # BxPx(shape of X[0])

    return grad


def get_wachter_grads(X_current: np.ndarray,
                      predict_class_fn: Callable,
                      distance_fn: Callable,
                      X_test: np.ndarray,
                      target_proba: float,
                      lam: float,
                      eps: Union[float, np.ndarray] = 1e-08
                      ) -> Tuple[np.ndarray, np.ndarray, Tuple]:
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
    eps
        Steps sizes for computing the gradient passed to the num_grad function
    Returns
    -------
    Loss and gradient of the Wachter loss

    """
    if isinstance(eps, float):
        eps = eps
    else:
        eps = None

    pred = predict_class_fn(X_current)
    logger.debug('Current prediction: p=%s', pred)

    # numerical gradient of the black-box prediction function (specific to the target class)
    prediction_grad = num_grad_batch(predict_class_fn, X_current, eps=eps)  # TODO feature-wise epsilons

    # numerical gradient of the distance function between the current point and the point to be explained
    distance_grad = num_grad_batch(distance_fn, X_current, args=tuple([X_test]),
                                   eps=eps)  # TODO epsilons

    norm_pred_grad = np.linalg.norm(prediction_grad.flatten())
    norm_dist_grad = np.linalg.norm(distance_grad.flatten())
    logger.debug('Norm of prediction_grad: %s', norm_pred_grad)
    logger.debug('Norm of distance_grad: %s', norm_dist_grad)
    logger.debug('pred - target_proba = %s', pred - target_proba)

    # numeric loss and gradient
    loss = (pred - target_proba) ** 2 + lam * distance_fn(X_current, X_test)
    grad_loss = 2 * (pred - target_proba) * prediction_grad + lam * distance_grad

    norm_grad = np.linalg.norm(grad_loss.flatten())
    logger.debug('Loss: %s', loss)
    logger.debug('Norm of grad_loss: %s', norm_grad)

    debug_info = tuple([norm_pred_grad, norm_dist_grad, norm_grad])

    return loss, grad_loss, debug_info


class CounterFactual:

    def __init__(self,
                 sess: tf.Session,
                 predict_fn: Callable,
                 data_shape: Tuple[int, ...],
                 distance_fn: str = 'l1',
                 target_proba: float = 0.95,
                 target_class: Union[str, int] = 'other',
                 max_iter: int = 1000,
                 lam_init: float = 1e-07,
                 lam_step: float = 10,
                 max_lam_steps: int = 10,
                 tol: float = 0.025,
                 learning_rate_init=0.1,
                 feature_range: Union[Tuple, str] = (-1e10, 1e10),  # important for positive features
                 eps: Union[float, np.ndarray] = 0.01,  # feature-wise epsilons
                 init: str = 'identity',
                 bisect=True,
                 decay=True):
        """
        Initialize counterfactual explanation method based on Wachter et al. (2017)

        Parameters
        ----------
        sess
            TensorFlow session
        predict_fn
            Keras or TensorFlow model or any other model's prediction function returning class probabilities
        data_shape
            Shape of input data starting with batch size
        distance_fn
            Distance function to use in the loss term
        target_proba
            Target probability for the counterfactual to reach
        target_class
            Target class for the counterfactual to reach, one of 'other', 'same' or an integer denoting
            desired class membership for the counterfactual instance
        max_iter
            Maximum number of interations to run the gradient descent for (inner loop)
        lam_init
            Initial regularization constant for the prediction part of the Wachter loss
        lam_step
            Regularization constant step size used in the search
        max_lam_steps
            Maximum number of times to increase the regularization constant (outer loop) before terminating the search
        tol
            Tolerance for the counterfactual target probability
        feature_range
            Tuple with min and max ranges to allow for perturbed instances. Min and max ranges can be floats or
            numpy arrays with dimension (1 x nb of features) for feature-wise ranges
        eps
            Gradient step sizes used in calculating numerical gradients, defaults to a single value for all
            features, but can be passed an array for feature-wise step sizes
        init
            Initialization method for the search of counterfactuals, one of 'random' or 'identity'
        """

        logger.warning('Counterfactual explainer currently only supports numeric features')
        self.sess = sess
        self.data_shape = data_shape
        self.batch_size = data_shape[0]
        self.target_proba = target_proba
        self.target_class = target_class

        # options for the optimizer
        self.max_iter = max_iter
        self.lam_init = lam_init
        self.lam_step = lam_step
        self.tol = tol
        self.max_lam_steps = max_lam_steps

        self.eps = eps
        self.init = init
        self.feature_range = feature_range

        self.bisect = bisect

        # TODO: support predict and predict_proba types for functions
        self.predict_fn = predict_fn
        if hasattr(predict_fn, 'predict'):  # Keras or TF model
            self.model = True
            # self.predict_fn = lambda x: predict_fn(tf.reshape(x, (1, -1)))  # Is this safe? batch?
            n_classes = self.sess.run(self.predict_fn(tf.convert_to_tensor(np.zeros(data_shape),
                                                                           dtype=tf.float32))).shape[1]
        else:
            self.model = False  # black-box model
            self.predict_fn = predict_fn  # x: predict_fn(x.reshape(1, -1))  # Is this safe?
            n_classes = self.predict_fn(np.zeros(data_shape)).shape[1]

        # TODO remove this entirely?
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
        with tf.variable_scope('cf_search', reuse=tf.AUTO_REUSE):

            # original instance, candidate counterfactual within feature range and target labels
            self.orig = tf.get_variable('original', shape=data_shape, dtype=tf.float32)
            self.cf = tf.get_variable('counterfactual', shape=data_shape,
                                      dtype=tf.float32,
                                      constraint=lambda x: tf.clip_by_value(x, feature_range[0], feature_range[1]))
            # TODO initialize when explain is called
            self.target = tf.get_variable('target', shape=(1, n_classes), dtype=tf.float32)

            # hyperparameter in the loss
            self.lam = tf.Variable(self.lam_init, name='lambda')

            # L1 distance and MAD constants
            self.l1 = tf.norm(tf.subtract(self.cf, self.orig), ord=1)
            # self.mads = tf.Variable(np.ones(), name='mads')  # TODO size

            # optimizer
            self.global_step = tf.Variable(0.0, trainable=False, name='global_step')
            if decay:
                self.learning_rate = tf.train.polynomial_decay(learning_rate_init, self.global_step,
                                                               self.max_iter, 0.0, power=0.5)
            else:
                self.learning_rate = learning_rate_init

            opt = tf.train.AdamOptimizer(
                self.learning_rate)  # TODO optional argument to change type, learning rate scheduler

            # training setup
            # grads_and_vars = opt.compute_gradients(, vars = [cf]) TODO if differentiable distance
            self.grad_ph = tf.placeholder(shape=data_shape, dtype=tf.float32)
            grad_and_var = [(self.grad_ph, self.cf)]  # could be cf.name

            self.apply_grad = opt.apply_gradients(grad_and_var,
                                                  global_step=self.global_step)  # TODO gradient clipping?

        self.tf_init = tf.variables_initializer(var_list=tf.global_variables(scope='cf_search'))

        return

    def _initialize(self, X: np.ndarray) -> np.ndarray:
        # TODO initialization strategies ("same", "random", "from_train")

        if self.init == 'identity':
            X_init = X
            logger.debug('Initializing search at the test point X')
        elif self.init == 'random':
            # TODO: handle ranges
            X_init = np.random.rand(*self.data_shape)
            logger.debug('Initializing search at a random test point')
        else:
            raise ValueError('Initialization method should be one of "random" or "identity"')

        return X_init

    def fit(self,
            X: np.ndarray,
            y: Optional[np.ndarray]) -> None:
        # TODO feature ranges, epsilons and MADs

        self.fitted = True

    def explain(self, X: np.ndarray) -> Dict:

        # make a prediction
        probas = self.predict_fn(X)
        pred_class = probas.argmax()
        logger.debug('Initial prediction: %s with p=%s', pred_class, probas.max())

        # define the class-specific prediction function
        self.predict_class_fn, t_class = _define_func(self.predict_fn, pred_class, self.target_class)

        if not self.fitted:
            logger.warning('Explain called before fit, explainer will operate in unsupervised mode.')

        # initialize with an instance
        X_init = self._initialize(X)

        # minimize loss iteratively
        exp_dict = self._minimize_wachter_loss(X, X_init)

        return exp_dict

    def _prob_condition(self, X_current):
        return np.abs(self.predict_class_fn(X_current) - self.target_proba) <= self.tol

    def _minimize_wachter_loss(self,
                               X: np.ndarray,
                               X_init: np.ndarray) -> Dict:
        Xs = []  # type: List[np.ndarray]
        losses = []  # type: List[float]
        grads = []  # type: List[np.ndarray]
        dists = []  # type: List[float]
        lambdas = []  # type: List[float]
        prob_cond = []  # type: List[bool]
        prob = []  # type: List[float]
        classes = []  # type: List[int]
        debug_info = []  # type: List[Tuple]

        cf_found = np.zeros((self.max_lam_steps), dtype=bool)

        return_dict = {'X_cf': X_init,
                       'loss': losses,
                       'grads': grads,
                       'dists': dists,
                       'Xs': Xs,
                       'lambdas': lambdas,
                       'prob_cond': prob_cond,
                       'prob': prob,
                       'classes': classes,
                       'debug_info': debug_info,
                       'cf_found': cf_found,
                       'success': False}

        lam = self.lam_init
        lam_lb = 0.0
        lam_ub = 1e10

        lam_steps = 0
        X_current = X_init
        # expanding = True  # flag to check if we are expanding the search or zooming in on a solution
        for l_step in range(self.max_lam_steps):
            self.sess.run(self.tf_init)  # where to put this

            # cf_found = False  # flag to use for increasing/decreasing lambda
            # TODO need some early stopping when lambda grows too big to satisfy prob_cond
            # while np.abs(self.predict_class_fn(X_current) - self.target_proba) > self.tol:
            lr = self.sess.run(self.learning_rate)
            logger.info('Starting outer loop: %s/%s with lambda=%s, lr=%s', lam_steps + 1, self.max_lam_steps, lam, lr)

            # if lam_steps == self.max_lam_steps:
            #    logger.warning(
            #        'Maximum number of iterations reached without finding a counterfactual.'
            #        'Increase max_lam_steps, tolerance or the lambda hyperparameter.')
            #    return return_dict

            num_iter = 0

            # number of gradient descent steps in each inner loop
            for i in range(self.max_iter):
                # minimize the loss
                num_iter += 1
                loss, gradients, d_info = get_wachter_grads(X_current=X_current, predict_class_fn=self.predict_class_fn,
                                                            distance_fn=self.distance_fn, X_test=X,
                                                            target_proba=self.target_proba,
                                                            lam=lam, eps=self.eps)
                # squeeze out class dimension
                gradients = gradients.reshape(self.data_shape)

                self.sess.run(self.apply_grad, feed_dict={self.grad_ph: gradients})
                X_current = self.sess.run(self.cf)

                # if probability condition satisfied, add to list of potential CFs TODO keep track of all for debugging
                # if self._prob_condition(X_current):
                Xs.append(X_current)
                losses.append(loss.squeeze())
                grads.append(gradients)
                dists.append(self.distance_fn(X, X_current).squeeze())
                lambdas.append(lam)

                cond = self._prob_condition(X_current).squeeze()
                if cond:
                    cf_found[l_step] = True
                    return_dict['X_cf'] = X_current
                    logger.debug('CF found')
                prob_cond.append(cond)

                probas = self.predict_fn(X_current)
                pred_class = probas.argmax()
                p = probas.max()

                prob.append(p)
                classes.append(pred_class)
                debug_info.append(d_info)

                logger.debug('Iteration: %s, cf pred_class: %s, cf proba: %s', lam_steps, pred_class, p)
                logger.debug('Iteration: %s, distance d(X_current, X): %s', lam_steps,
                             self.distance_fn(X, X_current))

                lr = self.sess.run(self.learning_rate)
                gs = self.sess.run(self.global_step)
                logger.info('Current learning rate lr=%s', lr)
                logger.info('Current global step: %s', gs)

            # adjust the lambda constant
            if self.bisect:
                if cf_found[l_step]:
                    # want to improve the solution by putting more weight on the distance term
                    # by increasing lambda
                    lam_lb = max(lam, lam_lb)
                    logger.debug('Lambda bounds: (%s, %s)', lam_lb, lam_ub)
                    if lam_ub < 1e9:
                        lam = (lam_lb + lam_ub) / 2
                    else:
                        lam *= self.lam_step
                        logger.debug('Changed lambda to %s', lam)

                elif not cf_found[l_step]:
                    if l_step == 0:
                        logger.warning('No solution found in the first iteration, try to reduce target_proba'
                                       'or increase tolerance.')
                        return return_dict

                    # want to refine the last solution by finding the closest solution that is
                    # still a counterfactual by bisecting lambda

                    lam_ub = min(lam_ub, lam)
                    logger.debug('Lambda bounds: (%s, %s)', lam_lb, lam_ub)
                    if lam_ub < 1e9:
                        lam = (lam_lb + lam_ub) / 2
                        logger.debug('Changed lambda to %s', lam)

                lam_steps += 1
            else:
                lam_steps += 1
                lam *= self.lam_step

        return_dict['success'] = True

        return return_dict
