import numpy as np
from typing import Callable, Dict, Optional, Tuple, Union
import keras
import tensorflow as tf
import keras
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


class CounterFactual:

    def __init__(self,
                 sess: tf.Session,
                 predict_fn: Union[Callable, tf.keras.Model, keras.Model],
                 data_shape: Tuple[int, ...],
                 distance_fn: str = 'l1',
                 target_proba: float = 1.0,
                 target_class: Union[str, int] = 'other',
                 max_iter: int = 1000,
                 lam_init: float = 1e-04,
                 max_lam_steps: int = 10,
                 tol: float = 0.05,
                 learning_rate_init=0.1,
                 feature_range: Union[Tuple, str] = (-1e10, 1e10),  # important for positive features
                 eps: Union[float, np.ndarray] = 0.01,  # feature-wise epsilons
                 init: str = 'identity',
                 decay: bool = False,
                 write_dir: str = None,
                 debug=False,
                 bisect: bool = True) -> None:
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
        max_lam_steps
            Maximum number of times to adjust the regularization constant (outer loop) before terminating the search
        tol
            Tolerance for the counterfactual target probability
        learning_rate_init
            Initial learning rate for each outer loop of lambda
        feature_range
            Tuple with min and max ranges to allow for perturbed instances. Min and max ranges can be floats or
            numpy arrays with dimension (1 x nb of features) for feature-wise ranges
        eps
            Gradient step sizes used in calculating numerical gradients, defaults to a single value for all
            features, but can be passed an array for feature-wise step sizes
        init
            Initialization method for the search of counterfactuals, one of 'random' or 'identity'
        decay
            Flag to decay learning rate to zero for each outer loop over lambda
        write_dir
            Directory to write Tensorboard files to
        debug
            Flag to write Tensorboard summaries for debugging
        """

        logger.warning('Counterfactual explainer currently only supports numeric features')
        self.sess = sess
        self.data_shape = data_shape
        self.batch_size = data_shape[0]
        self.target_class = target_class

        # options for the optimizer
        self.max_iter = max_iter
        self.lam_init = lam_init
        self.tol = tol
        self.max_lam_steps = max_lam_steps

        self.eps = eps
        self.init = init
        self.feature_range = feature_range
        self.target_proba_arr = target_proba * np.ones(self.batch_size)

        self.debug = debug
        self.bisect = bisect

        if isinstance(predict_fn, (tf.keras.Model, keras.Model)):  # Keras or TF model
            self.model = True
            self.predict_fn = predict_fn.predict  # array function
            self.predict_tn = predict_fn  # tensor function
            self.n_classes = self.sess.run(self.predict_tn(tf.convert_to_tensor(np.zeros(data_shape),
                                                                                dtype=tf.float32))).shape[1]
        else:  # black-box model
            self.predict_fn = predict_fn
            self.predict_tn = None
            self.model = False
            self.n_classes = self.predict_fn(np.zeros(data_shape)).shape[1]

        # flag to keep track if explainer is fit or not
        self.fitted = False

        # set up graph session for optimization (counterfactual search)
        with tf.variable_scope('cf_search', reuse=tf.AUTO_REUSE):

            # define variables for original and candidate counterfactual instances, target labels and lambda
            self.orig = tf.get_variable('original', shape=data_shape, dtype=tf.float32)
            self.cf = tf.get_variable('counterfactual', shape=data_shape,
                                      dtype=tf.float32,
                                      constraint=lambda x: tf.clip_by_value(x, feature_range[0], feature_range[1]))
            # the following will be a 1-hot encoding of the target class (as predicted by the model)
            self.target = tf.get_variable('target', shape=(self.batch_size, self.n_classes), dtype=tf.float32)

            # constant target probability and global step variable
            self.target_proba = tf.constant(target_proba * np.ones(self.batch_size), dtype=tf.float32,
                                            name='target_proba')
            self.global_step = tf.Variable(0.0, trainable=False, name='global_step')

            # lambda hyperparameter - annealed in the first epoch
            # if anneal_lam:
            #    self.lam_anneal = tf.reshape(tf.train.exponential_decay(self.lam_init, self.global_step,
            #                                          self.max_iter, 1e-10 / lam_init), (1, -1))
            # self.lam = tf.Variable(self.lam_init * np.ones(self.batch_size), name='lambda', dtype=tf.float32)
            self.lam = tf.placeholder(tf.float32, shape=(self.batch_size), name='lam')

            # define placeholders that will be assigned to relevant variables
            self.assign_orig = tf.placeholder(tf.float32, data_shape, name='assing_orig')
            self.assign_cf = tf.placeholder(tf.float32, data_shape, name='assign_cf')
            self.assign_target = tf.placeholder(tf.float32, shape=(self.batch_size, self.n_classes),
                                                name='assign_target')
            # self.assign_lam = tf.placeholder(tf.float32, shape=(self.batch_size), name='assign_lam')

            # L1 distance and MAD constants
            # TODO: refactor? MADs?
            ax_sum = list(np.arange(1, len(self.data_shape)))
            if distance_fn == 'l1':
                self.dist = tf.reduce_sum(tf.abs(self.cf - self.orig), axis=ax_sum, name='l1')
            else:
                logger.exception('Distance metric %s not supported', distance_fn)
                raise ValueError

            # distance loss
            self.loss_dist = self.lam * self.dist

            # prediction loss
            if not self.model:
                # will need to calculate gradients numerically
                self.loss_opt = self.loss_dist
            else:
                # autograd gradients throughout
                self.pred_proba = self.predict_tn(self.cf)

                # 3 cases for target_class
                if target_class == 'same':
                    self.pred_proba_class = tf.reduce_max(self.target * self.pred_proba, 1)
                elif target_class == 'other':
                    self.pred_proba_class = tf.reduce_max((1 - self.target) * self.pred_proba, 1)
                elif isinstance(target_class, int):
                    # if class is specified, this is known in advance
                    # TODO: try/except to handle invalid cases
                    self.pred_proba_class = tf.reduce_max(tf.one_hot(target_class, self.n_classes, dtype=tf.float32)
                                                          * self.pred_proba, 1)
                else:
                    logger.exception('Target class %s unknown', target_class)
                    raise ValueError

                self.loss_pred = tf.square(self.pred_proba_class - self.target_proba)

                self.loss_opt = self.loss_pred + self.loss_dist

            # optimizer
            if decay:
                self.learning_rate = tf.train.polynomial_decay(learning_rate_init, self.global_step,
                                                               self.max_iter, 0.0, power=0.5)
            else:
                self.learning_rate = tf.convert_to_tensor(learning_rate_init)
            # self.learning_rate = tf.placeholder(dtype=tf.float32, name='lr')

            # TODO optional argument to change type, learning rate scheduler
            opt = tf.train.AdamOptimizer(self.learning_rate)

            # first compute gradients, then apply them
            self.compute_grads = opt.compute_gradients(self.loss_opt, var_list=[self.cf])
            self.grad_ph = tf.placeholder(shape=data_shape, dtype=tf.float32, name='grad_cf')
            grad_and_var = [(self.grad_ph, self.cf)]
            self.apply_grads = opt.apply_gradients(grad_and_var, global_step=self.global_step)

        # variables to initialize
        self.setup = []  # type: list
        self.setup.append(self.orig.assign(self.assign_orig))
        self.setup.append(self.cf.assign(self.assign_cf))
        self.setup.append(self.target.assign(self.assign_target))
        # self.setup.append(self.lam.assign(self.assign_lam))

        self.tf_init = tf.variables_initializer(var_list=tf.global_variables(scope='cf_search'))

        # tensorboard
        if write_dir is not None:
            self.writer = tf.summary.FileWriter(write_dir, tf.get_default_graph())
            self.writer.add_graph(tf.get_default_graph())

        # return templates
        self.return_dict = {'cf': None, 'all': [], 'orig_class': None, 'orig_prob': None}
        self.instance_dict = dict.fromkeys(['X', 'distance', 'lambda', 'index', 'pred_class', 'prob', 'loss'])

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
        # TODO change init parameters on the fly

        if X.shape[0] != 1:
            logger.warning('Currently only single instance explanations supported (first dim = 1), '
                           'but first dim = %s', X.shape[0])

        # make a prediction
        if self.model:
            Y = self.sess.run(self.predict_tn(tf.convert_to_tensor(X, dtype=tf.float32)))
        else:
            Y = self.predict_fn(X)

        pred_class = Y.argmax()
        pred_prob = Y.max()
        self.return_dict['orig_class'] = pred_class
        self.return_dict['orig_prob'] = pred_prob

        logger.debug('Initial prediction: %s with p=%s', pred_class, Y.max())

        # define the class-specific prediction function
        self.predict_class_fn, t_class = _define_func(self.predict_fn, pred_class, self.target_class)

        # if not self.fitted:
        #    logger.warning('Explain called before fit, explainer will operate in unsupervised mode.')

        # initialize with an instance
        X_init = self._initialize(X)

        # minimize loss iteratively
        self._minimize_loss(X, X_init, Y)

        return_dict = self.return_dict.copy()
        self.return_dict = {'cf': None, 'all': [], 'orig_class': None, 'orig_prob': None}

        return return_dict

    def _prob_condition(self, X_current):
        return np.abs(self.predict_class_fn(X_current) - self.target_proba_arr) <= self.tol

    def _update_exp(self, i, l_step, lam, cf_found, X_current):
        cf_found[0][l_step] += 1  # TODO: batch support

        # populate the return dict
        self.instance_dict['X'] = X_current
        self.instance_dict['distance'] = self.sess.run(self.dist).item()
        self.instance_dict['lambda'] = lam[0]
        self.instance_dict['index'] = l_step * self.max_iter + i

        preds = self.predict_fn(X_current)
        pred_class = preds.argmax()
        prob = preds.max()
        self.instance_dict['pred_class'] = pred_class
        self.instance_dict['prob'] = prob

        self.instance_dict['loss'] = (self.instance_dict['prob'] - self.target_proba_arr[0]) ** 2 + \
                                     self.instance_dict[
                                         'lambda'] * self.instance_dict['distance']

        self.return_dict['cf'] = self.instance_dict.copy()
        self.return_dict['all'].append(self.instance_dict.copy())

        logger.debug('CF found at step %s', l_step * self.max_iter + i)

    def _write_tb(self, lam, lam_lb, lam_ub, cf_found, X_current, **kwargs):
        if self.model:
            scalars_tf = [self.global_step, self.learning_rate, self.dist[0],
                          self.loss_pred[0], self.loss_opt[0], self.pred_proba_class[0]]
            gs, lr, dist, loss_pred, loss_opt, pred = self.sess.run(scalars_tf, feed_dict={self.lam: lam})
        else:
            scalars_tf = [self.global_step, self.learning_rate, self.dist[0],
                          self.loss_opt[0]]
            gs, lr, dist, loss_opt = self.sess.run(scalars_tf, feed_dict={self.lam: lam})
            loss_pred = kwargs['loss_pred']
            pred = kwargs['pred']

        try:
            found = kwargs['found']
            not_found = kwargs['not_found']
        except KeyError:
            found = 0
            not_found = 0

        summary = tf.Summary()
        summary.value.add(tag='lr/global_step', simple_value=gs)
        summary.value.add(tag='lr/lr', simple_value=lr)

        summary.value.add(tag='lambda/lambda', simple_value=lam[0])
        summary.value.add(tag='lambda/l_bound', simple_value=lam_lb[0])
        summary.value.add(tag='lambda/u_bound', simple_value=lam_ub[0])

        summary.value.add(tag='losses/dist', simple_value=dist)
        summary.value.add(tag='losses/loss_pred', simple_value=loss_pred)
        summary.value.add(tag='losses/loss_opt', simple_value=loss_opt)
        summary.value.add(tag='losses/pred_div_dist', simple_value=loss_pred / (lam[0] * dist))

        summary.value.add(tag='Y/pred_proba_class', simple_value=pred)
        summary.value.add(tag='Y/pred_class_fn(X_current)', simple_value=self.predict_class_fn(X_current))
        summary.value.add(tag='Y/n_cf_found', simple_value=cf_found[0].sum())
        summary.value.add(tag='Y/found', simple_value=found)
        summary.value.add(tag='Y/not_found', simple_value=not_found)

        self.writer.add_summary(summary)
        self.writer.flush()

    def _bisect_lambda(self, cf_found, l_step, lam, lam_lb, lam_ub):

        for batch_idx in range(self.batch_size):  # TODO: batch not supported
            if cf_found[batch_idx][l_step] >= 5:  # minimum number of CF instances to warrant increasing lambda
                # want to improve the solution by putting more weight on the distance term
                # by increasing lambda
                lam_lb[batch_idx] = max(lam[batch_idx], lam_lb[batch_idx])
                logger.debug('Lambda bounds: (%s, %s)', lam_lb[batch_idx], lam_ub[batch_idx])
                if lam_ub[batch_idx] < 1e9:
                    lam[batch_idx] = (lam_lb[batch_idx] + lam_ub[batch_idx]) / 2
                    # lam[batch_idx] = lam_lb[batch_idx] + (lam_ub[batch_idx] - lam_lb[batch_idx]) * 0.1
                else:
                    lam[batch_idx] *= 10
                    logger.debug('Changed lambda to %s', lam[batch_idx])

            elif cf_found[batch_idx][l_step] < 5:
                # if not enough solutions found so far, decrease lambda by a factor of 10,
                # otherwise bisect up to the last known successful lambda
                lam_ub[batch_idx] = min(lam_ub[batch_idx], lam[batch_idx])
                logger.debug('Lambda bounds: (%s, %s)', lam_lb[batch_idx], lam_ub[batch_idx])
                if lam_lb[batch_idx] > 0:
                    lam[batch_idx] = (lam_lb[batch_idx] + lam_ub[batch_idx]) / 2
                    # lam[batch_idx] = lam_lb[batch_idx] + (lam_ub[batch_idx] - lam_lb[batch_idx]) * 0.1
                    logger.debug('Changed lambda to %s', lam[batch_idx])
                else:
                    lam[batch_idx] /= 10

        return lam, lam_lb, lam_ub

    def _minimize_loss(self,
                       X: np.ndarray,
                       X_init: np.ndarray,
                       Y: np.ndarray) -> None:

        # keep track of the number of CFs found for each lambda in outer loop
        cf_found = np.zeros((self.batch_size, self.max_lam_steps))

        # set the lower and upper bound for lamda to scale the distance loss term
        lam = np.ones(self.batch_size) * self.lam_init
        lam_lb = np.zeros(self.batch_size)
        lam_ub = np.ones(self.batch_size) * 1e10

        # make a one-hot vector of targets
        Y_ohe = np.zeros(Y.shape)
        np.put(Y_ohe, np.argmax(Y, axis=1), 1)

        # on first run estimate lambda bounds
        n_orders = 10
        n_steps = self.max_iter // n_orders
        lams = np.array([self.lam_init / 10 ** i for i in range(n_orders)])  # exponential decay
        cf_count = np.zeros_like(lams)
        logger.debug('Initial lambda sweep: %s', lams)

        X_current = X_init
        # TODO this whole initial loop should be optional?
        for ix, l_step in enumerate(lams):
            lam = np.ones(self.batch_size) * l_step
            self.sess.run(self.tf_init)
            self.sess.run(self.setup, {self.assign_orig: X,
                                       self.assign_cf: X_current,
                                       self.assign_target: Y_ohe})

            for i in range(n_steps):

                # numerical gradients
                grads_num = np.zeros(self.data_shape)
                if not self.model:
                    pred = self.predict_class_fn(X_current)
                    prediction_grad = num_grad_batch(self.predict_class_fn, X_current, eps=self.eps)

                    # squared difference prediction loss
                    loss_pred = (pred - self.target_proba.eval(session=self.sess)) ** 2
                    grads_num = 2 * (pred - self.target_proba.eval(session=self.sess)) * prediction_grad

                    grads_num = grads_num.reshape(self.data_shape)  # TODO? correct?

                # add values to tensorboard (1st item in batch only) every n steps
                if self.debug and not i % 50:
                    if not self.model:
                        self._write_tb(lam, lam_lb, lam_ub, cf_found, X_current, loss_pred=loss_pred, pred=pred)
                    else:
                        self._write_tb(lam, lam_lb, lam_ub, cf_found, X_current)

                # compute graph gradients
                grads_vars_graph = self.sess.run(self.compute_grads, feed_dict={self.lam: lam})
                grads_graph = [g for g, _ in grads_vars_graph][0]

                # apply gradients
                gradients = grads_graph + grads_num
                self.sess.run(self.apply_grads, feed_dict={self.grad_ph: gradients, self.lam: lam})

                # does the counterfactual condition hold?
                X_current = self.sess.run(self.cf)
                cond = self._prob_condition(X_current).squeeze()
                if cond:
                    cf_count[ix] += 1

        # find the lower bound
        logger.debug('cf_count: %s', cf_count)
        # lb_ix = np.where(cf_count == n_steps)[0][0]  # TODO is n_steps robust?
        lb_ix = np.where(cf_count > 0)[0][1]  # take the second order of magnitude with some CFs as lower-bound
        lam_lb = np.ones(self.batch_size) * lams[lb_ix]

        # find the upper bound
        ub_ix = np.where(cf_count == 0)[0][-1]  # TODO is 0 robust?
        lam_ub = np.ones(self.batch_size) * lams[ub_ix]

        # start the search in the middle
        lam = (lam_lb + lam_ub) / 2

        logger.debug('Found upper and lower bounds: %s, %s', lam_lb[0], lam_ub[0])

        # on subsequent runs bisect lambda within the bounds found initially
        X_current = X_init
        for l_step in range(self.max_lam_steps):
            self.sess.run(self.tf_init)

            # assign variables for the current iteration
            self.sess.run(self.setup, {self.assign_orig: X,
                                       self.assign_cf: X_current,
                                       self.assign_target: Y_ohe})

            found, not_found = 0, 0
            # number of gradient descent steps in each inner loop
            for i in range(self.max_iter):

                # numerical gradients
                grads_num = np.zeros(self.data_shape)
                if not self.model:
                    pred = self.predict_class_fn(X_current)
                    prediction_grad = num_grad_batch(self.predict_class_fn, X_current, eps=self.eps)

                    # squared difference prediction loss
                    loss_pred = (pred - self.target_proba.eval(session=self.sess)) ** 2
                    grads_num = 2 * (pred - self.target_proba.eval(session=self.sess)) * prediction_grad

                    grads_num = grads_num.reshape(self.data_shape)  # TODO? correct?

                # add values to tensorboard (1st item in batch only) every n steps
                if self.debug and not i % 50:
                    if not self.model:
                        self._write_tb(lam, lam_lb, lam_ub, cf_found, X_current, found=found, not_found=not_found,
                                       loss_pred=loss_pred, pred=pred)
                    else:
                        self._write_tb(lam, lam_lb, lam_ub, cf_found, X_current, found=found, not_found=not_found)

                # compute graph gradients
                grads_vars_graph = self.sess.run(self.compute_grads, feed_dict={self.lam: lam})
                grads_graph = [g for g, _ in grads_vars_graph][0]

                # apply gradients
                gradients = grads_graph + grads_num
                self.sess.run(self.apply_grads, feed_dict={self.grad_ph: gradients, self.lam: lam})

                # does the counterfactual condition hold?
                X_current = self.sess.run(self.cf)
                cond = self._prob_condition(X_current)
                if cond:
                    self._update_exp(i, l_step, lam, cf_found, X_current)
                    found += 1
                    not_found = 0
                else:
                    found = 0
                    not_found += 1

                # early stopping criterion - if no solutions or enough solutions found, change lambda
                if found >= 50 or not_found >= 50:
                    break

            # adjust the lambda constant via bisection at the end of the outer loop
            if self.bisect:
                self._bisect_lambda(cf_found, l_step, lam, lam_lb, lam_ub)

            self.return_dict['success'] = True
