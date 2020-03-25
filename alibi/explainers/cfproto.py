# flake8: noqa F841

import copy
import logging
import numpy as np
import sys
import tensorflow as tf
from typing import Callable, Dict, List, Tuple, Union, TYPE_CHECKING, Sequence

from alibi.api.interfaces import Explainer, Explanation, FitMixin
from alibi.api.defaults import DEFAULT_META_CFP, DEFAULT_DATA_CFP
from alibi.confidence import TrustScore
from alibi.utils.discretizer import Discretizer
from alibi.utils.distance import abdm, mvdm, multidim_scaling
from alibi.utils.gradients import perturb
from alibi.utils.mapping import ohe_to_ord_shape, ord_to_num, num_to_ord, ohe_to_ord, ord_to_ohe
from alibi.utils.tf import _check_keras_or_tf, argmax_grad, argmin_grad, one_hot_grad, round_grad

if TYPE_CHECKING:  # pragma: no cover
    import keras

logger = logging.getLogger(__name__)


class CounterFactualProto(Explainer, FitMixin):

    def __init__(self,
                 predict: Union[Callable, tf.keras.Model, 'keras.Model'],
                 shape: tuple,
                 kappa: float = 0.,
                 beta: float = .1,
                 feature_range: tuple = (-1e10, 1e10),
                 gamma: float = 0.,
                 ae_model: Union[tf.keras.Model, 'keras.Model'] = None,
                 enc_model: Union[tf.keras.Model, 'keras.Model'] = None,
                 theta: float = 0.,
                 cat_vars: dict = None,
                 ohe: bool = False,
                 use_kdtree: bool = False,
                 learning_rate_init: float = 1e-2,
                 max_iterations: int = 1000,
                 c_init: float = 10.,
                 c_steps: int = 10,
                 eps: tuple = (1e-3, 1e-3),
                 clip: tuple = (-1000., 1000.),
                 update_num_grad: int = 1,
                 write_dir: str = None,
                 sess: tf.compat.v1.Session = None) -> None:
        """
        Initialize prototypical counterfactual method.

        Parameters
        ----------
        predict
            Keras or TensorFlow model or any other model's prediction function returning class probabilities
        shape
            Shape of input data starting with batch size
        kappa
            Confidence parameter for the attack loss term
        beta
            Regularization constant for L1 loss term
        feature_range
            Tuple with min and max ranges to allow for perturbed instances. Min and max ranges can be floats or
            numpy arrays with dimension (1x nb of features) for feature-wise ranges
        gamma
            Regularization constant for optional auto-encoder loss term
        ae_model
            Optional auto-encoder model used for loss regularization
        enc_model
            Optional encoder model used to guide instance perturbations towards a class prototype
        theta
            Constant for the prototype search loss term
        cat_vars
            Dict with as keys the categorical columns and as values
            the number of categories per categorical variable.
        ohe
            Whether the categorical variables are one-hot encoded (OHE) or not. If not OHE, they are
            assumed to have ordinal encodings.
        use_kdtree
            Whether to use k-d trees for the prototype loss term if no encoder is available
        learning_rate_init
            Initial learning rate of optimizer
        max_iterations
            Maximum number of iterations for finding a counterfactual
        c_init
            Initial value to scale the attack loss term
        c_steps
            Number of iterations to adjust the constant scaling the attack loss term
        eps
            If numerical gradients are used to compute dL/dx = (dL/dp) * (dp/dx), then eps[0] is used to
            calculate dL/dp and eps[1] is used for dp/dx. eps[0] and eps[1] can be a combination of float values and
            numpy arrays. For eps[0], the array dimension should be (1x nb of prediction categories) and for
            eps[1] it should be (1x nb of features)
        clip
            Tuple with min and max clip ranges for both the numerical gradients and the gradients
            obtained from the TensorFlow graph
        update_num_grad
            If numerical gradients are used, they will be updated every update_num_grad iterations
        write_dir
            Directory to write tensorboard files to
        sess
            Optional Tensorflow session that will be used if passed instead of creating or inferring one internally
        """
        super().__init__(meta=copy.deepcopy(DEFAULT_META_CFP))
        params = locals()
        remove = ['self', 'predict', 'ae_model', 'enc_model', 'sess', '__class__']
        for key in remove:
            params.pop(key)
        self.meta['params'].update(params)

        self.predict = predict

        # check whether the model, encoder and auto-encoder are Keras or TF models and get session
        is_model, is_model_keras, model_sess = _check_keras_or_tf(predict)
        is_ae, is_ae_keras, ae_sess = _check_keras_or_tf(ae_model)
        is_enc, is_enc_keras, enc_sess = _check_keras_or_tf(enc_model)
        self.meta['params'].update(is_model=is_model, is_model_keras=is_model_keras)
        self.meta['params'].update(is_ae=is_ae, is_ae_keras=is_ae_keras)
        self.meta['params'].update(is_enc=is_enc, is_enc_keras=is_enc_keras)

        # TODO: check ae, enc and model are all compatible

        # if session provided, use it
        if isinstance(sess, tf.compat.v1.Session):
            self.sess = sess
        else:
            self.sess = model_sess

        if is_model:  # Keras or TF model
            self.model = True
            self.classes = self.predict.predict(np.zeros(shape)).shape[1]  # type: ignore
        else:  # black-box model
            self.model = False
            self.classes = self.predict(np.zeros(shape)).shape[1]

        if is_enc:
            self.enc_model = True
        else:
            self.enc_model = False

        if is_ae:
            self.ae_model = True
        else:
            self.ae_model = False

        if use_kdtree and self.enc_model:
            logger.warning('Both an encoder and k-d trees enabled. Using the encoder for the prototype loss term.')

        if use_kdtree or self.enc_model:
            self.enc_or_kdtree = True
        else:
            self.enc_or_kdtree = False
        self.meta['params'].update(enc_or_kdtree=self.enc_or_kdtree)

        if cat_vars:
            self.is_cat = True
        else:
            self.is_cat = False
        self.meta['params'].update(is_cat=self.is_cat)

        self.shape = shape
        self.kappa = kappa
        self.beta = beta
        self.gamma = gamma
        self.theta = theta
        self.ae = ae_model
        self.enc = enc_model
        self.cat_vars = cat_vars
        self.ohe = ohe
        self.use_kdtree = use_kdtree
        self.batch_size = shape[0]
        self.max_iterations = max_iterations
        self.c_init = c_init
        self.c_steps = c_steps
        self.feature_range = feature_range
        self.update_num_grad = update_num_grad
        self.eps = eps
        self.clip = clip
        self.write_dir = write_dir

        # compute dimensionality after conversion from OHE to ordinal encoding
        shape = ohe_to_ord_shape(shape, cat_vars=cat_vars, is_ohe=self.ohe)

        if self.is_cat:

            # define ragged tensor for mapping from categorical to numerical values
            self.map_cat_to_num = tf.ragged.constant([np.zeros(v) for _, v in self.cat_vars.items()])

            # define placeholder for mapping which can be fed after the fit step
            max_key = max(cat_vars, key=cat_vars.get)
            self.max_cat = cat_vars[max_key]
            cat_keys = list(cat_vars.keys())
            n_cat = len(cat_keys)
            self.assign_map = tf.placeholder(tf.float32, (n_cat, self.max_cat), name='assign_map')
            self.map_var = tf.Variable(np.zeros((n_cat, self.max_cat)), dtype=tf.float32, name='map_var')

            # update ragged mapping tensor
            lengths = [v for k, v in cat_vars.items()]
            map_cat_to_num_val = tf.RaggedTensor.from_tensor(self.map_var, lengths=list(lengths))
            self.map_cat_to_num = tf.ragged.map_flat_values(tf.add, self.map_cat_to_num, map_cat_to_num_val)

            # store categorical columns assuming ordinal encoding
            # used for the mapping between numerical values and categories
            if self.ohe:
                cat_vars_ord = {}
                c, k = 0, 0
                while c < self.shape[-1]:
                    if c in cat_keys:
                        v = cat_vars[c]
                        cat_vars_ord[k] = v
                        k += 1
                        c += v
                        continue
                    k += 1
                    c += 1
                cat_keys_ord = list(cat_vars_ord.keys())
                cat_cols_ord = tf.constant(cat_keys_ord, name='cat_keys_ord')
            else:
                cat_cols_ord = tf.constant(cat_keys, name='cat_keys_ord')

            # mapping from numerical values to categories and vice versa
            # supports mapping to and from both ordinal encoding and OHE
            def is_eq(col, cat_cols):
                """
                Check if a column represents a categorical variable.

                Parameters
                ----------
                col
                    Column index to check.
                cat_cols
                    Indices of categorical variables.

                Returns
                -------
                Boolean whether the column is a categorical variable.
                """
                eq = tf.math.equal(col, cat_cols)
                eq_any = tf.reduce_any(eq)
                return tf.equal(eq_any, tf.constant(True))

            def cond_loop(icol, iohe, icat, adv_to_map, adv_map, map_cols):
                """
                Condition for while loop, only iterate over columns of instance.

                Parameters
                ----------
                icol
                    Iteration over columns of instance.
                adv_to_map
                    Instance that needs to be mapped from categories to numerical values or vice versa.
                map_cols
                    Number of columns in instance to be mapped.

                Returns
                -------
                Boolean whether condition is met.
                """
                return tf.less(icol, tf.minimum(map_cols, tf.shape(adv_to_map)[1]))

            def body_ord_to_num(icol, iohe, icat, adv_to_map, adv_map, map_cols):
                """
                Body executed in while loop when mapping ordinal categories to numerical values.

                Parameters
                ----------
                icol
                    Iteration over columns of instance.
                icat
                    Iteration over categorical variables.
                adv_to_map
                    Instance that needs to be mapped from categories to numerical values or vice versa.
                adv_map
                    Mapped instance from categories to numerical values.
                """
                # check if icol is a categorical variable
                eq_any_true = is_eq(icol, cat_cols_ord)

                # map category to its numerical value
                def true_fn():
                    try:
                        return self.map_cat_to_num[icat][adv_to_map[0, icol]]
                    except:  # the value of adv_to_map[0, icol] is a float
                        # TODO: add error type
                        idx = round_grad(adv_to_map[0, icol])
                        return self.map_cat_to_num[icat][idx]

                def false_fn():
                    return adv_to_map[0, icol]

                # write column to array
                adv_map_col = tf.cond(eq_any_true, true_fn, false_fn)
                adv_map = adv_map.write(icol, adv_map_col)

                # increment
                icol = tf.add(icol, 1)
                icat = tf.cond(eq_any_true, lambda: tf.add(icat, 1), lambda: icat)  # if categorical variable

                return [icol, iohe, icat, adv_to_map, adv_map, map_cols]

            def body_num_to_ord(icol, iohe, icat, adv_to_map, adv_map, map_cols):
                """
                Body executed in while loop when mapping numerical values to ordinal categories.

                Parameters
                ----------
                icol
                    Iteration over columns of instance.
                icat
                    Iteration over categorical variables.
                adv_to_map
                    Instance that needs to be mapped from categories to numerical values or vice versa.
                adv_map
                    Mapped instance from numerical values to categories.
                """
                # check if icol is a categorical variable
                eq_any_true = is_eq(icol, cat_cols_ord)

                # map numerical value to category
                def true_fn():
                    return argmin_grad(adv_to_map[0, icol], self.map_cat_to_num[icat])

                def false_fn():
                    return adv_to_map[0, icol]

                # write column to array
                adv_map_col = tf.cond(eq_any_true, true_fn, false_fn)
                adv_map = adv_map.write(icol, adv_map_col)

                # increment
                icol = tf.add(icol, 1)
                icat = tf.cond(eq_any_true, lambda: tf.add(icat, 1), lambda: icat)  # if categorical variable

                return [icol, iohe, icat, adv_to_map, adv_map, map_cols]

            def body_ohe_to_num(icol, iohe, icat, adv_to_map, adv_map, map_cols):
                """
                Body executed in while loop when mapping OHE categories to numerical values.

                Parameters
                ----------
                icol
                    Iteration over columns of instance.
                iohe
                    Iteration over OHE columns of instance.
                icat
                    Iteration over categorical variables.
                adv_to_map
                    Instance that needs to be mapped from categories to numerical values or vice versa.
                adv_map
                    Mapped instance from categories to numerical values.
                """
                # check if icol is a categorical variable
                eq_any_true = is_eq(icol, cat_cols_ord)

                # nb of categories
                v = tf.cond(eq_any_true,
                            lambda: tf.shape(self.map_cat_to_num[icat])[0],
                            lambda: tf.constant(1))

                # map category to its numerical value
                def true_fn():
                    adv_ord = argmax_grad(adv_to_map[0, iohe:iohe + v])  # map to ord
                    return self.map_cat_to_num[icat][adv_ord]  # map to num

                def false_fn():
                    return adv_to_map[0, iohe]

                # write column to array
                adv_map_col = tf.cond(eq_any_true, true_fn, false_fn)
                adv_map = adv_map.write(icol, adv_map_col)

                # increment
                icol = tf.add(icol, 1)
                iohe = tf.add(iohe, v)
                icat = tf.cond(eq_any_true, lambda: tf.add(icat, 1), lambda: icat)  # if categorical variable

                return [icol, iohe, icat, adv_to_map, adv_map, map_cols]

            def body_num_to_ohe(icol, iohe, icat, adv_to_map, adv_map, map_cols):
                """
                Body executed in while loop when mapping numerical values to OHE categories.

                Parameters
                ----------
                icol
                    Iteration over columns of instance.
                iohe
                    Iteration over OHE columns of instance.
                icat
                    Iteration over categorical variables.
                adv_to_map
                    Instance that needs to be mapped from categories to numerical values or vice versa.
                adv_map
                    Mapped instance from numerical values to categories.
                """
                # check if icol is a categorical variable
                eq_any_true = is_eq(icol, cat_cols_ord)

                def true_fn():
                    cat_ord = argmin_grad(adv_to_map[0, icol], self.map_cat_to_num[icat])  # map to ord
                    cat_ohe = one_hot_grad(cat_ord, self.map_cat_to_num[icat])  # map to OHE
                    return cat_ohe

                def false_fn():
                    return tf.reshape(adv_to_map[0, icol], (1,))

                # get OHE mapped columns
                adv_map_col = tf.cond(eq_any_true, true_fn, false_fn)

                def while_ohe(i_ohe, i_ohe_cat, adv_ohe):
                    return tf.less(i_ohe_cat, tf.shape(adv_map_col)[0])

                def body_ohe(i_ohe, i_ohe_cat, adv_ohe):
                    i_write = tf.add(i_ohe_cat, i_ohe)
                    adv_ohe = adv_ohe.write(i_write, adv_map_col[i_ohe_cat])
                    i_ohe_cat = tf.add(i_ohe_cat, 1)
                    return [i_ohe, i_ohe_cat, adv_ohe]

                # write OHE columns to array
                iohe, iohecat, adv_map = tf.while_loop(while_ohe, body_ohe, [iohe, tf.constant(0), adv_map])

                # increment
                icol = tf.add(icol, 1)
                iohe = tf.add(iohe, iohecat)
                icat = tf.cond(eq_any_true, lambda: tf.add(icat, 1), lambda: icat)  # if categorical variable

                return [icol, iohe, icat, adv_to_map, adv_map, map_cols]

            def apply_map(adv_to_map, to_num):
                """
                Apply mapping from numerical to ordinal or OHE categorical variables
                or vice versa for an instance.

                Parameters
                ----------
                adv_to_map
                    Instance to map.
                to_num
                    Map from categorical to numerical values if True, vice versa if False.

                Returns
                -------
                Mapped instance.
                """
                icol = tf.constant(0)
                iohe = tf.constant(0)
                icat = tf.constant(0)

                if self.ohe:
                    body_to_num, body_to_cat = body_ohe_to_num, body_num_to_ohe
                else:
                    body_to_num, body_to_cat = body_ord_to_num, body_num_to_ord

                if self.ohe and not to_num:
                    shape_adv_map = self.shape
                else:
                    shape_adv_map = shape

                adv_map = tf.TensorArray(dtype=tf.float32, size=shape_adv_map[1])
                loop_vars = (icol, iohe, icat, adv_to_map, adv_map, shape_adv_map[1])

                if to_num:  # map from categorical to numerical values
                    _, _, _, _, adv_map, _ = tf.while_loop(cond_loop, body_to_num, loop_vars,
                                                           parallel_iterations=1, back_prop=True)
                else:  # map from numerical to categorical values
                    _, _, _, _, adv_map, _ = tf.while_loop(cond_loop, body_to_cat, loop_vars,
                                                           parallel_iterations=1, back_prop=True)
                adv_map_stack = tf.reshape(adv_map.stack(), shape_adv_map)
                return adv_map_stack

        # define tf variables for original and perturbed instances, and target labels
        self.orig = tf.Variable(np.zeros(shape), dtype=tf.float32, name='orig')
        self.adv = tf.Variable(np.zeros(shape), dtype=tf.float32, name='adv')
        self.adv_s = tf.Variable(np.zeros(shape), dtype=tf.float32, name='adv_s')
        self.target = tf.Variable(np.zeros((self.batch_size, self.classes)), dtype=tf.float32, name='target')

        # variable for target class proto
        if self.enc_model:
            self.shape_enc = self.enc.predict(np.zeros(self.shape)).shape
        else:
            self.shape_enc = shape

        self.target_proto = tf.Variable(np.zeros(self.shape_enc), dtype=tf.float32, name='target_proto')

        # define tf variable for constant used in FISTA optimization
        self.const = tf.Variable(np.zeros(self.batch_size), dtype=tf.float32, name='const')
        self.global_step = tf.Variable(0.0, trainable=False, name='global_step')

        # define placeholders that will be assigned to relevant variables
        self.assign_orig = tf.placeholder(tf.float32, shape, name='assign_orig')
        self.assign_adv = tf.placeholder(tf.float32, shape, name='assign_adv')
        self.assign_adv_s = tf.placeholder(tf.float32, shape, name='assign_adv_s')
        self.assign_target = tf.placeholder(tf.float32, (self.batch_size, self.classes), name='assign_target')
        self.assign_const = tf.placeholder(tf.float32, [self.batch_size], name='assign_const')
        self.assign_target_proto = tf.placeholder(tf.float32, self.shape_enc, name='assign_target_proto')

        # define conditions and values for element-wise shrinkage thresholding
        with tf.name_scope('shrinkage_thresholding') as scope:
            cond = [tf.cast(tf.greater(tf.subtract(self.adv_s, self.orig), self.beta), tf.float32),
                    tf.cast(tf.less_equal(tf.abs(tf.subtract(self.adv_s, self.orig)), self.beta), tf.float32),
                    tf.cast(tf.less(tf.subtract(self.adv_s, self.orig), tf.negative(self.beta)), tf.float32)]
            upper = tf.minimum(tf.subtract(self.adv_s, self.beta), tf.cast(feature_range[1], tf.float32))
            lower = tf.maximum(tf.add(self.adv_s, self.beta), tf.cast(feature_range[0], tf.float32))
            self.assign_adv = tf.multiply(cond[0], upper) + tf.multiply(cond[1], self.orig) + tf.multiply(cond[2],
                                                                                                          lower)

        # perturbation update and vector projection on correct feature range set
        with tf.name_scope('perturbation_y') as scope:
            self.zt = tf.divide(self.global_step, self.global_step + tf.cast(3, tf.float32))
            self.assign_adv_s = self.assign_adv + tf.multiply(self.zt, self.assign_adv - self.adv)
            # map to feature space
            self.assign_adv_s = tf.minimum(self.assign_adv_s, tf.cast(feature_range[1], tf.float32))
            self.assign_adv_s = tf.maximum(self.assign_adv_s, tf.cast(feature_range[0], tf.float32))

        # assign counterfactual of step k+1 to k
        with tf.name_scope('update_adv') as scope:
            self.adv_updater = tf.assign(self.adv, self.assign_adv)
            self.adv_updater_s = tf.assign(self.adv_s, self.assign_adv_s)

        # from perturbed instance, derive deviation delta
        with tf.name_scope('update_delta') as scope:
            self.delta = self.orig - self.adv
            self.delta_s = self.orig - self.adv_s

        # define L1 and L2 loss terms; L1+L2 is later used as an optimization constraint for FISTA
        ax_sum = list(np.arange(1, len(shape)))
        with tf.name_scope('loss_l1_l2') as scope:
            self.l2 = tf.reduce_sum(tf.square(self.delta), axis=ax_sum)
            self.l2_s = tf.reduce_sum(tf.square(self.delta_s), axis=ax_sum)
            self.l1 = tf.reduce_sum(tf.abs(self.delta), axis=ax_sum)
            self.l1_s = tf.reduce_sum(tf.abs(self.delta_s), axis=ax_sum)
            self.l1_l2 = self.l2 + tf.multiply(self.l1, self.beta)
            self.l1_l2_s = self.l2_s + tf.multiply(self.l1_s, self.beta)

            # sum losses
            self.loss_l1 = tf.reduce_sum(self.l1)
            self.loss_l1_s = tf.reduce_sum(self.l1_s)
            self.loss_l2 = tf.reduce_sum(self.l2)
            self.loss_l2_s = tf.reduce_sum(self.l2_s)

        if self.is_cat:  # map adv and adv_s to categories
            self.adv_cat = apply_map(self.adv, to_num=False)
            self.adv_cat_s = apply_map(self.adv_s, to_num=False)
        else:
            self.adv_cat = self.adv
            self.adv_cat_s = self.adv_s

        with tf.name_scope('loss_ae') as scope:
            # gamma * AE loss
            if self.ae_model:
                # run autoencoder
                self.adv_ae = self.ae(self.adv_cat)
                self.adv_ae_s = self.ae(self.adv_cat_s)
                if self.is_cat:  # map output autoencoder back to numerical values
                    self.adv_ae = apply_map(self.adv_ae, to_num=True)
                    self.adv_ae_s = apply_map(self.adv_ae_s, to_num=True)
                # compute loss
                self.loss_ae = self.gamma * tf.square(tf.norm(self.adv_ae - self.adv))
                self.loss_ae_s = self.gamma * tf.square(tf.norm(self.adv_ae_s - self.adv_s))
            else:  # no auto-encoder available
                self.loss_ae = tf.constant(0.)
                self.loss_ae_s = tf.constant(0.)

        with tf.name_scope('loss_attack') as scope:
            if not self.model:
                self.loss_attack = tf.placeholder(tf.float32)
            elif self.c_init == 0. and self.c_steps == 1:  # prediction loss term not used
                # make predictions on perturbed instance
                self.pred_proba = self.predict(self.adv_cat)
                self.pred_proba_s = self.predict(self.adv_cat_s)

                self.loss_attack = tf.constant(0.)
                self.loss_attack_s = tf.constant(0.)
            else:
                # make predictions on perturbed instance
                self.pred_proba = self.predict(self.adv_cat)
                self.pred_proba_s = self.predict(self.adv_cat_s)

                # probability of target label prediction
                self.target_proba = tf.reduce_sum(self.target * self.pred_proba, 1)
                target_proba_s = tf.reduce_sum(self.target * self.pred_proba_s, 1)

                # max probability of non target label prediction
                self.nontarget_proba_max = tf.reduce_max((1 - self.target) * self.pred_proba - (self.target * 10000), 1)
                nontarget_proba_max_s = tf.reduce_max((1 - self.target) * self.pred_proba_s - (self.target * 10000), 1)

                # loss term f(x,d)
                loss_attack = tf.maximum(0.0, -self.nontarget_proba_max + self.target_proba + self.kappa)
                loss_attack_s = tf.maximum(0.0, -nontarget_proba_max_s + target_proba_s + self.kappa)

                # c * f(x,d)
                self.loss_attack = tf.reduce_sum(self.const * loss_attack)
                self.loss_attack_s = tf.reduce_sum(self.const * loss_attack_s)

        with tf.name_scope('loss_prototype') as scope:
            if self.enc_model:
                self.loss_proto = self.theta * tf.square(tf.norm(self.enc(self.adv_cat) - self.target_proto))
                self.loss_proto_s = self.theta * tf.square(tf.norm(self.enc(self.adv_cat_s) - self.target_proto))
            elif self.use_kdtree:
                self.loss_proto = self.theta * tf.square(tf.norm(self.adv - self.target_proto))
                self.loss_proto_s = self.theta * tf.square(tf.norm(self.adv_s - self.target_proto))
            else:  # no encoder available and no k-d trees used
                self.loss_proto = tf.constant(0.)
                self.loss_proto_s = tf.constant(0.)

        with tf.name_scope('loss_combined') as scope:
            # no need for L1 term in loss to optimize when using FISTA
            if self.model:
                self.loss_opt = self.loss_attack_s + self.loss_l2_s + self.loss_ae_s + self.loss_proto_s
            else:  # separate numerical computation of loss attack gradient
                self.loss_opt = self.loss_l2_s + self.loss_ae_s + self.loss_proto_s

            # add L1 term to overall loss; this is not the loss that will be directly optimized
            self.loss_total = (self.loss_attack + self.loss_l2 + self.loss_ae +
                               tf.multiply(self.beta, self.loss_l1) + self.loss_proto)

        with tf.name_scope('training') as scope:
            self.learning_rate = tf.train.polynomial_decay(learning_rate_init, self.global_step,
                                                           self.max_iterations, 0, power=0.5)
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            start_vars = set(x.name for x in tf.global_variables())

            # first compute, then apply grads
            self.compute_grads = optimizer.compute_gradients(self.loss_opt, var_list=[self.adv_s])
            self.grad_ph = tf.placeholder(tf.float32, name='grad_adv_s')
            var = [tvar for tvar in tf.trainable_variables() if tvar.name.startswith('adv_s')][-1]  # get the last in
            # case explainer is re-initialized and a new graph is created
            grad_and_var = [(self.grad_ph, var)]
            self.apply_grads = optimizer.apply_gradients(grad_and_var, global_step=self.global_step)
            end_vars = tf.global_variables()
            new_vars = [x for x in end_vars if x.name not in start_vars]

        # variables to initialize
        self.setup = []  # type: list
        self.setup.append(self.orig.assign(self.assign_orig))
        self.setup.append(self.target.assign(self.assign_target))
        self.setup.append(self.const.assign(self.assign_const))
        self.setup.append(self.adv.assign(self.assign_adv))
        self.setup.append(self.adv_s.assign(self.assign_adv_s))
        self.setup.append(self.target_proto.assign(self.assign_target_proto))
        if self.is_cat:
            self.setup.append(self.map_var.assign(self.assign_map))

        self.init = tf.variables_initializer(var_list=[self.global_step] + [self.adv_s] + [self.adv] + new_vars)

        if self.write_dir is not None:
            self.writer = tf.summary.FileWriter(write_dir, tf.get_default_graph())
            self.writer.add_graph(tf.get_default_graph())
        else:
            self.writer = None

    def fit(self, train_data: np.ndarray, trustscore_kwargs: dict = None, d_type: str = 'abdm',
            w: float = None, disc_perc: Sequence[Union[int, float]] = (25, 50, 75), standardize_cat_vars: bool = False,
            smooth: float = 1., center: bool = True, update_feature_range: bool = True) -> "CounterFactualProto":
        """
        Get prototypes for each class using the encoder or k-d trees.
        The prototypes are used for the encoder loss term or to calculate the optional trust scores.

        Parameters
        ----------
        train_data
            Representative sample from the training data.
        trustscore_kwargs
            Optional arguments to initialize the trust scores method.
        d_type
            Pairwise distance metric used for categorical variables. Currently, 'abdm', 'mvdm' and 'abdm-mvdm'
            are supported. 'abdm' infers context from the other variables while 'mvdm' uses the model predictions.
            'abdm-mvdm' is a weighted combination of the two metrics.
        w
            Weight on 'abdm' (between 0. and 1.) distance if d_type equals 'abdm-mvdm'.
        disc_perc
            List with percentiles used in binning of numerical features used for the 'abdm'
            and 'abdm-mvdm' pairwise distance measures.
        standardize_cat_vars
            Standardize numerical values of categorical variables if True.
        smooth
            Smoothing exponent between 0 and 1 for the distances. Lower values of l will smooth the difference in
            distance metric between different features.
        center
            Whether to center the scaled distance measures. If False, the min distance for each feature
            except for the feature with the highest raw max distance will be the lower bound of the
            feature range, but the upper bound will be below the max feature range.
        update_feature_range
            Update feature range with scaled values.
        """
        # get params for storage in meta
        params = locals()
        remove = ['self', 'train_data']
        for key in remove:
            params.pop(key)
        # update metadata
        self.meta['params'].update(params)

        if self.model:
            preds = np.argmax(self.predict.predict(train_data), axis=1)  # type: ignore
        else:
            preds = np.argmax(self.predict(train_data), axis=1)

        self.cat_vars_ord = None
        if self.is_cat:  # compute distance metrics for categorical variables

            if self.ohe:  # convert OHE to ordinal encoding
                train_data_ord, self.cat_vars_ord = ohe_to_ord(train_data, self.cat_vars)
            else:
                train_data_ord, self.cat_vars_ord = train_data, self.cat_vars

            # bin numerical features to compute the pairwise distance matrices
            cat_keys = list(self.cat_vars_ord.keys())
            n_ord = train_data_ord.shape[1]
            numerical_feats = [feat for feat in range(n_ord) if feat not in cat_keys]
            if d_type in ['abdm', 'abdm-mvdm'] and len(cat_keys) != n_ord:
                fnames = [str(_) for _ in range(n_ord)]
                disc = Discretizer(train_data_ord, numerical_feats, fnames, percentiles=disc_perc)
                train_data_bin = disc.discretize(train_data_ord)
                cat_vars_bin = {k: len(disc.feature_intervals[k]) for k in range(n_ord) if k not in cat_keys}
            else:
                train_data_bin = train_data_ord
                cat_vars_bin = {}

            if d_type not in ['abdm', 'mvdm', 'abdm-mvdm']:
                raise ValueError('d_type needs to be "abdm", "mvdm" or "abdm-mvdm". '
                                 '{} is not supported.'.format(d_type))

            # pairwise distances for categorical variables
            if d_type == 'abdm':
                d_pair = abdm(train_data_bin, self.cat_vars_ord, cat_vars_bin)
            elif d_type == 'mvdm':
                d_pair = mvdm(train_data_ord, preds, self.cat_vars_ord, alpha=1)

            # combined distance measure
            if d_type == 'abdm-mvdm':
                # pairwise distances
                d_abdm = abdm(train_data_bin, self.cat_vars_ord, cat_vars_bin)
                d_mvdm = mvdm(train_data_ord, preds, self.cat_vars_ord, alpha=1)

                # multidim scaled distances
                d_abs_abdm, _ = multidim_scaling(d_abdm, n_components=2, use_metric=True,
                                                 feature_range=self.feature_range,
                                                 standardize_cat_vars=standardize_cat_vars,
                                                 smooth=smooth, center=center,
                                                 update_feature_range=False)

                d_abs_mvdm, _ = multidim_scaling(d_mvdm, n_components=2, use_metric=True,
                                                 feature_range=self.feature_range,
                                                 standardize_cat_vars=standardize_cat_vars,
                                                 smooth=smooth, center=center,
                                                 update_feature_range=False)

                # combine abdm and mvdm
                self.d_abs = {}  # type: Dict
                new_feature_range = tuple([f.copy() for f in self.feature_range])
                for k, v in d_abs_abdm.items():
                    self.d_abs[k] = v * w + d_abs_mvdm[k] * (1 - w)
                    if center:  # center the numerical feature values
                        self.d_abs[k] -= .5 * (self.d_abs[k].max() + self.d_abs[k].min())
                    if update_feature_range:
                        new_feature_range[0][0, k] = self.d_abs[k].min()
                        new_feature_range[1][0, k] = self.d_abs[k].max()
                if update_feature_range:  # assign updated feature range
                    self.feature_range = new_feature_range
            else:  # apply multidimensional scaling for the abdm or mvdm distances
                self.d_abs, self.feature_range = multidim_scaling(d_pair, n_components=2, use_metric=True,
                                                                  feature_range=self.feature_range,
                                                                  standardize_cat_vars=standardize_cat_vars,
                                                                  smooth=smooth, center=center,
                                                                  update_feature_range=update_feature_range)

            # create array used for ragged tensor placeholder
            self.d_abs_ragged = []  # type: List
            for _, v in self.d_abs.items():
                n_pad = self.max_cat - len(v)
                v_pad = np.pad(v, (0, n_pad), 'constant')
                self.d_abs_ragged.append(v_pad)
            self.d_abs_ragged = np.array(self.d_abs_ragged)

        if self.enc_model:
            enc_data = self.enc.predict(train_data)
            self.class_proto = {}  # type: dict
            self.class_enc = {}  # type: dict
            for i in range(self.classes):
                idx = np.where(preds == i)[0]
                self.class_proto[i] = np.expand_dims(np.mean(enc_data[idx], axis=0), axis=0)
                self.class_enc[i] = enc_data[idx]
        elif self.use_kdtree:
            logger.warning('No encoder specified. Using k-d trees to represent class prototypes.')
            if trustscore_kwargs is not None:
                ts = TrustScore(**trustscore_kwargs)
            else:
                ts = TrustScore()
            if self.is_cat:  # map categorical to numerical data
                train_data = ord_to_num(train_data_ord, self.d_abs)
            ts.fit(train_data, preds, classes=self.classes)
            self.kdtrees = ts.kdtrees
            self.X_by_class = ts.X_kdtree

        return self

    def loss_fn(self, pred_proba: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Compute the attack loss.

        Parameters
        ----------
        pred_proba
            Prediction probabilities of an instance
        Y
            One-hot representation of instance labels

        Returns
        -------
        Loss of the attack.
        """
        # probability of target label prediction
        target_proba = np.sum(pred_proba * Y)
        # max probability of non target label prediction
        nontarget_proba_max = np.max((1 - Y) * pred_proba - 10000 * Y)
        # loss term f(x,d)
        loss = np.maximum(0., - nontarget_proba_max + target_proba + self.kappa)
        # c * f(x,d)
        loss_attack = np.sum(self.const.eval(session=self.sess) * loss)
        return loss_attack

    def get_gradients(self, X: np.ndarray, Y: np.ndarray, grads_shape: tuple,
                      cat_vars_ord: dict = None) -> np.ndarray:
        """
        Compute numerical gradients of the attack loss term:
        dL/dx = (dL/dP)*(dP/dx) with L = loss_attack_s; P = predict; x = adv_s

        Parameters
        ----------
        X
            Instance around which gradient is evaluated
        Y
            One-hot representation of instance labels
        grads_shape
            Shape of gradients.
        cat_vars_ord
            Dict with as keys the categorical columns and as values
            the number of categories per categorical variable.

        Returns
        -------
        Array with gradients.
        """
        # map back to categories to make predictions
        if self.is_cat:
            X_pred = num_to_ord(X, self.d_abs)
            if self.ohe:
                X_pred = ord_to_ohe(X_pred, cat_vars_ord)[0]
        else:
            X_pred = X

        # N = gradient batch size; F = nb of features; P = nb of prediction classes; B = instance batch size
        # dL/dP -> BxP
        preds = self.predict(X_pred)  # NxP
        preds_pert_pos, preds_pert_neg = perturb(preds, self.eps[0], proba=True)  # (N*P)xP

        def f(preds_pert):
            return np.sum(Y * preds_pert, axis=1)

        def g(preds_pert):
            return np.max((1 - Y) * preds_pert, axis=1)

        # find instances where the gradient is 0
        idx_nograd = np.where(f(preds) - g(preds) <= - self.kappa)[0]
        if len(idx_nograd) == X.shape[0]:
            return np.zeros(X.shape)

        dl_df = f(preds_pert_pos) - f(preds_pert_neg)  # N*P
        dl_dg = g(preds_pert_pos) - g(preds_pert_neg)  # N*P
        dl_dp = dl_df - dl_dg  # N*P
        dl_dp = np.reshape(dl_dp, (X.shape[0], -1)) / (2 * self.eps[0])  # NxP

        # dP/dx -> PxF
        X_pert_pos, X_pert_neg = perturb(X, self.eps[1], proba=False)  # (N*F)x(shape of X[0])
        X_pert = np.concatenate([X_pert_pos, X_pert_neg], axis=0)
        if self.is_cat:
            X_pert = num_to_ord(X_pert, self.d_abs)
        if self.ohe:
            X_pert = ord_to_ohe(X_pert, cat_vars_ord)[0]
        preds_concat = self.predict(X_pert)
        n_pert = X_pert_pos.shape[0]
        dp_dx = preds_concat[:n_pert] - preds_concat[n_pert:]  # (N*F)*P
        dp_dx = np.reshape(np.reshape(dp_dx, (X.shape[0], -1)),
                           (X.shape[0], preds.shape[1], -1), order='F') / (2 * self.eps[1])  # NxPxF

        # dL/dx -> Bx(shape of X[0])
        grads = np.einsum('ij,ijk->ik', dl_dp, dp_dx)  # NxF
        # set instances where gradient is 0 to 0
        if len(idx_nograd) > 0:
            grads[idx_nograd] = np.zeros(grads.shape[1:])
        grads = np.mean(grads, axis=0)  # B*F
        grads = np.reshape(grads, (self.batch_size,) + grads_shape)  # B*(shape of X[0])
        return grads

    def score(self, X: np.ndarray, adv_class: int, orig_class: int, eps: float = 1e-10) -> float:
        """
        Parameters
        ----------
        X
            Instance to encode and calculate distance metrics for
        adv_class
            Predicted class on the perturbed instance
        orig_class
            Predicted class on the original instance
        eps
            Small number to avoid dividing by 0

        Returns
        -------
        Ratio between the distance to the prototype of the predicted class for the original instance and
        the prototype of the predicted class for the perturbed instance.
        """
        if self.enc_model:
            if self.is_cat:
                X = num_to_ord(X, self.d_abs)
            if self.ohe:
                X = ord_to_ohe(X, self.cat_vars_ord)
            X_enc = self.enc.predict(X)
            adv_proto = self.class_proto[adv_class]
            orig_proto = self.class_proto[orig_class]
            dist_adv = np.linalg.norm(X_enc - adv_proto)
            dist_orig = np.linalg.norm(X_enc - orig_proto)
        elif self.use_kdtree:
            dist_adv = self.kdtrees[adv_class].query(X, k=1)[0]
            dist_orig = self.kdtrees[orig_class].query(X, k=1)[0]
        else:
            logger.warning('Need either an encoder or the k-d trees enabled to compute distance scores.')
        return dist_orig / (dist_adv + eps)

    def attack(self, X: np.ndarray, Y: np.ndarray, target_class: list = None, k: int = None, k_type: str = 'mean',
               threshold: float = 0., verbose: bool = False, print_every: int = 100, log_every: int = 100) \
            -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Find a counterfactual (CF) for instance X using a fast iterative shrinkage-thresholding algorithm (FISTA).

        Parameters
        ----------
        X
            Instance to attack
        Y
            Labels for X as one-hot-encoding
        target_class
            List with target classes used to find closest prototype. If None, the nearest prototype
            except for the predict class on the instance is used.
        k
            Number of nearest instances used to define the prototype for a class. Defaults to using all
            instances belonging to the class if an encoder is used and to 1 for k-d trees.
        k_type
            Use either the average encoding of the k nearest instances in a class (k_type='mean') or
            the k-nearest encoding in the class (k_type='point') to define the prototype of that class.
            Only relevant if an encoder is used to define the prototypes.
        threshold
            Threshold level for the ratio between the distance of the counterfactual to the prototype of the
            predicted class for the original instance over the distance to the prototype of the predicted class
            for the counterfactual. If the trust score is below the threshold, the proposed counterfactual does
            not meet the requirements.
        verbose
            Print intermediate results of optimization if True
        print_every
            Print frequency if verbose is True
        log_every
            Tensorboard log frequency if write directory is specified

        Returns
        -------
        Overall best attack and gradients for that attack.
        """

        # make sure nb of instances in X equals batch size
        assert self.batch_size == X.shape[0]

        def compare(x: Union[float, int, np.ndarray], y: int) -> bool:
            """
            Compare predictions with target labels and return whether counterfactual conditions hold.

            Parameters
            ----------
            x
                Predicted class probabilities or labels
            y
                Target or predicted labels

            Returns
            -------
            Bool whether counterfactual conditions hold.
            """
            if not isinstance(x, (float, int, np.int64)):
                x = np.copy(x)
                x[y] += self.kappa
                x = np.argmax(x)
            return x != y

        # define target classes for prototype if not specified yet
        if target_class is None:
            target_class = list(range(self.classes))
            target_class.remove(np.argmax(Y, axis=1))
            if verbose:
                print('Predicted class: {}'.format(np.argmax(Y, axis=1)))
                print('Target classes: {}'.format(target_class))

        if self.is_cat and self.ohe:  # map categorical to numerical data
            X_ord = ohe_to_ord(X, self.cat_vars)[0]
            X_num = ord_to_num(X_ord, self.d_abs)
        elif self.is_cat:
            X_num = ord_to_num(X, self.d_abs)
        else:
            X_num = X

        # find closest prototype in the target class list
        dist_proto = {}
        if self.enc_model:

            X_enc = self.enc.predict(X)
            class_dict = self.class_proto if k is None else self.class_enc

            for c, v in class_dict.items():
                if c not in target_class:
                    continue
                if k is None:
                    dist_proto[c] = np.linalg.norm(X_enc - v)
                elif k is not None:
                    dist_k = np.linalg.norm(X_enc.reshape(X_enc.shape[0], -1) -
                                            v.reshape(v.shape[0], -1), axis=1)
                    idx = np.argsort(dist_k)[:k]
                    if k_type == 'mean':
                        dist_proto[c] = np.mean(dist_k[idx])
                    else:
                        dist_proto[c] = dist_k[idx[-1]]
                    self.class_proto[c] = np.expand_dims(np.mean(v[idx], axis=0), axis=0)
        elif self.use_kdtree:
            if k is None:
                k = 1
            self.class_proto = {}
            for c in range(self.classes):
                if c not in target_class:
                    continue
                dist_c, idx_c = self.kdtrees[c].query(X_num, k=k)
                dist_proto[c] = dist_c[0][-1]
                self.class_proto[c] = self.X_by_class[c][idx_c[0][-1]].reshape(1, -1)

        if self.enc_or_kdtree:
            self.id_proto = min(dist_proto, key=dist_proto.get)
            proto_val = self.class_proto[self.id_proto]
            if verbose:
                print('Prototype class: {}'.format(self.id_proto))
        else:  # no prototype loss term used
            proto_val = np.zeros(self.shape_enc)

        # set shape for perturbed instance and gradients
        pert_shape = ohe_to_ord_shape(self.shape, cat_vars=self.cat_vars, is_ohe=self.ohe)

        # set the lower and upper bounds for the constant 'c' to scale the attack loss term
        # these bounds are updated for each c_step iteration
        const_lb = np.zeros(self.batch_size)
        const = np.ones(self.batch_size) * self.c_init
        const_ub = np.ones(self.batch_size) * 1e10

        # init values for the best attack instances for each instance in the batch
        overall_best_dist = [1e10] * self.batch_size
        overall_best_attack = [np.zeros(self.shape[1:])] * self.batch_size
        overall_best_grad = (np.zeros(self.shape), np.zeros(self.shape))

        # keep track of counterfactual evolution
        self.cf_global = {i: [] for i in range(self.c_steps)}  # type: dict

        # iterate over nb of updates for 'c'
        for _ in range(self.c_steps):

            # init variables
            self.sess.run(self.init)

            # reset current best distances and scores
            current_best_dist = [1e10] * self.batch_size
            current_best_proba = [-1] * self.batch_size

            # assign variables for the current iteration
            feed_dict = {self.assign_orig: X_num,
                         self.assign_target: Y,
                         self.assign_const: const,
                         self.assign_adv: X_num,
                         self.assign_adv_s: X_num,
                         self.assign_target_proto: proto_val}
            if self.is_cat:
                feed_dict[self.assign_map] = self.d_abs_ragged
            self.sess.run(self.setup, feed_dict=feed_dict)

            X_der_batch, X_der_batch_s = [], []

            for i in range(self.max_iterations):

                # numerical gradients
                grads_num = np.zeros(pert_shape)
                grads_num_s = np.zeros(pert_shape)

                # check if numerical gradient computation is needed
                if not self.model and (self.c_init != 0. or self.c_steps > 1):
                    X_der = self.adv.eval(session=self.sess)
                    X_der_s = self.adv_s.eval(session=self.sess)
                    X_der_batch.append(X_der)
                    X_der_batch_s.append(X_der_s)

                    if i % self.update_num_grad == 0 and i > 0:  # compute numerical gradients
                        c = self.const.eval(session=self.sess)
                        X_der_batch = np.concatenate(X_der_batch)
                        X_der_batch_s = np.concatenate(X_der_batch_s)
                        grads_num = self.get_gradients(X_der_batch, Y, cat_vars_ord=self.cat_vars_ord,
                                                       grads_shape=pert_shape[1:]) * c
                        grads_num_s = self.get_gradients(X_der_batch_s, Y, cat_vars_ord=self.cat_vars_ord,
                                                         grads_shape=pert_shape[1:]) * c
                        # clip gradients
                        grads_num = np.clip(grads_num, self.clip[0], self.clip[1])
                        grads_num_s = np.clip(grads_num_s, self.clip[0], self.clip[1])
                        X_der_batch, X_der_batch_s = [], []

                # compute and clip gradients defined in graph
                grads_vars_graph = self.sess.run(self.compute_grads)
                grads_graph = [g for g, _ in grads_vars_graph][0]
                grads_graph = np.clip(grads_graph, self.clip[0], self.clip[1])

                # apply gradients
                grads = grads_graph + grads_num_s
                self.sess.run(self.apply_grads, feed_dict={self.grad_ph: grads})

                # update adv and adv_s with perturbed instances
                self.sess.run([self.adv_updater, self.adv_updater_s, self.delta, self.delta_s])

                # compute overall and attack loss, L1+L2 loss, prediction probabilities
                # on perturbed instances and new adv
                # L1+L2 and prediction probabilities used to see if adv is better than the current best adv under FISTA
                if self.model:
                    loss_tot, loss_attack, loss_l1_l2, pred_proba, adv = \
                        self.sess.run([self.loss_total, self.loss_attack, self.l1_l2, self.pred_proba, self.adv])
                else:
                    X_der = self.adv.eval(session=self.sess)  # get updated perturbed instances
                    if self.is_cat:  # map back to categories to make predictions
                        X_der = num_to_ord(X_der, self.d_abs)
                    if self.ohe:
                        X_der = ord_to_ohe(X_der, self.cat_vars_ord)[0]
                    pred_proba = self.predict(X_der)

                    # compute attack, total and L1+L2 losses as well as new perturbed instance
                    loss_attack = self.loss_fn(pred_proba, Y)
                    feed_dict = {self.loss_attack: loss_attack}
                    loss_tot, loss_l1_l2, adv = self.sess.run([self.loss_total, self.l1_l2, self.adv],
                                                              feed_dict=feed_dict)

                if i % log_every == 0 or i % print_every == 0:
                    loss_l2, loss_l1, loss_ae, loss_proto = \
                        self.sess.run([self.loss_l2, self.loss_l1, self.loss_ae, self.loss_proto])
                    target_proba = np.sum(pred_proba * Y)
                    nontarget_proba_max = np.max((1 - Y) * pred_proba)
                    loss_opt = loss_l1_l2 + loss_attack + loss_ae + loss_proto

                if i % log_every == 0 and self.writer is not None:
                    lr, zt, gs = self.sess.run([self.learning_rate, self.zt, self.global_step])

                    # add values and images to tensorboard
                    summary = tf.Summary()
                    summary.value.add(tag='loss/Optimized', simple_value=loss_opt)
                    summary.value.add(tag='loss/Total', simple_value=loss_tot)
                    summary.value.add(tag='loss/L1', simple_value=loss_l1)
                    summary.value.add(tag='loss/L2', simple_value=loss_l2)
                    summary.value.add(tag='loss/AutoEncoder', simple_value=loss_ae)
                    summary.value.add(tag='loss/ClassPrototype', simple_value=loss_proto)
                    summary.value.add(tag='loss/PredScale', simple_value=const[0])
                    summary.value.add(tag='loss/PredLoss', simple_value=loss_attack)
                    summary.value.add(tag='training/lr', simple_value=lr)
                    summary.value.add(tag='training/z', simple_value=zt)
                    summary.value.add(tag='training/GlobalStep', simple_value=gs)
                    self.writer.add_summary(summary)
                    self.writer.flush()

                if verbose and i % print_every == 0:
                    print('\nIteration: {}; Const: {}'.format(i, const[0]))
                    print('Loss total: {:.3f}, loss attack: {:.3f}'.format(loss_tot, loss_attack))
                    print('L2: {:.3f}, L1: {:.3f}, loss AE: {:.3f}'.format(loss_l2, loss_l1, loss_ae))
                    print('Loss proto: {:.3f}'.format(loss_proto))
                    print('Target proba: {:.2f}, max non target proba: {:.2f}'.format(target_proba,
                                                                                      nontarget_proba_max))
                    print('Gradient graph min/max: {:.3f}/{:.3f}'.format(grads_graph.min(), grads_graph.max()))
                    print('Gradient graph mean/abs mean: {:.3f}/{:.3f}'.format(np.mean(grads_graph),
                                                                               np.mean(np.abs(grads_graph))))
                    if not self.model:
                        print('Gradient numerical attack min/max: {:.3f}/{:.3f}'.format(grads_num.min(),
                                                                                        grads_num.max()))
                        print('Gradient numerical mean/abs mean: {:.3f}/{:.3f}'.format(np.mean(grads_num),
                                                                                       np.mean(np.abs(grads_num))))
                    sys.stdout.flush()

                # update best perturbation (distance) and class probabilities
                # if beta * L1 + L2 < current best and predicted label is different from the initial label:
                # update best current step or global perturbations
                for batch_idx, (dist, proba, adv_idx) in enumerate(zip(loss_l1_l2, pred_proba, adv)):
                    Y_class = np.argmax(Y[batch_idx])
                    adv_class = np.argmax(proba)
                    adv_idx = np.expand_dims(adv_idx, axis=0)

                    if self.is_cat:  # map back to categories
                        adv_idx = num_to_ord(adv_idx, self.d_abs)

                    if self.ohe:  # map back from ordinal to OHE
                        adv_idx = ord_to_ohe(adv_idx, self.cat_vars_ord)[0]

                    # calculate trust score
                    if threshold > 0.:
                        score = self.score(adv_idx, np.argmax(pred_proba), Y_class)
                        above_threshold = score > threshold
                    else:
                        above_threshold = True

                    # current step
                    if (dist < current_best_dist[batch_idx] and compare(proba, Y_class) and above_threshold
                            and adv_class in target_class):
                        current_best_dist[batch_idx] = dist
                        current_best_proba[batch_idx] = adv_class

                    # global
                    if (dist < overall_best_dist[batch_idx] and compare(proba, Y_class) and above_threshold
                            and adv_class in target_class):
                        if verbose:
                            print('\nNew best counterfactual found!')
                        overall_best_dist[batch_idx] = dist
                        overall_best_attack[batch_idx] = adv_idx
                        overall_best_grad = (grads_graph, grads_num)
                        self.best_attack = True
                        self.cf_global[_].append(adv_idx)

            # adjust the 'c' constant for the first loss term
            for batch_idx in range(self.batch_size):
                if (compare(current_best_proba[batch_idx], np.argmax(Y[batch_idx])) and
                        current_best_proba[batch_idx] != -1):
                    # want to refine the current best solution by putting more emphasis on the regularization terms
                    # of the loss by reducing 'c'; aiming to find a perturbation closer to the original instance
                    const_ub[batch_idx] = min(const_ub[batch_idx], const[batch_idx])
                    if const_ub[batch_idx] < 1e9:
                        const[batch_idx] = (const_lb[batch_idx] + const_ub[batch_idx]) / 2
                else:
                    # no valid current solution; put more weight on the first loss term to try and meet the
                    # prediction constraint before finetuning the solution with the regularization terms
                    const_lb[batch_idx] = max(const_lb[batch_idx], const[batch_idx])  # update lower bound to constant
                    if const_ub[batch_idx] < 1e9:
                        const[batch_idx] = (const_lb[batch_idx] + const_ub[batch_idx]) / 2
                    else:
                        const[batch_idx] *= 10

        # return best overall attack
        best_attack = np.concatenate(overall_best_attack, axis=0)
        if best_attack.shape != self.shape:
            best_attack = np.expand_dims(best_attack, axis=0)

        return best_attack, overall_best_grad

    def explain(self, X: np.ndarray, Y: np.ndarray = None, target_class: list = None, k: int = None,
                k_type: str = 'mean', threshold: float = 0., verbose: bool = False,
                print_every: int = 100, log_every: int = 100) -> Explanation:
        """
        Explain instance and return counterfactual with metadata.

        Parameters
        ----------
        X
            Instances to attack
        Y
            Labels for X as one-hot-encoding
        target_class
            List with target classes used to find closest prototype. If None, the nearest prototype
            except for the predict class on the instance is used.
        k
            Number of nearest instances used to define the prototype for a class. Defaults to using all
            instances belonging to the class if an encoder is used and to 1 for k-d trees.
        k_type
            Use either the average encoding of the k nearest instances in a class (k_type='mean') or
            the k-nearest encoding in the class (k_type='point') to define the prototype of that class.
            Only relevant if an encoder is used to define the prototypes.
        threshold
            Threshold level for the ratio between the distance of the counterfactual to the prototype of the
            predicted class for the original instance over the distance to the prototype of the predicted class
            for the counterfactual. If the trust score is below the threshold, the proposed counterfactual does
            not meet the requirements.
        verbose
            Print intermediate results of optimization if True
        print_every
            Print frequency if verbose is True
        log_every
            Tensorboard log frequency if write directory is specified

        Returns
        -------
        explanation
            Dictionary containing the counterfactual with additional metadata
        """
        # get params for storage in meta
        params = locals()
        remove = ['self', 'X', 'Y']
        for key in remove:
            params.pop(key)

        if X.shape[0] != 1:
            logger.warning('Currently only single instance explanations supported (first dim = 1), '
                           'but first dim = %s', X.shape[0])

        # output explanation dictionary
        data = copy.deepcopy(DEFAULT_DATA_CFP)

        if Y is None:
            if self.model:
                Y_proba = self.predict.predict(X)  # type: ignore
            else:
                Y_proba = self.predict(X)
            Y_ohe = np.zeros(Y_proba.shape)
            Y_class = np.argmax(Y_proba, axis=1)
            Y_ohe[np.arange(Y_proba.shape[0]), Y_class] = 1
            Y = Y_ohe.copy()
            data['orig_proba'] = Y_proba
        else:  # provided one-hot-encoding of prediction on X
            data['orig_proba'] = None
        data['orig_class'] = np.argmax(Y, axis=1)[0]

        # find best counterfactual
        self.best_attack = False
        best_attack, grads = self.attack(X, Y=Y, target_class=target_class, k=k, k_type=k_type,
                                         verbose=verbose, threshold=threshold,
                                         print_every=print_every, log_every=log_every)

        if self.enc_or_kdtree:
            data['id_proto'] = self.id_proto

        # add to explanation dict
        if not self.best_attack:
            logger.warning('No counterfactual found!')

            # create explanation object
            explanation = Explanation(meta=copy.deepcopy(self.meta), data=data)
            return explanation

        data['all'] = self.cf_global
        data['cf'] = {}
        data['cf']['X'] = best_attack
        if self.model:
            Y_pert = self.predict.predict(best_attack)  # type: ignore
        else:
            Y_pert = self.predict(best_attack)
        data['cf']['class'] = np.argmax(Y_pert, axis=1)[0]
        data['cf']['proba'] = Y_pert
        data['cf']['grads_graph'], data['cf']['grads_num'] = grads[0], grads[1]

        # create explanation object
        explanation = Explanation(meta=copy.deepcopy(self.meta), data=data)

        return explanation
