from .datafeeder import DataFeeder
import logging
import numpy as np
import os
from typing import List, Sequence, Optional, Dict, Union, Any
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.python.ops.gradients_impl import _hessian_vector_product

logger = logging.getLogger(__name__)


class InfluenceKeras:
    """
    This class implements the Influence Function approach for explaining decisions of neural networks [1]_.
    This implementation is Keras specific and takes inspiration both from the original code [2]_
    and the implementation available in [3]_.


    References
    ----------
    .. [1] Koh, Pang Wei and Liang, Percy, "Understanding Black-box Predictions via Influence Functions",
        ICML 2017

    .. [2] https://github.com/kohpangwei/influence-release

    .. [3] https://github.com/darkonhub/darkon

    """

    def __init__(self, model: keras.models.Model, workspace: str, datafeeder: DataFeeder,
                 ihvp_config: Optional[Dict[str, Any]] = None) -> None:
        self.model = model
        self.workspace = workspace
        self.datafeeder = datafeeder

        # extract losses and operations
        self.loss_op = model.total_loss
        self.grad_op = K.gradients(model.total_loss, model.weights)
        input_tensors = model.inputs + model.targets + [K.learning_phase()]
        self.grad_op_num = K.function(inputs=input_tensors, outputs=self.grad_op)

        # placeholders for calculations
        trainable_vars = model.weights
        self.v_test_grad = [tf.placeholder(tf.float32, shape=w.get_shape()) for w in trainable_vars]

        # hessian vector product operation
        self.v_cur_estimate = [tf.placeholder(tf.float32, shape=w.get_shape()) for w in trainable_vars]
        self.hessian_vector_op = _hessian_vector_product(self.loss_op, trainable_vars, self.v_cur_estimate)
        input_tensors = model.inputs + model.targets + self.v_cur_estimate + [K.learning_phase()]
        self.hessian_vector_op_num = K.function(inputs=input_tensors,
                                                outputs=self.hessian_vector_op)

        # Hessian vector product hyperparameters
        if ihvp_config is None:
            self.ihvp_config = {
                'scale': 1e4,
                'damping': 0.01,
                'num_repeats': 1,
                'recursion_batch_size': 10,
                'recursion_depth': 10000
            }
        else:
            self.ihvp_config = ihvp_config

        # gradients = model.optimizer.get_gradients(model.total_loss, trainable_vars)  # same as K.gradients or tf.gradients
        # input_tensors = [model.inputs[0],  # TODO extend to multiple inputs/outputs?
        #                 model.sample_weights[0],
        #                 model.targets[0],
        #                 K.learning_phase()]  # 0 for test, 1 for train mode

        # gradient of the loss operation (both test and train mode via flag)
        # self.grad_op = K.function(inputs=input_tensors, outputs=gradients)

        # self.loss_op_train = None
        # self.hess_vec_op = None

    def get_test_grad_loss(self, test_indices: Sequence[int]) -> np.ndarray:  # TODO support batch?
        """
        This function calculates the gradient of the loss of the test samples:

        .. math::
            \\nabla_{\\theta}L(z_{\\text{test}}, \\theta)

        Returns
        -------
        Gradient of the loss for each test sample of interest

        """
        # get test samples
        X_test, y_test = self.datafeeder.test_batch(test_indices)
        # test_feed_dict = {self.x_placeholder: X_test,
        #                  self.y_placeholder: y_test}
        # input_tensors = [self.model.inputs[0],  # TODO extend to multiple inputs/outputs?
        #                 self.model.targets[0],
        #                 K.learning_phase()]
        # kfunc = K.function(inputs=input_tensors, outputs=self.grad_op)

        # calculate gradients
        # test_grad_loss = self.sess.run(self.grad_op, feed_dict=test_feed_dict)  # TODO K.function vs sess.run ?
        test_grad_loss = self.grad_op_num([X_test, y_test, 0])
        logger.debug("test_grad_loss shapes: %s", [g.shape for g in test_grad_loss])
        # TODO scale for multiple test samples

        # convert to array
        test_grad_loss = np.array(test_grad_loss)

        return test_grad_loss

    def get_inverse_hvp_lissa(self, test_grad_loss: np.ndarray) -> np.ndarray:
        """
        Calculates the inverse Hessian vector product:

        .. math::
            H_{\\theta}^{-1}\\nabla_{\\theta}L(z_{\\text{test}}, \\theta)

        This uses the LISSA method developed in [4]_.

        Parameters
        ----------
        test_grad_loss
            Gradient of the loss for each test sample of interest

        Returns
        -------
        Inverse Hessian vector product with the gradient of the loss for test samples

        References
        -------
        .. [4] Agarwal, N., Bullins, B. and Hazan, E., "Second-Order Stochastic Optimization
            for Machine Learning in Linear Time"

        """
        params = self.ihvp_config
        ihvp = None

        # number of times to estimate
        for _ in range(params['num_repeats']):
            cur_estimate = test_grad_loss

            # number of steps in the estimation
            for j in range(params['recursion_depth']):
                X_train, y_train = self.datafeeder.train_batch(params['recursion_batch_size'])
                logger.debug("X_train shape: %s, y_train shape: %s", X_train.shape, y_train.shape)

                hessian_vector_val = self.hessian_vector_op_num([X_train, y_train] + list(cur_estimate) + [1])
                logger.debug("hessian_vector_val shapes: %s", [h.shape for h in hessian_vector_val])

                # convert to arrays to be able to calculate estimate
                hessian_vector_val = np.array(hessian_vector_val)
                # test_grad_loss = np.array(test_grad_loss)
                # cur_estimate = np.array(cur_estimate)

                cur_estimate = test_grad_loss + (1 - params['damping']) * cur_estimate - hessian_vector_val / params[
                    'scale']
                # TODO: need to monitor convergence

            if ihvp is None:
                ihvp = cur_estimate / params['scale']
            else:
                ihvp += cur_estimate / params['scale']

        ihvp /= params['num_repeats']
        return ihvp

    def run_ihvp(self, test_indices: Sequence[int], force_refresh: bool = False):
        # TODO: better paths
        ihvp_path = self.workspace + '/ihvp.npz'
        if not os.path.exists(ihvp_path) or force_refresh:
            self.datafeeder.reset()
            test_grad_loss = self.get_test_grad_loss(test_indices)
            self.ihvp = self.get_inverse_hvp_lissa(test_grad_loss)
            np.savez(ihvp_path, ihvp=self.ihvp, encoding='bytes')
            logger.info("Saved IHVP to %s", ihvp_path)
        else:
            self.ihvp = np.load(ihvp_path, encoding='bytes')['ihvp']
            logger.info("Loaded IHVP from %s", ihvp_path)

    def grad_dot_ihvp(self):
        # use all training data unless specified otherwise
        # if num_batches is None:
        #    num_batches = int(np.ceil(len(self.feeder.X_train) / train_batch_size))

        n_train = len(self.datafeeder.X_train)

        # flatten everything into a single vector
        ihvp = np.concatenate([i.reshape(-1) for i in self.ihvp])

        # placeholder for results
        grad_dot_ihvp = np.zeros([len(self.datafeeder.X_train)])

        # for batch in range(num_batches):
        #    X_train, y_train = self.datafeeder.train_batch(train_batch_size)
        #    pass

        # calculate grad_dot_ihvp separately for every sample
        for ix in range(n_train):
            X, y = self.datafeeder.train_batch(1)
            train_grads = self.single_grad_dot_ihvp(X, y)
            grad_dot_ihvp[ix] = np.dot(ihvp, train_grads)

        return grad_dot_ihvp

    def single_grad_dot_ihvp(self, X: np.ndarray, y: np.ndarray):
        train_grads = self.grad_op_num([X, y, 1])
        train_grads = np.concatenate([g.reshape(-1) for g in train_grads])
        return train_grads

    def influence_scores(self, test_indices: Sequence[int], force_refresh: bool = False) -> np.ndarray:

        # calculate the Inverse Hessian vector product
        self.run_ihvp(test_indices, force_refresh)
        self.datafeeder.reset()

        # dot product with the gradient of training loss
        scores = self.grad_dot_ihvp()

        return scores
