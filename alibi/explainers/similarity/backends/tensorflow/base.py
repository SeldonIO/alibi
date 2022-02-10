"""Tensorflow backend for similarity explainers.

Methods unique to the Tensorflow backend are defined here. The interface this class defines syncs with the torch backend
in order to ensure that the similarity methods only require to match this interface.
"""

import random
import os
from typing import Callable

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


class _TensorFlowBackend:
    device = None

    @staticmethod
    def get_grads(
            model: keras.Model,
            x: tf.Tensor,
            y: tf.Tensor,
            loss_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
    ) -> np.ndarray:
        """
        Computes the gradients of the loss function with respect to the model's parameters for a single training and
        target pair.

        Parameters:
        -----------
        model: keras.Model
            The model to compute gradients for.
        x: tf.Tensor
            The input data point.
        y: tf.Tensor
            The target data point.
        loss_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor]
            The loss function to use.

        Returns:
        --------
        grads: np.ndarray
            The gradients of the loss function with respect to the model's parameters. This is returned as a flattened
            array.
        """

        with tf.device(_TensorFlowBackend.device):
            with tf.GradientTape() as tape:
                output = model(x, training=False)
                loss = loss_fn(y, output)

            # compute gradients of the loss w.r.t the weights
            grad_x_train = tape.gradient(loss, model.trainable_weights)
            grad_x_train = np.concatenate([w.numpy().reshape(-1) for w in grad_x_train])
        return grad_x_train

    @staticmethod
    def to_tensor(x: np.ndarray, **kwargs) -> tf.Tensor:
        """Converts a numpy array to a torch tensor."""
        return tf.convert_to_tensor(x)

    @staticmethod
    def set_device(device: str = 'cpu:0') -> None:
        """Sets the device to use for the backend.

        Sets te device value on the class. Any subsequent calls to the backend will use this device.
        """
        _TensorFlowBackend.device = device

    @staticmethod
    def to_numpy(x: tf.Tensor) -> tf.Tensor:
        """Converts a tensor to a numpy array."""
        return x.numpy()

    @staticmethod
    def argmax(x: tf.Tensor) -> tf.Tensor:
        """Returns the index of the maximum value in a tensor."""
        x = tf.math.argmax(x, axis=1)
        return x

    @staticmethod
    def set_seed(seed: int = 13):
        # TODO: align with CFRL backend
        """
        Sets a seed to ensure reproducibility. Does NOT ensure reproducibility.

        Parameters
        ----------
        seed
            seed to be set
        """
        # others
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)

        # tf random
        tf.random.set_seed(seed)
