"""Tensor flow backend for similarity explainers.

Methods unique to the Tensorflow backend are defined here.
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from typing import Callable
import random
import os


def get_grads(
        model: keras.Model,
        x: tf.Tensor,
        y: tf.Tensor,
        loss_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
) -> np.ndarray:
    """
    Computes the gradients of the loss function with respect to the model's parameters for a single training and target
    pair.

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

    Notes:
    ------
    x is assumed to be of shape (n, p) where n is the number of samples and p is the number of parameters. y is assumed
    to be of shape (n, 1).
    """

    with tf.GradientTape() as tape:
        output = model(x, training=False)
        loss = loss_fn(y, output)

    # compute gradients of the loss w.r.t the weights
    grad_x_train = tape.gradient(loss, model.trainable_weights)
    return np.concatenate([w.numpy().reshape(-1) for w in grad_x_train])


def to_tensor(x: np.ndarray, **kwargs) -> tf.Tensor:
    # TODO: align with CFRL backend
    return tf.convert_to_tensor(x)


def get_device(device: str = 'cpu:0') -> tf.device:
    return tf.device(device)


def to_numpy(x: tf.Tensor) -> tf.Tensor:
    # TODO: align with CFRL backend
    return x.numpy()


def argmax(x: tf.Tensor) -> tf.Tensor:
    x = tf.math.argmax(x, axis=1)
    return x


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
