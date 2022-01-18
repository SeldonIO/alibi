"""Tensor flow backend for similarity explainers.

Methods unique to the Tensorflow backend are defined here.
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from typing import Dict, Any, Callable, Optional, Union


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
    """

    with tf.GradientTape() as tape:
        output = model(x, training=False)
        # print(output[None].shape, y.shape)
        loss = loss_fn(y, output)

    # compute gradients of the loss w.r.t the weights
    grad_x_train = tape.gradient(loss, model.trainable_weights)
    return np.concatenate([w.numpy().reshape(-1) for w in grad_x_train])


def to_tensor(x: np.ndarray) -> tf.Tensor:
    return tf.convert_to_tensor(x, dtype=tf.float32)
