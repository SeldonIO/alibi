"""`tensorflow` backend for similarity explainers.

Methods unique to the `tensorflow` backend are defined here. The interface this class defines syncs with the `pytorch`
backend in order to ensure that the similarity methods only require to match this interface.
"""

from typing import Callable, Optional, Union

import numpy as np
import tensorflow as tf
from tensorflow import keras


class _TensorFlowBackend:
    device: Optional[str] = None  # device used by `tensorflow` backend

    @staticmethod
    def get_grads(
            model: keras.Model,
            X: tf.Tensor,
            Y: tf.Tensor,
            loss_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
    ) -> np.ndarray:
        """
        Computes the gradients of the loss function with respect to the model's parameters for a single training and
        target pair.

        Parameters
        -----------
        model
            The model to compute gradients for.
        X
            The input data point.
        Y
            The target data point.
        loss_fn
            The loss function to use.

        Returns
        --------
        grads
            The gradients of the loss function with respect to the model's parameters. This is returned as a flattened \
            array.
        """

        with tf.device(_TensorFlowBackend.device):
            with tf.GradientTape() as tape:
                output = model(X, training=False)
                loss = loss_fn(Y, output)

            # compute gradients of the loss w.r.t the weights
            grad_X_train = tape.gradient(loss, model.trainable_weights)
            grad_X_train = np.concatenate([_TensorFlowBackend._grad_to_numpy(w, getattr(w, 'name', None))
                                           for w in grad_X_train])
        return grad_X_train

    @staticmethod
    def _grad_to_numpy(grad: Union[tf.IndexedSlices, tf.Tensor], name: Optional[str] = None) -> np.ndarray:
        """Convert gradient to `np.ndarray`.

        Converts gradient tensor to flat `numpy` array. If the gradient is a sparse tensor, it is converted to a dense
        tensor first.
        """

        if isinstance(grad, tf.IndexedSlices):
            # see https://github.com/SeldonIO/alibi/issues/828
            grad = tf.convert_to_tensor(grad)

        if not hasattr(grad, 'numpy'):
            name = f' for the named tensor: {name}' if name else ''
            raise TypeError((f'Could not convert gradient to `numpy` array{name}. To ignore these '
                             'gradients in the similarity computation set ``trainable=False`` on the '
                             'corresponding parameter.'))
        return grad.numpy().reshape(-1)

    @staticmethod
    def to_tensor(X: np.ndarray) -> tf.Tensor:
        """Converts a `numpy` array to a `tensorflow` tensor."""
        return tf.convert_to_tensor(X)

    @staticmethod
    def set_device(device: Union[str, None] = None) -> None:
        """Sets the device to use for the backend.

        Sets the device value on the class. Any subsequent calls to the backend will use this device.
        """
        if device is None or isinstance(device, str):
            _TensorFlowBackend.device = device
        else:
            raise TypeError(f"`device` must be a `string` or ``None``. Got {type(device)} instead.")

    @staticmethod
    def to_numpy(X: tf.Tensor) -> np.ndarray:
        """Converts a tensor to `np.ndarray`."""
        return X.numpy()

    @staticmethod
    def argmax(X: tf.Tensor, dim=-1) -> tf.Tensor:
        """Returns the index of the maximum value in a tensor."""
        X = tf.math.argmax(X, axis=dim)
        return X

    @staticmethod
    def _count_non_trainable(model: keras.Model) -> int:
        """Returns number of non trainable parameters.

        Returns the number of parameters that are non trainable. If no trainable parameter exists we raise
        a `ValueError`.
        """

        if len(model.trainable_weights) == 0:
            raise ValueError("The model has no trainable weights. This method requires at least "
                             "one trainable parameter to compute the gradients for. "
                             "Set ``trainable=True`` on the model or a model weight.")
        return len(model.non_trainable_weights)
