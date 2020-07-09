from typing import Callable, Union, Optional, Tuple, Dict

import numpy as np
import tensorflow as tf

from alibi.utils.gradients import numerical_gradient


def perturb_tensorflow(X: tf.Tensor, eps: Union[float, np.ndarray] = 1e-08) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    An implementation of `perturb` for TensorFlow 2.x. See perturb for details. This function does not support
    the `proba` kwarg.
    """
    batch_size, *datapoint_shape = X.shape
    # flatten datapoint
    X = tf.reshape(X, (batch_size, -1))  # N x F
    n_features = X.shape[1]
    # create perturbation for each  F features
    pert = tf.tile(tf.eye(n_features, dtype=X.dtype)*eps, (batch_size, 1))
    # create F pertrubed copies of X. Only one feature is pertrubed in each copy.
    X_pert_pos, X_pert_neg = X + pert, X - pert
    shape = (n_features * batch_size, *datapoint_shape)
    X_pert_pos = tf.reshape(X_pert_pos, shape)  # (N * F)*X[0].shape
    X_pert_neg = tf.reshape(X_pert_neg, shape)

    return X_pert_pos, X_pert_neg


@numerical_gradient(framework='tensorflow')
def central_difference(func: Callable,
                       X: tf.Tensor,
                       eps: Union[float, np.ndarray] = 1e-08,
                       fcn_args: Optional[Tuple] = None,
                       fcn_kwargs: Optional[Dict] = None) -> np.ndarray:
    """
    Calculate the numerical gradients of a vector-valued function (typically a prediction function in classification)
    with respect to a batch of arrays X. Unlike the numpy version this function:

        - Allows passing `kwargs` to `func`
        - Infers the prediction shape during gradient computation, avoiding an additional call to `func`.

    Parameters
    ----------
    func
        Function to be differentiated
    X
        A batch of vectors at which to evaluate the gradient of the function
    fcn_args
        Any additional arguments to pass to the function
    fcn_kwargs
        Any additional key-word arguments to pass to the function
    eps
        Gradient step to use in the numerical calculation, can be a single float or one for each feature

    Returns
    -------
    An array of gradients at each point in the batch X

    """
    if fcn_args is None:
        fcn_args = ()
    if fcn_kwargs is None:
        fcn_kwargs = {}

    # N = gradient batch size; F = nb of features in X, P = nb of prediction classes, B = instance batch size
    batch_size, *data_shape = X.shape
    X_pert_pos, X_pert_neg = perturb_tensorflow(X, eps)  # (N*F)x(shape of X[0])
    n_pert = X_pert_pos.shape[0]
    X_pert = tf.concat([X_pert_pos, X_pert_neg], axis=0)
    preds_concat = func(X_pert, *fcn_args, **fcn_kwargs)
    # scalar output
    if len(preds_concat.shape) == 1:
        preds_concat = tf.expand_dims(preds_concat, axis=-1)
    preds_shape = (batch_size, preds_concat.shape[-1])

    grad_numerator = preds_concat[:n_pert] - preds_concat[n_pert:]  # (N*F)*P
    grad_numerator = tf.reshape(grad_numerator, (batch_size, -1))  # -> Nx(F*P)
    grad_numerator = tf.reshape(grad_numerator, (-1, preds_shape[-1], batch_size))  # -> FxPxN
    grad_numerator = tf.transpose(grad_numerator)  # -> NxPxF

    grad = grad_numerator / (2 * eps)
    grad = tf.reshape(grad, (*preds_shape, *data_shape))  # BxPx(shape of X[0])

    return grad
