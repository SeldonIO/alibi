import tensorflow as tf

from alibi.utils.wrappers import blackbox_wrapper
from functools import wraps
from typing import Callable


@blackbox_wrapper(framework='tensorflow')
def wrap_blackbox_predictor(func: Callable) -> Callable:
    """
    A decorator that converts the first argument to `func` to a `np.array` object and converts the output to a tensor.
    """
    @wraps(func)
    def wrap(*args, **kwargs):
        X, *others = args
        if isinstance(X, tf.Tensor) or isinstance(X, tf.Variable):
            X = X.numpy()
        result = func(X, *others, **kwargs)
        return tf.identity(result)
    return wrap
