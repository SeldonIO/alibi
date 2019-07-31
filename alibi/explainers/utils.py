import tensorflow as tf
from typing import Callable, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import keras  # noqa


def _check_keras_or_tf(predict_fn: Union[Callable, tf.keras.Model, 'keras.Model']) -> Tuple[bool, bool]:
    """
    Test if the prediction function is a tf.keras or keras model

    Parameters
    ----------
    predict_fn
        Prediction function or a tf.keras or keras model

    Returns
    -------
    (is_model, is_keras)
        Tuple of boolean values indicating whether the prediction function is a model or a black-box function and if
        it is a tf.keras or keras model
    """
    try:
        import keras  # noqa
        is_model = isinstance(predict_fn, keras.Model)
        if is_model:
            is_keras = True
        else:
            is_model = isinstance(predict_fn, tf.keras.Model)
            is_keras = False
    except ImportError:
        is_model = isinstance(predict_fn, tf.keras.Model)
        is_keras = False

    return is_model, is_keras
