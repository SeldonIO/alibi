import tensorflow as tf
import sys
import os
from typing import Callable, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import keras  # noqa


def _check_keras_or_tf(predict_fn: Union[Callable, tf.keras.Model, 'keras.Model']) -> \
        Tuple[bool, bool, tf.compat.v1.Session]:
    """
    Test if the prediction function is a tf.keras or keras model and return the associated TF session.

    Parameters
    ----------
    predict_fn
        Prediction function or a tf.keras or keras model

    Returns
    -------
    (is_model, is_keras, sess)
        Tuple of boolean values indicating whether the prediction function is a model or a black-box function and if
        it is a tf.keras or keras model, also returns the associated session (or a new one if the model is black-box).
    """
    tfsess = tf.keras.backend.get_session()

    try:
        # workaround to suppress keras backend message, see https://github.com/keras-team/keras/issues/1406
        stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        import keras  # noqa
        sys.stderr = stderr

        ksess = keras.backend.get_session()
        is_model = isinstance(predict_fn, keras.Model)
        if is_model:
            #  keras model, return keras session
            is_keras = True
            return is_model, is_keras, ksess

        else:
            is_model = isinstance(predict_fn, tf.keras.Model)
            is_keras = False
            if is_model:
                # tf model, return tf.keras session
                return is_model, is_keras, tfsess
            else:
                # at this point could be K.predict, TF.predict or vanilla blackbox
                # we cant distinguish between K.predict or TF.predict, so we support TF.predict by default
                # support for K.predict would be provided by passing in the user session as an argument
                return is_model, is_keras, tfsess

    except ImportError:
        # no keras found, so only possibilities are TF model, blackbox model or TF.predict blackbox
        is_model = isinstance(predict_fn, tf.keras.Model)
        is_keras = False
        tfsess = tf.keras.backend.get_session()
        return is_model, is_keras, tfsess
