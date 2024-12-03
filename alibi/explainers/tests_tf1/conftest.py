import os
import pytest

@pytest.fixture
def set_env_variables():
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # force CPU usage

    import tensorflow as tf
    tf.compat.v1.disable_v2_behavior() # disable TF2 behaviour as alibi code still relies on TF1 constructs 

    print('TF version: ', tf.__version__)
    print('Eager execution enabled: ', tf.executing_eagerly()) # False

    yield
    tf.keras.backend.clear_session()