# flake8: noqa E731
import keras

import tensorflow as tf

from alibi.utils.tf import _check_keras_or_tf
from unittest import mock

blackbox_model = lambda x: x
keras_model = keras.Model()
tf_model = tf.keras.Model()
blackbox_keras = lambda x: keras_model.predict(x)
blackbox_tf = lambda x: tf_model.predict(x)


def test_blackbox_check_keras_or_tf():
    is_model, is_keras, sess = _check_keras_or_tf(blackbox_model)
    assert not is_model
    assert not is_keras


def test_keras_check_keras_or_tf():
    is_model, is_keras, sess = _check_keras_or_tf(keras_model)
    assert is_model
    assert is_keras


def test_tf_check_keras_or_tf():
    is_model, is_keras, sess = _check_keras_or_tf(tf_model)
    assert is_model
    assert not is_keras


def test_tf_bb_check_keras_or_tf():
    is_model, is_keras, sess = _check_keras_or_tf(blackbox_tf)
    assert not is_model
    assert not is_keras


def test_keras_bb_check_keras_or_tf():
    is_model, is_keras, sess = _check_keras_or_tf(blackbox_keras)
    assert not is_model
    assert not is_keras


def test_tf_check_keras_or_tf_no_keras_import():
    with mock.patch.dict('sys.modules', {'keras': None}):
        is_model, is_keras, sess = _check_keras_or_tf(tf_model)
        assert is_model
        assert not is_keras


def test_blackbox_check_keras_or_tf_no_keras_import():
    with mock.patch.dict('sys.modules', {'keras': None}):
        is_model, is_keras, sess = _check_keras_or_tf(blackbox_model)
        assert not is_model
        assert not is_keras
