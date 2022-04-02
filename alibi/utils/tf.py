import tensorflow.compat.v1 as tf


@tf.custom_gradient
def argmin_grad(x, y):
    abs_diff = tf.abs(tf.subtract(x, y))
    argmin = tf.cast(tf.argmin(abs_diff), tf.float32)

    def grad(dy):
        """ Let gradients pass through. """
        return dy, None

    return argmin, grad


@tf.custom_gradient
def one_hot_grad(x, y):
    cat_ohe = tf.one_hot(indices=tf.cast(x, tf.int32), depth=tf.shape(y)[0])

    def grad(dy):
        """ Let gradients pass through. """
        return tf.reduce_sum(dy), None

    return cat_ohe, grad


@tf.custom_gradient
def argmax_grad(x):
    argmax = tf.argmax(x)

    def grad(dy):
        """ Let gradients pass through. """
        return tf.ones(tf.shape(x)) * dy

    return argmax, grad


@tf.custom_gradient
def round_grad(x):
    idx = tf.cast(tf.round(x), tf.int32)

    def grad(dy):
        """ Let gradients pass through. """
        return dy

    return idx, grad
