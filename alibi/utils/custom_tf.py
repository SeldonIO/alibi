import tensorflow as tf


@tf.custom_gradient
def argmin_custom_grad(x, y):
    abs_diff = tf.abs(tf.subtract(x, y))
    argmin = tf.cast(tf.argmin(abs_diff), tf.float32)

    def grad(dy):
        """ Let gradients pass through. """
        return dy, None

    return argmin, grad


@tf.custom_gradient
def one_hot_custom_grad(x, y):
    cat_ohe = tf.one_hot(indices=tf.cast(x, tf.int32), depth=tf.shape(y)[0])

    def grad(dy):
        """ Let gradients pass through. """
        return tf.reduce_sum(dy), None  # TODO: check if reduce_sum is correct!
        #return dy[tf.cast(x, tf.int32)], None
        #return dy[1], None

    return cat_ohe, grad


@tf.custom_gradient
def argmax_custom_grad(x):
    argmax = tf.argmax(x)

    def grad(dy):
        """ Let gradients pass through. """
        return tf.ones(tf.shape(x)) * dy  # TODO: check if gradient broadcasting makes sense!

    return argmax, grad
