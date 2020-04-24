import numpy as np
import pytest
from alibi.explainers import IntegratedGradients
from alibi.api.interfaces import Explanation
import tensorflow as tf 
tf.compat.v1.enable_eager_execution()


def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)
    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    # Example
    ```python
    # Consider an array of 5 labels out of a set of 3 classes {0, 1, 2}:
    > labels
    array([0, 2, 1, 2, 0])
    # `to_categorical` converts this into a matrix with as many
    # columns as there are classes. The number of rows
    # stays the same.
    > to_categorical(labels)
    array([[ 1.,  0.,  0.],
           [ 0.,  0.,  1.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.],
           [ 1.,  0.,  0.]], dtype=float32)
    ```
    """

    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


X = np.random.rand(100, 4)
y = (X[:, 0] + X[:, 1] > 1).astype(int)
y = to_categorical(y)
X_train, y_train = X[:90, :], y[:90, :]
X_test, y_test = X[90:, :], y[90:, :]
test_labels = np.argmax(y_test, axis=1)


@pytest.fixture(scope='module')
def hacky_cnn():

    inputs = tf.keras.Input(shape=(X.shape[1:]))
    x = tf.keras.layers.Dense(20, activation='linear')(inputs)
    outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
  
    # train model
    model.fit(X_train,
              y_train,
              epochs=1,
              batch_size=256,
              verbose=0,
              validation_data=(X_test, y_test)
              )

    return model


@pytest.mark.eager
@pytest.mark.parametrize('method', ('gausslegendre',
                                    "riemann_left",
                                    "riemann_right",
                                    "riemann_middle",
                                    "riemann_trapezoid"))
@pytest.mark.parametrize('rcd', (True, False))
@pytest.mark.parametrize('rp', (True, False))
@pytest.mark.parametrize('fn', (None, ['feat_{}'.format(i) for i in range(4)]))
def test_integratedgradients(hacky_cnn, method, rcd, rp, fn):
    model = hacky_cnn
    ig = IntegratedGradients(model, n_steps=50, method=method, return_convergence_delta=rcd,
                             return_predictions=rp)

    explanations = ig.explain(X_test,
                              baselines=None,
                              features_names=fn,
                              target=test_labels)

    assert isinstance(explanations, Explanation)
    assert explanations['data']['attributions'].shape == X_test.shape
    if rcd:
        assert 'deltas' in explanations['data'].keys()
        assert explanations['data']['deltas'].shape[0] == X_test.shape[0]
    if rp:
        assert 'predictions' in explanations['data'].keys()
        assert explanations['data']['predictions'].shape[0] == X_test.shape[0]
    if fn is not None:
        assert len(fn) == X_test.reshape(X_test.shape[0], -1).shape[1]


@pytest.mark.eager
@pytest.mark.parametrize('method', ('gausslegendre',
                                    "riemann_left",
                                    "riemann_right",
                                    "riemann_middle",
                                    "riemann_trapezoid"))
@pytest.mark.parametrize('rcd', (True, False))
@pytest.mark.parametrize('rp', (True, False))
@pytest.mark.parametrize('fn', (None, ['feat_{}'.format(i) for i in range(4)]))
@pytest.mark.parametrize('layer_nb', (None, 1))
def test_layer_integratedgradients(hacky_cnn, method, rcd, rp, fn, layer_nb):

    model = hacky_cnn
    if layer_nb is not None:
        layer = model.layers[layer_nb]
    else:
        layer = None

    ig = IntegratedGradients(model, layer=layer,
                             n_steps=50, method=method, return_convergence_delta=rcd,
                             return_predictions=rp)

    explanations = ig.explain(X_test,
                              baselines=None,
                              features_names=fn,
                              target=test_labels)

    assert isinstance(explanations, Explanation)
    if layer is not None:
        layer_out = layer(X_test).numpy()
        assert explanations['data']['attributions'].shape == layer_out.shape
    else:
        assert explanations['data']['attributions'].shape == X_test.shape
    if rcd:
        assert 'deltas' in explanations['data'].keys()
        assert explanations['data']['deltas'].shape[0] == X_test.shape[0]
    if rp:
        assert 'predictions' in explanations['data'].keys()
        assert explanations['data']['predictions'].shape[0] == X_test.shape[0]
    if fn is not None:
        assert len(fn) == X_test.reshape(X_test.shape[0], -1).shape[1]
