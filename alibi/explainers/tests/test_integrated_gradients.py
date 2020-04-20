import numpy as np
import tensorflow as tf
import pytest
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from alibi.explainers import IntegratedGradients
from alibi.api.interfaces import Explanation

X = np.random.rand(100, 4)
y = (X[:, 0] + X[:, 1] > 1).astype(int)
y = to_categorical(y)
X_train, y_train = X[:90, :], y[:90, :]
X_test, y_test = X[90:, :], y[90:, :]
test_labels = np.argmax(y_test, axis=1)

inputs = Input(shape=(X.shape[1:]), dtype=tf.float64)
x = Dense(20, activation='linear')(inputs)
outputs = Dense(2, activation='softmax')(x)
model = Model(inputs=inputs, outputs=outputs)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# train model
model.fit(X_train,
          y_train,
          epochs=2,
          batch_size=256,
          verbose=0,
          validation_data=(X_test, y_test)
          )


@pytest.mark.parametrize('method', ('gausslegendre',
                                    "riemann_left",
                                    "riemann_right",
                                    "riemann_middle",
                                    "riemann_trapezoid"))
@pytest.mark.parametrize('rcd', (True, False))
@pytest.mark.parametrize('rp', (True, False))
@pytest.mark.parametrize('fn', (None, ['feat_{}'.format(i) for i in range(4)]))
def test_integratedgradients(method, rcd, rp, fn):

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


@pytest.mark.parametrize('method', ('gausslegendre',
                                    "riemann_left",
                                    "riemann_right",
                                    "riemann_middle",
                                    "riemann_trapezoid"))
@pytest.mark.parametrize('rcd', (True, False))
@pytest.mark.parametrize('rp', (True, False))
@pytest.mark.parametrize('fn', (None, ['feat_{}'.format(i) for i in range(4)]))
@pytest.mark.parametrize('layer_nb', (None, 1))
def test_layer_integratedgradients(method, rcd, rp, fn, layer_nb):

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
