import numpy as np
import pytest
from alibi.explainers import IntegratedGradients
from alibi.api.interfaces import Explanation
import tensorflow as tf
tf.compat.v1.enable_eager_execution()


X = np.random.rand(100, 4)
y_class = (X[:, 0] + X[:, 1] > 1).astype(int)
y_reg = X[:, 0] + X[:, 1]
y = tf.keras.utils.to_categorical(y_class)

X_train_c, y_train_c, y_train_c_2 = X[:90, :], y[:90, :], y_class[:90]
X_test_c, y_test_c, y_test_c_2 = X[90:, :], y[90:, :], y_class[90:]

X_train_r, y_train_r = X[:90, :], y_reg[:90]
X_test_r, y_test_r = X[90:, :], y_reg[90:]
test_labels = np.argmax(y_test_c, axis=1)


@pytest.fixture(scope='module')
def hacky_class():

    inputs = tf.keras.Input(shape=(X.shape[1:]))
    x = tf.keras.layers.Dense(20, activation='linear')(inputs)
    outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # train model
    model.fit(X_train_c,
              y_train_c,
              epochs=1,
              batch_size=256,
              verbose=0,
              validation_data=(X_test_c, y_test_c)
              )

    return model


@pytest.fixture(scope='module')
def hacky_class_2():

    inputs = tf.keras.Input(shape=(X.shape[1:]))
    x = tf.keras.layers.Dense(20, activation='linear')(inputs)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # train model
    model.fit(X_train_c,
              y_train_c_2,
              epochs=1,
              batch_size=256,
              verbose=0,
              validation_data=(X_test_c, y_test_c_2)
              )

    return model


@pytest.fixture(scope='module')
def hacky_class_3():

    inputs = tf.keras.Input(shape=(X.shape[1:]))
    x = tf.keras.layers.Dense(20, activation='linear')(inputs)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    outputs = tf.keras.layers.Reshape(())(outputs)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # train model
    model.fit(X_train_c,
              y_train_c_2,
              epochs=1,
              batch_size=256,
              verbose=0,
              validation_data=(X_test_c, y_test_c_2)
              )

    return model


@pytest.fixture(scope='module')
def hacky_reg():

    inputs = tf.keras.Input(shape=(X.shape[1:]))
    x = tf.keras.layers.Dense(20, activation='linear')(inputs)
    outputs = tf.keras.layers.Dense(1, activation='linear')(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['mean_squared_error'])

    # train model
    model.fit(X_train_r,
              y_train_r,
              epochs=1,
              batch_size=256,
              verbose=0,
              validation_data=(X_test_r, y_test_r)
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
def test_integratedgradients(hacky_class, method, rcd, rp, fn):
    model = hacky_class
    ig = IntegratedGradients(model, n_steps=50, method=method)

    explanations = ig.explain(X_test_c,
                              baselines=None,
                              target=test_labels,
                              features_names=fn,
                              return_convergence_delta=rcd,
                              return_predictions=rp)

    assert isinstance(explanations, Explanation)
    assert explanations['data']['attributions'].shape == X_test_c.shape
    if rcd:
        assert 'deltas' in explanations['data'].keys()
        assert explanations['data']['deltas'].shape[0] == X_test_c.shape[0]
    if rp:
        assert 'predictions' in explanations['data'].keys()
        assert explanations['data']['predictions'].shape[0] == X_test_c.shape[0]
    if fn is not None:
        assert len(fn) == X_test_c.reshape(X_test_c.shape[0], -1).shape[1]


@pytest.mark.eager
@pytest.mark.parametrize('method', ('gausslegendre',
                                    "riemann_left"))
@pytest.mark.parametrize('rcd', (True, False))
@pytest.mark.parametrize('rp', (True, False))
@pytest.mark.parametrize('fn', (None, ['feat_{}'.format(i) for i in range(4)]))
def test_integratedgradients_2(hacky_class_2, method, rcd, rp, fn):
    model = hacky_class_2
    ig = IntegratedGradients(model, n_steps=50, method=method)

    explanations = ig.explain(X_test_c,
                              baselines=None,
                              target=test_labels,
                              features_names=fn,
                              return_convergence_delta=rcd,
                              return_predictions=rp)

    assert isinstance(explanations, Explanation)
    assert explanations['data']['attributions'].shape == X_test_c.shape
    if rcd:
        assert 'deltas' in explanations['data'].keys()
        assert explanations['data']['deltas'].shape[0] == X_test_c.shape[0]
    if rp:
        assert 'predictions' in explanations['data'].keys()
        assert explanations['data']['predictions'].shape[0] == X_test_c.shape[0]
    if fn is not None:
        assert len(fn) == X_test_c.reshape(X_test_c.shape[0], -1).shape[1]


@pytest.mark.eager
@pytest.mark.parametrize('method', ('gausslegendre',
                                    "riemann_left"))
@pytest.mark.parametrize('rcd', (True, False))
@pytest.mark.parametrize('rp', (True, False))
@pytest.mark.parametrize('fn', (None, ['feat_{}'.format(i) for i in range(4)]))
def test_integratedgradients_3(hacky_class_3, method, rcd, rp, fn):
    model = hacky_class_3
    ig = IntegratedGradients(model, n_steps=50, method=method)

    explanations = ig.explain(X_test_c,
                              baselines=None,
                              target=test_labels,
                              features_names=fn,
                              return_convergence_delta=rcd,
                              return_predictions=rp)

    assert isinstance(explanations, Explanation)
    assert explanations['data']['attributions'].shape == X_test_c.shape
    if rcd:
        assert 'deltas' in explanations['data'].keys()
        assert explanations['data']['deltas'].shape[0] == X_test_c.shape[0]
    if rp:
        assert 'predictions' in explanations['data'].keys()
        assert explanations['data']['predictions'].shape[0] == X_test_c.shape[0]
    if fn is not None:
        assert len(fn) == X_test_c.reshape(X_test_c.shape[0], -1).shape[1]


@pytest.mark.eager
@pytest.mark.parametrize('method', ('gausslegendre',
                                    "riemann_left"))
@pytest.mark.parametrize('rcd', (True, False))
@pytest.mark.parametrize('rp', (True, False))
@pytest.mark.parametrize('fn', (None, ['feat_{}'.format(i) for i in range(4)]))
@pytest.mark.parametrize('layer_nb', (None, 1))
def test_layer_integratedgradients(hacky_class, method, rcd, rp, fn, layer_nb):

    model = hacky_class
    if layer_nb is not None:
        layer = model.layers[layer_nb]
    else:
        layer = None

    ig = IntegratedGradients(model, layer=layer,
                             n_steps=50, method=method)

    explanations = ig.explain(X_test_c,
                              baselines=None,
                              target=test_labels,
                              features_names=fn,
                              return_convergence_delta=rcd,
                              return_predictions=rp)

    assert isinstance(explanations, Explanation)
    if layer is not None:
        layer_out = layer(X_test_c).numpy()
        assert explanations['data']['attributions'].shape == layer_out.shape
    else:
        assert explanations['data']['attributions'].shape == X_test_c.shape
    if rcd:
        assert 'deltas' in explanations['data'].keys()
        assert explanations['data']['deltas'].shape[0] == X_test_c.shape[0]
    if rp:
        assert 'predictions' in explanations['data'].keys()
        assert explanations['data']['predictions'].shape[0] == X_test_c.shape[0]
    if fn is not None:
        assert len(fn) == X_test_c.reshape(X_test_c.shape[0], -1).shape[1]


@pytest.mark.eager
@pytest.mark.parametrize('method', ('gausslegendre',
                                    "riemann_left"))
@pytest.mark.parametrize('rcd', (True, False))
@pytest.mark.parametrize('rp', (True, False))
@pytest.mark.parametrize('fn', (None, ['feat_{}'.format(i) for i in range(4)]))
def test_integratedgradients_reg(hacky_reg, method, rcd, rp, fn):
    model = hacky_reg
    ig = IntegratedGradients(model, n_steps=50, method=method)

    explanations = ig.explain(X_test_r,
                              baselines=None,
                              target=None,
                              features_names=fn,
                              return_convergence_delta=rcd,
                              return_predictions=rp)

    assert isinstance(explanations, Explanation)
    assert explanations['data']['attributions'].shape == X_test_r.shape
    if rcd:
        assert 'deltas' in explanations['data'].keys()
        assert explanations['data']['deltas'].shape[0] == X_test_r.shape[0]
    if rp:
        assert 'predictions' in explanations['data'].keys()
        assert explanations['data']['predictions'].shape[0] == X_test_r.shape[0]
    if fn is not None:
        assert len(fn) == X_test_r.reshape(X_test_r.shape[0], -1).shape[1]
