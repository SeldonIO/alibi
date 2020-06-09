import numpy as np
import pytest
from alibi.explainers import IntegratedGradients
from alibi.api.interfaces import Explanation
import tensorflow as tf

tf.compat.v1.enable_eager_execution()

# generate some dummy data
N = 100
N_TRAIN = 90
N_FEATURES = 4
X = np.random.rand(N, N_FEATURES)
X_train, X_test = X[:N_TRAIN, :], X[N_TRAIN:, :]

# regression labels
y_regression = X[:, 0] + X[:, 1]
y_train_regression = y_regression[:N_TRAIN]

# classification labels
y_classification_ordinal = (X[:, 0] + X[:, 1] > 1).astype(int)
y_classification_categorical = tf.keras.utils.to_categorical(y_classification_ordinal)

y_train_classification_ordinal = y_classification_ordinal[:N_TRAIN]
y_train_classification_categorical = y_classification_categorical[:N_TRAIN, :]

test_labels = y_classification_ordinal[N_TRAIN:]

# integral method used shouldn't affect wrapper functionality
INTEGRAL_METHODS = ['gausslegendre', 'riemann_middle']


@pytest.fixture()
def ffn_model(request):
    """
    Simple feed-forward model with configurable data, loss funciton, output activation and dimension
    """
    config = request.param
    inputs = tf.keras.Input(shape=config['X_train'].shape[1:])
    x = tf.keras.layers.Dense(20, activation='relu')(inputs)
    outputs = tf.keras.layers.Dense(config['output_dim'], activation=config['activation'])(x)
    if config.get('squash_output', False):
        outputs = tf.keras.layers.Reshape(())(outputs)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(loss=config['loss'],
                  optimizer='adam')

    model.fit(config['X_train'], config['y_train'], epochs=1, batch_size=256, verbose=0)

    return model


@pytest.mark.eager
@pytest.mark.parametrize('ffn_model', [({'output_dim': 2,
                                         'activation': 'softmax',
                                         'loss': 'categorical_crossentropy',
                                         'X_train': X_train,
                                         'y_train': y_train_classification_categorical})], indirect=True)
@pytest.mark.parametrize('method', INTEGRAL_METHODS, ids='method={}'.format)
def test_integrated_gradients_binary_classification(ffn_model, method):
    model = ffn_model
    ig = IntegratedGradients(model, n_steps=50, method=method)

    explanations = ig.explain(X_test,
                              baselines=None,
                              target=test_labels)

    assert isinstance(explanations, Explanation)
    assert explanations['data']['attributions'].shape == X_test.shape

    assert 'deltas' in explanations['data'].keys()
    assert explanations['data']['deltas'].shape[0] == X_test.shape[0]

    assert 'predictions' in explanations['data'].keys()
    assert explanations['data']['predictions'].shape[0] == X_test.shape[0]


@pytest.mark.eager
@pytest.mark.parametrize('ffn_model', [({'output_dim': 1,
                                         'activation': 'sigmoid',
                                         'loss': 'binary_crossentropy',
                                         'X_train': X_train,
                                         'y_train': y_train_classification_ordinal})], indirect=True)
@pytest.mark.parametrize('method', INTEGRAL_METHODS)
def test_integrated_gradients_binary_classification_single_output(ffn_model, method):
    model = ffn_model
    ig = IntegratedGradients(model, n_steps=50, method=method)

    explanations = ig.explain(X_test,
                              baselines=None,
                              target=test_labels)

    assert isinstance(explanations, Explanation)
    assert explanations['data']['attributions'].shape == X_test.shape

    assert 'deltas' in explanations['data'].keys()
    assert explanations['data']['deltas'].shape[0] == X_test.shape[0]

    assert 'predictions' in explanations['data'].keys()
    assert explanations['data']['predictions'].shape[0] == X_test.shape[0]


@pytest.mark.eager
@pytest.mark.parametrize('ffn_model', [({'output_dim': 1,
                                         'activation': 'sigmoid',
                                         'loss': 'binary_crossentropy',
                                         'X_train': X_train,
                                         'y_train': y_train_classification_ordinal,
                                         'squash_output': True})], indirect=True)
@pytest.mark.parametrize('method', INTEGRAL_METHODS)
def test_integrated_gradients_binary_classification_single_output_squash_output(ffn_model, method):
    model = ffn_model
    ig = IntegratedGradients(model, n_steps=50, method=method)

    explanations = ig.explain(X_test,
                              baselines=None,
                              target=test_labels)

    assert isinstance(explanations, Explanation)
    assert explanations['data']['attributions'].shape == X_test.shape

    assert 'deltas' in explanations['data'].keys()
    assert explanations['data']['deltas'].shape[0] == X_test.shape[0]

    assert 'predictions' in explanations['data'].keys()
    assert explanations['data']['predictions'].shape[0] == X_test.shape[0]


@pytest.mark.eager
@pytest.mark.parametrize('ffn_model', [({'output_dim': 2,
                                         'activation': 'softmax',
                                         'loss': 'categorical_crossentropy',
                                         'X_train': X_train,
                                         'y_train': y_train_classification_categorical})], indirect=True)
@pytest.mark.parametrize('method', INTEGRAL_METHODS)
@pytest.mark.parametrize('layer_nb', (None, 1))
def test_integrated_gradients_binary_classification_layer(ffn_model, method, layer_nb):
    model = ffn_model
    if layer_nb is not None:
        layer = model.layers[layer_nb]
    else:
        layer = None

    ig = IntegratedGradients(model, layer=layer,
                             n_steps=50, method=method)

    explanations = ig.explain(X_test,
                              baselines=None,
                              target=test_labels)

    assert isinstance(explanations, Explanation)
    if layer is not None:
        layer_out = layer(X_test).numpy()
        assert explanations['data']['attributions'].shape == layer_out.shape
    else:
        assert explanations['data']['attributions'].shape == X_test.shape

    assert 'deltas' in explanations['data'].keys()
    assert explanations['data']['deltas'].shape[0] == X_test.shape[0]

    assert 'predictions' in explanations['data'].keys()
    assert explanations['data']['predictions'].shape[0] == X_test.shape[0]


@pytest.mark.eager
@pytest.mark.parametrize('ffn_model', [({'output_dim': 1,
                                         'activation': 'linear',
                                         'loss': 'mean_squared_error',
                                         'X_train': X_train,
                                         'y_train': y_train_regression})], indirect=True)
@pytest.mark.parametrize('method', INTEGRAL_METHODS)
def test_integrated_gradients_regression(ffn_model, method):
    model = ffn_model
    ig = IntegratedGradients(model, n_steps=50, method=method)

    explanations = ig.explain(X_test,
                              baselines=None,
                              target=None)

    assert isinstance(explanations, Explanation)
    assert explanations['data']['attributions'].shape == X_test.shape

    assert 'deltas' in explanations['data'].keys()
    assert explanations['data']['deltas'].shape[0] == X_test.shape[0]

    assert 'predictions' in explanations['data'].keys()
    assert explanations['data']['predictions'].shape[0] == X_test.shape[0]
