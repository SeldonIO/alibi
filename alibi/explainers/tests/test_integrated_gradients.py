import numpy as np
import pytest
from alibi.explainers import IntegratedGradients
from alibi.explainers.integrated_gradients import _run_forward_to_layer, _run_forward_from_layer
from alibi.api.interfaces import Explanation
import tensorflow as tf
from tensorflow.keras import Model

# generate some dummy data
N = 100
N_TRAIN = 90
N_FEATURES = 4
N_TEST = N - N_TRAIN
BASELINES = [None, 1, np.random.rand(N_TEST, N_FEATURES)]

X = np.random.rand(N, N_FEATURES)
X_train, X_test = X[:N_TRAIN, :], X[N_TRAIN:, :]
KWARGS = [None, {'mask': np.zeros(X_test.shape, dtype=np.float32)}]

# multi inputs features
X0 = np.random.rand(N, 10, N_FEATURES)
X_multi_inputs = [X0, X]
X_train_multi_inputs, X_test_multi_inputs = [X0[:N_TRAIN, :], X[:N_TRAIN, :]], [X0[N_TRAIN:, :], X[N_TRAIN:, :]]
BASELINES_MULTI_INPUTS = [None, [1, 1],
                          [np.random.random(X_test_multi_inputs[0].shape),
                           np.random.random(X_test_multi_inputs[1].shape)]]

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
    Simple feed-forward model with configurable data, loss function, output activation and dimension
    """
    config = request.param
    inputs = tf.keras.Input(shape=config['X_train'].shape[1:])
    x = tf.keras.layers.Dense(20, activation='relu')(inputs)
    x = tf.keras.layers.Dense(20, activation='relu')(x)
    outputs = tf.keras.layers.Dense(config['output_dim'], activation=config['activation'])(x)
    if config.get('squash_output', False):
        outputs = tf.keras.layers.Reshape(())(outputs)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(loss=config['loss'],
                  optimizer='adam')

    model.fit(config['X_train'], config['y_train'], epochs=1, batch_size=256, verbose=0)

    return model


@pytest.fixture()
def ffn_model_multi_inputs(request):
    """
    Simple multi-inputs feed-forward model with configurable data, loss function, output activation and dimension
    """
    config = request.param
    input0 = tf.keras.Input(shape=config['X_train_multi_inputs'][0].shape[1:])
    input1 = tf.keras.Input(shape=config['X_train_multi_inputs'][1].shape[1:])

    x = tf.keras.layers.Flatten()(input0)
    x = tf.keras.layers.Concatenate()([x, input1])

    x = tf.keras.layers.Dense(20, activation='relu')(x)
    outputs = tf.keras.layers.Dense(config['output_dim'], activation=config['activation'])(x)
    if config.get('squash_output', False):
        outputs = tf.keras.layers.Reshape(())(outputs)
    model = tf.keras.models.Model(inputs=[input0, input1], outputs=outputs)
    model.compile(loss=config['loss'],
                  optimizer='adam')

    model.fit(config['X_train_multi_inputs'], config['y_train'], epochs=1, batch_size=256, verbose=0)

    return model


@pytest.fixture()
def ffn_model_subclass(request):
    """
    Simple subclassed feed-forward model with configurable data, loss function, output activation and dimension
    """
    config = request.param

    class Linear(Model):

        def __init__(self, output_dim, activation):
            super(Linear, self).__init__()
            self.dense_1 = tf.keras.layers.Dense(20, activation='relu')
            self.dense_2 = tf.keras.layers.Dense(20, activation='relu')
            self.dense_3 = tf.keras.layers.Dense(output_dim, activation)

        def call(self, inputs, mask=None):
            if mask is not None:
                x = tf.math.multiply(inputs, mask)
                x = self.dense_1(x)
            else:
                x = self.dense_1(inputs)
            x = self.dense_2(x)
            outputs = self.dense_3(x)
            return outputs

    model = Linear(config['output_dim'], activation=config['activation'])
    model.compile(loss=config['loss'],
                  optimizer='adam')

    model.fit(config['X_train'], config['y_train'], epochs=1, batch_size=256, verbose=1)

    return model


@pytest.fixture()
def ffn_model_subclass_list_input(request):
    """
    Simple subclassed, multi-input feed-forward model with configurable data,
    loss function, output activation and dimension
    """
    config = request.param

    class Linear(Model):

        def __init__(self, output_dim, activation):
            super(Linear, self).__init__()
            self.flat = tf.keras.layers.Flatten()
            self.concat = tf.keras.layers.Concatenate()
            self.dense_1 = tf.keras.layers.Dense(20, activation='relu')
            self.dense_2 = tf.keras.layers.Dense(output_dim, activation)

        def call(self, inputs):
            inp0 = self.flat(inputs[0])
            inp1 = self.flat(inputs[1])
            x = self.concat([inp0, inp1])
            x = self.dense_1(x)
            outputs = self.dense_2(x)
            return outputs

    model = Linear(config['output_dim'], activation=config['activation'])
    model.compile(loss=config['loss'],
                  optimizer='adam')

    model.fit(config['X_train_multi_inputs'], config['y_train'], epochs=1, batch_size=256, verbose=1)

    return model


@pytest.fixture()
def ffn_model_sequential(request):
    """
    Simple sequential feed-forward model with configurable data, loss function, output activation and dimension
    """
    config = request.param
    layers = [
        tf.keras.layers.InputLayer(input_shape=config['X_train'].shape[1:]),
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dense(config['output_dim'], activation=config['activation'])
        ]
    if config.get('squash_output', False):
        layers.append(tf.keras.layers.Reshape(()))
    model = tf.keras.models.Sequential(layers)
    model.compile(loss=config['loss'],
                  optimizer='adam')

    model.fit(config['X_train'], config['y_train'], epochs=1, batch_size=256, verbose=1)

    return model


@pytest.mark.parametrize('ffn_model_sequential', [({'output_dim': 2,
                                                    'activation': 'softmax',
                                                    'loss': 'categorical_crossentropy',
                                                    'X_train': X_train,
                                                    'y_train': y_train_classification_categorical})], indirect=True)
@pytest.mark.parametrize('method', INTEGRAL_METHODS, ids='method={}'.format)
@pytest.mark.parametrize('baselines', BASELINES)
def test_integrated_gradients_model_sequential(ffn_model_sequential, method, baselines):
    model = ffn_model_sequential
    ig = IntegratedGradients(model, n_steps=50, method=method)

    explanations = ig.explain(X_test,
                              baselines=baselines,
                              target=test_labels)

    assert isinstance(explanations, Explanation)
    assert explanations['data']['attributions'][0].shape == X_test.shape

    assert 'deltas' in explanations['data'].keys()
    assert explanations['data']['deltas'].shape[0] == X_test.shape[0]

    assert 'predictions' in explanations['data'].keys()
    assert explanations['data']['predictions'].shape[0] == X_test.shape[0]


@pytest.mark.parametrize('ffn_model_subclass', [({'output_dim': 2,
                                                  'activation': 'softmax',
                                                  'loss': 'categorical_crossentropy',
                                                  'X_train': X_train,
                                                  'y_train': y_train_classification_categorical})], indirect=True)
@pytest.mark.parametrize('method', INTEGRAL_METHODS, ids='method={}'.format)
@pytest.mark.parametrize('baselines', BASELINES)
@pytest.mark.parametrize('kwargs', KWARGS)
def test_integrated_gradients_model_subclass(ffn_model_subclass, method, baselines, kwargs):
    model = ffn_model_subclass
    ig = IntegratedGradients(model, n_steps=50, method=method)

    explanations = ig.explain(X_test,
                              baselines=baselines,
                              target=test_labels,
                              forward_kwargs=kwargs)

    assert isinstance(explanations, Explanation)
    assert explanations['data']['attributions'][0].shape == X_test.shape

    assert 'deltas' in explanations['data'].keys()
    assert explanations['data']['deltas'].shape[0] == X_test.shape[0]

    assert 'predictions' in explanations['data'].keys()
    assert explanations['data']['predictions'].shape[0] == X_test.shape[0]


@pytest.mark.parametrize('ffn_model_subclass_list_input', [({'output_dim': 2,
                                                             'activation': 'softmax',
                                                             'loss': 'categorical_crossentropy',
                                                             'X_train_multi_inputs': X_train_multi_inputs,
                                                             'y_train': y_train_classification_categorical})],
                         indirect=True)
@pytest.mark.parametrize('method', INTEGRAL_METHODS, ids='method={}'.format)
@pytest.mark.parametrize('baselines', BASELINES_MULTI_INPUTS)
def test_integrated_gradients_model_subclass_list_input(ffn_model_subclass_list_input, method, baselines):
    model = ffn_model_subclass_list_input
    ig = IntegratedGradients(model, n_steps=50, method=method)

    explanations = ig.explain(X_test_multi_inputs,
                              baselines=baselines,
                              target=test_labels)

    assert isinstance(explanations, Explanation)
    assert max([len(x) for x in X_test_multi_inputs]) == min([len(x) for x in X_test_multi_inputs])
    assert (max([len(x) for x in explanations['data']['attributions']]) ==
            min([len(x) for x in explanations['data']['attributions']]))
    assert len(explanations['data']['attributions'][0]) == N_TEST
    assert len(X_test_multi_inputs[0]) == N_TEST

    attrs = explanations['data']['attributions']
    for i in range(len(attrs)):
        assert attrs[i].shape == X_test_multi_inputs[i].shape

    assert 'deltas' in explanations['data'].keys()
    assert explanations['data']['deltas'].shape[0] == N_TEST

    assert 'predictions' in explanations['data'].keys()
    assert explanations['data']['predictions'].shape[0] == N_TEST


@pytest.mark.parametrize('ffn_model_multi_inputs', [({'output_dim': 2,
                                                      'activation': 'softmax',
                                                      'loss': 'categorical_crossentropy',
                                                      'X_train_multi_inputs': X_train_multi_inputs,
                                                      'y_train': y_train_classification_categorical})], indirect=True)
@pytest.mark.parametrize('method', INTEGRAL_METHODS, ids='method={}'.format)
@pytest.mark.parametrize('baselines', BASELINES_MULTI_INPUTS)
def test_integrated_gradients_binary_classification_multi_inputs(ffn_model_multi_inputs, method, baselines):
    model = ffn_model_multi_inputs
    ig = IntegratedGradients(model, n_steps=50, method=method)

    explanations = ig.explain(X_test_multi_inputs,
                              baselines=baselines,
                              target=test_labels)

    assert isinstance(explanations, Explanation)
    assert max([len(x) for x in X_test_multi_inputs]) == min([len(x) for x in X_test_multi_inputs])
    assert (max([len(x) for x in explanations['data']['attributions']]) ==
            min([len(x) for x in explanations['data']['attributions']]))
    assert len(explanations['data']['attributions'][0]) == N_TEST
    assert len(X_test_multi_inputs[0]) == N_TEST

    attrs = explanations['data']['attributions']
    for i in range(len(attrs)):
        assert attrs[i].shape == X_test_multi_inputs[i].shape

    assert 'deltas' in explanations['data'].keys()
    assert explanations['data']['deltas'].shape[0] == N_TEST

    assert 'predictions' in explanations['data'].keys()
    assert explanations['data']['predictions'].shape[0] == N_TEST


@pytest.mark.parametrize('ffn_model_multi_inputs', [({'output_dim': 1,
                                                      'activation': 'sigmoid',
                                                      'loss': 'binary_crossentropy',
                                                      'X_train_multi_inputs': X_train_multi_inputs,
                                                      'y_train': y_train_classification_ordinal})], indirect=True)
@pytest.mark.parametrize('method', INTEGRAL_METHODS)
@pytest.mark.parametrize('baselines', BASELINES_MULTI_INPUTS)
def test_integrated_gradients_binary_classification_single_output_multi_inputs(ffn_model_multi_inputs,
                                                                               method,
                                                                               baselines):
    model = ffn_model_multi_inputs
    ig = IntegratedGradients(model, n_steps=50, method=method)

    explanations = ig.explain(X_test_multi_inputs,
                              baselines=baselines,
                              target=test_labels)

    assert isinstance(explanations, Explanation)
    assert max([len(x) for x in X_test_multi_inputs]) == min([len(x) for x in X_test_multi_inputs])
    assert (max([len(x) for x in explanations['data']['attributions']]) ==
            min([len(x) for x in explanations['data']['attributions']]))
    assert len(explanations['data']['attributions'][0]) == N_TEST
    assert len(X_test_multi_inputs[0]) == N_TEST

    attrs = explanations['data']['attributions']
    for i in range(len(attrs)):
        assert attrs[i].shape == X_test_multi_inputs[i].shape

    assert 'deltas' in explanations['data'].keys()
    assert explanations['data']['deltas'].shape[0] == N_TEST

    assert 'predictions' in explanations['data'].keys()
    assert explanations['data']['predictions'].shape[0] == N_TEST


@pytest.mark.parametrize('ffn_model', [({'output_dim': 2,
                                         'activation': 'softmax',
                                         'loss': 'categorical_crossentropy',
                                         'X_train': X_train,
                                         'y_train': y_train_classification_categorical})], indirect=True)
@pytest.mark.parametrize('method', INTEGRAL_METHODS, ids='method={}'.format)
@pytest.mark.parametrize('baselines', BASELINES)
def test_integrated_gradients_binary_classification(ffn_model, method, baselines):
    model = ffn_model
    ig = IntegratedGradients(model, n_steps=50, method=method)

    explanations = ig.explain(X_test,
                              baselines=baselines,
                              target=test_labels)

    assert isinstance(explanations, Explanation)
    assert explanations['data']['attributions'][0].shape == X_test.shape

    assert 'deltas' in explanations['data'].keys()
    assert explanations['data']['deltas'].shape[0] == X_test.shape[0]

    assert 'predictions' in explanations['data'].keys()
    assert explanations['data']['predictions'].shape[0] == X_test.shape[0]


@pytest.mark.parametrize('ffn_model', [({'output_dim': 1,
                                         'activation': 'sigmoid',
                                         'loss': 'binary_crossentropy',
                                         'X_train': X_train,
                                         'y_train': y_train_classification_ordinal})], indirect=True)
@pytest.mark.parametrize('method', INTEGRAL_METHODS)
@pytest.mark.parametrize('baselines', BASELINES)
def test_integrated_gradients_binary_classification_single_output(ffn_model, method, baselines):
    model = ffn_model
    ig = IntegratedGradients(model, n_steps=50, method=method)

    explanations = ig.explain(X_test,
                              baselines=baselines,
                              target=test_labels)

    assert isinstance(explanations, Explanation)
    assert explanations['data']['attributions'][0].shape == X_test.shape

    assert 'deltas' in explanations['data'].keys()
    assert explanations['data']['deltas'].shape[0] == X_test.shape[0]

    assert 'predictions' in explanations['data'].keys()
    assert explanations['data']['predictions'].shape[0] == X_test.shape[0]


@pytest.mark.parametrize('ffn_model', [({'output_dim': 1,
                                         'activation': 'sigmoid',
                                         'loss': 'binary_crossentropy',
                                         'X_train': X_train,
                                         'y_train': y_train_classification_ordinal,
                                         'squash_output': True})], indirect=True)
@pytest.mark.parametrize('method', INTEGRAL_METHODS)
@pytest.mark.parametrize('baselines', BASELINES)
def test_integrated_gradients_binary_classification_single_output_squash_output(ffn_model, method, baselines):
    model = ffn_model
    ig = IntegratedGradients(model, n_steps=50, method=method)

    explanations = ig.explain(X_test,
                              baselines=baselines,
                              target=test_labels)

    assert isinstance(explanations, Explanation)
    assert explanations['data']['attributions'][0].shape == X_test.shape

    assert 'deltas' in explanations['data'].keys()
    assert explanations['data']['deltas'].shape[0] == X_test.shape[0]

    assert 'predictions' in explanations['data'].keys()
    assert explanations['data']['predictions'].shape[0] == X_test.shape[0]


@pytest.mark.parametrize('ffn_model', [({'output_dim': 2,
                                         'activation': 'softmax',
                                         'loss': 'categorical_crossentropy',
                                         'X_train': X_train,
                                         'y_train': y_train_classification_categorical})], indirect=True)
@pytest.mark.parametrize('method', INTEGRAL_METHODS)
@pytest.mark.parametrize('layer_nb', (None, 1))
@pytest.mark.parametrize('baselines', BASELINES)
@pytest.mark.parametrize('layer_inputs_attributions', (False, True))
def test_integrated_gradients_binary_classification_layer(ffn_model,
                                                          method,
                                                          layer_nb,
                                                          baselines,
                                                          layer_inputs_attributions):
    model = ffn_model
    if layer_nb is not None:
        layer = model.layers[layer_nb]
    else:
        layer = None

    ig = IntegratedGradients(model, layer=layer,
                             n_steps=50, method=method)

    explanations = ig.explain(X_test,
                              baselines=baselines,
                              target=test_labels,
                              attribute_to_layer_inputs=layer_inputs_attributions)

    assert isinstance(explanations, Explanation)
    if layer is not None:
        orig_call = layer.call
        layer_out = _run_forward_to_layer(model,
                                          layer,
                                          orig_call,
                                          X_test,
                                          run_to_layer_inputs=layer_inputs_attributions)

        if isinstance(layer_out, tuple):
            for i in range(len(layer_out)):
                assert explanations['data']['attributions'][i].shape == layer_out[i].shape
        else:
            assert explanations['data']['attributions'][0].shape == layer_out.shape
    elif layer is None:
        assert explanations['data']['attributions'][0].shape == X_test.shape

    assert 'deltas' in explanations['data'].keys()
    assert explanations['data']['deltas'].shape[0] == X_test.shape[0]

    assert 'predictions' in explanations['data'].keys()
    assert explanations['data']['predictions'].shape[0] == X_test.shape[0]


@pytest.mark.parametrize('ffn_model_subclass', [({'output_dim': 2,
                                                  'activation': 'softmax',
                                                  'loss': 'categorical_crossentropy',
                                                  'X_train': X_train,
                                                  'y_train': y_train_classification_categorical})], indirect=True)
@pytest.mark.parametrize('method', INTEGRAL_METHODS)
@pytest.mark.parametrize('layer_nb', (None, 1))
@pytest.mark.parametrize('baselines', BASELINES)
@pytest.mark.parametrize('kwargs', KWARGS)
@pytest.mark.parametrize('layer_inputs_attributions', (False, True))
def test_integrated_gradients_binary_classification_layer_subclass(ffn_model_subclass,
                                                                   method,
                                                                   layer_nb,
                                                                   baselines,
                                                                   kwargs,
                                                                   layer_inputs_attributions):
    model = ffn_model_subclass
    if layer_nb is not None:
        layer = model.layers[layer_nb]
    else:
        layer = None

    ig = IntegratedGradients(model, layer=layer,
                             n_steps=50, method=method)

    explanations = ig.explain(X_test,
                              baselines=baselines,
                              target=test_labels,
                              forward_kwargs=kwargs,
                              attribute_to_layer_inputs=layer_inputs_attributions)

    assert isinstance(explanations, Explanation)
    if layer is not None:
        orig_call = layer.call
        layer_out = _run_forward_to_layer(model,
                                          layer,
                                          orig_call,
                                          X_test,
                                          run_to_layer_inputs=layer_inputs_attributions)

        if isinstance(layer_out, tuple):
            for i in range(len(layer_out)):
                assert explanations['data']['attributions'][i].shape == layer_out[i].shape
        else:
            assert explanations['data']['attributions'][0].shape == layer_out.shape
    elif layer is None:
        assert explanations['data']['attributions'][0].shape == X_test.shape

    assert 'deltas' in explanations['data'].keys()
    assert explanations['data']['deltas'].shape[0] == X_test.shape[0]

    assert 'predictions' in explanations['data'].keys()
    assert explanations['data']['predictions'].shape[0] == X_test.shape[0]


@pytest.mark.parametrize('ffn_model', [({'output_dim': 1,
                                         'activation': 'linear',
                                         'loss': 'mean_squared_error',
                                         'X_train': X_train,
                                         'y_train': y_train_regression})], indirect=True)
@pytest.mark.parametrize('method', INTEGRAL_METHODS)
@pytest.mark.parametrize('baselines', BASELINES)
def test_integrated_gradients_regression(ffn_model, method, baselines):
    model = ffn_model
    ig = IntegratedGradients(model, n_steps=50, method=method)

    explanations = ig.explain(X_test,
                              baselines=baselines,
                              target=None)

    assert isinstance(explanations, Explanation)
    assert explanations['data']['attributions'][0].shape == X_test.shape

    assert 'deltas' in explanations['data'].keys()
    assert explanations['data']['deltas'].shape[0] == X_test.shape[0]

    assert 'predictions' in explanations['data'].keys()
    assert explanations['data']['predictions'].shape[0] == X_test.shape[0]


@pytest.mark.parametrize('layer_nb', (1, ))
@pytest.mark.parametrize('run_from_layer_inputs', (False, True))
def test_run_forward_from_layer(layer_nb,
                                run_from_layer_inputs):

    # One layer ffn with all weights = 1.
    inputs = tf.keras.Input(shape=(16,))
    out = tf.keras.layers.Dense(8,
                                kernel_initializer=tf.keras.initializers.Ones(),
                                name='linear1')(inputs)
    out = tf.keras.layers.Dense(1,
                                kernel_initializer=tf.keras.initializers.Ones(),
                                name='linear3')(out)
    model = tf.keras.Model(inputs=inputs, outputs=out)

    # Select layer
    layer = model.layers[layer_nb]
    orig_call = layer.call

    dummy_input = np.zeros((1, 16))

    if run_from_layer_inputs:
        x_layer = [tf.convert_to_tensor(np.ones((2, ) + (layer.input_shape[1:])))]
        expected_shape = (2, 1)
        expected_values = 128
    else:
        x_layer = tf.convert_to_tensor(np.ones((3, ) + (layer.output_shape[1:])))
        expected_shape = (3, 1)
        expected_values = 8

    preds_from_layer = _run_forward_from_layer(model,
                                               layer,
                                               orig_call,
                                               dummy_input,
                                               x_layer,
                                               None,
                                               run_from_layer_inputs=run_from_layer_inputs,
                                               select_target=False)
    preds_from_layer = preds_from_layer.numpy()

    assert preds_from_layer.shape == expected_shape
    assert np.allclose(preds_from_layer, expected_values)

#####################################################################################################################


@pytest.mark.skip(reason='Not testing as multi-layers will not be supported in the future')
@pytest.mark.parametrize('ffn_model', [({'output_dim': 2,
                                         'activation': 'softmax',
                                         'loss': 'categorical_crossentropy',
                                         'X_train': X_train,
                                         'y_train': y_train_classification_categorical})], indirect=True)
@pytest.mark.parametrize('method', INTEGRAL_METHODS)
@pytest.mark.parametrize('baselines', BASELINES)
def test_integrated_gradients_binary_classification_multi_layer(ffn_model, method, baselines):
    model = ffn_model

    layer = [model.layers[1], model.layers[2]]

    ig = IntegratedGradients(model, layer=layer,
                             n_steps=50, method=method)

    explanations = ig.explain(X_test,
                              baselines=baselines,
                              target=test_labels)

    assert isinstance(explanations, Explanation)


@pytest.mark.skip(reason='Not testing as multi-layers will not be supported in the future')
@pytest.mark.parametrize('ffn_model_subclass', [({'output_dim': 2,
                                                  'activation': 'softmax',
                                                  'loss': 'categorical_crossentropy',
                                                  'X_train': X_train,
                                                  'y_train': y_train_classification_categorical})], indirect=True)
@pytest.mark.parametrize('method', INTEGRAL_METHODS)
@pytest.mark.parametrize('baselines', BASELINES)
def test_integrated_gradients_binary_classification_multi_layer_subclassed(ffn_model_subclass, method, baselines):
    model = ffn_model_subclass

    layer = [model.layers[0], model.layers[1]]

    ig = IntegratedGradients(model, layer=layer,
                             n_steps=50, method=method)

    explanations = ig.explain(X_test,
                              baselines=baselines,
                              target=test_labels)

    assert isinstance(explanations, Explanation)
