import pytest
from pytest_lazyfixture import lazy_fixture

import numpy as np
from numpy.testing import assert_allclose
from typing import Union, List

import tensorflow as tf
import tensorflow.keras as keras
from alibi.explainers import CounterfactualRLTabular
from alibi.explainers.backends.cfrl_base import get_hard_distribution
from alibi.explainers.backends.cfrl_tabular import get_he_preprocessor, split_ohe, get_numerical_conditional_vector,\
    get_categorical_conditional_vector, get_statistics, get_conditional_vector, sample


@pytest.mark.parametrize('dataset', [lazy_fixture('iris_data'),
                                     lazy_fixture('adult_data'),
                                     lazy_fixture('boston_data')])
def test_he_preprocessor(dataset):
    """ Test the heterogeneous preprocessor and inverse preprocessor. """
    # Unpack dataset.
    X = dataset["X_train"]
    feature_names = dataset["metadata"]["feature_names"]
    category_map = dataset["metadata"].get("category_map", dict())

    # Get heterogeneous preprocessor.
    preprocessor, inv_preprocessor = get_he_preprocessor(X=X, feature_names=feature_names, category_map=category_map)

    # Preprocess the dataset.
    X_ohe = preprocessor(X)

    # Check if the number of columns in the is the one expected.
    numerical_cols = [i for i in range(len(feature_names)) if i not in category_map]
    total_num_cols = len(numerical_cols) + sum([len(v) for v in category_map.values()])
    assert total_num_cols == X_ohe.shape[1]

    # Test if the inverse preprocessor maps the ohe back to the original input.
    inv_X = inv_preprocessor(X_ohe)
    assert_allclose(X.astype(np.float32), inv_X.astype(np.float32))


@pytest.mark.parametrize('dataset', [lazy_fixture("iris_data"),
                                     lazy_fixture("adult_data"),
                                     lazy_fixture("boston_data")])
def test_split_ohe(dataset):
    """ Test the one-hot encoding splitting of a dataset. """

    # Unpack data
    X = dataset["X_train"]
    feature_names = dataset["metadata"]["feature_names"]
    category_map = dataset["metadata"].get("category_map", {})

    # Get heterogeneous preprocessor
    preprocessor, _ = get_he_preprocessor(X=X, feature_names=feature_names, category_map=category_map)

    # Preprocess the dataset
    X_ohe = preprocessor(X)

    # Split the ohe representation in multiple heads
    num_heads, cat_heads = split_ohe(X_ohe=X_ohe, category_map=category_map)

    # Check the number of numerical heads.
    if len(category_map) < len(feature_names):
        assert len(num_heads) == 1
    else:
        assert len(num_heads) == 0

    # Check the number of categorical heads.
    assert len(cat_heads) == len(category_map)


@pytest.mark.parametrize('dataset', [lazy_fixture("iris_data"),
                                     lazy_fixture("adult_data"),
                                     lazy_fixture("boston_data")])
def test_get_numerical_condition(dataset):
    """ Test the training numerical conditional generator. """

    # Unpack dataset
    X = dataset["X_train"]
    feature_names = dataset["metadata"]["feature_names"]
    category_map = dataset["metadata"].get("category_map", {})

    # Get heterogeneous preprocessor
    preprocessor, _ = get_he_preprocessor(X=X, feature_names=feature_names, category_map=category_map)

    # Define numerical ranges.
    ranges = {}

    for i, fn in enumerate(feature_names):
        if i not in category_map:
            ranges[fn] = [-np.random.rand(), np.random.rand()]

    # Compute dataset statistics
    stats = get_statistics(X=X, preprocessor=preprocessor, category_map=category_map)

    # Generate numerical conditional vector
    C = get_numerical_conditional_vector(X=X[:5],
                                         condition={},
                                         preprocessor=preprocessor,
                                         feature_names=feature_names,
                                         category_map=category_map,
                                         stats=stats,
                                         ranges=ranges,
                                         immutable_features=[])

    # Check that values in C are included in the intervals defined in ranges.
    numerical_features = [feature_names[i] for i in range(len(feature_names)) if i not in category_map]

    if numerical_features:
        for i, nf in enumerate(numerical_features):
            assert np.all(C[2 * i] >= ranges[nf][0])
            assert np.all(C[2 * i + 1] <= ranges[nf][1])
    else:
        assert len(C) == 0


@pytest.mark.parametrize('dataset', [lazy_fixture("iris_data"),
                                     lazy_fixture("adult_data"),
                                     lazy_fixture("boston_data")])
def test_get_categorical_condition(dataset):
    """ Test the training categorical conditional generator. """

    # Unpack dataset
    X = dataset["X_train"]
    feature_names = dataset["metadata"]["feature_names"]
    category_map = dataset["metadata"].get("category_map", {})

    # Get heterogeneous preprocessor
    preprocessor, _ = get_he_preprocessor(X=X, feature_names=feature_names, category_map=category_map)

    # Generate set of immutable features
    categorical_cols = [feature_names[i] for i in category_map]

    if len(category_map):
        immutable_features = np.random.choice(categorical_cols,
                                              size=np.random.randint(low=0, high=len(categorical_cols)))
    else:
        immutable_features = []

    # Generate conditional vector
    C = get_categorical_conditional_vector(X=X,
                                           condition={},
                                           preprocessor=preprocessor,
                                           feature_names=feature_names,
                                           category_map=category_map,
                                           immutable_features=immutable_features)

    if category_map:
        for i, fn in zip(range(len(C)), categorical_cols):
            # Check immutable features.
            if fn in immutable_features:
                assert np.all(np.sum(C[i], axis=1) == 1)
            else:
                # Check that the masks contain the original value
                assert np.all(C[i][np.arange(X.shape[0]), X[:, feature_names.index(fn)]] == 1)

    else:
        assert len(C) == 0


@pytest.mark.parametrize('seed', [0, 1, 2, 3])
@pytest.mark.parametrize('dataset', [lazy_fixture("iris_data"),
                                     lazy_fixture("adult_data"),
                                     lazy_fixture("boston_data")])
def test_sample(dataset, seed):
    """ Test sampling reconstruction. """

    # Unpack dataset.
    X = dataset["X_train"]
    feature_names = dataset["metadata"]["feature_names"]
    category_map = dataset["metadata"].get("category_map", {})

    # Get heterogeneous preprocessor.
    preprocessor, inv_preprocessor = get_he_preprocessor(X=X, feature_names=feature_names, category_map=category_map)

    # Preprocess dataset.
    X_ohe = preprocessor(X)
    X_ohe_split_num, X_ohe_split_cat = split_ohe(X_ohe=X_ohe, category_map=category_map)
    X_ohe_split = X_ohe_split_num + X_ohe_split_cat

    # Compute dataset statistics.
    stats = get_statistics(X=X, preprocessor=preprocessor, category_map=category_map)

    # Define numerical ranges.
    ranges = {}
    for i, fn in enumerate(feature_names):
        if i not in category_map:
            ranges[fn] = [-np.random.rand(), np.random.rand()]

    # Define list of immutable attributes.
    np.random.seed(seed)
    immutable_attributes = np.random.choice(feature_names, size=np.random.randint(low=0, high=len(feature_names)))

    # Generate conditional vector.
    C = get_conditional_vector(X=X,
                               condition={},
                               preprocessor=preprocessor,
                               feature_names=feature_names,
                               category_map=category_map,
                               stats=stats,
                               ranges=ranges,
                               immutable_features=immutable_attributes)

    # Generate some random reconstruction
    X_hat = np.random.randn(*X_ohe.shape)
    X_hat_split_num, X_hat_split_cat = split_ohe(X_ohe=X_hat, category_map=category_map)
    X_hat_split = X_hat_split_num + X_hat_split_cat

    # Sample reconstruction.
    X_hat_ohe_split = sample(X_hat_split=X_hat_split,
                             X_ohe=X_ohe,
                             C=C,
                             category_map=category_map,
                             stats=stats)

    # Check that all sampled categorical features are one-hot encoded.
    offset = 1 if len(feature_names) > len(category_map) else 0
    for i in range(offset, len(X_ohe_split)):
        assert np.all(np.sum(X_hat_ohe_split[i], axis=1) == 1)

    # Check that the immutable features did not change.
    X_hat = inv_preprocessor(np.concatenate(X_hat_ohe_split, axis=1))
    for i, fn in enumerate(feature_names):
        if fn in immutable_attributes:
            assert_allclose(X[:, i].astype(np.float32), X_hat[:, i].astype(np.float32))


@pytest.mark.parametrize('Y_shape, num_classes', [(5, None), (10, None), (1, 5)])
def test_hard_distribution(Y_shape, num_classes):
    """ Test transforming a soft labels or label encodings to one-hot encoding. """
    if Y_shape > 1:
        Y = np.random.randn(10, Y_shape)
        Y_ohe = get_hard_distribution(Y)
    else:
        Y = np.random.randint(low=0, high=num_classes, size=10)
        Y_ohe = get_hard_distribution(Y, num_classes=num_classes)

    assert np.all(np.sum(Y_ohe, axis=1) == 1)


@pytest.fixture
def tf_keras_iris_explainer(models, iris_data, rf_classifier):
    # Define explainer constants
    LATENT_DIM = 2
    COEFF_SPARSITY = 0.0
    COEFF_CONSISTENCY = 0.0
    TRAIN_STEPS = 100
    BATCH_SIZE = 100

    # Define encoder. `tanh` is added to work with DDPG.
    encoder = keras.Sequential([
        models[1],
        keras.layers.Activation(tf.math.tanh)
    ])

    # Define decoder. `atanh` is added to work with DDPG
    decoder = keras.Sequential([
        keras.layers.Activation(tf.math.atan),
        models[0].layers[2]
    ])

    # need to define a wrapper for the decoder to return a list of tensors
    class DecoderList(tf.keras.Model):
        def __init__(self, decoder: tf.keras.Model, **kwargs):
            super().__init__(**kwargs)
            self.decoder = decoder

        def call(self, input: Union[tf.Tensor, List[tf.Tensor]], **kwargs):
            return [self.decoder(input, **kwargs)]

    # Redefine the call method to return a list of tensors.
    decoder = DecoderList(decoder)

    # Define predictor.
    predictor = lambda x: rf_classifier[0].predict_proba(x)  # noqa: E731

    # Define explainer.
    explainer = CounterfactualRLTabular(encoder=encoder,
                                        decoder=decoder,
                                        latent_dim=LATENT_DIM,
                                        encoder_preprocessor=lambda x: x,
                                        decoder_inv_preprocessor=lambda x: x,
                                        predictor=predictor,
                                        coeff_sparsity=COEFF_SPARSITY,
                                        coeff_consistency=COEFF_CONSISTENCY,
                                        category_map=iris_data["metadata"].get("category_map", dict()),
                                        feature_names=iris_data["metadata"].get("feature_names"),
                                        ranges=dict(),
                                        immutable_features=[],
                                        conditional_func=lambda x: None,
                                        train_steps=TRAIN_STEPS,
                                        batch_size=BATCH_SIZE,
                                        backend="tensorflow")

    return explainer


@pytest.mark.parametrize('models', [('iris-ae-tf2.2.0', 'iris-enc-tf2.2.0')], ids='model={}'.format, indirect=True)
@pytest.mark.parametrize('rf_classifier',
                         [lazy_fixture('iris_data')],
                         indirect=True,
                         ids='clf=rf_{}'.format)
def test_explainer(tf_keras_iris_explainer, iris_data):
    explainer = tf_keras_iris_explainer

    # Check that the encoding representation is between [-1, 1]
    encoder_preprocessor = explainer.params["encoder_preprocessor"]
    encoder = explainer.params["encoder"]
    decoder = explainer.params["decoder"]

    Z = encoder(encoder_preprocessor(iris_data["X_train"]))
    assert tf.math.reduce_min(Z) >= -1 and tf.math.reduce_max(Z) <= 1

    X_hat = decoder(Z)
    assert isinstance(X_hat, list)

    # Fit the explainer
    explainer.fit(X=iris_data["X_train"])

    # Construct explanation object.
    explainer.explain(X=iris_data["X_test"], Y_t=np.array([2]), C=None)


@pytest.mark.parametrize('backend', ['tensorflow', 'pytorch'])
@pytest.mark.parametrize('dataset', [lazy_fixture("adult_data")])
def test_sample_differentiable(dataset, backend):
    SIZE = 100
    feature_names = dataset["metadata"]["feature_names"]
    category_map = dataset["metadata"].get("category_map", {})

    # define number of numerical
    num_features = len(feature_names) - len(category_map)

    # define random autoencoder reconstruction
    X_hat_split = []

    # generate numerical feature reconstruction
    X_hat_split.append(np.random.randn(SIZE, num_features).astype(np.float32))

    # for each categorical feature generate random reconstruction
    for cat_col in category_map:
        X_hat_split.append(np.random.rand(SIZE, len(category_map[cat_col])).astype(np.float32))

    if backend == "tensorflow":
        from alibi.explainers.backends.tensorflow.cfrl_base import to_tensor, to_numpy
        from alibi.explainers.backends.tensorflow.cfrl_tabular import sample_differentiable
        device = None
    else:
        import torch
        from alibi.explainers.backends.pytorch.cfrl_base import to_tensor, to_numpy
        from alibi.explainers.backends.pytorch.cfrl_tabular import sample_differentiable
        device = torch.device("cpu")

    for i in range(len(X_hat_split)):
        X_hat_split[i] = to_tensor(X_hat_split[i], device=device)

    # sample output differentiable
    X_ohe_split = sample_differentiable(X_hat_split, category_map)

    # convert back to numpy arrays
    X_ohe_split = to_numpy(X_ohe_split)
    X_hat_split = to_numpy(X_hat_split)

    # check if the numerical feature are unchanged
    assert_allclose(X_ohe_split[0], X_hat_split[0])

    # check the categorical ones
    for i in range(1, len(X_ohe_split)):
        assert np.all(np.argmax(X_ohe_split[i], axis=1) == np.argmax(X_hat_split[i], axis=1))
        assert np.all(X_ohe_split[i][np.arange(SIZE), np.argmax(X_ohe_split[i], axis=1)] == 1)
        assert np.all(np.sum(X_ohe_split[i], axis=1) == 1)


@pytest.mark.parametrize('backend', ['tensorflow', 'pytorch'])
@pytest.mark.parametrize('reduction', ['none', 'mean', 'sum'])
def test_l0_loss(reduction, backend):
    NUM_CLASSES = 5
    NUM_SAMPLES = 1000

    def generate_random_labels():
        y = np.random.randint(0, NUM_CLASSES, size=NUM_SAMPLES)
        y_ohe = np.zeros((NUM_SAMPLES, NUM_CLASSES), dtype=np.float32)
        y_ohe[np.arange(NUM_SAMPLES), y] = 1
        assert np.all(np.sum(y_ohe, axis=1) == 1)
        return y, y_ohe

    y1, y1_ohe = generate_random_labels()
    y2, y2_ohe = generate_random_labels()

    if backend == 'tensorflow':
        from alibi.explainers.backends.tensorflow.cfrl_base import to_tensor, to_numpy
        from alibi.explainers.backends.tensorflow.cfrl_tabular import l0_ohe
        device = None
    else:
        import torch
        from alibi.explainers.backends.pytorch.cfrl_base import to_tensor, to_numpy
        from alibi.explainers.backends.pytorch.cfrl_tabular import l0_ohe
        device = torch.device("cpu")

    y1_ohe, y2_ohe = to_tensor(y1_ohe, device=device), to_tensor(y2_ohe, device=device)
    l0_ohe_loss = l0_ohe(y1_ohe, y2_ohe, reduction=reduction)
    l0_ohe_loss = to_numpy(l0_ohe_loss)
    l0_loss = (y1 != y2)

    if reduction == 'none':
        assert_allclose(l0_loss, l0_ohe_loss, atol=1e-5)
    elif reduction == 'sum':
        assert np.isclose(np.mean(l0_loss), l0_ohe_loss / NUM_SAMPLES)
    else:
        assert np.isclose(np.mean(l0_loss), l0_ohe_loss)


@pytest.mark.parametrize('backend', ['tensorflow', 'pytorch'])
@pytest.mark.parametrize('reduction', ['none', 'mean', 'sum'])
def test_l1_loss(reduction, backend):
    NUM_SAMPLES = 100
    SIZE = 10

    # generate random tensors
    y1 = np.random.randn(NUM_SAMPLES, SIZE)
    y2 = np.random.randn(NUM_SAMPLES, SIZE)

    if backend == 'tensorflow':
        from alibi.explainers.backends.tensorflow.cfrl_tabular import l1_loss
        from alibi.explainers.backends.tensorflow.cfrl_base import to_tensor, to_numpy
        device = None
    else:
        import torch
        from alibi.explainers.backends.pytorch.cfrl_tabular import l1_loss
        from alibi.explainers.backends.pytorch.cfrl_base import to_tensor, to_numpy
        device = torch.device("cpu")

    y1_tensor, y2_tensor = to_tensor(y1, device=device), to_tensor(y2, device=device)
    l1_backend = l1_loss(y1_tensor, y2_tensor, reduction=reduction)
    l1_backend = to_numpy(l1_backend)
    l1 = np.abs(y1 - y2)

    if reduction == 'none':
        assert_allclose(l1, l1_backend)
    elif reduction == 'sum':
        assert np.isclose(np.sum(l1), l1_backend)
    else:
        assert np.isclose(np.mean(l1), l1_backend)


@pytest.mark.parametrize('backend', ['tensorflow', 'pytorch'])
def test_consistency_loss(backend):
    NUM_SAMPLES = 100
    SIZE = 10

    z1 = np.random.randn(NUM_SAMPLES, SIZE)
    z2 = np.random.randn(NUM_SAMPLES, SIZE)

    if backend == 'tensorflow':
        from alibi.explainers.backends.tensorflow.cfrl_tabular import consistency_loss
        from alibi.explainers.backends.tensorflow.cfrl_base import to_tensor, to_numpy
        device = None
    else:
        import torch
        from alibi.explainers.backends.pytorch.cfrl_tabular import consistency_loss
        from alibi.explainers.backends.pytorch.cfrl_base import to_tensor, to_numpy
        device = torch.device("cpu")

    z1_tensor, z2_tensor = to_tensor(z1, device=device), to_tensor(z2, device=device)
    closs_backend = consistency_loss(z1_tensor, z2_tensor)["consistency_loss"]
    closs_backend = to_numpy(closs_backend)
    closs = np.mean((z1 - z2)**2)
    assert np.isclose(closs, closs_backend)


@pytest.mark.parametrize('backend', ['tensorflow', 'pytorch'])
@pytest.mark.parametrize('dataset', [lazy_fixture("adult_data")])
def test_sparsity_loss(dataset, backend):
    SIZE = 100
    feature_names = dataset["metadata"]["feature_names"]
    category_map = dataset["metadata"].get("category_map", {})

    # define number of numerical
    num_features = len(feature_names) - len(category_map)

    # define random autoencoder reconstruction and ohe
    X_hat_split = []
    X_ohe_split = []

    # generate numerical feature reconstruction
    X_num = np.random.randn(SIZE, num_features).astype(np.float32)
    X_hat_split.append(X_num)
    X_ohe_split.append(X_num)

    # for each categorical feature generate random reconstruction
    for cat_col in category_map:
        X_cat = np.random.rand(SIZE, len(category_map[cat_col])).astype(np.float32)
        X_hat_split.append(X_cat)

        X_ohe_cat = np.zeros_like(X_cat)
        X_ohe_cat[np.arange(SIZE), np.argmax(X_cat, axis=1)] = 1
        X_ohe_split.append(X_ohe_cat)

    if backend == 'tensorflow':
        from alibi.explainers.backends.tensorflow.cfrl_tabular import sparsity_loss
        from alibi.explainers.backends.tensorflow.cfrl_base import to_tensor, to_numpy
        device = None
    else:
        import torch
        from alibi.explainers.backends.pytorch.cfrl_tabular import sparsity_loss
        from alibi.explainers.backends.pytorch.cfrl_base import to_tensor, to_numpy
        device = torch.device('cpu')

    for i in range(len(X_hat_split)):
        X_hat_split[i] = to_tensor(X_hat_split[i], device=device)

    X_ohe = np.concatenate(X_ohe_split, axis=1)
    X_ohe = to_tensor(X_ohe, device=device)

    losses = sparsity_loss(X_hat_split, X_ohe, category_map)
    losses = {key: to_numpy(val) for key, val in losses.items()}
    assert np.isclose(losses["sparsity_num_loss"], 0, atol=1e-5)
    assert np.isclose(losses["sparsity_cat_loss"], 0, atol=1e-5)
