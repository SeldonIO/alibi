import pytest
from pytest_lazyfixture import lazy_fixture

import numpy as np
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
    assert np.linalg.norm(X - inv_X) < 1e-4


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
def test_generate_numerical_condition(dataset):
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
def test_generate_categorical_condition(dataset):
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
    X = dataset["X_train"][:5]
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
            assert np.linalg.norm(X[:, i] - X_hat[:, i]) < 1e-4


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
    COEFF_SPARSITY = 0.1
    COEFF_CONSISTENCY = 0.0
    TRAIN_STEPS = 1000
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

    # Need to define a decorator for the decoder to return a list of tensors
    def call_decorator(call):
        def inner(inputs, *args, **kwargs):
            return [call(inputs, *args, **kwargs)]
        return inner

    # Redefine the call method to return a list of tensors.
    decoder.call = call_decorator(decoder.call)

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
    explanation = explainer.explain(X=iris_data["X_test"], Y_t=np.array([0]))

    # Compute counterfactual accuracy.
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(explanation.data["cf"]["class"].reshape(-1),
                              explanation.data["target"].reshape(-1))
    assert accuracy > 0.9
