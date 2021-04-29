import numpy as np
from numpy.testing import assert_allclose
import pytest
from pytest_lazyfixture import lazy_fixture
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import tempfile
import tensorflow as tf

from alibi.explainers import (
    ALE,
    AnchorImage,
    AnchorTabular,
    AnchorText,
    IntegratedGradients,
    KernelShap,
    TreeShap
)
from alibi.saving import load_explainer
from alibi_testing.data import get_adult_data, get_iris_data, get_movie_sentiment_data
import alibi_testing
from alibi.utils.download import spacy_model
from alibi.explainers.tests.utils import predict_fcn


# TODO: consolidate fixtures with those in explainers/tests/conftest.py

@pytest.fixture(scope='module')
def english_spacy_model():
    import spacy
    model = 'en_core_web_md'
    spacy_model(model=model)
    nlp = spacy.load(model)
    return nlp


@pytest.fixture(scope='module')
def adult_data():
    return get_adult_data()


@pytest.fixture(scope='module')
def iris_data():
    return get_iris_data()


@pytest.fixture(scope='module')
def movie_sentiment_data():
    return get_movie_sentiment_data()


@pytest.fixture(scope='module')
def lr_classifier(request):
    data = request.param
    is_preprocessor = False
    if data['preprocessor']:
        is_preprocessor = True
        preprocessor = data['preprocessor']

    clf = LogisticRegression()
    if is_preprocessor:
        clf.fit(preprocessor.transform(data['X_train']), data['y_train'])
    else:
        clf.fit(data['X_train'], data['y_train'])
    return clf


@pytest.fixture(scope='module')
def rf_classifier(request):
    data = request.param
    is_preprocessor = False
    if data['preprocessor']:
        is_preprocessor = True
        preprocessor = data['preprocessor']

    np.random.seed(0)
    clf = RandomForestClassifier(n_estimators=50)
    if is_preprocessor:
        clf.fit(preprocessor.transform(data['X_train']), data['y_train'])
    else:
        clf.fit(data['X_train'], data['y_train'])
    return clf


@pytest.fixture(scope='module')
def ffn_classifier(request):
    data = request.param
    inputs = tf.keras.Input(shape=data['X_train'].shape[1:])
    x = tf.keras.layers.Dense(20, activation='relu')(inputs)
    x = tf.keras.layers.Dense(20, activation='relu')(x)
    outputs = tf.keras.layers.Dense(config['output_dim'], activation=config['activation'])(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss=config['loss'], optimizer='adam')
    model.fit(data['X_train'], tf.keras.utils.to_categorical(data['y_train']), epochs=1)
    return model


@pytest.fixture(scope='module')
def mnist_predictor():
    model = alibi_testing.load('mnist-cnn-tf2.2.0')
    predictor = lambda x: model.predict(x)  # noqa
    return predictor


@pytest.fixture(scope='module')
def ale_explainer(iris_data, lr_classifier):
    ale = ALE(predictor=lr_classifier.predict_proba,
              feature_names=iris_data['metadata']['feature_names'])
    return ale


@pytest.fixture(scope='module')
def ig_explainer(iris_data, ffn_classifier):
    ig = IntegratedGradients(model=ffn_classifier)
    return ig


def mnist_segmentation_fn(image, size=(4, 7)):
    segments = np.zeros([image.shape[0], image.shape[1]])
    row_idx, col_idx = np.where(segments == 0)
    for i, j in zip(row_idx, col_idx):
        segments[i, j] = int((image.shape[1] / size[1]) * (i // size[0]) + j // size[1])
    return segments


@pytest.fixture(scope='module')
def ai_explainer(mnist_predictor, request):
    segmentation_fn = request.param
    ai = AnchorImage(predictor=mnist_predictor,
                     image_shape=(28, 28, 1),
                     segmentation_fn=segmentation_fn)
    return ai


@pytest.fixture(scope='module')
def atext_explainer(lr_classifier, english_spacy_model, movie_sentiment_data):
    predictor = predict_fcn(predict_type='class',
                            clf=lr_classifier,
                            preproc=movie_sentiment_data['preprocessor'])
    atext = AnchorText(nlp=english_spacy_model,
                       predictor=predictor)
    return atext


@pytest.fixture(scope='module')
def atab_explainer(lr_classifier, adult_data):
    predictor = predict_fcn(predict_type='class',
                            clf=lr_classifier,
                            preproc=adult_data['preprocessor'])
    atab = AnchorTabular(predictor=predictor,
                         feature_names=adult_data['metadata']['feature_names'],
                         categorical_names=adult_data['metadata']['category_map'])
    atab.fit(adult_data['X_train'], disc_perc=(25, 50, 75))
    return atab


@pytest.fixture(scope='module')
def kshap_explainer(lr_classifier, adult_data):
    predictor = predict_fcn(predict_type='proba',
                            clf=lr_classifier,
                            preproc=adult_data['preprocessor'])
    kshap = KernelShap(predictor=predictor,
                       link='logit',
                       feature_names=adult_data['metadata']['feature_names'])
    kshap.fit(adult_data['X_train'][:100])
    return kshap


@pytest.fixture(scope='module')
def tree_explainer(rf_classifier, iris_data):
    treeshap = TreeShap(predictor=rf_classifier,
                        model_output='probability',
                        feature_names=iris_data['metadata']['feature_names'])
    treeshap.fit(iris_data['X_train'])
    return treeshap


@pytest.mark.parametrize('lr_classifier', [lazy_fixture('iris_data')], indirect=True)
def test_save_ALE(ale_explainer, lr_classifier, iris_data):
    X = iris_data['X_test']
    exp0 = ale_explainer.explain(X)
    with tempfile.TemporaryDirectory() as temp_dir:
        ale_explainer.save(temp_dir)
        ale_explainer1 = load_explainer(temp_dir, predictor=lr_classifier.predict_proba)

        assert isinstance(ale_explainer1, ALE)
        # TODO: cannot pass as meta updated after explain
        # assert ale_explainer.meta == ale_explainer1.meta
        exp1 = ale_explainer1.explain(X)

        # ALE explanations are deterministic
        assert exp0.meta == exp1.meta
        # assert exp0.data == exp1.data # cannot compare as many types instide TODO: define equality for explanations?
        # or compare pydantic schemas?
        assert np.all(exp0.ale_values[0] == exp1.ale_values[0])


config = {'output_dim': 3, 'loss': 'categorical_crossentropy', 'activation': 'softmax'}


# TODO: figure out a way to pass both lazy_fixture('iris_data') and config...
@pytest.mark.parametrize('ffn_classifier', [lazy_fixture('iris_data')], indirect=True)
def test_save_IG(ig_explainer, ffn_classifier, iris_data):
    X = iris_data['X_test']
    target = iris_data['y_test']
    exp0 = ig_explainer.explain(X, target=target)

    with tempfile.TemporaryDirectory() as temp_dir:
        ig_explainer.save(temp_dir)
        ig_explainer1 = load_explainer(temp_dir, predictor=ffn_classifier)

        assert isinstance(ig_explainer1, IntegratedGradients)
        assert ig_explainer.meta == ig_explainer1.meta

        exp1 = ig_explainer.explain(X, target=target)
        assert exp0.meta == exp1.meta

        # IG is deterministic
        assert np.all(exp0.attributions[0] == exp1.attributions[0])


# test with black-box and built-in segmentation function
@pytest.mark.parametrize('ai_explainer', [mnist_segmentation_fn, 'slic'], indirect=True)
def test_save_AnchorImage(ai_explainer, mnist_predictor):
    X = np.random.rand(28, 28, 1)

    exp0 = ai_explainer.explain(X)

    with tempfile.TemporaryDirectory() as temp_dir:
        ai_explainer.save(temp_dir)
        ai_explainer1 = load_explainer(temp_dir, predictor=mnist_predictor)

        assert isinstance(ai_explainer1, AnchorImage)
        assert ai_explainer.meta == ai_explainer1.meta

        exp1 = ai_explainer1.explain(X)
        assert exp0.meta == exp1.meta


@pytest.mark.parametrize('lr_classifier', [lazy_fixture('movie_sentiment_data')], indirect=True)
def test_save_AnchorText(atext_explainer, lr_classifier, movie_sentiment_data):
    predictor = predict_fcn(predict_type='class',
                            clf=lr_classifier,
                            preproc=movie_sentiment_data['preprocessor'])
    X = movie_sentiment_data['X_test'][0]

    exp0 = atext_explainer.explain(X)

    with tempfile.TemporaryDirectory() as temp_dir:
        atext_explainer.save(temp_dir)
        atext_explainer1 = load_explainer(temp_dir, predictor=predictor)

        assert isinstance(atext_explainer1, AnchorText)
        assert atext_explainer.meta == atext_explainer1.meta

        exp1 = atext_explainer1.explain(X)
        assert exp0.meta == exp1.meta


@pytest.mark.parametrize('lr_classifier', [lazy_fixture('adult_data')], indirect=True)
def test_save_AnchorTabular(atab_explainer, lr_classifier, adult_data):
    predictor = predict_fcn(predict_type='class',
                            clf=lr_classifier,
                            preproc=adult_data['preprocessor'])
    X = adult_data['X_test'][0]

    exp0 = atab_explainer.explain(X)

    with tempfile.TemporaryDirectory() as temp_dir:
        atab_explainer.save(temp_dir)
        atab_explainer1 = load_explainer(temp_dir, predictor=predictor)

        assert isinstance(atab_explainer1, AnchorTabular)
        assert atab_explainer.meta == atab_explainer1.meta

        exp1 = atab_explainer1.explain(X)
        assert exp0.meta == exp1.meta


@pytest.mark.parametrize('lr_classifier', [lazy_fixture('adult_data')], indirect=True)
def test_save_KernelShap(kshap_explainer, lr_classifier, adult_data):
    predictor = predict_fcn(predict_type='proba',
                            clf=lr_classifier,
                            preproc=adult_data['preprocessor'])
    X = adult_data['X_test'][:2]

    exp0 = kshap_explainer.explain(X)

    with tempfile.TemporaryDirectory() as temp_dir:
        kshap_explainer.save(temp_dir)
        kshap_explainer1 = load_explainer(temp_dir, predictor=predictor)

        assert isinstance(kshap_explainer1, KernelShap)
        assert kshap_explainer.meta == kshap_explainer1.meta

        exp1 = kshap_explainer.explain(X)
        assert exp0.meta == exp1.meta


@pytest.mark.parametrize('rf_classifier', [lazy_fixture('iris_data')], indirect=True)
def test_save_TreeShap(tree_explainer, rf_classifier, iris_data):
    X = iris_data['X_test']

    exp0 = tree_explainer.explain(X)

    with tempfile.TemporaryDirectory() as temp_dir:
        tree_explainer.save(temp_dir)
        tree_explainer1 = load_explainer(temp_dir, predictor=rf_classifier)

        assert isinstance(tree_explainer1, TreeShap)
        assert tree_explainer.meta == tree_explainer1.meta

        exp1 = tree_explainer1.explain(X)
        assert exp0.meta == exp1.meta

        # TreeShap is deterministic
        assert_allclose(exp0.shap_values[0], exp1.shap_values[0])
