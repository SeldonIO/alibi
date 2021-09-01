import numpy as np
from numpy.testing import assert_allclose
import pytest
from pytest_lazyfixture import lazy_fixture
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import tempfile
import tensorflow as tf
from typing import List, Union

from alibi.explainers import (
    ALE,
    AnchorImage,
    AnchorTabular,
    AnchorText,
    IntegratedGradients,
    KernelShap,
    TreeShap,
    CounterfactualRLTabular
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
def language_model():
    from alibi.utils.lang_model import DistilbertBaseUncased
    return DistilbertBaseUncased()


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
def iris_ae(iris_data):
    from alibi.models.tensorflow.autoencoder import AE

    # define encoder
    encoder = tf.keras.Sequential([
        tf.keras.layers.Dense(2),
        tf.keras.layers.Activation('tanh')
    ])

    # define decoder
    decoder = tf.keras.Sequential([
        tf.keras.layers.Dense(4)
    ])

    # define autoencoder, compile and fit
    ae = AE(encoder=encoder, decoder=decoder)
    ae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=tf.keras.losses.MeanSquaredError())
    ae.fit(iris_data['X_train'], iris_data['X_train'], epochs=1)
    return ae


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
def atext_explainer_nlp(lr_classifier, english_spacy_model, movie_sentiment_data):
    predictor = predict_fcn(predict_type='class',
                            clf=lr_classifier,
                            preproc=movie_sentiment_data['preprocessor'])
    atext = AnchorText(nlp=english_spacy_model,
                       predictor=predictor,
                       sampling_strategy="unknown")
    return atext


@pytest.fixture(scope='module')
def atext_explainer_lm(lr_classifier, language_model, movie_sentiment_data):
    predictor = predict_fcn(predict_type='class',
                            clf=lr_classifier,
                            preproc=movie_sentiment_data['preprocessor'])
    atext = AnchorText(language_model=language_model,
                       predictor=predictor,
                       sampling_strategy="language_model",
                       sample_proba=1.0)
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


@pytest.fixture(scope='module')
def cfrl_explainer(rf_classifier, iris_ae, iris_data):
    # define explainer constants
    LATENT_DIM = 2
    COEFF_SPARSITY = 0.1
    COEFF_CONSISTENCY = 0.0
    TRAIN_STEPS = 100
    BATCH_SIZE = 100

    # need to define a wrapper for the decoder to return a list of tensors
    class DecoderList(tf.keras.Model):
        def __init__(self, decoder: tf.keras.Model, **kwargs):
            super().__init__(**kwargs)
            self.decoder = decoder

        def call(self, input: Union[tf.Tensor, List[tf.Tensor]], **kwargs):
            return [self.decoder(input, **kwargs)]

    # redefine the call method to return a list of tensors.
    iris_ae.decoder = DecoderList(iris_ae.decoder)

    # define predictor.
    predictor = lambda x: rf_classifier[0].predict_proba(x)  # noqa: E731

    # define explainer.
    explainer = CounterfactualRLTabular(encoder=iris_ae.encoder,
                                        decoder=iris_ae.decoder,
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

    # fit the explainer
    explainer.fit(X=iris_data['X_train'])
    return explainer


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
@pytest.mark.parametrize('atext_explainer', [lazy_fixture('atext_explainer_nlp'), lazy_fixture('atext_explainer_lm')])
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


@pytest.mark.parametrize('rf_classifier', [lazy_fixture('iris_data')], indirect=True)
def test_save_cfrl(cfrl_explainer, rf_classifier, iris_data):
    X = iris_data['X_test']
    exp0 = cfrl_explainer.explain(X=X, Y_t=np.array([0]), C=[])

    with tempfile.TemporaryDirectory() as temp_dir:
        # save explainer
        cfrl_explainer.save(temp_dir)

        # define predictor and load the explainer
        predictor = lambda x: rf_classifier[0].predict_proba(x)  # noqa: E731
        cfrl_explainer1 = load_explainer(temp_dir, predictor=predictor)
        assert isinstance(cfrl_explainer1, CounterfactualRLTabular)

        # Check metadata. Loading a model can change its class, so we have to remove the encoder, decoder,
        # actor and critic metadata which are the class name.
        # See: https://www.tensorflow.org/api_docs/python/tf/saved_model/load for more details.
        # Also have to remove the optimizers and callbacks since those are not saved.
        keys_to_remove = ['encoder', 'decoder', 'actor', 'critic', 'optimizer_actor', 'optimizer_critic', 'callbacks']
        for key in keys_to_remove:
            cfrl_explainer.meta["params"].pop(key)
            cfrl_explainer1.meta["params"].pop(key)
        assert cfrl_explainer.meta == cfrl_explainer1.meta

        exp1 = cfrl_explainer1.explain(X=X, Y_t=np.array([0]), C=[])
        assert exp0.meta == exp1.meta

        # cfrl is determinstic
        assert_allclose(exp0.cf["X"].astype(np.float32), exp1.cf["X"].astype(np.float32))
        assert_allclose(exp0.cf["class"].astype(np.float32), exp1.cf["class"].astype(np.float32))
