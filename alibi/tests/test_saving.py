import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture
from sklearn.linear_model import LogisticRegression
import tempfile
import tensorflow as tf

from alibi.explainers import ALE, IntegratedGradients
from alibi.saving import load_explainer
from alibi_testing.data import get_iris_data


@pytest.fixture(scope='module')
def iris_data():
    return get_iris_data()


@pytest.fixture(scope='module')
def lr_classifier(request):
    data = request.param
    clf = LogisticRegression()
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
def ale_explainer(iris_data, lr_classifier):
    ale = ALE(predictor=lr_classifier.predict_proba,
              feature_names=iris_data['metadata']['feature_names'])
    return ale


@pytest.fixture(scope='module')
def ig_explainer(iris_data, ffn_classifier):
    ig = IntegratedGradients(model=ffn_classifier)
    return ig


@pytest.mark.parametrize('lr_classifier', [lazy_fixture('iris_data')], indirect=True)
def test_save_ALE(ale_explainer, lr_classifier, iris_data):
    X = iris_data['X_test']
    exp0 = ale_explainer.explain(X)

    with tempfile.TemporaryDirectory() as temp_dir:
        ale_explainer.save(temp_dir)
        ale_explainer1 = load_explainer(temp_dir, predictor=lr_classifier.predict_proba)

        assert isinstance(ale_explainer1, ALE)
        # assert ale_explainer.meta == ale_explainer1.meta # cannot pass as meta updated after explain TODO: fix
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

        exp1 = ig_explainer.explain(X, target=target)
        assert exp0.meta == exp1.meta

        # IG is deterministic
        assert np.all(exp0.attributions[0] == exp1.attributions[0])
