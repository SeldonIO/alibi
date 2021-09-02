import pytest
import logging
import shap

import spacy
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression

from alibi.explainers import ALE
from alibi.explainers import AnchorTabular
from alibi.explainers import KernelShap, TreeShap
from alibi.explainers.tests.utils import predict_fcn, MockTreeExplainer
from alibi.tests.utils import MockPredictor
from alibi.utils.lang_model import BertBaseUncased, DistilbertBaseUncased, RobertaBase
from alibi.utils.download import spacy_model

import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier

import alibi_testing
from alibi_testing.data import get_adult_data, get_iris_data, get_boston_data, get_mnist_data, \
    get_movie_sentiment_data


# A file containing fixtures that can be used across tests

# Fixtures that return datasets can be combined with classifier
# fixtures to generate models for testing.


@pytest.fixture
def models(request):
    """
    This fixture loads a list of pre-trained test-models by name from the
    alibi-testing helper package.
    """
    models = []
    for name in request.param:
        models.append(alibi_testing.load(name))
    return models


@pytest.fixture(scope='module')
def mnist_data():
    return get_mnist_data()


@pytest.fixture(scope='module')
def boston_data():
    return get_boston_data()


@pytest.fixture(scope='module')
def iris_data():
    """
    This fixture can be passed to a classifier fixture to return
    a trained classifier on the Iris dataset. Because it is scoped
    at module level, the state of this  fixture should not be
    mutated during testing - if you need to do so, please copy the
    objects returned first.
    """
    return get_iris_data()


@pytest.fixture(scope='module')
def adult_data():
    """
    This fixture can be passed to a classifier fixture to return
    a trained classifier on the Adult dataset. Because it is scoped
    at module level, the state of this  fixture should not be
    mutated during testing - if you need to do so, please copy the
    objects returned first.
    """
    return get_adult_data()


@pytest.fixture(scope='module')
def movie_sentiment_data():
    return get_movie_sentiment_data()


# The classifier fixtures accept a dictionary that
# contains data and a preprocessor and return a fitted model
# See the *_dataset functions in alibi.tests.utils for
# examples with the expected data type

# TODO: The classifier training code is identical so should be
#  ble to parametrize with module name and param dict to have only 1
#  such fixture

@pytest.fixture(scope='module')
def rf_classifier(request):
    """
    Trains a random forest classifier. Because it is scoped
    at module level, the state of this  fixture should not be
    mutated during test - if you need to do so, please copy the
    objects returned.
    """

    is_preprocessor = False
    preprocessor = None
    # this fixture should be parametrised with a fixture that
    # returns a dataset dictionary with specified attributes
    # see test_anchor_tabular for a usage example
    data = request.param

    if data['preprocessor']:
        is_preprocessor = True
        preprocessor = data['preprocessor']

    np.random.seed(0)
    clf = RandomForestClassifier(n_estimators=50)

    if is_preprocessor:
        clf.fit(preprocessor.transform(data['X_train']), data['y_train'])
    else:
        clf.fit(data['X_train'], data['y_train'])

    return clf, preprocessor


@pytest.fixture(scope='module')
def lr_classifier(request):
    """
    Trains a logistic regression classifier. Because it is scoped
    at module level, the state of this  fixture should not be
    mutated during test - if you need to do so, please copy the
    objects returned.
    """

    is_preprocessor = False
    preprocessor = False
    # see test_anchor_text for an example on how this
    # fixture can be parametrized
    data = request.param
    if data['preprocessor']:
        is_preprocessor = True
        preprocessor = data['preprocessor']

    clf = LogisticRegression()

    if is_preprocessor:
        clf.fit(preprocessor.transform(data['X_train']), data['y_train'])
    else:
        clf.fit(data['X_train'], data['y_train'])

    return clf, preprocessor


@pytest.fixture(scope='module')
def lr_regressor(request):
    """
    Trains a linear regression model.
    """
    is_preprocessor = False
    preprocessor = False

    data = request.param
    if data['preprocessor']:
        is_preprocessor = True
        preprocessor = data['preprocessor']

    model = LinearRegression()

    if is_preprocessor:
        model.fit(preprocessor.transform(data['X_train']), data['y_train'])
    else:
        model.fit(data['X_train'], data['y_train'])

    return model, preprocessor


# Following fixtures are related to Anchor explainers testing


@pytest.fixture(scope='module')
def at_defaults(request):
    """
    Default config for explainers. Because it is scoped
    at module level, the state of this  fixture should not be
    mutated during test - if you need to do so, please copy the
    objects returned.
    """

    desired_confidence = request.param

    return {
        'delta': 0.1,
        'epsilon': 0.15,
        'batch_size': 100,
        'desired_confidence': desired_confidence,
        'max_anchor_size': None,
        'coverage_samples': 9999,
        'n_covered_ex': 5,
        'seed': 0
    }


@pytest.fixture(params=['proba', 'class'], ids='predictor_type={}'.format)
def at_iris_explainer(iris_data, rf_classifier, request):
    """
    Instantiates and fits an AnchorTabular explainer for the Iris dataset.
    """

    predict_type = request.param
    data = iris_data
    clf, _ = rf_classifier  # preprocessor not necessary

    # instantiate and fit explainer
    pred_fn = predict_fcn(predict_type, clf)
    explainer = AnchorTabular(pred_fn, data['metadata']['feature_names'])
    explainer.fit(data['X_train'], disc_perc=(25, 50, 75))

    return data['X_test'], explainer, pred_fn, predict_type


@pytest.fixture(params=['proba', 'class'], ids='predictor_type={}'.format)
def at_adult_explainer(adult_data, rf_classifier, request):
    """
    Instantiates and fits an AnchorTabular explainer for the Adult dataset.
    """

    # fit random forest classifier
    predict_type = request.param
    data = adult_data
    clf, preprocessor = rf_classifier

    # instantiate and fit explainer
    pred_fn = predict_fcn(predict_type, clf, preprocessor)
    explainer = AnchorTabular(
        pred_fn,
        data['metadata']['feature_names'],
        categorical_names=data['metadata']['category_map']
    )
    explainer.fit(data['X_train'], disc_perc=(25, 50, 75))

    return data['X_test'], explainer, pred_fn, predict_type


@pytest.fixture
def mock_kernel_shap_explainer(request):
    """
    Instantiates a KernelShap explainer with a mock predictor.
    """

    pred_out_dim, link, distributed_opts = request.param
    predictor = MockPredictor(out_dim=pred_out_dim, seed=0)
    explainer = KernelShap(predictor=predictor, link=link, distributed_opts=distributed_opts, seed=0)

    return explainer


@pytest.fixture
def mock_ale_explainer(request):
    """
    Instantiates an ALE explainer with a mock predictor.
    """
    out_dim, out_type = request.param
    predictor = MockPredictor(out_dim=out_dim, out_type=out_type, seed=0)
    explainer = ALE(predictor)

    return explainer


@pytest.fixture
def mock_tree_shap_explainer(monkeypatch, request):
    """
    Instantiates a TreeShap explainer where both the `_explainer` and `model` atttributes
    are mocked

    Parameters
    ----------
    request:
        Tuple containing number of outputs and model_output string (see TreeShap constructor).
    """

    n_outs, model_output = request.param
    seed = 0
    predictor = MockPredictor(out_dim=n_outs, out_type=model_output, seed=seed)
    tree_explainer = MockTreeExplainer(predictor, seed=seed)
    monkeypatch.setattr(shap, "TreeExplainer", tree_explainer)
    explainer = TreeShap(predictor, model_output=model_output)

    return explainer


# High level fixtures that help us check if the code logs any warnings/correct

@pytest.fixture
def no_warnings(caplog):
    """
    This fixture should be passed to any test function in order to check if any warnings are raised.
    """

    caplog.set_level(logging.WARNING)
    yield
    warnings = [record for record in caplog.get_records('call') if record.levelno == logging.WARNING]
    assert not warnings


@pytest.fixture
def no_errors(caplog):
    """
    This fixture should be passed to any test function in order to check if any errors are raised.
    """

    caplog.set_level(logging.ERROR)
    yield
    errors = [record for record in caplog.get_records('call') if record.levelno == logging.ERROR]
    assert not errors


# hooks to skip test configurations that don't make sense
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "uncollect_if(*, func): function to unselect tests from parametrization"
    )


def pytest_collection_modifyitems(config, items):
    removed = []
    kept = []
    for item in items:
        m = item.get_closest_marker('uncollect_if')
        if m:
            func = m.kwargs['func']
            if func(**item.callspec.params):
                removed.append(item)
                continue
        kept.append(item)
    if removed:
        config.hook.pytest_deselected(items=removed)
        items[:] = kept


@pytest.fixture
def disable_tf2():
    """
    Fixture for disabling TF2.x functionality for test functions which
    rely on TF1.x style code (Counterfactual, CEM, CounterfactualProto).

    Because of restrictions in TF, the teardown does not contain code
    to enable v2 behaviour back. Instead, we need to run two sets of
    tests - one with marker "tf1" and another with marker "not tf1".
    """
    tf.compat.v1.disable_v2_behavior()
    yield


@pytest.fixture(scope='module')
def lang_model(request):
    model_name = request.param

    if model_name == "RobertaBase":
        return RobertaBase()

    if model_name == "BertBaseUncased":
        return BertBaseUncased()

    if model_name == "DistilbertBaseUncased":
        return DistilbertBaseUncased()

    return None


@pytest.fixture(scope='module')
def nlp():
    model = 'en_core_web_md'
    spacy_model(model=model)
    nlp = spacy.load(model)
    return nlp
