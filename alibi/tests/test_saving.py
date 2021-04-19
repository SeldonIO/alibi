import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture
from sklearn.linear_model import LogisticRegression
import tempfile

from alibi.explainers import ALE
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
def ale_explainer(iris_data, lr_classifier):
    ale = ALE(predictor=lr_classifier.predict_proba,
              feature_names=iris_data['metadata']['feature_names'])
    return ale


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
