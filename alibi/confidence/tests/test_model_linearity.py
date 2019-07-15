import pytest
import numpy as np
from sklearn.datasets import load_iris, load_boston
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVR
from alibi.confidence.model_linearity import linearity_measure, LinearityMeasure


@pytest.mark.parametrize('method', ('knn', 'gridSampling'))
@pytest.mark.parametrize('epsilon', (0.04, 0.9))
@pytest.mark.parametrize('res', (10, 100))
@pytest.mark.parametrize('nb_instances', (1, 10))
def test_linearity_measure_class(method, epsilon, res, nb_instances):

    iris = load_iris()
    X_train = iris.data
    y_train = iris.target
    x = X_train[0: nb_instances].reshape(nb_instances, -1)

    lg = LogisticRegression()
    lg.fit(X_train, y_train)

    def predict_fn(x):
        return lg.predict_proba(x)

    lin = linearity_measure(predict_fn, x, method=method, epsilon=epsilon, X_train=X_train, res=res,
                            model_type='classifier')
    assert lin.shape[0] == nb_instances, 'Checking shapes'
    assert (lin >= 0).all(), 'Linearity measure must be >= 0'

    features_range = [[0, 1] for _ in range(X_train.shape[1])]
    lin_2 = linearity_measure(predict_fn, x, method='gridSampling', epsilon=epsilon, features_range=features_range,
                              res=res, model_type='classifier')
    assert lin_2.shape[0] == nb_instances, 'Checking shapes'
    assert (lin_2 >= 0).all(), 'Linearity measure must be >= 0'


@pytest.mark.parametrize('method', ('knn', 'gridSampling'))
@pytest.mark.parametrize('epsilon', (0.04, 0.9))
@pytest.mark.parametrize('res', (10, 100))
@pytest.mark.parametrize('nb_instances', (1, 10))
def test_linearity_measure_reg(method, epsilon, res, nb_instances):

    boston = load_boston()
    X_train, y_train = boston.data, boston.target
    x = X_train[0: nb_instances].reshape(nb_instances, -1)

    lg = LinearRegression()
    lg.fit(X_train, y_train)
    svr = SVR(kernel='linear')
    svr.fit(X_train, y_train)

    def predict_fn_svr(x):
        return svr.predict(x)

    def predict_fn(x):
        return lg.predict(x)

    lin = linearity_measure(predict_fn, x, method=method, epsilon=epsilon, X_train=X_train, res=res,
                            model_type='regressor')
    assert lin.shape[0] == nb_instances, 'Checking shapes'
    assert (lin >= 0).all(), 'Linearity measure must be >= 0'
    assert np.allclose(lin, np.zeros(lin.shape))

    lin_svr = linearity_measure(predict_fn_svr, x, method=method, epsilon=epsilon, X_train=X_train,
                                res=res, model_type='regressor')
    assert lin_svr.shape[0] == nb_instances, 'Checking shapes'
    assert (lin_svr >= 0).all(), 'Linearity measure must be >= 0'
    # assert np.allclose(lin_svr, np.zeros(lin_svr.shape))

    features_range = [[0, 1] for _ in range(X_train.shape[1])]
    lin_2 = linearity_measure(predict_fn, x, method='gridSampling', epsilon=epsilon, features_range=features_range,
                              res=res, model_type='regressor')
    assert lin_2.shape[0] == nb_instances, 'Checking shapes'
    assert (lin_2 >= 0).all(), 'Linearity measure must be >= 0'
    assert np.allclose(lin_2, np.zeros(lin_2.shape))

    features_range = [[0, 1] for _ in range(X_train.shape[1])]
    lin_2_svr = linearity_measure(predict_fn_svr, x, method='gridSampling', epsilon=epsilon,
                                  features_range=features_range, res=res, model_type='regressor')
    assert lin_2_svr.shape[0] == nb_instances, 'Checking shapes'
    assert (lin_2_svr >= 0).all(), 'Linearity measure must be >= 0'
    # assert np.allclose(lin_2_svr, np.zeros(lin_2_svr.shape))


@pytest.mark.parametrize('method', ('knn', 'gridSampling'))
@pytest.mark.parametrize('epsilon', (0.04, 0.9))
@pytest.mark.parametrize('res', (10, 100))
@pytest.mark.parametrize('nb_instances', (1, 10))
def test_LinearityMeasure_class(method, epsilon, res, nb_instances):

    iris = load_iris()
    X_train = iris.data
    y_train = iris.target
    x = X_train[0: nb_instances].reshape(nb_instances, -1)

    lg = LogisticRegression()
    lg.fit(X_train, y_train)

    def predict_fn(x):
        return lg.predict_proba(x)

    lm = LinearityMeasure(method=method, epsilon=epsilon, res=res, model_type='classifier')
    lm.fit(X_train)
    lin = lm.score(predict_fn, x)
    assert lin.shape[0] == nb_instances, 'Checking shapes'
    assert (lin >= 0).all(), 'Linearity measure must be >= 0'


@pytest.mark.parametrize('method', ('knn', 'gridSampling'))
@pytest.mark.parametrize('epsilon', (0.04, 0.9))
@pytest.mark.parametrize('res', (10, 100))
@pytest.mark.parametrize('nb_instances', (1, 10))
def test_LinearityMeasure_reg(method, epsilon, res, nb_instances):

    boston = load_boston()
    X_train, y_train = boston.data, boston.target
    x = X_train[0: nb_instances].reshape(nb_instances, -1)

    lg = LinearRegression()
    lg.fit(X_train, y_train)

    def predict_fn(x):
        return lg.predict(x)

    lm = LinearityMeasure(method=method, epsilon=epsilon, res=res, model_type='regressor')
    lm.fit(X_train)
    lin = lm.score(predict_fn, x)
    assert lin.shape[0] == nb_instances, 'Checking shapes'
    assert (lin >= 0).all(), 'Linearity measure must be >= 0'
    assert np.allclose(lin, np.zeros(lin.shape))
