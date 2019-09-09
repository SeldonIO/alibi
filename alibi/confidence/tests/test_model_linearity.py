import pytest
import numpy as np
from sklearn.datasets import load_iris, load_boston
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVR
from alibi.confidence.model_linearity import linearity_measure, LinearityMeasure
from alibi.confidence.model_linearity import _linear_superposition, _sample_grid, _sample_knn
from functools import reduce


@pytest.mark.parametrize('input_shape', ((3,), (4, 4, 1)))
@pytest.mark.parametrize('nb_instances', (1, 10))
def test_linear_superposition(input_shape, nb_instances):
    alphas = np.array([0.5, 0.5])

    vecs_list = []
    for i in range(nb_instances):
        v0 = np.zeros((1,) + input_shape)
        v1 = np.ones((1,) + input_shape)
        vec = np.stack((v0, v1), axis=1)
        vecs_list.append(vec)
    vecs = reduce(lambda x, y: np.vstack((x, y)), vecs_list)

    summ = _linear_superposition(alphas, vecs, input_shape)

    assert summ.shape[0] == nb_instances
    assert summ.shape[1:] == input_shape
    assert (summ == 0.5).all()


@pytest.mark.parametrize('nb_instances', (1, 5))
@pytest.mark.parametrize('nb_samples', (2, 10))
def test_sample_knn(nb_instances, nb_samples):

    iris = load_iris()
    X_train = iris.data
    input_shape = X_train.shape[1:]
    x = np.ones((nb_instances, ) + input_shape)

    X_samples = _sample_knn(x=x, X_train=X_train, nb_samples=nb_samples)

    assert X_samples.shape[0] == nb_instances
    assert X_samples.shape[1] == nb_samples


@pytest.mark.parametrize('nb_instances', (5, ))
@pytest.mark.parametrize('nb_samples', (3, ))
@pytest.mark.parametrize('input_shape', ((3,), (4, 4, 1)))
def test_sample_grid(nb_instances, nb_samples, input_shape):

    x = np.ones((nb_instances, ) + input_shape)
    nb_features = x.reshape(x.shape[0], -1).shape[1]
    feature_range = np.array([[0, 1] for _ in range(nb_features)])

    X_samples = _sample_grid(x, feature_range, nb_samples=nb_samples)

    assert X_samples.shape[0] == nb_instances
    assert X_samples.shape[1] == nb_samples


@pytest.mark.parametrize('method', ('knn', 'grid'))
@pytest.mark.parametrize('epsilon', (0.04,))
@pytest.mark.parametrize('res', (100,))
@pytest.mark.parametrize('nb_instances', (1, 10))
@pytest.mark.parametrize('agg', ('global', 'pairwise'))
def test_linearity_measure_class(method, epsilon, res, nb_instances, agg):

    iris = load_iris()
    X_train = iris.data
    y_train = iris.target
    x = X_train[0: nb_instances].reshape(nb_instances, -1)

    lg = LogisticRegression()
    lg.fit(X_train, y_train)

    def predict_fn(x):
        return lg.predict_proba(x)

    lin = linearity_measure(predict_fn, x, method=method, epsilon=epsilon, X_train=X_train, res=res,
                            model_type='classifier', agg=agg)
    assert lin.shape[0] == nb_instances, 'Checking shapes'
    assert (lin >= 0).all(), 'Linearity measure must be >= 0'

    feature_range = [[0, 1] for _ in range(X_train.shape[1])]
    lin_2 = linearity_measure(predict_fn, x, method='grid', epsilon=epsilon, feature_range=feature_range,
                              res=res, model_type='classifier', agg=agg)
    assert lin_2.shape[0] == nb_instances, 'Nb of linearity values returned different from number of instances'
    assert (lin_2 >= 0).all(), 'Linearity measure must be >= 0'


@pytest.mark.parametrize('method', ('knn', 'grid'))
@pytest.mark.parametrize('epsilon', (0.04,))
@pytest.mark.parametrize('res', (100,))
@pytest.mark.parametrize('nb_instances', (1, 10))
@pytest.mark.parametrize('agg', ('global', 'pairwise'))
def test_linearity_measure_reg(method, epsilon, res, nb_instances, agg):

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
                            model_type='regressor', agg=agg)
    assert lin.shape[0] == nb_instances, 'Checking shapes'
    assert (lin >= 0).all(), 'Linearity measure must be >= 0'
    assert np.allclose(lin, np.zeros(lin.shape))

    lin_svr = linearity_measure(predict_fn_svr, x, method=method, epsilon=epsilon, X_train=X_train,
                                res=res, model_type='regressor', agg=agg)
    assert lin_svr.shape[0] == nb_instances, 'Checking shapes'
    assert (lin_svr >= 0).all(), 'Linearity measure must be >= 0'

    feature_range = [[0, 1] for _ in range(X_train.shape[1])]
    lin_2 = linearity_measure(predict_fn, x, method='grid', epsilon=epsilon, feature_range=feature_range,
                              res=res, model_type='regressor', agg=agg)
    assert lin_2.shape[0] == nb_instances, 'Checking shapes'
    assert (lin_2 >= 0).all(), 'Linearity measure must be >= 0'
    assert np.allclose(lin_2, np.zeros(lin_2.shape))

    feature_range = [[0, 1] for _ in range(X_train.shape[1])]
    lin_2_svr = linearity_measure(predict_fn_svr, x, method='grid', epsilon=epsilon,
                                  feature_range=feature_range, res=res, model_type='regressor', agg=agg)
    assert lin_2_svr.shape[0] == nb_instances, 'Checking shapes'
    assert (lin_2_svr >= 0).all(), 'Linearity measure must be >= 0'

    y_train_multi = np.stack((y_train, y_train), axis=1)
    lg_multi = LinearRegression()
    lg_multi.fit(X_train, y_train_multi)

    def predict_fn_multi(x):
        return lg_multi.predict(x)

    lm_multi = LinearityMeasure(method=method, epsilon=epsilon, res=res, model_type='regressor', agg=agg)
    lm_multi.fit(X_train)
    lin_multi = lm_multi.score(predict_fn_multi, x)
    assert lin_multi.shape[0] == nb_instances, 'Checking shapes'
    assert (lin_multi >= 0).all(), 'Linearity measure must be >= 0'
    assert np.allclose(lin_multi, np.zeros(lin_multi.shape))


@pytest.mark.parametrize('method', ('knn', 'grid'))
@pytest.mark.parametrize('epsilon', (0.04,))
@pytest.mark.parametrize('res', (100,))
@pytest.mark.parametrize('nb_instances', (1, 10))
@pytest.mark.parametrize('agg', ('global', 'pairwise'))
def test_LinearityMeasure_class(method, epsilon, res, nb_instances, agg):

    iris = load_iris()
    X_train = iris.data
    y_train = iris.target
    x = X_train[0: nb_instances].reshape(nb_instances, -1)

    lg = LogisticRegression()
    lg.fit(X_train, y_train)

    def predict_fn(x):
        return lg.predict_proba(x)

    lm = LinearityMeasure(method=method, epsilon=epsilon, res=res, model_type='classifier', agg=agg)
    lm.fit(X_train)
    lin = lm.score(predict_fn, x)
    assert lin.shape[0] == nb_instances, 'Checking shapes'
    assert (lin >= 0).all(), 'Linearity measure must be >= 0'


@pytest.mark.parametrize('method', ('knn', 'grid'))
@pytest.mark.parametrize('epsilon', (0.04,))
@pytest.mark.parametrize('res', (100,))
@pytest.mark.parametrize('nb_instances', (1, 10))
@pytest.mark.parametrize('agg', ('global', 'pairwise'))
def test_LinearityMeasure_reg(method, epsilon, res, nb_instances, agg):

    boston = load_boston()
    X_train, y_train = boston.data, boston.target
    x = X_train[0: nb_instances].reshape(nb_instances, -1)

    lg = LinearRegression()
    lg.fit(X_train, y_train)

    def predict_fn(x):
        return lg.predict(x)

    y_train_multi = np.stack((y_train, y_train), axis=1)
    lg_multi = LinearRegression()
    lg_multi.fit(X_train, y_train_multi)

    def predict_fn_multi(x):
        return lg_multi.predict(x)

    lm = LinearityMeasure(method=method, epsilon=epsilon, res=res, model_type='regressor', agg=agg)
    lm.fit(X_train)
    lin = lm.score(predict_fn, x)
    assert lin.shape[0] == nb_instances, 'Checking shapes'
    assert (lin >= 0).all(), 'Linearity measure must be >= 0'
    assert np.allclose(lin, np.zeros(lin.shape))

    lm_multi = LinearityMeasure(method=method, epsilon=epsilon, res=res, model_type='regressor', agg=agg)
    lm_multi.fit(X_train)
    lin_multi = lm_multi.score(predict_fn_multi, x)
    assert lin_multi.shape[0] == nb_instances, 'Checking shapes'
    assert (lin_multi >= 0).all(), 'Linearity measure must be >= 0'
    assert np.allclose(lin_multi, np.zeros(lin_multi.shape))
