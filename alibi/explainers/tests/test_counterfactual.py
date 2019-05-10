# flake8: noqa E731
import pytest
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import cityblock
import tensorflow as tf

from alibi.explainers.counterfactual import _define_func, \
    num_grad_batch, cityblock_batch, get_wachter_grads
from alibi.explainers import CounterFactual


@pytest.fixture
def logistic_iris():
    X, y = load_iris(return_X_y=True)
    lr = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=200).fit(X, y)
    return X, y, lr


@pytest.fixture
def iris_explainer(request, logistic_iris):
    X, y, lr = logistic_iris
    predict_fn = lambda x: lr.predict_proba(x.reshape(1, -1))
    sess = tf.Session()

    cf_explainer = CounterFactual(sess=sess, predict_fn=predict_fn, data_shape=(4,),
                                  target_class=request.param, lam_init=1000, max_iter=2000)

    yield cf_explainer
    tf.reset_default_graph()
    sess.close()


@pytest.mark.parametrize('target_class', ['other', 'same', 0, 1, 2])
def test_define_func(logistic_iris, target_class):
    X, y, model = logistic_iris

    x = X[0].reshape(1, -1)
    predict_fn = model.predict_proba
    probas = predict_fn(x)
    pred_class = probas.argmax(axis=1)[0]
    pred_prob = probas[:, pred_class][0]

    func, target = _define_func(predict_fn, pred_class, target_class)

    if target_class == 'same':
        assert target == pred_class
        assert func(x) == pred_prob
    elif isinstance(target_class, int):
        assert target == target_class
        assert func(x) == probas[:, target]
    elif target_class == 'other':
        assert target == 'other'
        # highest probability different to the class of x
        ix2 = np.argsort(-probas)[:, 1]
        assert func(x) == probas[:, ix2]


@pytest.mark.parametrize('shape', [(1,), (2, 3), (1, 3, 5)])
@pytest.mark.parametrize('batch_size', [1, 3, 10])
def test_get_batch_num_gradients_cityblock(shape, batch_size):
    u = np.random.rand(batch_size, *shape)
    v = np.random.rand(1, *shape)

    grad_true = np.sign(u - v).reshape(batch_size, 1, *shape)  # expand dims to incorporate 1-d scalar response
    grad_approx = num_grad_batch(cityblock_batch, u, args=tuple([v]))

    assert grad_approx.shape == grad_true.shape
    assert np.allclose(grad_true, grad_approx)


@pytest.mark.parametrize('batch_size', [1, 2, 5])
def test_get_batch_num_gradients_logistic_iris(logistic_iris, batch_size):
    X, y, lr = logistic_iris
    predict_fn = lr.predict_proba
    x = X[0:batch_size]
    probas = predict_fn(x)

    # true gradient of the logistic regression wrt x
    grad_true = np.zeros((batch_size, 3, 4))
    for i, p in enumerate(probas):
        p = p.reshape(1, 3)
        grad = (p.T * (np.eye(3, 3) - p) @ lr.coef_)
        grad_true[i, :, :] = grad
    assert grad_true.shape == (batch_size, 3, 4)

    grad_approx = num_grad_batch(predict_fn, x)

    assert grad_approx.shape == grad_true.shape
    assert np.allclose(grad_true, grad_approx)


def test_get_wachter_grads(logistic_iris):
    X, y, lr = logistic_iris
    predict_fn = lr.predict_proba
    x = X[0].reshape(1, -1)
    probas = predict_fn(x)
    pred_class = probas.argmax()
    func, target = _define_func(predict_fn, pred_class, 'same')

    loss, grad_loss, debug_info = get_wachter_grads(X_current=x, predict_class_fn=func, distance_fn=cityblock_batch,
                                        X_test=x, target_proba=0.1, lam=1)

    assert loss.shape == (1, 1)
    assert grad_loss.shape == x.reshape(1, 1, 4).shape


@pytest.mark.skip(reason='This will change')
@pytest.mark.parametrize('iris_explainer',
                         ['other', 'same', 0, 1, 2],
                         indirect=True)
def test_cf_explainer_iris(iris_explainer, logistic_iris):
    X, y, lr = logistic_iris
    x = X[0]
    probas = iris_explainer.predict_fn(x)
    pred_class = probas.argmax()

    assert iris_explainer.data_shape == (4,)

    # test explanation
    exp = iris_explainer.explain(x)
    x_cf = exp['X_cf']
    assert x.shape == x_cf.shape

    probas_cf = iris_explainer.predict_fn(x_cf)
    pred_class_cf = probas_cf.argmax()

    # get attributes for testing
    target_class = iris_explainer.target_class
    target_proba = iris_explainer.target_proba
    tol = iris_explainer.tol
    max_iter = iris_explainer.max_iter
    pred_class_fn = iris_explainer.predict_class_fn

    # check if target_class condition is met
    if target_class == 'same':
        assert pred_class == pred_class_cf
    elif target_class == 'other':
        assert pred_class != pred_class_cf
    elif exp['success']:
        assert pred_class_cf == target_class

    # check if probability is within tolerance
    # if exp['success']:
    # assert exp['n_iter'] <= max_iter
    assert exp['success']
    assert np.abs(pred_class_fn(x_cf) - target_proba) <= tol
    # else:
    # assert exp['n_iter'] == max_iter
    #    assert np.abs(pred_class_fn(x_cf) - target_proba) > tol
