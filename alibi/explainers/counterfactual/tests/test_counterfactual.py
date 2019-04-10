import pytest
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import cityblock

from alibi.explainers.counterfactual.counterfactual import _define_func, num_grad


@pytest.fixture
def logistic_iris():
    X, y = load_iris(return_X_y=True)
    lr = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=200).fit(X, y)
    return X, y, lr


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


@pytest.mark.parametrize('dim', [1, 2, 3, 10])
def test_get_num_gradients_cityblock(dim):
    u = np.random.rand(dim)
    v = np.random.rand(dim)

    grad_true = np.sign(u - v)
    grad_approx = num_grad(cityblock, u, args=tuple([v])).flatten()  # promote 0-D to 1-D

    assert grad_approx.shape == grad_true.shape
    assert np.allclose(grad_true, grad_approx)


def test_get_num_gradients_logistic_iris(logistic_iris):
    X, y, lr = logistic_iris
    predict_fn = lr.predict_proba
    x = X[0].reshape(1, -1)
    probas = predict_fn(x)
    pred_class = probas.argmax()

    # true gradient of the softmax function wrt x
    grad_true = (probas.T * (np.eye(3, 3) - probas) @ lr.coef_).sum(axis=1)
    grad_approx = num_grad(predict_fn, x)

    assert grad_approx.shape == grad_true.shape
    assert np.allclose(grad_true, grad_approx)

    # now restrict to just one class
    func, target = _define_func(predict_fn, pred_class, 'same')
    grad_true = grad_true[0]
    grad_approx = num_grad(func, x)

    assert grad_approx.shape == grad_true.shape
    assert np.allclose(grad_true, grad_approx)
