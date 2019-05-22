import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from alibi.utils.distance import cityblock_batch
from alibi.utils.gradients import num_grad_batch


@pytest.fixture
def logistic_iris():
    X, y = load_iris(return_X_y=True)
    lr = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=200).fit(X, y)
    return X, y, lr


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
