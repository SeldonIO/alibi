import pytest
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import tensorflow as tf

from alibi.api.defaults import DEFAULT_META_CF, DEFAULT_DATA_CF
from alibi.explainers.counterfactual import _define_func
from alibi.explainers import Counterfactual


@pytest.fixture
def logistic_iris():
    X, y = load_iris(return_X_y=True)
    lr = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=200).fit(X, y)
    return X, y, lr


@pytest.fixture
def cf_iris_explainer(request, logistic_iris):
    X, y, lr = logistic_iris
    predict_fn = lr.predict_proba
    cf_explainer = Counterfactual(predict_fn=predict_fn, shape=(1, 4),
                                  target_class=request.param, lam_init=1e-1, max_iter=1000,
                                  max_lam_steps=10)

    yield X, y, lr, cf_explainer
    tf.keras.backend.clear_session()


@pytest.fixture
def keras_mnist_cf_explainer(request, models):
    cf_explainer = Counterfactual(predict_fn=models[0], shape=(1, 28, 28, 1),
                                  target_class=request.param, lam_init=1e-1, max_iter=1000,
                                  max_lam_steps=10)
    yield models[0], cf_explainer
    tf.keras.backend.clear_session()


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


@pytest.mark.tf1
@pytest.mark.parametrize('cf_iris_explainer',
                         ['other', 'same', 0, 1, 2],
                         ids='target={}'.format,
                         indirect=True)
def test_cf_explainer_iris(disable_tf2, cf_iris_explainer):
    X, y, lr, cf = cf_iris_explainer
    x = X[0].reshape(1, -1)
    probas = cf.predict_fn(x)
    pred_class = probas.argmax()

    assert cf.data_shape == (1, 4)

    # test explanation
    exp = cf.explain(x)
    assert exp.meta.keys() == DEFAULT_META_CF.keys()
    assert exp.data.keys() == DEFAULT_DATA_CF.keys()

    x_cf = exp.cf['X']
    assert x.shape == x_cf.shape

    probas_cf = cf.predict_fn(x_cf)
    pred_class_cf = probas_cf.argmax()

    # get attributes for testing
    target_class = cf.target_class
    target_proba = cf.sess.run(cf.target_proba)
    tol = cf.tol
    pred_class_fn = cf.predict_class_fn

    # check if target_class condition is met
    if target_class == 'same':
        assert pred_class == pred_class_cf
    elif target_class == 'other':
        assert pred_class != pred_class_cf
    elif isinstance(target_class, int):
        assert pred_class_cf == target_class

    if exp.success:
        assert np.abs(pred_class_fn(x_cf) - target_proba) <= tol


@pytest.mark.tf1
@pytest.mark.parametrize('keras_mnist_cf_explainer',
                         ['other', 'same', 4, 9],
                         ids='target={}'.format,
                         indirect=True)
@pytest.mark.parametrize('models',
                         [('mnist-logistic-tf2.2.0',), ('mnist-logistic-tf1.15.2.h5',)],
                         ids='model={}'.format,
                         indirect=True)
def test_keras_mnist_explainer(disable_tf2, keras_mnist_cf_explainer, mnist_data):
    model, cf = keras_mnist_cf_explainer
    X = mnist_data['X_train']

    x = X[0:1]
    probas = cf.predict_fn(x)
    pred_class = probas.argmax()

    assert cf.data_shape == (1, 28, 28, 1)

    # test explanation
    exp = cf.explain(x)
    assert exp.meta.keys() == DEFAULT_META_CF.keys()
    assert exp.data.keys() == DEFAULT_DATA_CF.keys()

    x_cf = exp.cf['X']
    assert x.shape == x_cf.shape

    probas_cf = cf.predict_fn(x_cf)
    pred_class_cf = probas_cf.argmax()

    # get attributes for testing
    target_class = cf.target_class
    target_proba = cf.sess.run(cf.target_proba)
    tol = cf.tol
    pred_class_fn = cf.predict_class_fn

    # check if target_class condition is met
    if target_class == 'same':
        assert pred_class == pred_class_cf
    elif target_class == 'other':
        assert pred_class != pred_class_cf
    elif isinstance(target_class, int):
        assert pred_class_cf == target_class

    if exp.success:
        assert np.abs(pred_class_fn(x_cf) - target_proba) <= tol
