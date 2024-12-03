import pytest
import numpy as np
from typing import Tuple
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression


def logistic_iris():
    X, y = load_iris(return_X_y=True)
    lr = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=200).fit(X, y)
    return X, y, lr


def cf_iris_explainer(predict_fn, target_class):
    from alibi.explainers import Counterfactual
    return Counterfactual(
        predict_fn=predict_fn,
        shape=(1, 4),
        target_class=target_class,
        lam_init=1e-1, max_iter=1000,
        max_lam_steps=10
    )


@pytest.mark.parametrize('target_class', ['other', 'same', 0, 1, 2])
def test_define_func(target_class):
    X, y, model = logistic_iris()

    x = X[0].reshape(1, -1)
    predict_fn = model.predict_proba
    probas = predict_fn(x)
    pred_class = probas.argmax(axis=1)[0]
    pred_prob = probas[:, pred_class][0]

    from alibi.explainers.counterfactual import _define_func
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


@pytest.mark.parametrize('target_class', ['same', 0, 1, 2])
def test_cf_explainer_iris(set_env_variables, target_class):
    X, _, lr = logistic_iris()
    cf = cf_iris_explainer(lr.predict_proba, target_class)

    x = X[0].reshape(1, -1)
    probas = cf.predict_fn(x)
    pred_class = probas.argmax()

    assert cf.data_shape == (1, 4)

    # test explanation
    exp = cf.explain(x)
    from alibi.api.defaults import DEFAULT_META_CF, DEFAULT_DATA_CF
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


def tf_models(tf_models: Tuple[str]):
    """
    This fixture loads a list of pre-trained test-models by name from the
    alibi-testing helper package.
    """
    from alibi_testing.loading import load
    return [load(name) for name in tf_models]


def tf_keras_mnist_cf_explainer(models, target_class):
    from alibi.explainers import Counterfactual
    cf_explainer = Counterfactual(
        predict_fn=models[0],
        shape=(1, 28, 28, 1),
        target_class=target_class,
        lam_init=1e-1, max_iter=1000,
        max_lam_steps=10
    )
    return models[0], cf_explainer


@pytest.mark.parametrize('tf_model_names', [('mnist-logistic-tf2.18.0.h5',)],)
@pytest.mark.parametrize('target_class', ['other', 'same', 4, 9])
def test_keras_mnist_explainer(set_env_variables, tf_model_names, target_class):
    # load data
    from alibi_testing.data import get_mnist_data
    mnist_data = get_mnist_data()

    # load models
    models = tf_models(tf_model_names)

    # load explaine
    _, cf = tf_keras_mnist_cf_explainer(models, target_class)
    assert cf.data_shape == (1, 28, 28, 1)

    # instance to be explained
    X = mnist_data['X_train']
    x = X[0:1]

    probas = cf.predict_fn(x)
    pred_class = probas.argmax()

    # test explanation
    exp = cf.explain(x)
    from alibi.api.defaults import DEFAULT_META_CF, DEFAULT_DATA_CF
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
