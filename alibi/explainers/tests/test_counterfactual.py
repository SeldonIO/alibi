# flake8: noqa E731
import pytest
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
import keras

from alibi.api.defaults import DEFAULT_META_CF, DEFAULT_DATA_CF
from alibi.explainers.counterfactual import _define_func
from alibi.explainers import CounterFactual


@pytest.fixture
def logistic_iris():
    X, y = load_iris(return_X_y=True)
    lr = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=200).fit(X, y)
    return X, y, lr


@pytest.fixture
def cf_iris_explainer(request, logistic_iris):
    X, y, lr = logistic_iris
    predict_fn = lr.predict_proba
    cf_explainer = CounterFactual(predict_fn=predict_fn, shape=(1, 4),
                                  target_class=request.param, lam_init=1e-1, max_iter=1000,
                                  max_lam_steps=10)

    yield X, y, lr, cf_explainer
    keras.backend.clear_session()
    tf.keras.backend.clear_session()


@pytest.fixture
def keras_logistic_mnist(request):
    if request.param == 'keras':
        k = keras
    elif request.param == 'tf':
        k = tf.keras
    else:
        raise ValueError('Unknown parameter')

    (X_train, y_train), (X_test, y_test) = k.datasets.mnist.load_data()
    input_dim = 784
    output_dim = nb_classes = 10

    X = X_train.reshape(60000, input_dim)[:1000]  # only train on 1000 instances
    X = X.astype('float32')
    X /= 255

    y = k.utils.to_categorical(y_train[:1000], nb_classes)

    model = k.models.Sequential([
        k.layers.Dense(output_dim,
                       input_dim=input_dim,
                       activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X, y, epochs=5)

    yield X, y, model
    keras.backend.clear_session()
    tf.keras.backend.clear_session()


@pytest.fixture
def keras_mnist_cf_explainer(request, keras_logistic_mnist):
    X, y, model = keras_logistic_mnist
    cf_explainer = CounterFactual(predict_fn=model, shape=(1, 784),
                                  target_class=request.param, lam_init=1e-1, max_iter=1000,
                                  max_lam_steps=10)
    yield X, y, model, cf_explainer
    keras.backend.clear_session()
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


@pytest.mark.parametrize('cf_iris_explainer', ['other', 'same', 0, 1, 2], indirect=True)
def test_cf_explainer_iris(cf_iris_explainer):
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


@pytest.mark.parametrize('keras_logistic_mnist', ['keras', 'tf'], indirect=True)
@pytest.mark.parametrize('keras_mnist_cf_explainer', ['other', 'same', 4, 9], indirect=True)
def test_keras_logistic_mnist_explainer(keras_logistic_mnist, keras_mnist_cf_explainer):
    X, y, model, cf = keras_mnist_cf_explainer
    x = X[0].reshape(1, -1)
    probas = cf.predict_fn(x)
    pred_class = probas.argmax()

    assert cf.data_shape == (1, 784)

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
