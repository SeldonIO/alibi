# flake8: noqa E731
import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import keras
from alibi.api.defaults import DEFAULT_META_CFP, DEFAULT_DATA_CFP
from alibi.datasets import fetch_adult
from alibi.explainers import CounterFactualProto
from alibi.utils.mapping import ord_to_ohe, ohe_to_ord, ord_to_num


@pytest.fixture
def tf_keras_iris_model(request):
    if request.param == 'keras':
        k = keras
    elif request.param == 'tf':
        k = tf.keras
    else:
        raise ValueError('Unknown parameter')

    x_in = k.layers.Input(shape=(4,))
    x = k.layers.Dense(10, activation='relu')(x_in)
    x_out = k.layers.Dense(3, activation='softmax')(x)
    model = k.models.Model(inputs=x_in, outputs=x_out)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return model


@pytest.fixture
def tf_keras_iris_ae(request):
    if request.param == 'keras':
        k = keras
    elif request.param == 'tf':
        k = tf.keras
    else:
        raise ValueError('Unknown parameter')

    # encoder
    x_in = k.layers.Input(shape=(4,))
    x = k.layers.Dense(5, activation='relu')(x_in)
    encoded = k.layers.Dense(2, activation=None)(x)
    encoder = k.models.Model(x_in, encoded)

    # decoder
    dec_in = k.layers.Input(shape=(2,))
    x = k.layers.Dense(5, activation='relu')(dec_in)
    decoded = k.layers.Dense(4, activation=None)(x)
    decoder = k.models.Model(dec_in, decoded)

    # autoencoder = encoder + decoder
    x_out = decoder(encoder(x_in))
    autoencoder = k.models.Model(x_in, x_out)
    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder, encoder, decoder


@pytest.fixture
def tf_keras_iris(tf_keras_iris_model, tf_keras_iris_ae):
    X, y = load_iris(return_X_y=True)
    X = (X - X.mean(axis=0)) / X.std(axis=0)  # scale dataset

    idx = 145
    X_train, y_train = X[:idx, :], y[:idx]
    # y_train = to_categorical(y_train) # TODO: fine to leave as is?

    # set random seed
    np.random.seed(1)
    tf.set_random_seed(1)

    model = tf_keras_iris_model
    model.fit(X_train, y_train, batch_size=128, epochs=500, verbose=0)

    ae, enc, _ = tf_keras_iris_ae
    ae.fit(X_train, X_train, batch_size=32, epochs=100, verbose=0)

    return X_train, model, ae, enc


@pytest.fixture
def tf_keras_iris_explainer(request, tf_keras_iris):
    X_train, model, ae, enc = tf_keras_iris

    if request.param[0]:  # use k-d trees
        ae = None
        enc = None

    shape = (1, 4)
    cf_explainer = CounterFactualProto(model, shape, gamma=100, theta=100,
                                       ae_model=ae, enc_model=enc, use_kdtree=request.param[0],
                                       max_iterations=1000, c_init=request.param[1], c_steps=request.param[2],
                                       feature_range=(X_train.min(axis=0).reshape(shape),
                                                      X_train.max(axis=0).reshape(shape)))
    yield X_train, model, cf_explainer


@pytest.mark.parametrize('tf_keras_iris_explainer,use_kdtree,k', [
    ((False, 0., 1), False, None),
    ((False, 1., 3), False, None),
    ((False, 0., 1), False, 2),
    ((False, 1., 3), False, 2),
    ((True, 0., 1), True, None),
    ((True, 1., 3), True, None),
    ((True, 0., 1), True, 2),
    ((True, 1., 3), True, 2)
], indirect=['tf_keras_iris_explainer'])
@pytest.mark.parametrize('tf_keras_iris_model,tf_keras_iris_ae', [('tf', 'tf'), ('keras', 'keras')],
                         indirect=True)
def test_tf_keras_iris_explainer(tf_keras_iris_explainer, use_kdtree, k):
    X_train, model, cf = tf_keras_iris_explainer

    # instance to be explained
    x = X_train[0].reshape(1, -1)
    pred_class = np.argmax(model.predict(x))
    not_pred_class = np.argmin(model.predict(x))

    # test fit
    cf.fit(X_train)
    if use_kdtree:  # k-d trees
        assert len(cf.kdtrees) == cf.classes  # each class has a k-d tree
        n_by_class = 0
        for c in range(cf.classes):
            n_by_class += cf.X_by_class[c].shape[0]
        assert n_by_class == X_train.shape[0]  # all training instances are stored in the trees
        assert cf.kdtrees[pred_class].query(x, k=1)[0] == 0.  # nearest distance to own class equals 0
        assert cf.score(x, not_pred_class, pred_class) == 0.  # test score fn
    else:  # encoder
        assert len(list(cf.class_proto.keys())) == cf.classes
        assert [True for _ in range(cf.classes)] == [v.shape == (1, 2) for _, v in cf.class_proto.items()]
        n_by_class = 0
        for c in range(cf.classes):
            n_by_class += cf.class_enc[c].shape[0]
        assert n_by_class == X_train.shape[0]  # all training instances encoded

    # test explanation
    explanation = cf.explain(x, k=k)
    assert cf.id_proto != pred_class
    assert np.argmax(model.predict(explanation.cf['X'])) == explanation.cf['class']
    assert explanation.cf['grads_num'].shape == explanation.cf['grads_graph'].shape == x.shape
    assert explanation.meta.keys() == DEFAULT_META_CFP.keys()
    assert explanation.data.keys() == DEFAULT_DATA_CFP.keys()

    # test gradient shapes
    y = np.zeros((1, cf.classes))
    np.put(y, pred_class, 1)
    cf.predict = cf.predict.predict  # make model black box
    grads = cf.get_gradients(x, y, x.shape[1:])
    assert grads.shape == x.shape


@pytest.fixture
def tf_keras_adult_model(request):
    if request.param == 'keras':
        k = keras
    elif request.param == 'tf':
        k = tf.keras
    else:
        raise ValueError('Unknown parameter')

    x_in = k.layers.Input(shape=(57,))
    x = k.layers.Dense(60, activation='relu')(x_in)
    x = k.layers.Dense(60, activation='relu')(x)
    x_out = k.layers.Dense(2, activation='softmax')(x)
    model = k.models.Model(inputs=x_in, outputs=x_out)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


@pytest.fixture
def tf_keras_adult(tf_keras_adult_model):
    # fetch data
    adult = fetch_adult()
    X = adult.data
    X_ord = np.c_[X[:, 1:8], X[:, 11], X[:, 0], X[:, 8:11]]
    y = adult.target

    # scale numerical features
    X_num = X_ord[:, -4:].astype(np.float32, copy=False)
    xmin, xmax = X_num.min(axis=0), X_num.max(axis=0)
    rng = (-1., 1.)
    X_num_scaled = (X_num - xmin) / (xmax - xmin) * (rng[1] - rng[0]) + rng[0]

    # OHE categorical features
    X_cat = X_ord[:, :-4].copy()
    ohe = OneHotEncoder()
    ohe.fit(X_cat)
    X_cat_ohe = ohe.transform(X_cat)

    # combine categorical and numerical data
    X_comb = np.c_[X_cat_ohe.todense(), X_num_scaled].astype(np.float32, copy=False)

    # split in train and test set
    idx = 30000
    X_train, y_train = X_comb[:idx, :], y[:idx]

    assert X_train.shape[1] == 57

    # set random seed
    np.random.seed(1)
    tf.set_random_seed(1)

    model = tf_keras_adult_model
    model.fit(X_train, to_categorical(y_train), batch_size=128, epochs=5, verbose=0)

    # create categorical variable dict
    cat_vars_ord = {}
    n_categories = 8
    for i in range(n_categories):
        cat_vars_ord[i] = len(np.unique(X_ord[:, i]))
    cat_vars_ohe = ord_to_ohe(X_ord, cat_vars_ord)[1]

    return X_train, model, cat_vars_ohe


@pytest.fixture
def tf_keras_adult_explainer(request, tf_keras_adult):
    X_train, model, cat_vars_ohe = tf_keras_adult

    shape = (1, 57)
    cf_explainer = CounterFactualProto(model, shape, beta=.01, cat_vars=cat_vars_ohe, ohe=True,
                                       use_kdtree=request.param[0], max_iterations=1000,
                                       c_init=request.param[1], c_steps=request.param[2],
                                       feature_range=(-1 * np.ones((1, 12)), np.ones((1, 12))))
    yield X_train, model, cf_explainer


@pytest.mark.parametrize('tf_keras_adult_explainer,use_kdtree,k,d_type', [
    ((False, 1., 3), False, None, 'mvdm'),
    ((True, 1., 3), True, 2, 'mvdm'),
    ((True, 1., 3), True, 2, 'abdm'),
], indirect=['tf_keras_adult_explainer'])
@pytest.mark.parametrize('tf_keras_adult_model', ['tf', 'keras'], indirect=True)
def test_tf_keras_adult_explainer(tf_keras_adult_explainer, use_kdtree, k, d_type):
    X_train, model, cf = tf_keras_adult_explainer

    # instance to be explained
    x = X_train[0].reshape(1, -1)
    pred_class = np.argmax(model.predict(x))
    not_pred_class = np.argmin(model.predict(x))

    # test fit
    cf.fit(X_train, d_type=d_type)

    # checked ragged tensor shape
    n_cat = len(list(cf.cat_vars_ord.keys()))
    max_key = max(cf.cat_vars_ord, key=cf.cat_vars_ord.get)
    max_cat = cf.cat_vars_ord[max_key]
    assert cf.d_abs_ragged.shape == (n_cat, max_cat)

    if use_kdtree:  # k-d trees
        assert len(cf.kdtrees) == cf.classes  # each class has a k-d tree
        n_by_class = 0
        for c in range(cf.classes):
            n_by_class += cf.X_by_class[c].shape[0]
        assert n_by_class == X_train.shape[0]  # all training instances are stored in the trees

    # test explanation
    explanation = cf.explain(x, k=k)
    if use_kdtree:
        assert cf.id_proto != pred_class
    assert np.argmax(model.predict(explanation.cf['X'])) == explanation.cf['class']
    num_shape = (1, 12)
    assert explanation.cf['grads_num'].shape == explanation.cf['grads_graph'].shape == num_shape
    assert explanation.meta.keys() == DEFAULT_META_CFP.keys()
    assert explanation.data.keys() == DEFAULT_DATA_CFP.keys()

    # test gradient shapes
    y = np.zeros((1, cf.classes))
    np.put(y, pred_class, 1)
    cf.predict = cf.predict.predict  # make model black box
    # convert instance to numerical space
    x_ord = ohe_to_ord(x, cf.cat_vars)[0]
    x_num = ord_to_num(x_ord, cf.d_abs)
    # check gradients
    grads = cf.get_gradients(x_num, y, num_shape[1:], cf.cat_vars_ord)
    assert grads.shape == num_shape
