# flake8: noqa E731
import numpy as np
import pytest
from sklearn.datasets import load_iris
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

from alibi.explainers import CounterFactualProto

@pytest.fixture
def tf_keras_iris_model():
    x_in = Input(shape=(4,))
    x = Dense(10, activation='relu')(x_in)
    x_out = Dense(3, activation='softmax')(x)
    model = Model(inputs=x_in, outputs=x_out)
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return model

@pytest.fixture
def tf_keras_iris_ae():
    # encoder
    x_in = Input(shape=(4,))
    x = Dense(5, activation='relu')(x_in)
    encoded = Dense(2, activation=None)(x)
    encoder = Model(x_in, encoded)

    # decoder
    dec_in = Input(shape=(2,))
    x = Dense(5, activation='relu')(dec_in)
    decoded = Dense(4, activation=None)(x)
    decoder = Model(dec_in, decoded)

    # autoencoder = encoder + decoder
    x_out = decoder(encoder(x_in))
    autoencoder = Model(x_in, x_out)
    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder, encoder, decoder


@pytest.fixture
def tf_keras_iris(tf_keras_iris_model, tf_keras_iris_ae):
    X, y = load_iris(return_X_y=True)
    X = (X - X.mean(axis=0)) / X.std(axis=0)  # scale dataset

    idx = 145
    X_train, y_train = X[:idx, :], y[:idx]
    y_train = to_categorical(y_train)

    # set random seed
    np.random.seed(1)
    tf.set_random_seed(1)

    # init tf session
    sess = tf.Session()
    K.set_session(sess)
    sess.run(tf.global_variables_initializer())

    model = tf_keras_iris_model
    model.fit(X_train, y_train, batch_size=128, epochs=500, verbose=0)

    ae, enc, _ = tf_keras_iris_ae
    ae.fit(X_train, X_train, batch_size=32, epochs=100, verbose=0)

    return X_train, model, ae, enc


@pytest.fixture
def tf_keras_iris_explainer(request, tf_keras_iris):
    X_train, model, ae, enc = tf_keras_iris
    sess = K.get_session()

    if request.param[0]:  # use k-d trees
        ae = None
        enc = None

    shape = (1, 4)
    cf_explainer = CounterFactualProto(sess, model, shape, gamma=100, theta=100,
                                       ae_model=ae, enc_model=enc, use_kdtree=request.param[0],
                                       max_iterations=1000, c_init=request.param[1], c_steps=request.param[2],
                                       feature_range=(X_train.min(axis=0).reshape(shape),
                                                      X_train.max(axis=0).reshape(shape)))
    yield X_train, model, cf_explainer, sess


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
def test_tf_keras_iris_explainer(tf_keras_iris_explainer, use_kdtree, k):
    X_train, model, cf, sess = tf_keras_iris_explainer

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
    assert np.argmax(model.predict(explanation['cf']['X'])) == explanation['cf']['class']
    assert explanation['cf']['grads_num'].shape == explanation['cf']['grads_graph'].shape == x.shape

    # test gradient shapes
    y = np.zeros((1, cf.classes))
    np.put(y, pred_class, 1)
    cf.predict = cf.predict.predict  # make model black box
    grads = cf.get_gradients(x, y)
    assert grads.shape == x.shape

    tf.reset_default_graph()
    sess.close()
