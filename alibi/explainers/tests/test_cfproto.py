import numpy as np
import pytest
import tensorflow as tf
from alibi.api.defaults import DEFAULT_META_CFP, DEFAULT_DATA_CFP
from alibi.explainers import CounterfactualProto
from alibi.utils.mapping import ohe_to_ord, ord_to_num


@pytest.fixture
def tf_keras_iris_explainer(request, models, iris_data):
    X_train = iris_data['X_train']
    model, ae, enc = models
    if request.param[0]:  # use k-d trees
        ae = None
        enc = None

    shape = (1, 4)
    cf_explainer = CounterfactualProto(model, shape, gamma=100, theta=100,
                                       ae_model=ae, enc_model=enc, use_kdtree=request.param[0],
                                       max_iterations=1000, c_init=request.param[1], c_steps=request.param[2],
                                       feature_range=(X_train.min(axis=0).reshape(shape),
                                                      X_train.max(axis=0).reshape(shape)))
    yield model, cf_explainer
    tf.keras.backend.clear_session()


@pytest.mark.tf1
@pytest.mark.parametrize('tf_keras_iris_explainer, use_kdtree, k',
                         [((False, 0., 1), False, None),
                          ((False, 1., 3), False, None),
                          ((False, 0., 1), False, 2),
                          ((False, 1., 3), False, 2),
                          ((True, 0., 1), True, None),
                          ((True, 1., 3), True, None),
                          ((True, 0., 1), True, 2),
                          ((True, 1., 3), True, 2)],
                         indirect=['tf_keras_iris_explainer'])
@pytest.mark.parametrize('models',
                         [('iris-ffn-tf2.2.0', 'iris-ae-tf2.2.0', 'iris-enc-tf2.2.0'),
                          ('iris-ffn-tf1.15.2.h5', 'iris-ae-tf1.15.2.h5', 'iris-enc-tf1.15.2.h5')],
                         indirect=True)
def test_tf_keras_iris_explainer(disable_tf2, iris_data, tf_keras_iris_explainer, use_kdtree, k):
    model, cf = tf_keras_iris_explainer
    X_train = iris_data['X_train']

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
def tf_keras_adult_explainer(request, models, adult_data):
    shape = (1, 57)
    cat_vars_ohe = adult_data['metadata']['cat_vars_ohe']
    cf_explainer = CounterfactualProto(models[0], shape, beta=.01, cat_vars=cat_vars_ohe, ohe=True,
                                       use_kdtree=request.param[0], max_iterations=1000,
                                       c_init=request.param[1], c_steps=request.param[2],
                                       feature_range=(-1 * np.ones((1, 12)), np.ones((1, 12))))
    yield models[0], cf_explainer
    tf.keras.backend.clear_session()


@pytest.mark.tf1
@pytest.mark.parametrize('tf_keras_adult_explainer, use_kdtree, k, d_type',
                         [((False, 1., 3), False, None, 'mvdm'),
                          ((True, 1., 3), True, 2, 'mvdm'),
                          ((True, 1., 3), True, 2, 'abdm')],
                         indirect=['tf_keras_adult_explainer'])
@pytest.mark.parametrize('models',
                         [('adult-ffn-tf2.2.0',), ('adult-ffn-tf1.15.2.h5',)],
                         ids='model={}'.format,
                         indirect=True)
def test_tf_keras_adult_explainer(disable_tf2, adult_data, tf_keras_adult_explainer, use_kdtree, k, d_type):
    model, cf = tf_keras_adult_explainer
    X_train = adult_data['preprocessor'].transform(adult_data['X_train']).toarray()

    # instance to be explained
    x = X_train[0].reshape(1, -1)
    pred_class = np.argmax(model.predict(x))

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
