# flake8: noqa E731
import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from alibi.explainers import CounterFactualProto

@pytest.fixture
def logistic_iris():
    X, y = load_iris(return_X_y=True)
    X = (X - X.mean(axis=0)) / X.std(axis=0)  # scale dataset
    lr = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=200).fit(X, y)
    return X, y, lr


@pytest.fixture
def iris_explainer(logistic_iris):
    X, y, clf = logistic_iris

    # define prediction function
    predict_fn = lambda x: clf.predict_proba(x)

    # initialize explainer
    shape = (1, 4)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    cf = CounterFactualProto(sess, predict_fn, (1, 4), use_kdtree=True, max_iterations=500, theta=10.,
                             feature_range=(X.min(axis=0).reshape(shape), X.max(axis=0).reshape(shape)),
                             c_init=1., c_steps=3)
    yield X, y, clf, predict_fn, cf
    sess.close()


def test_cfproto(iris_explainer):
    X, y, clf, predict_fn, cf = iris_explainer
    # instance to be explained
    X_expl = X[0].reshape((1,) + X[0].shape)
    pred_class = clf.predict(X_expl)[0]
    not_pred_class = np.argmin(predict_fn(X_expl), axis=1)[0]

    # test fit
    cf.fit(X)
    assert len(cf.kdtrees) == cf.classes  # each class has a k-d tree
    n_by_class = 0
    for c in range(cf.classes):
        n_by_class += cf.X_by_class[c].shape[0]
    assert n_by_class == X.shape[0]  # all training instances are stored in the trees
    assert cf.kdtrees[pred_class].query(X_expl, k=1)[0] == 0.  # nearest distance to own class equals 0

    # test score fn
    assert cf.score(X_expl, not_pred_class, pred_class) == 0.

    # test explanation
    explanation = cf.explain(X_expl)
    assert cf.id_proto != pred_class
    assert clf.predict(explanation['CF']) == cf.id_proto == explanation['CF_pred']
    assert (explanation['X'] == X_expl).all()
    assert explanation['grads_num'].shape == explanation['grads_graph'].shape == X_expl.shape

    # test gradient shapes
    Y_expl = np.zeros((1, cf.classes))
    np.put(Y_expl, pred_class, 1)
    grads = cf.get_gradients(X_expl, Y_expl)
    assert grads.shape == X_expl.shape
