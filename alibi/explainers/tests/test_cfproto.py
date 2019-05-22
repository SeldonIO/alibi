# flake8: noqa E731
import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from alibi.explainers import CounterFactualProto

def test_cfproto():
    # load iris dataset
    dataset = load_iris()

    # scale dataset
    dataset.data = (dataset.data - dataset.data.mean(axis=0)) / dataset.data.std(axis=0)

    # define train and test set
    X, Y = dataset.data, dataset.target

    # fit random forest to training data
    np.random.seed(0)
    clf = LogisticRegression(solver='liblinear')
    clf.fit(X, Y)

    # define prediction function
    predict_fn = lambda x: clf.predict_proba(x)

    # instance to be explained
    idx = 0
    X_expl = X[idx].reshape((1,) + X[idx].shape)
    pred_class = clf.predict(X_expl)[0]
    not_pred_class = np.argmin(predict_fn(X_expl), axis=1)[0]

    # initialize explainer
    shape = (1, 4)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    cf = CounterFactualProto(sess, predict_fn, shape, use_kdtree=True, max_iterations=1000, theta=10.,
                             feature_range=(X.min(axis=0).reshape(shape), X.max(axis=0).reshape(shape)),
                             c_init=1., c_steps=5)

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

    sess.close()
