# flake8: noqa E731

from alibi.api.defaults import DEFAULT_META_CEM, DEFAULT_DATA_CEM
from alibi.explainers import CEM
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression


def test_cem():
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

    # test explainer initialization
    shape = (1, 4)
    feature_range = (X.min(axis=0).reshape(shape) - .1, X.max(axis=0).reshape(shape) + .1)
    cem = CEM(predict_fn, 'PN', shape, feature_range=feature_range, max_iterations=10, no_info_val=-1.)
    explanation = cem.explain(X_expl, verbose=False)

    assert not cem.model
    assert set(explanation.data.keys()) >= {'X', 'X_pred', 'PN', 'PN_pred', 'grads_graph', 'grads_num'}
    assert (explanation.X != explanation.PN).astype(int).sum() > 0
    assert explanation.X_pred != explanation.PN_pred
    assert explanation.grads_graph.shape == explanation.grads_num.shape
    assert explanation.meta.keys() == DEFAULT_META_CEM.keys()
    assert explanation.data.keys() == DEFAULT_DATA_CEM.keys()