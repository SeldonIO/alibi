# flake8: noqa E731

from alibi.explainers import CounterFactualAdversarialSearch
from alibi.explainers import CounterFactualRandomSearch
import numpy as np
import pytest
from sklearn import svm, datasets


# @pytest.mark.parametrize('predict_type', ('proba', 'class'))
# @pytest.mark.parametrize('threshold', (0.9, 0.95))
def test_iris_adversarial():

    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    clf = svm.SVC(gamma='scale', probability=True)
    clf.fit(X, y)

    x = X[:1]
    cf = CounterFactualAdversarialSearch(clf)
    cf.fit(X_train=X)

    assert hasattr(cf, 'f_ranges')
    assert hasattr(cf, 'mads')
    assert len(cf.f_ranges) == x.shape[1]
    assert len(cf.mads) == x.shape[1]

    expl = cf.explain(x)
    assert (expl.shape[1:] == x.shape[1:]), 'different shapes. expl shape: {}, x shape: {}'.format(expl.shape, x.shape)


def test_iris_randomsearch():
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    clf = svm.SVC(gamma='scale', probability=True)
    clf.fit(X, y)

    x = X[:1]
    cf = CounterFactualRandomSearch(clf)
    cf.fit(X_train=X)
    assert hasattr(cf, 'f_ranges')
    assert len(cf.f_ranges) == x.shape[1]

    expl = cf.explain(x)
    assert (expl.shape[1] == x.shape[1]), 'different shapes. expl shape: {}, x shape: {}'.format(expl.shape, x.shape)



