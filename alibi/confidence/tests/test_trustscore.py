from alibi.confidence import TrustScore
from keras.utils import to_categorical
import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression


@pytest.mark.parametrize('filter_type', (None, 'distance_knn', 'probability_knn'))
def test_trustscore(filter_type):
    # load iris dataset
    dataset = load_iris()

    # define train and test set
    idx = 140
    X_train, Y_train = dataset.data[:idx, :], dataset.target[:idx]
    X_test, Y_test = dataset.data[idx + 1:, :], dataset.target[idx + 1:]  # noqa F841

    # fit logistic regression to training data
    np.random.seed(0)
    clf = LogisticRegression(solver='liblinear', multi_class='auto')
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    Y_pred_proba = clf.predict_proba(X_test)

    alpha = .1

    # test filtering methods and fitted KDTrees
    kdtree_class_len = [50, 50, 40]
    if filter_type == 'distance_knn':
        kdtree_class_len = [int(tree * (1 - alpha)) for tree in kdtree_class_len]

    ts = TrustScore(k_filter=5, alpha=alpha, filter_type=filter_type)
    ts.fit(X_train, Y_train, classes=3)

    n_tree = 0
    for i, tree in enumerate(ts.kdtrees):
        if filter_type != 'probability_knn':
            assert tree.get_arrays()[0].shape[0] == kdtree_class_len[i]
        else:
            n_tree += tree.get_arrays()[0].shape[0]

    if filter_type == 'probability_knn':
        assert n_tree < X_train.shape[0]

    assert len(ts.kdtrees) == 3

    # check distances for the first class to itself and the first nearest neighbor in the KDTrees
    if filter_type is None:
        assert (ts.kdtrees[0].query(X_train, k=2)[0][:50, 0] != 0).astype(int).sum() == 0
        assert (ts.kdtrees[0].query(X_train, k=2)[0][:50, 1] == 0).astype(int).sum() == 0

    # test one-hot encoding of Y vs. class labels
    ts = TrustScore()
    ts.fit(X_train, Y_train, classes=3)
    score_class, _ = ts.score(X_test, Y_pred)
    ts = TrustScore()
    ts.fit(X_train, to_categorical(Y_train), classes=3)
    score_ohe, _ = ts.score(X_test, Y_pred_proba)
    assert (score_class != score_ohe).astype(int).sum() == 0
