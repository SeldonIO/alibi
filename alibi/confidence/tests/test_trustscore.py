from alibi.confidence import TrustScore
import pytest
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression


def test_trustscore():
    # load iris dataset
    dataset = load_iris()

    # define train and test set
    idx = 145
    X_train, Y_train = dataset.data[:idx, :], dataset.target[:idx]
    X_test, Y_test = dataset.data[idx + 1:, :], dataset.target[idx + 1:]  # noqa F841

    # fit logistic regression to training data
    np.random.seed(0)
    clf = LogisticRegression(solver='liblinear', multi_class='auto')
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)

    #