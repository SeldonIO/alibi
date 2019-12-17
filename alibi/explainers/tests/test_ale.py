import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from alibi.explainers.ale import ale_num


def test_ale_num_linear_regression():
    """
    ALE of a linear regression should equal the learnt coefficients
    """
    X, y = load_boston(return_X_y=True)
    lr = LinearRegression().fit(X, y)
    for feature in range(X.shape[1]):
        # in the following we reduce num_intervals as some dimensions don't have enough data points and will
        # result in an error, some dimensions are also ordinally encoded categorical variables but we don't
        # distinguish those here
        q, ale = ale_num(lr.predict, X, feature=feature, num_intervals=10)
        assert np.allclose((ale[-1] - ale[0]) / (X[:, feature].max() - X[:, feature].min()), lr.coef_[feature])
