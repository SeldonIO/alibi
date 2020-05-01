import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
import pytest
from alibi.explainers.ale import ale_num


@pytest.mark.parametrize("min_bin_points", [1, 4, 10])
def test_ale_num_linear_regression(min_bin_points):
    """
    The slope of the ALE of linear regression should equal the learnt coefficients
    """
    X, y = load_boston(return_X_y=True)
    lr = LinearRegression().fit(X, y)
    for feature in range(X.shape[1]):
        q, ale, _ = ale_num(lr.predict, X, feature=feature, min_bin_points=min_bin_points)
        assert np.allclose((ale[-1] - ale[0]) / (X[:, feature].max() - X[:, feature].min()), lr.coef_[feature])
