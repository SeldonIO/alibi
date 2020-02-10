import pytest
import numpy as np

from alibi.explainers.kernel_shap import sum_categories


def get_random_maxtrix(*, n_rows=500, n_cols=100):
    return np.random.random(size=(n_rows, n_cols))


sum_categories_inputs = [
    (50, [3, 6, 4, 4], None),
    (50, None, [0, 6, 5, 12]),
    (100, [3, 6, 4, 4],  [0, 6, 15, 22]),
    (5, [3, 2, 4], [0, 5, 9]),
    (10, [3, 3, 4], [0, 3, 6])
]


@pytest.mark.parametrize('n_feats, feat_enc_dim, start_idx', sum_categories_inputs)
def test_sum_categories(n_feats, feat_enc_dim, start_idx):
    """
    This function tests if the summing the columns
    corresponding to a categorical variables into
    one variable works properly.
    """

    # create inputs to feed the function
    X = get_random_maxtrix(n_cols=n_feats)

    # check a value error is raised if start indices or
    # encoding lengths are not provided
    if feat_enc_dim is None or start_idx is None:
        with pytest.raises(ValueError) as exc_info:
            summ_X = sum_categories(X, start_idx, feat_enc_dim)
            assert exc_info.type is ValueError
    elif len(feat_enc_dim) != len(start_idx):
        with pytest.raises(ValueError) as exc_info:
            summ_X = sum_categories(X, start_idx, feat_enc_dim)
            assert exc_info.type is ValueError

    # check if sum of encodings greater than num columns raises value error
    elif sum(feat_enc_dim) > n_feats:
        with pytest.raises(ValueError) as exc_info:
            summ_X = sum_categories(X, start_idx, feat_enc_dim)
            assert exc_info.type is ValueError

    # check that if inputs are correct, we retrive the sum in the correct col
    else:
        summ_X = sum_categories(X, start_idx, feat_enc_dim)
        assert summ_X.shape[1] == X.shape[1] - sum(feat_enc_dim) + len(feat_enc_dim)
        for i, enc_dim in enumerate(feat_enc_dim):
            # work out the index of the summed column in the returned matrix
            sum_col_idx = start_idx[i] - sum(feat_enc_dim[:i]) + len(feat_enc_dim[:i])
            diff = summ_X[:, sum_col_idx] - np.sum(X[:, start_idx[i]:start_idx[i] + feat_enc_dim[i]], axis=1)
            assert diff.sum() == 0.0
