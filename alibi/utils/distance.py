import numpy as np


def cityblock_batch(X: np.ndarray,
                    y: np.ndarray) -> np.ndarray:
    """
    Calculate the L1 distances between a batch of arrays X and an array of the same shape y.

    Parameters
    ----------
    X
        Batch of arrays to calculate the distances from
    y
        Array to calculate the distance to

    Returns
    -------
    Array of distances from each array in X to y

    """
    X_dim = len(X.shape)
    y_dim = len(y.shape)

    if X_dim == y_dim:
        assert y.shape[0] == 1, 'y must have batch size equal to 1'
    else:
        assert X.shape[1:] == y.shape, 'X and y must have matching shapes'

    return np.abs(X - y).sum(axis=tuple(np.arange(1, X_dim))).reshape(X.shape[0], -1)

def mvdm(X: np.ndarray,
         y: np.ndarray,
         cat_vars: dict,
         alpha: int = 1) -> np.ndarray:
    """
    Calculate the pair-wise distances between categories of a categorical variable using
    the Modified Value Difference Measure.
    https://link.springer.com/content/pdf/10.1023/a:1022664626993.pdf

    Parameters
    ----------
    X
        Batch of arrays.
    y
        Batch of labels or predictions.
    cat_vars
        Dict with as keys the categorical columns and as optional values
        the number of categories per categorical variable.
    alpha
        Power of absolute difference between conditional probabilities.

    Returns
    -------
    Dict with as keys the categorical columns and as values the pairwise distance matrix for the variable.
    """
    # TODO: vectorize the damn thing!
    # TODO: expand for y as ordinal or OHE of labels + support probabilities
    n_y = len(np.unique(y))
    cat_cols = list(cat_vars.keys())
    for col in cat_cols:
        if cat_vars[col] is not None:
            continue
        cat_vars[col] = len(np.unique(X[:, col]))

    # conditional probabilities and pairwise distance matrix
    d_pair = {}
    for col, n_cat in cat_vars.items():
        d_pair_col = np.zeros([n_cat, n_cat])
        p_cond_col = np.zeros([n_cat, n_y])
        for i in range(n_cat):
            idx = np.where(X[:, col] == i)
            for i_y in range(n_y):
                p_cond_col[i, i_y] = np.sum(y[idx] == i_y) / (y[idx].shape[0] + 1e-12)

        # not efficient now b/c matrix is symmetrical
        for i in range(n_cat):
            for j in range(n_cat):
                d_pair_col[i, j] = np.sum(np.abs(p_cond_col[i, :] - p_cond_col[j, :]) ** alpha)
        d_pair[col] = d_pair_col
    return d_pair


def abdm(X: np.ndarray,
         cat_vars: dict):
    """
    Calculate the pair-wise distances between categories of a categorical variable using
    the Association-Based Distance Metric.
    http://www.jaist.ac.jp/~bao/papers/N26.pdf

    Parameters
    ----------
    X
        Batch of arrays.
    cat_vars
        Dict with as keys the categorical columns and as optional values
        the number of categories per categorical variable.

    Returns
    -------
    Dict with as keys the categorical columns and as values the pairwise distance matrix for the variable.
    """
    # TODO: vectorize the damn thing!
    # TODO: proper numerical stabilization instead of current hack for KL divergence
    # https://mathoverflow.net/questions/72668/how-to-compute-kl-divergence-when-pmf-contains-0s
    # https://stats.stackexchange.com/questions/1028/questions-about-kl-divergence
    # https://www.sciencedirect.com/topics/mathematics/pairwise-distance
    # https://stats.stackexchange.com/questions/14127/how-to-compute-the-kullback-leibler-divergence-when-the-pmf-contains-0s

    # numerical stability variables
    p_cond_floor = 1e-5
    eps = 1e-12
    d_pair_ceil = 100000

    cat_cols = list(cat_vars.keys())
    for col in cat_cols:
        if cat_vars[col] is not None:
            continue
        cat_vars[col] = len(np.unique(X[:, col]))

    d_pair = {}
    X_cat_eq = {}
    for col, n_cat in cat_vars.items():
        X_cat_eq[col] = []
        for i in range(n_cat):
            idx = np.where(X[:, col] == i)[0]
            X_cat_eq[col].append(X[idx, :])

        # conditional probabilities
        p_cond = []
        for col_t, n_cat_t in cat_vars.items():
            if col == col_t:
                continue
            p_cond_t = np.zeros([n_cat_t, n_cat])
            for i in range(n_cat_t):
                for j, X_cat_j in enumerate(X_cat_eq[col]):
                    idx = np.where(X_cat_j[:, col_t] == i)[0]
                    p_cond_t[i, j] = len(idx) / (X_cat_j.shape[0] + 1e-12)
            p_cond.append(p_cond_t)

        # pairwise distance matrix
        # already doing double counting because of symmetry of matrix
        d_pair_col = np.zeros([n_cat, n_cat])
        for i in range(n_cat):
            for j in range(n_cat):
                if i == j:  # diagonal = 0
                    continue
                d_ij_tmp = 0
                for p in p_cond:  # loop over other categorical variables
                    for t in range(p.shape[0]):  # loop over categories of each categorical variable
                        if p[t, j] < p_cond_floor or p[t, i] < p_cond_floor:
                            continue  # numerical stability hack
                        d_ij_t = (p[t, i] * p[t, i] / (p[t, j] + eps) +
                                  p[t, j] * p[t, j] / (p[t, i] + eps))
                        d_ij_tmp += d_ij_t
                d_pair_col[i, j] = min(d_ij_tmp, d_pair_ceil)
        d_pair[col] = d_pair_col
    return d_pair
