import numpy as np
from sklearn.manifold import MDS
from typing import Dict, Tuple


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
    the Modified Value Difference Measure based on Cost et al (1993).
    https://link.springer.com/article/10.1023/A:1022664626993

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
    # TODO: handle triangular inequality
    # infer number of categories per categorical variable
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
            idx = np.where(X[:, col] == i)[0]
            for i_y in range(n_y):
                p_cond_col[i, i_y] = np.sum(y[idx] == i_y) / (y[idx].shape[0] + 1e-12)

        for i in range(n_cat):
            j = 0
            while j < i:  # symmetrical matrix
                d_pair_col[i, j] = np.sum(np.abs(p_cond_col[i, :] - p_cond_col[j, :]) ** alpha)
                j += 1
        d_pair_col += d_pair_col.T
        d_pair[col] = d_pair_col
    return d_pair


def abdm(X: np.ndarray,
         cat_vars: dict,
         cat_vars_bin: dict = dict()):
    """
    Calculate the pair-wise distances between categories of a categorical variable using
    the Association-Based Distance Metric based on Le et al (2005).
    http://www.jaist.ac.jp/~bao/papers/N26.pdf

    Parameters
    ----------
    X
        Batch of arrays.
    cat_vars
        Dict with as keys the categorical columns and as optional values
        the number of categories per categorical variable.
    cat_vars_bin
        Dict with as keys the binned numerical columns and as optional values
        the number of bins per variable.

    Returns
    -------
    Dict with as keys the categorical columns and as values the pairwise distance matrix for the variable.
    """
    # TODO: handle triangular inequality
    # ensure numerical stability
    eps = 1e-12

    # infer number of categories per categorical variable
    cat_cols = list(cat_vars.keys())
    for col in cat_cols:
        if cat_vars[col] is not None:
            continue
        cat_vars[col] = len(np.unique(X[:, col]))

    # combine dict for categorical with binned features
    cat_vars_combined = {**cat_vars, **cat_vars_bin}

    d_pair = {}  # type: Dict
    X_cat_eq = {}  # type: Dict
    for col, n_cat in cat_vars.items():
        X_cat_eq[col] = []
        for i in range(n_cat):  # for each category in categorical variable, store instances of each category
            idx = np.where(X[:, col] == i)[0]
            X_cat_eq[col].append(X[idx, :])

        # conditional probabilities, also use the binned numerical features
        p_cond = []
        for col_t, n_cat_t in cat_vars_combined.items():
            if col == col_t:
                continue
            p_cond_t = np.zeros([n_cat_t, n_cat])
            for i in range(n_cat_t):
                for j, X_cat_j in enumerate(X_cat_eq[col]):
                    idx = np.where(X_cat_j[:, col_t] == i)[0]
                    p_cond_t[i, j] = len(idx) / (X_cat_j.shape[0] + eps)
            p_cond.append(p_cond_t)

        # pairwise distance matrix
        d_pair_col = np.zeros([n_cat, n_cat])
        for i in range(n_cat):
            j = 0
            while j < i:
                d_ij_tmp = 0
                for p in p_cond:  # loop over other categorical variables
                    for t in range(p.shape[0]):  # loop over categories of each categorical variable
                        a, b = p[t, i], p[t, j]
                        d_ij_t = a * np.log((a + eps) / (b + eps)) + b * np.log((b + eps) / (a + eps))  # KL divergence
                        d_ij_tmp += d_ij_t
                d_pair_col[i, j] = d_ij_tmp
                j += 1
        d_pair_col += d_pair_col.T
        d_pair[col] = d_pair_col
    return d_pair


def multidim_scaling(d_pair: dict,
                     n_components: int = 2,
                     use_metric: bool = True,
                     standardize_cat_vars: bool = True,
                     feature_range: tuple = None,
                     smooth: float = 1.,
                     center: bool = True,
                     update_feature_range: bool = True) -> Tuple[dict, tuple]:
    """
    Apply multidimensional scaling to pairwise distance matrices.

    Parameters
    ----------
    d_pair
        Dict with as keys the column index of the categorical variables and as values
        a pairwise distance matrix for the categories of the variable.
    n_components
        Number of dimensions in which to immerse the dissimilarities.
    use_metric
        If True, perform metric MDS; otherwise, perform nonmetric MDS.
    standardize_cat_vars
        Standardize numerical values of categorical variables if True.
    feature_range
        Tuple with min and max ranges to allow for perturbed instances. Min and max ranges can be floats or
        numpy arrays with dimension (1 x nb of features) for feature-wise ranges.
    smooth
        Smoothing exponent between 0 and 1 for the distances. Lower values of l will smooth the difference in
        distance metric between different features.
    center
        Whether to center the scaled distance measures. If False, the min distance for each feature
        except for the feature with the highest raw max distance will be the lower bound of the
        feature range, but the upper bound will be below the max feature range.
    update_feature_range
        Update feature range with scaled values.

    Returns
    -------
    Dict with multidimensional scaled version of pairwise distance matrices.
    """
    d_abs = {}
    d_min, d_max = 1e10, 0
    for k, v in d_pair.items():
        # distance smoothening
        v **= smooth
        # fit multi-dimensional scaler
        mds = MDS(n_components=n_components, max_iter=5000, eps=1e-9, random_state=0, n_init=4,
                  dissimilarity="precomputed", metric=use_metric)
        d_fit = mds.fit(v)
        emb = d_fit.embedding_  # coordinates in embedding space
        # use biggest single observation Frobenius norm as origin
        origin = np.argsort(np.linalg.norm(emb, axis=1))[-1]
        # calculate distance from origin for each category
        d_origin = np.linalg.norm(emb - emb[origin].reshape(1, -1), axis=1)
        # assign to category
        d_abs[k] = d_origin
        # update min and max distance
        d_min_k, d_max_k = d_origin.min(), d_origin.max()
        d_min = d_min_k if d_min_k < d_min else d_min
        d_max = d_max_k if d_max_k > d_max else d_max

    d_abs_scaled = {}
    new_feature_range = tuple([f.copy() for f in feature_range])
    for k, v in d_abs.items():
        if standardize_cat_vars:  # scale numerical values for the category
            d_scaled = (v - v.mean()) / (v.std() + 1e-12)
        else:  # scale by overall min and max
            try:
                rng = (feature_range[0][0, k], feature_range[1][0, k])
            except TypeError:
                raise TypeError('Feature-wise min and max ranges need to be specified.')
            d_scaled = (v - d_min) / (d_max - d_min) * (rng[1] - rng[0]) + rng[0]
            if center:  # center the numerical feature values between the min and max feature range
                d_scaled -= .5 * (d_scaled.max() + d_scaled.min())
        if update_feature_range:
            new_feature_range[0][0, k] = d_scaled.min()
            new_feature_range[1][0, k] = d_scaled.max()
        d_abs_scaled[k] = d_scaled  # scaled distance from the origin for each category

    if update_feature_range:
        feature_range = new_feature_range

    return d_abs_scaled, feature_range
