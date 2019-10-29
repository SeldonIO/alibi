import numpy as np
from typing import Tuple, List


def ohe_to_ord_shape(shape: tuple, cat_vars: dict = None, is_ohe: bool = False) -> tuple:
    """
    Infer shape of instance if the categorical variables have ordinal instead of on-hot encoding.

    Parameters
    ----------
    shape
        Instance shape, starting with batch dimension.
    cat_vars
        Dict with as keys the categorical columns and as values
        the number of categories per categorical variable.
    is_ohe
        Whether instance is OHE.

    Returns
    -------
    Tuple with shape of instance with ordinal encoding of categorical variables.
    """
    if not is_ohe:
        return shape
    else:
        n_cols_ohe = 0
        for _, v in cat_vars.items():
            n_cols_ohe += v - 1
        shape = (shape[0],) + (shape[-1] - n_cols_ohe,)
        return shape


def ord_to_num(data: np.ndarray, dist: dict) -> np.ndarray:
    """
    Transform categorical into numerical values using a mapping.

    Parameters
    ----------
    data
        Numpy array with the categorical data.
    dist
        Dict with as keys the categorical variables and as values
        the numerical value for each category.

    Returns
    -------
    Numpy array with transformed categorical data into numerical values.
    """
    rng = data.shape[0]
    X = data.astype(np.float32, copy=True)
    for k, v in dist.items():
        cat_col = X[:, k].copy()
        cat_col = np.array([v[int(cat_col[i])] for i in range(rng)])
        if type(X) == np.matrix:
            X[:, k] = cat_col.reshape(-1, 1)
        else:
            X[:, k] = cat_col
    return X.astype(np.float32)


def num_to_ord(data: np.ndarray, dist: dict) -> np.ndarray:
    """
    Transform numerical values into categories using the map calculated under the fit method.

    Parameters
    ----------
    data
        Numpy array with the numerical data.
    dist
        Dict with as keys the categorical variables and as values
        the numerical value for each category.

    Returns
    -------
    Numpy array with transformed numerical data into categories.
    """
    X = data.copy()
    for k, v in dist.items():
        num_col = np.repeat(X[:, k].reshape(-1, 1), v.shape[0], axis=1)
        diff = np.abs(num_col - v.reshape(1, -1))
        X[:, k] = np.argmin(diff, axis=1)
    return X


def ord_to_ohe(X_ord: np.ndarray, cat_vars_ord: dict) -> Tuple[np.ndarray, dict]:
    """
    Convert ordinal to one-hot encoded variables.

    Parameters
    ----------
    X_ord
        Data with mixture of ordinal encoded and numerical variables.
    cat_vars_ord
        Dict with as keys the categorical columns and as values
        the number of categories per categorical variable.

    Returns
    -------
    One-hot equivalent of ordinal encoded data and dict with categorical columns and number of categories.
    """
    n, cols = X_ord.shape
    ord_vars_keys = list(cat_vars_ord.keys())
    X_list = []
    c = 0
    k = 0
    cat_vars_ohe = {}
    while c < cols:
        if c in ord_vars_keys:
            v = cat_vars_ord[c]
            X_ohe_c = np.zeros((n, v), dtype=np.float32)
            X_ohe_c[np.arange(n), X_ord[:, c].astype(int)] = 1.
            cat_vars_ohe[k] = v
            k += v
            X_list.append(X_ohe_c)
        else:
            X_list.append(X_ord[:, c].reshape(n, 1))
            k += 1
        c += 1
    X_ohe = np.concatenate(X_list, axis=1)
    return X_ohe, cat_vars_ohe


def ohe_to_ord(X_ohe: np.ndarray, cat_vars_ohe: dict) -> Tuple[np.ndarray, dict]:
    """
    Convert one-hot encoded variables to ordinal encodings.

    Parameters
    ----------
    X_ohe
        Data with mixture of one-hot encoded and numerical variables.
    cat_vars_ohe
        Dict with as keys the first column index for each one-hot encoded categorical variable
        and as values the number of categories per categorical variable.

    Returns
    -------
    Ordinal equivalent of one-hot encoded data and dict with categorical columns and number of categories.
    """
    n, cols = X_ohe.shape
    ohe_vars_keys = list(cat_vars_ohe.keys())
    X_list = []  # type: List
    c = 0
    cat_vars_ord = {}
    while c < cols:
        if c in ohe_vars_keys:
            v = cat_vars_ohe[c]
            X_ohe_c = X_ohe[:, c:c + v]
            assert int(np.sum(X_ohe_c, axis=1).sum()) == n
            X_ord_c = np.argmax(X_ohe_c, axis=1)
            cat_vars_ord[len(X_list)] = v
            X_list.append(X_ord_c.reshape(n, 1))
            c += v
            continue
        X_list.append(X_ohe[:, c].reshape(n, 1))
        c += 1
    X_ord = np.concatenate(X_list, axis=1)
    return X_ord, cat_vars_ord
