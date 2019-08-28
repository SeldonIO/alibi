# TODO: probably remove file if already in mapping.py
import numpy as np
from typing import Tuple


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
        Dict with as keys the categorical columns and as values
        the number of categories per categorical variable.

    Returns
    -------
    Ordinal equivalent of one-hot encoded data and dict with categorical columns and number of categories.
    """
    n, cols = X_ohe.shape
    ohe_vars_keys = list(cat_vars_ohe.keys())
    X_list = []
    c = 0
    cat_vars_ord = {}
    while c < cols:
        if c in ohe_vars_keys:
            v = cat_vars_ohe[c]
            X_ohe_c = X_ohe[:, c:c+v]
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
