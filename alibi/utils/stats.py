import numpy as np


def median_abs_deviation(X: np.ndarray) -> np.ndarray:
    """
    Computes the median of the feature-wise median absolute deviation from `X`

    Parameters
    ----------
    X
        Input array.

    Returns
    -------
        An array containing the median of the feature-wise median absolute deviation.
    """

    # TODO: ALEX: TBD: THROW WARNINGS IF THE FEATURE SCALE IS EITHER VERY LARGE OR VERY SMALL?
    # TODO: ALEX: TBD: MOVE TO UTILITY FILE, CREATE A SCALING DECORATOR, AUTOIMPORT THEN APPLY

    feat_median = np.median(X, axis=0)

    return np.median(np.abs(X - feat_median), axis=0)


def kl_bernoulli(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Compute KL-divergence between 2 probabilities p and q. len(p) divergences are calculated
    simultaneously.

    Parameters
    ----------
    p
        Probability.
    q
        Probability.

    Returns
    -------
        Array with the KL-divergence between p and q.
    """

    m = np.clip(p, 0.0000001, 0.9999999999999999).astype(float)
    n = np.clip(q, 0.0000001, 0.9999999999999999).astype(float)

    return m * np.log(m / n) + (1. - m) * np.log((1. - m) / (1. - n))
