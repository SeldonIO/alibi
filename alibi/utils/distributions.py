import numpy as np


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
