import numpy as np
from typing import Callable, Tuple, Union
from statsmodels.tools.numdiff import approx_fprime
import logging

logger = logging.getLogger(__name__)


def _define_func(predict_fn: Callable,
                 pred_class: int,
                 target_class: Union[str, int] = 'same') -> Tuple[Callable, Union[str, int]]:
    """
    Define the class-specific prediction function to be used in the optimization.

    Parameters
    ----------
    predict_fn
        Classifier prediction function
    pred_class
        Predicted class of the instance to be explained
    target_class
        Target class of the explanation, one of 'same', 'other' or an integer class

    Returns
    -------
        Class-specific prediction function and the target class used.

    """
    if target_class == 'other':

        def func(X):
            probas = predict_fn(X)
            sorted = np.argsort(-probas)  # class indices in decreasing order of probability

            # take highest probability class different from class predicted for X
            if sorted[0, 0] == pred_class:
                target_class = sorted[0, 1]
            else:
                target_class = sorted[0, 0]

            logger.debug('Current best target class: %s', target_class)
            return predict_fn(X)[:, target_class]

        return func, target_class

    elif target_class == 'same':
        target_class = pred_class

    def func(X):  # type: ignore
        return predict_fn(X)[:, target_class]

    return func, target_class


def num_grad(func: Callable, X: np.ndarray, args: Tuple = (), epsilon: float = 1e-08) -> np.ndarray:
    """
    Compute the numerical gradient using the symmetric difference. Currently wraps statsmodels implementation.
    Parameters
    ----------
    func
        Function to differentiate
    X
        Point at which to compute the gradient
    args
        Additional arguments to the function
    epsilon
        Step size for computing the gradient
    Returns
    -------
    Numerical gradient
    """
    gradient = approx_fprime(X, func, epsilon=epsilon, args=args, centered=True)
    return gradient


def get_wachter_grads(): ...


def minimize_wachter_loss(): ...


class CounterFactual:

    def __init__(self): ...

    def _initialize(self): ...

    def fit(self): ...

    def explain(self): ...
