import numpy as np
from functools import singledispatch, update_wrapper
from typing import Callable, Union


class Predictor:

    def __init__(self, clf, preprocessor=None):

        if not hasattr(clf, 'predict'):
            raise AttributeError('Classifier object is expected to have a predict method!')

        self.clf = clf
        self.predict_fcn = clf.predict
        self.preprocessor = preprocessor

    def __call__(self, x):
        if self.preprocessor:
            return self.predict_fcn(self.preprocessor.transform(x))
        return self.predict_fcn(x)


class ArgmaxTransformer:
    """
    A transformer for converting classification output probability tensors to class labels. It assumes the predictor is
    a callable that can be called with a N-tensor of data points `x` and produces an N-tensor of outputs.
    """

    def __init__(self, predictor):
        self.predictor = predictor

    def __call__(self, x):
        pred = np.atleast_2d(self.predictor(x))
        return np.argmax(pred, axis=1)


def methdispatch(func):
    """
    A decorator that is used to support singledispatch style functionality
    for instance methods. By default, singledispatch selects a function to
    call from registered based on the type of args[0]::
        def wrapper(*args, **kw):
            return dispatch(args[0].__class__)(*args, **kw)
    This uses singledispatch to do achieve this but instead uses args[1]
    since args[0] will always be self.
    """

    dispatcher = singledispatch(func)

    def wrapper(*args, **kw):
        return dispatcher.dispatch(args[1].__class__)(*args, **kw)

    wrapper.register = dispatcher.register
    update_wrapper(wrapper, dispatcher)

    return wrapper


blackbox_wrappers = {'pytorch': None, 'tensorflow': None}
"""dict: A registry for wrappers that cast the input to a black-box function to `np.array` object and the output to a
frame-work specific tensor.
"""


def blackbox_wrapper(framework: str = 'tensorflow') -> Callable:
    """
    A decorator that registers a wrapper for a black-box model. In the  future, the functionality may be extended so
    that the wrapper can also be parametrized by the algorithm name, if necessary.

    Parameters
    ----------
    framework: {'pytorch', 'tensorflow'}
        The framework in which the the predictor wrapped is implemented.
    """

    def register_wrapper(func):
        if framework not in ['pytorch', 'tensorflow']:
            raise ValueError(
                f"Unknown value {framework} for framework. Black-box model wrappers are only registered for 'pytorch' "
                f"and 'tensorflow' frameworks!"
            )
        blackbox_wrappers[framework] = func
        return func

    return register_wrapper


def get_blackbox_wrapper(framework: str) -> Union[Callable, None]:
    """
    Returns a wrapper for a black-box function. The role of a wrapper is to convert tensors to numpy arrays and the
    output of the predictor to a tensor, specific to `framework`. The wrapper is returned from a registry,
    which is updated via the `alibi.utils.wrappers.blackbox_wrapper` decorator, which is parametrized by `framework`.
    Only one wrapper for each framework is defined. In the future, the registry can be customised to provide specialised
    wrappers for various algorithms.

    Parameter
    ---------
    framework: {'pytorch', 'tensorflow'}
        Framework for the optimisation algorithm.

    Returns
    -------
    A decorator that can be applied functionally to a predictor.

    Examples
    --------
    In a class that receives a `predictor` whose inputs/outputs are `np.array` objects, the wrapper can be used to cast
    the types as follows:

    >>> # retrieve wrapper
    >>> wrapper = get_blackbox_wrapper('tensorflow')
    >>> # apply it
    >>> wrapped_predictor = wrapper(predictor)
    >>> # wrapped predictor can be called with `tf.Variable` objects and returns `tf.Tensor` objects
    """

    return blackbox_wrappers[framework]
