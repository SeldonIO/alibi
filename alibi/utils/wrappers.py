import numpy as np

from functools import singledispatch, update_wrapper


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
    A transformer for converting classification output probability
    tensors to class labels. It assumes the predictor is a callable
    that can be called with a N-tensor of data points `x` and produces
    an N-tensor of outputs.
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
