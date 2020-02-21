import numpy as np


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
