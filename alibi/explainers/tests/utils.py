# flake8: noqa: E731
# A file containing functions that can be used by multiple tests
import numpy as np


def predict_fcn(predict_type, clf, preproc=None):
    """
    Define a prediction function given a classifier Optionally a preprocessor
    with a .transform method can be passed.
    """

    if preproc:
        if not hasattr(preproc, "transform"):
            raise AttributeError(
                "Passed preprocessor to predict_fn but the "
                "preprocessor did not have a .transform method. "
                "Are you sure you passed the correct object?"
            )
    if predict_type == 'proba':
        if preproc:
            predict_fn = lambda x: clf.predict_proba(preproc.transform(x))
        else:
            predict_fn = lambda x: clf.predict_proba(x)
    elif predict_type == 'class':
        if preproc:
            predict_fn = lambda x: clf.predict(preproc.transform(x))
        else:
            predict_fn = lambda x: clf.predict(x)

    return predict_fn


def get_random_matrix(*, n_rows=500, n_cols=100):
    """
    Generates a random matrix with uniformly distributed
    numbers between 0 and 1 for testing puposes.
    """

    if n_rows == 0:
        sz = (n_cols,)
    elif n_cols == 0:
        sz = (n_rows,)
    else:
        sz = (n_rows, n_cols)

    return np.random.random(size=sz)


class MockTreeExplainer:
    """
    A class that can be used to replace shap.TreeExplainer so that
    the wrapper can be tested without fitting a model and instantiating
    an actual explainer.
    """

    def __init__(self, predictor, seed=None, *args, **kwargs):

        self.seed = seed
        np.random.seed(self.seed)
        self.model = predictor
        self.n_outputs = predictor.out_dim

    def shap_values(self, X, *args, **kwargs):
        """
        Returns random numbers simulating shap values.
        """

        self._check_input(X)

        if self.n_outputs == 1:
            return np.random.random(X.shape)
        return [np.random.random(X.shape) for _ in range(self.n_outputs)]

    def shap_interaction_values(self, X, *args, **kwargs):
        """
        Returns random numbers simulating shap interaction values.
        """

        self._check_input(X)

        if self.n_outputs == 1:
            return np.random.random((X.shape[0], X.shape[1], X.shape[1]))
        shap_output = []
        for _ in range(self.n_outputs):
            shap_output.append(np.random.random((X.shape[0], X.shape[1], X.shape[1])))
        return shap_output

    def _set_expected_value(self):
        """
        Set random expected value for the explainer.
        """

        if self.n_outputs == 1:
            self.expected_value = np.random.random()
        else:
            self.expected_value = [np.random.random() for _ in range(self.n_outputs)]

    def __call__(self, *args, **kwargs):
        self._set_expected_value()
        return self

    def _check_input(self, X):
        if not hasattr(X, 'shape'):
            raise TypeError("Input X has no attribute shape!")
