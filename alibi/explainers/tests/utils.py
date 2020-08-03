# flake8: noqa: E731
# A file containing functions that can be used by multiple tests
import numpy as np

from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from alibi.datasets import fetch_adult


def get_adult_data():
    """
    Loads and preprocesses Adult dataset.
    """

    # load raw data
    adult = fetch_adult()
    data = adult.data
    target = adult.target
    feature_names = adult.feature_names
    category_map = adult.category_map

    # split it
    idx = 30000
    X_train, Y_train = data[:idx, :], target[:idx]
    X_test, Y_test = data[idx + 1:, :], target[idx + 1:]

    # Create feature transformation pipeline
    ordinal_features = [x for x in range(len(feature_names)) if x not in list(category_map.keys())]
    ordinal_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]
    )

    categorical_features = list(category_map.keys())
    categorical_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', ordinal_transformer, ordinal_features),
            ('cat', categorical_transformer, categorical_features)]
    )
    preprocessor.fit(X_train)

    return {
        'X_train': X_train,
        'y_train': Y_train,
        'X_test': X_test,
        'y_test': Y_test,
        'preprocessor': preprocessor,
        'metadata': {
            'feature_names': feature_names,
            'category_map': category_map,
            'name': 'adult'
        }
    }


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
