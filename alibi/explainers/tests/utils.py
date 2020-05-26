# flake8: noqa: E731
# A file containing functions that can be used by multiple tests
import numpy as np
import tensorflow.keras as keras

from tensorflow.keras.utils import to_categorical
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_iris, load_boston
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from alibi.datasets import fetch_movie_sentiment, fetch_adult


SUPPORTED_DATASETS = ['adult', 'fashion_mnist', 'iris', 'movie_sentiment']


# When registring a dataset, add the dataset name in ['metadata']['name'] and
# add its name to SUPPORTED_DATASETS. Follow the convention for the naming
# of the function and the output as shown below

def adult_dataset():
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


def fashion_mnist_dataset():
    """
    Load and prepare Fashion MNIST dataset.
    """

    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x_train = x_train.astype('float32') / 255
    x_train = np.reshape(x_train, x_train.shape + (1,))
    y_train = to_categorical(y_train)

    return {
        'X_train': x_train,
        'y_train': y_train,
        'X_test': x_test,
        'y_test': y_test,
        'preprocessor': None,
        'metadata': {'name': 'fashion_mnist'},
    }


def iris_dataset():
    """
    Loads the Iris dataset.
    """

    dataset = load_iris()
    feature_names = dataset.feature_names
    # define train and test set
    idx = 145
    X_train, Y_train = dataset.data[:idx, :], dataset.target[:idx]
    X_test, Y_test = dataset.data[idx + 1:, :], dataset.target[idx + 1:]  # noqa F841

    return {
        'X_test': X_test,
        'X_train': X_train,
        'y_train': Y_train,
        'y_test': Y_test,
        'preprocessor': None,
        'metadata': {
            'feature_names': feature_names,
            'name': 'iris'
        }
    }


def boston_dataset():
    """
    Load the Boston housing dataset.
    """
    dataset = load_boston()
    data = dataset.data
    labels = dataset.target
    feature_names = dataset.feature_names
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=.2, random_state=0)

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'preprocessor': None,
        'metadata': {
            'feature_names': feature_names,
            'name': 'boston'}
    }


def movie_sentiment_dataset():
    """
    Load and prepare movie sentiment data.
    """

    movies = fetch_movie_sentiment()
    data = movies.data
    labels = movies.target
    train, test, train_labels, test_labels = train_test_split(data, labels, test_size=.2, random_state=0)
    train_labels = np.array(train_labels)
    vectorizer = CountVectorizer(min_df=1)
    vectorizer.fit(train)

    return {
        'X_train': train,
        'y_train': train_labels,
        'X_test': test,
        'y_test': test_labels,
        'preprocessor': vectorizer,
        'metadata': {'name': 'movie_sentiment'},
    }


def get_dataset(name):
    """
    Returns a dataset given the name, which must be a member of
    SUPPORTED_DATASETS.
    """

    if name == 'adult':
        return adult_dataset()
    elif name == 'movie_sentiment':
        return movie_sentiment_dataset()
    elif name == 'fashion_mnist':
        return fashion_mnist_dataset()
    else:
        fmt = "Value of name parameters is {}. Supported datasets are {}!"
        raise ValueError(fmt.format(name, SUPPORTED_DATASETS))


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
