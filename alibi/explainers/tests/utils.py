# flake8: noqa: E731
# A file containing functions that can be used by multiple tests
import keras
import numpy as np
from keras.utils import to_categorical
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from alibi.datasets import fetch_movie_sentiment


def predict_fcn(predict_type, clf, preproc=None):
    """ Define a prediction function given a classifier Optionally a preprocessor
    with a .transform method can be passed."""

    if preproc:
        if not hasattr(preproc, "transform"):
            raise AttributeError("Passed preprocessor to predict_fn but the "
                                 "preprocessor did not have a .transform method. "
                                 "Are you sure you passed the correct object?")
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


def get_fashion_mnist_dataset():
    """
    Load and prepare Fashion MNIST data
    :return:
    """

    (x_train, y_train), (_, _) = keras.datasets.fashion_mnist.load_data()
    x_train = x_train.astype('float32') / 255
    x_train = np.reshape(x_train, x_train.shape + (1,))
    y_train = to_categorical(y_train)

    return x_train, y_train


def get_movie_sentiment_dataset():
    """
    Load and prepare movie sentiment data
    """

    movies = fetch_movie_sentiment()
    data = movies.data
    labels = movies.target
    train, test, train_labels, test_labels = train_test_split(data, labels, test_size=.2, random_state=0)
    train_labels = np.array(train_labels)
    # apply CountVectorizer
    vectorizer = CountVectorizer(min_df=1)
    vectorizer.fit(train)

    return train, train_labels, vectorizer
