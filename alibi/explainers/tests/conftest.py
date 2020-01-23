import pytest

import numpy as np
from sklearn.linear_model import LogisticRegression

from alibi.explainers import AnchorTabular
from alibi.explainers.tests.utils import predict_fcn
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, Input
from keras.models import Model
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# A file containing fixtures that can be used across tests


@pytest.fixture(scope='module')
def get_iris_dataset():
    """ Loads the iris dataset."""

    dataset = load_iris()
    feature_names = dataset.feature_names
    # define train and test set
    idx = 145
    X_train, Y_train = dataset.data[:idx, :], dataset.target[:idx]
    X_test, Y_test = dataset.data[idx + 1:, :], dataset.target[idx + 1:]  # noqa F841

    return X_test, X_train, Y_train, feature_names


@pytest.fixture(scope='module')
def iris_rf_classifier(get_iris_dataset):
    """Fits random forrest classifier on Iris dataset."""

    X_test, X_train, Y_train, feature_names = get_iris_dataset
    np.random.seed(0)
    clf = RandomForestClassifier(n_estimators=50)
    clf.fit(X_train, Y_train)

    return clf


@pytest.fixture(scope='module')
def at_defaults(request):
    """Default config for explainers."""

    desired_confidence = request.param

    return {
        'delta': 0.1,
        'epsilon': 0.15,
        'batch_size': 100,
        'desired_confidence': desired_confidence,
        'max_anchor_size': None,
        'coverage_samples': 9999,
        'n_covered_ex': 5,
        'seed': 0
    }


@pytest.fixture(scope='module')
def at_iris_explainer(get_iris_dataset, iris_rf_classifier, request):
    """Instantiates and fits an AnchorTabular explainer for the Iris dataset."""

    predict_type = request.param
    X_test, X_train, _, feature_names = get_iris_dataset
    # fit random forest to training data
    clf = iris_rf_classifier
    predict_fn = predict_fcn(predict_type, clf)
    # test explainer initialization
    explainer = AnchorTabular(predict_fn, feature_names)
    # test explainer fit: shape and binning of ordinal features
    explainer.fit(X_train, disc_perc=(25, 50, 75))

    return X_test, explainer, predict_fn, predict_type


@pytest.fixture(scope='module')
def conv_net(request):

    x_train, y_train = request.param

    def model():

        x_in = Input(shape=(28, 28, 1))
        x = Conv2D(filters=8, kernel_size=2, padding='same', activation='relu')(x_in)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.3)(x)
        x = Flatten()(x)
        x_out = Dense(10, activation='softmax')(x)
        cnn = Model(inputs=x_in, outputs=x_out)
        cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return cnn

    cnn = model()
    cnn.fit(x_train, y_train, batch_size=256, epochs=1)

    return cnn


@pytest.fixture(scope='module')
def movie_sentiment_lr_classifier(request):
    """Trains a logistic regression model."""

    is_vectorizer = False
    if len(request.param) == 3:
        is_vectorizer = True
        train, train_labels, vectorizer = request.param
    else:
        train, train_labels = request.param

    clf = LogisticRegression()
    if is_vectorizer:
        clf.fit(vectorizer.transform(train), train_labels)
    else:
        clf.fit(train, train_labels)

    return clf
