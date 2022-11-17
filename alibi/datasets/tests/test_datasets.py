import numpy as np
import pytest
from alibi.datasets import (fetch_adult, fetch_fashion_mnist, fetch_imagenet,
                            fetch_imagenet_10, fetch_movie_sentiment,
                            load_cats)
from requests import RequestException

# TODO use mocking instead of relying on external services

ADULT_DIM = 2
ADULT_FEATURES = 12
ADULT_CLASSES = 2


@pytest.mark.parametrize('return_X_y', [True, False])
def test_adult(return_X_y):
    try:
        data = fetch_adult(return_X_y=return_X_y)
    except RequestException:
        pytest.skip('Adult dataset URL down')
    if return_X_y:
        assert len(data) == 2
        X, y = data
    else:
        assert len(data) == 5
        X = data.data
        y = data.target

    assert X.ndim == ADULT_DIM
    assert X.shape[1] == ADULT_FEATURES
    assert len(X) == len(y)
    assert len(set(y)) == ADULT_CLASSES


MOVIE_CLASSES = 2


@pytest.mark.parametrize('return_X_y', [True, False])
def test_movie_sentiment(return_X_y):
    try:
        data = fetch_movie_sentiment(return_X_y=return_X_y)
    except RequestException:
        pytest.skip('Movie sentiment dataset URL down')
    if return_X_y:
        assert len(data) == 2
        X, y = data
    else:
        assert len(data) == 3
        X = data.data
        y = data.target

    assert len(X) == len(y)
    assert len(set(y)) == MOVIE_CLASSES


@pytest.mark.parametrize('return_X_y', [True, False])
@pytest.mark.parametrize('target_size', [(299, 299)])
def test_cats(return_X_y, target_size):
    data = load_cats(target_size=target_size, return_X_y=return_X_y)
    if return_X_y:
        assert len(data) == 2
        X, y = data
    else:
        assert len(data) == 3
        X = data.data
        y = data.target
    assert len(X) == len(y)
    assert X.shape[1:] == target_size + (3,)  # 3 channels


@pytest.mark.parametrize('target_size', [(224, 224)])
@pytest.mark.parametrize('num_classes', [10])
def test_imagenet_10(target_size, num_classes):
    try:
        data = fetch_imagenet_10()
    except RequestException:
        pytest.skip('ImageNet10 dataset URL down.')

    assert isinstance(data, dict)

    # check if the expected keys exist in the dict
    keys = ['trainset', 'testset', 'int_to_str_labels', 'str_to_int_labels', 'mean_channels']
    assert all([key in data for key in keys])

    # check the types of the values in the dict
    assert isinstance(data['trainset'], tuple)
    assert isinstance(data['testset'], tuple)
    assert isinstance(data['int_to_str_labels'], dict)
    assert isinstance(data['str_to_int_labels'], dict)
    assert isinstance(data['mean_channels'], np.ndarray)

    assert len(data['trainset']) == 2
    assert len(data['testset']) == 2
    assert data['mean_channels'].shape[-1] == 3

    X_train, y_train = data['trainset']
    X_test, y_test = data['testset']

    # check dataset size
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)

    # check input shape
    assert X_train.shape[1:] == target_size + (3, )
    assert X_test.shape[1:] == target_size + (3, )

    # check labels
    train_class_indices = np.unique(y_train)
    test_class_indices = np.unique(y_test)
    assert len(train_class_indices) == num_classes
    assert len(test_class_indices) == num_classes

    class_names = {'stingray', 'trilobite', 'centipede', 'slug', 'snail', 'Rhodesian ridgeback',
                   'beagle', 'golden retriever', 'sea lion', 'espresso'}
    train_class_names = set([data['int_to_str_labels'][i] for i in train_class_indices])
    test_class_names = set([data['int_to_str_labels'][i] for i in test_class_indices])
    assert train_class_names == class_names
    assert test_class_names == class_names


def test_imagenet():
    with pytest.warns() as warn:
        fetch_imagenet()
    assert 'The Imagenet API is no longer publicly available' in str(warn[0].message)


FASHION_MNIST_DIM = 3
FASHION_MNIST_FEATURES = 784
FASHION_MNIST_CLASSES = 10


@pytest.mark.parametrize('return_X_y', [True, False])
def test_fashion_mnist(return_X_y):

    try:
        data = fetch_fashion_mnist(return_X_y=return_X_y)
    except RequestException:
        pytest.skip('Fashion MNIST dataset URL down')

    if return_X_y:
        assert len(data) == 2
        X, y = data
    else:
        assert len(data) == 3
        X = data.data
        y = data.target

    assert X.ndim == FASHION_MNIST_DIM
    assert X.reshape(X.shape[0], -1).shape[1] == FASHION_MNIST_FEATURES
    assert len(X) == len(y)
    assert len(set(y)) == FASHION_MNIST_CLASSES
