import pytest
from requests import RequestException
from alibi.datasets import fetch_adult, fetch_imagenet, fetch_movie_sentiment

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


@pytest.mark.parametrize('nb_images', [3])
@pytest.mark.parametrize('category', [
    'Persian cat', 'volcano', 'strawberry', 'Siamese cat', 'Siamese', 'Siamese cat, Siamese'])
@pytest.mark.parametrize('return_X_y', [True, False])
def test_imagenet(nb_images, category, return_X_y):
    try:
        data = fetch_imagenet(category=category, nb_images=nb_images, target_size=(299, 299), return_X_y=return_X_y)
    except RequestException:
        pytest.skip('Imagenet API down')

    if return_X_y:
        X, y = data
    else:
        X = data.data
        y = data.target

    assert X.shape == (nb_images, 299, 299, 3)  # 3 color channels
    assert X.max() <= 255  # RGB limits
    assert X.min() >= 0

    assert len(y) == nb_images


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
