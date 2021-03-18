import pytest
from requests import RequestException
from alibi.datasets import fetch_adult, fetch_movie_sentiment, load_cats

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
