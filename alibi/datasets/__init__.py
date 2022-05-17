from alibi.utils.missing_optional_dependency import import_optional
from .default import fetch_adult, fetch_imagenet, fetch_movie_sentiment, load_cats, fetch_imagenet_10

fetch_fashion_mnist = import_optional('alibi.datasets.tensorflow', names=['fetch_fashion_mnist'])

__all__ = ['fetch_adult',
           'fetch_fashion_mnist',
           'fetch_imagenet',
           'fetch_movie_sentiment',
           'load_cats',
           'fetch_imagenet_10']
