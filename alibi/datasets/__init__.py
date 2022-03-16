from alibi.utils.missing_optional_dependency import MissingOptionalDependency

from .default import fetch_adult, fetch_imagenet, fetch_movie_sentiment, load_cats

try:
    from .tensorflow import fetch_fashion_mnist
except ImportError as err:
    fetch_fashion_mnist = MissingOptionalDependency(err, 'fetch_fashion_mnist', 'tensorflow')  # type: ignore

__all__ = ['fetch_adult',
           'fetch_fashion_mnist',
           'fetch_imagenet',
           'fetch_movie_sentiment',
           'load_cats']
