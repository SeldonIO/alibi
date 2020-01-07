import numpy as np

from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.utils import to_categorical


class ImageNetPreprocessor:

    def __int__(self):
        pass

    def __call__(self, X, add_batch_dim=False, *args, **kwargs):
        return preprocess_input(X)


class FashionMnistProcessor:

    def __init__(self, normalise=True, normaliser='max', add_singleton_channel=True):

        if normalise:
            self.normalizer = normaliser

        self.add_singleton_channel = add_singleton_channel

    def __call__(self, X, add_batch_dim=False,*args, **kwargs):
        return self.transform_x(X, add_batch_dim=add_batch_dim)

    def transform_x(self, X, add_batch_dim=False):

        if self.normalizer == 'max':
            X = self._max_normalise(X)
        else:
            raise ValueError("Only division by max is implemented. Please implement "
                             "your transform as an additional method to use this "
                             "normalizer!")

        if self.add_singleton_channel:
            X = X[..., np.newaxis]     # (H, W) -> (H, W, 1)

        if add_batch_dim:
            X = X[np.newaxis, ...]     # (H, W, 1) -> (1, H, W, 1)

        return X

    @staticmethod
    def transform_y(y):
        return to_categorical(y)

    @staticmethod
    def _max_normalise(X):
        return X.astype('float32') / X.max()
