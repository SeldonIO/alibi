from typing import Tuple, Union

import tensorflow.keras as keras
import numpy as np

from alibi.utils.data import Bunch


def fetch_fashion_mnist(return_X_y: bool = False
                        ) -> Union[Bunch, Tuple[np.ndarray, np.ndarray]]:
    """
    Loads the Fashion MNIST dataset.

    Parameters
    ----------
    return_X_y:
        If ``True``, an `N x M x P` array of data points and `N`-array of labels are returned
        instead of a dict.

    Returns
    -------
    If ``return_X_y=False``, a Bunch object with fields 'data', 'targets' and 'target_names'
    is returned. Otherwise an array with data points and an array of labels is returned.
    """

    target_names = {
        0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat',
        5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot',
    }

    data, labels = keras.datasets.fashion_mnist.load_data()[0]

    if return_X_y:
        return data, labels

    return Bunch(data=data, target=labels, target_names=target_names)
