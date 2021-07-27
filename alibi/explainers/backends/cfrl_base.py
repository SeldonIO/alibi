import numpy as np
from typing import Callable
from abc import ABC, abstractmethod


class CounterfactualRLDataset(ABC):
    @staticmethod
    def predict_batches(x: np.ndarray, predict_func: Callable, batch_size: int) -> np.ndarray:
        """
        Infer the classification labels of the input dataset. This is performed in batches.

        Parameters
        ----------
        x
            Input to be classified.
        predict_func
            Prediction function.
        batch_size
            Maximum batch size to be used during each inference step.

        Returns
        -------
        Classification labels.
        """
        n_minibatch = int(np.ceil(x.shape[0] / batch_size))
        y_m = np.zeros(x.shape[0])

        for i in range(n_minibatch):
            istart, istop = i * batch_size, min((i + 1) * batch_size, x.shape[0])
            y_m[istart:istop] = predict_func(x[istart:istop])

        return y_m

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, item):
        raise NotImplementedError
