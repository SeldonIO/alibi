import numpy as np
from typing import Callable
from abc import ABC, abstractmethod


class CounterfactualRLDataset(ABC):
    @staticmethod
    def predict_batches(X: np.ndarray, predictor: Callable, batch_size: int) -> np.ndarray:
        """
        Infer the classification labels of the input dataset. This is performed in batches.

        Parameters
        ----------
        X
            Input to be classified.
        predictor
            Prediction function.
        batch_size
            Maximum batch size to be used during each inference step.

        Returns
        -------
        Classification labels.
        """
        n_minibatch = int(np.ceil(X.shape[0] / batch_size))
        Y_m = np.zeros(X.shape[0])

        for i in range(n_minibatch):
            istart, istop = i * batch_size, min((i + 1) * batch_size, X.shape[0])
            Y_m[istart:istop] = predictor(X[istart:istop])

        return Y_m

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, item):
        pass
