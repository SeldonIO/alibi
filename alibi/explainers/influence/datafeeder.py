from abc import ABC, abstractmethod
from typing import Tuple, Any, Sequence
import numpy as np


class DataFeeder(ABC):
    """
    Simple interface for batch-feeding data to ML models
    """

    @abstractmethod
    def reset(self) -> None:
        """ Reset the dataset
        """
        pass

    @abstractmethod
    def train_batch(selfself, batch_size: int) -> Tuple[Any, Any]:
        """

        Parameters
        ----------
        batch_size
            Number of samples to return

        Returns
        -------
        X, y
            Batch of features and batch of labels from the training set

        """
        pass

    @abstractmethod
    def test_batch(self, test_indices: Sequence[int]) -> Tuple[Any, Any]:
        """

        Parameters
        ----------
        test_indices
            Indices of the test set to return

        Returns
        -------
        X, y
            Batch of features and batch of labels from the test set

        """
        pass


class NumpyFeeder(DataFeeder):
    def __init__(self, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> None:
        """
        Initialize the Numpy data feeder

        Parameters
        ----------
        X_train
            Training set
        X_test
            Test set
        y_train
            Training set labels
        y_test
            Test set labels
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.train_offset = 0

    def reset(self) -> None:
        self.train_offset = 0

    def train_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return a batch of training data

        Parameters
        ----------
        batch_size
            Number of samples in a batch

        Returns
        -------
        X, y
            Batch of training features and labels

        """
        # calculate offset
        start = self.train_offset
        end = start + batch_size
        self.train_offset += batch_size

        return self.X_train[start:end, ...], self.y_train[start:end, ...]

    def test_batch(self, test_indices: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return a batch of test data with given index

        Parameters
        ----------
        test_indices
            Indices of the test data to return

        Returns
        -------
        X, y
            Batch of test features and labels

        """
        return self.X_test[test_indices], self.y_test[test_indices]
