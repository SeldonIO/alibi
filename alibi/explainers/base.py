from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Union

# input and output types
Data = Union[Dict, List]


class BaseExplainer(ABC):
    """
    Base class for explainers.
    """

    def __repr__(self):
        # TODO get and pretty print all attributes a la sklearn
        return self.__class__.__name__


class Explainer(BaseExplainer):

    @property
    def meta(self) -> Dict:
        return self._meta

    @meta.setter
    # TODO validation here
    def meta(self, value):
        if not isinstance(value, dict):
            raise TypeError('meta must be a dictionary')
        self._meta = value

    @abstractmethod
    def explain(self, X: np.ndarray, y: np.ndarray = None) -> "Explanation":
        pass


class FitMixin(Explainer):
    @abstractmethod
    def fit(self, X: np.ndarray = None, y: np.ndarray = None) -> "Explainer":
        return self


class Explanation(ABC):
    """
    Base class for explanations returned by explainers
    """

    @property
    def meta(self) -> Dict:
        return self._meta

    @meta.setter
    # TODO validation here
    def meta(self, value):
        if not isinstance(value, dict):
            raise TypeError('meta must be a dictionary')
        self._meta = value

    @property
    def data(self) -> Data:
        return self._data

    @data.setter
    # TODO validation here
    def data(self, value):
        if not isinstance(value, dict):
            raise TypeError('data must be a dictionary')
        self._data = value
