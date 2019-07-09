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
    @abstractmethod
    def explainer_type(self) -> str:
        pass

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
    @abstractmethod
    def explanation_type(self) -> str:
        pass

    @abstractmethod
    def data(self, key: int = None) -> Data:
        pass
