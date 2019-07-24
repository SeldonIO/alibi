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

    def __init__(self):
        self.meta = {}

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

    def __init__(self):
        self.meta = {}
        self.data = {"local": None, "global": None}

    @property
    def meta(self) -> Dict:
        return self._meta

    @meta.setter
    # TODO validation here
    def meta(self, value):
        _validate_meta(value)
        self._meta = value

    @property
    def data(self) -> Data:
        return self._data

    @data.setter
    # TODO validation here
    def data(self, value):
        _validate_data(value)
        self._data = value


class DataException(Exception):
    pass


class MetaException(Exception):
    pass


def _validate_meta(meta):
    if not isinstance(meta, dict):
        raise MetaException('Meta must be a dictionary')


def _validate_data(data):
    if not isinstance(data, dict):
        raise DataException('Data must be a dictionary')
    if set(data.keys()) != {'local', 'global'}:
        raise DataException('Data must have `local` and `global` as top level fields')
