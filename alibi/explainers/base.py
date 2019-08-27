from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Dict, List, Union
import copy

# input and output types
Data = Union[Dict, List]

# default data and metadata
DEFAULT_META = {"name": None}  # type: Dict
DEFAULT_DATA = {"overall": None, "local": []}  # type: Dict


class Base(ABC):
    """
    Base class for explainers and explanations.
    """

    def __repr__(self):
        # TODO get and pretty print all attributes a la sklearn
        return self.__class__.__name__


class BaseExplainer(Base):

    def __init__(self):
        self.meta = copy.deepcopy(DEFAULT_META)
        self.meta["name"] = self.__class__.__name__

    @property
    def meta(self) -> Dict:
        return self._meta

    @meta.setter
    def meta(self, value):
        if not isinstance(value, dict):
            raise TypeError('meta must be a dictionary')
        self._meta = value

    @abstractmethod
    def explain(self, X: Any) -> "BaseExplanation":
        pass


class FitMixin(ABC):
    @abstractmethod
    def fit(self, X: Any) -> "BaseExplainer":
        pass


class BaseExplanation(Base, Sequence):
    """
    Base class for explanations returned by explainers
    """

    def __init__(self):
        self.meta = copy.deepcopy(DEFAULT_META)
        self.meta["name"] = self.__class__.__name__
        self.data = copy.deepcopy(DEFAULT_DATA)

    def __len__(self):
        return len(self.data['local'])

    def __getitem__(self, key):
        return self.data['local'][key]

    @property
    def meta(self) -> Dict:
        return self._meta

    @meta.setter
    def meta(self, value):
        _validate_meta(value)
        self._meta = value

    @property
    def data(self) -> dict:
        return self._data

    @data.setter
    def data(self, value):
        _validate_data(value)
        self._data = value

    @property
    def local(self) -> List:
        return self._data['local']

    @property
    def overall_(self) -> Data:
        return self._data['overall']


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
    if set(data.keys()) != {'local', 'overall'}:
        raise DataException('Data must have `local` and `overall` as top level fields')
    if not isinstance(data['local'], list):
        raise DataException('data[local] must be a list')
