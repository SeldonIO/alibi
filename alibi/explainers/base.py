from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Dict, List, Union
import copy
import json
import numpy as np

# input and output types
Data = Union[Dict, List]

# default data and metadata
DEFAULT_META = {"name": None}  # type: Dict
DEFAULT_DATA = {"overall": None, "local": []}  # type: Dict


class BaseExplainer(ABC):
    """
    Base class for explainer algorithms
    """

    def __init__(self):
        self.meta = copy.deepcopy(DEFAULT_META)
        self.meta["name"] = self.__class__.__name__

    def __repr__(self):
        # TODO get and pretty print all attributes a la sklearn
        return self.__class__.__name__

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


class BaseExplanation(Sequence):
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

    def __repr__(self):
        name = self.__class__.__name__
        is_overall = False
        len_loc = 0
        # check what is explained
        if self.overall is not None:
            is_overall = True
        if self.local != []:
            len_loc = len(self.local)
        dictrepr = ' {{overall: {}, local: {}}}'.format(is_overall, len_loc)
        return '<' + name + dictrepr + '>'

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
    def overall(self) -> Data:
        return self._data['overall']

    def serialize(self) -> str:
        """
        Serialize the explanation data and metadata into a json format.

        Returns
        -------
        String containing json representation of the explanation
        """
        meta = self.meta
        data = self.data
        all = {"meta": meta, "data": data}
        return json.dumps(all, cls=NumpyEncoder)


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


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (
                np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32,
                np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
