import abc
import copy
import json
from collections import ChainMap
from typing import Any

import attr

import numpy as np

# default data and metadata
DEFAULT_META = {
    "name": None,
    "type": None,
    "explanations": [],
    "params": {},
}  # type: dict

DEFAULT_DATA = {}  # type: dict


@attr.s
class Explainer(abc.ABC):
    """
    Base class for explainer algorithms
    """

    meta: dict = attr.ib(default=copy.deepcopy(DEFAULT_META))

    def __attrs_post_init__(self):
        self.meta["name"] = self.__class__.__name__

    @abc.abstractmethod
    def explain(self, X: Any) -> "Explanation":
        pass


class FitMixin(abc.ABC):
    @abc.abstractmethod
    def fit(self, X: Any) -> "Explainer":
        pass


class Explanation(abc.ABC):
    """
    Base class for explanations returned by explainers.
    """

    def __attrs_post_init__(self):
        """
        Add a name attribute and expose keys stored in self.meta and self.data as attributes of the class.
        """
        self.meta["name"] = self.__class__.__name__
        for key, value in ChainMap(self.meta, self.data).items():
            setattr(self, key, value)

    def to_json(self) -> str:
        """
        Serialize the explanation data and metadata into a json format.

        Returns
        -------
        String containing json representation of the explanation
        """
        return json.dumps(attr.asdict(self), cls=NumpyEncoder)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(
                obj,
                (
                        np.int_,
                        np.intc,
                        np.intp,
                        np.int8,
                        np.int16,
                        np.int32,
                        np.int64,
                        np.uint8,
                        np.uint16,
                        np.uint32,
                        np.uint64,
                ),
        ):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class BaseExplainer:
    pass


class BaseExplanation:
    pass
