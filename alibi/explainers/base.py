import abc
import copy
import json
from collections import ChainMap
from typing import Any
import logging

import attr

import numpy as np

logger = logging.getLogger(__name__)

# default data and metadata
DEFAULT_META = {
    "name": None,
    "type": [],
    "explanations": [],
    "params": {},
}  # type: dict

DEFAULT_DATA = {}  # type: dict


@attr.s
class Explainer(abc.ABC):
    """
    Base class for explainer algorithms
    """

    meta = attr.ib(default=copy.deepcopy(DEFAULT_META))  # type: dict

    def __attrs_post_init__(self):
        self.meta["name"] = self.__class__.__name__

    @abc.abstractmethod
    def explain(self, X: Any) -> "Explanation":
        pass


class FitMixin(abc.ABC):
    @abc.abstractmethod
    def fit(self, X: Any) -> "Explainer":
        pass


@attr.s
class Explanation:
    """
    Explanation class returned by explainers.
    """
    meta = attr.ib()  # type: dict
    data = attr.ib()  # type: dict

    def __attrs_post_init__(self):
        """
        Epose keys stored in self.meta and self.data as attributes of the class.
        """
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

    @classmethod
    def from_json(cls, jsonrepr) -> "Explanation":
        """
        Create an instance of an Explanation class using a json representation of the Explanation.
        NB: this is intended to be used with subclasses inheriting from Explanation.
        Parameters
        ----------
        jsonrepr
            json representation of an explanation
        Returns
        -------
        An Explanation object
        """
        dictrepr = json.loads(jsonrepr)
        try:
            meta = dictrepr['meta']
            data = dictrepr['data']
        except KeyError:
            logger.exception("Invalid explanation representation")
        return cls(meta=meta, data=data)

    def __getitem__(self, item):
        """
        This method is purely for deprecating previous behaviour of accessing explanation
        data via items in the returned dictionary.
        """
        import warnings
        warnings.warn("The Explanation object is not a dictionary anymore and accessing elements should "
                      "be done via attribute access. Accessing via item will stop working in a future version.",
                      DeprecationWarning, stacklevel=2)
        return getattr(self, item)


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
