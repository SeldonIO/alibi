import abc
import json
from collections import ChainMap
from typing import Any
import logging
from functools import partial
import pprint

import attr
import numpy as np

logger = logging.getLogger(__name__)


# default metadata
def default_meta() -> dict:
    return {
        "name": None,
        "type": [],
        "explanations": [],
        "params": {},
    }


class AlibiPrettyPrinter(pprint.PrettyPrinter):
    """
    Overrides the built in dictionary pretty representation to look more similar to the external
    prettyprinter libary.
    """
    _dispatch = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # `sort_dicts` kwarg was only introduced in Python 3.8 so we just override it here.
        # Before Python 3.8 the printing was done in insertion order by default.
        self._sort_dicts = False

    def _pprint_dict(self, object, stream, indent, allowance, context, level):
        # Add a few newlines and the appropriate indentation to dictionary printing
        # compare with https://github.com/python/cpython/blob/3.9/Lib/pprint.py
        write = stream.write
        indent += self._indent_per_level
        write('{\n' + ' ' * (indent + 1))
        if self._indent_per_level > 1:
            write((self._indent_per_level - 1) * ' ')
        length = len(object)
        if length:
            if self._sort_dicts:
                items = sorted(object.items(), key=pprint._safe_tuple)
            else:
                items = object.items()
            self._format_dict_items(items, stream, indent, allowance + 1,
                                    context, level)
        write('}\n' + ' ' * (indent - 1))

    _dispatch[dict.__repr__] = _pprint_dict


alibi_pformat = partial(AlibiPrettyPrinter().pformat)


@attr.s
class Explainer(abc.ABC):
    """
    Base class for explainer algorithms
    """

    meta = attr.ib(default=attr.Factory(default_meta), repr=alibi_pformat)  # type: dict

    def __attrs_post_init__(self):
        # add a name to the metadata dictionary
        self.meta["name"] = self.__class__.__name__

        # expose keys stored in self.meta as attributes of the class.
        for key, value in self.meta.items():
            setattr(self, key, value)

    @abc.abstractmethod
    def explain(self, X: Any) -> "Explanation":
        pass

    def _update_metadata(self, data_dict: dict, params: bool = False) -> None:
        """
        Updates the metadata of the explainer using the data from the `data_dict`. If the params option
        is specified, then each key-value pair is added to the metadata `'params'` dictionary.

        Parameters
        ----------
        data_dict
            Contains the data to be stored in the metadata.
        params
            If True, the method updates the `'params'` attribute of the metatadata.
        """

        if params:
            for key in data_dict.keys():
                self.meta['params'].update([(key, data_dict[key])])
        else:
            self.meta.update(data_dict)


class FitMixin(abc.ABC):
    @abc.abstractmethod
    def fit(self, X: Any) -> "Explainer":
        pass


@attr.s
class Explanation:
    """
    Explanation class returned by explainers.
    """
    meta = attr.ib(repr=alibi_pformat)  # type: dict
    data = attr.ib(repr=alibi_pformat)  # type: dict

    def __attrs_post_init__(self):
        """
        Expose keys stored in self.meta and self.data as attributes of the class.
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
        msg = "The Explanation object is not a dictionary anymore and accessing elements should " \
              "be done via attribute access. Accessing via item will stop working in a future version."
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
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
