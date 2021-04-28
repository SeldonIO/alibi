import copy
import json
from os import PathLike
from pathlib import Path
import sys
from typing import Callable, TYPE_CHECKING, Union
import warnings

import dill
import numpy as np
import tensorflow as tf

if TYPE_CHECKING:
    from alibi.api.interfaces import Explainer
    from alibi.explainers import (
        AnchorImage,
        AnchorText,
        IntegratedGradients,
        KernelShap,
        TreeShap
    )
    import keras

from alibi.version import __version__

thismodule = sys.modules[__name__]
# https://www.python.org/dev/peps/pep-0519/#provide-specific-type-hinting-support
# PathLike = Union[str, bytes, os.PathLike]

NOT_SUPPORTED = ["DistributedAnchorTabular",
                 "CEM",
                 "CounterFactual",
                 "CounterFactualProto"]


def load_explainer(path: PathLike, predictor) -> 'Explainer':
    # load metadata
    with open(Path(path, 'meta.dill'), 'rb') as f:
        meta = dill.load(f)

    # check version
    if meta['version'] != __version__:
        warnings.warn(f'Trying to load explainer from version {meta["version"]} when using version {__version__}. '
                      f'This may lead to breaking code or invalid results.')

    name = meta['name']
    try:
        # get the explainer specific load function
        load_fn = getattr(thismodule, '_load_' + name)
    except AttributeError:
        load_fn = _simple_load
    return load_fn(path, predictor, meta)


def save_explainer(explainer: 'Explainer', path: PathLike) -> None:
    name = explainer.meta['name']
    if name in NOT_SUPPORTED:
        raise NotImplementedError(f'Saving for {name} not yet supported')

    path = Path(path)

    # create directory
    path.mkdir(parents=True, exist_ok=True)

    # save metadata
    meta = copy.deepcopy(explainer.meta)
    meta['version'] = explainer._version
    with open(Path(path, 'meta.dill'), 'wb') as f:
        dill.dump(meta, f)

    try:
        # get explainer specific save function
        save_fn = getattr(thismodule, '_save_' + name)
    except AttributeError:
        # no explainer specific functionality required, just set predictor to `None` and dump
        save_fn = _simple_save
    save_fn(explainer, path)


def _simple_save(explainer: 'Explainer', path: PathLike) -> None:
    predictor = explainer.predictor  # type: ignore
    explainer.predictor = None  # type: ignore
    with open(Path(path, 'explainer.dill'), 'wb') as f:
        dill.dump(explainer, f, recurse=True)
    explainer.predictor = predictor  # type: ignore


def _simple_load(path: PathLike, predictor, meta) -> 'Explainer':
    with open(Path(path, 'explainer.dill'), 'rb') as f:
        explainer = dill.load(f)
    explainer.reset_predictor(predictor)
    return explainer


def _load_IntegratedGradients(path: PathLike, predictor: Union[tf.keras.Model, 'keras.Model'],
                              meta: dict) -> 'IntegratedGradients':
    layer_num = meta['params']['layer']
    if layer_num == 0:
        layer = None
    else:
        layer = predictor.layers[layer_num]

    with open(Path(path, 'explainer.dill'), 'rb') as f:
        explainer = dill.load(f)
    explainer.reset_predictor(predictor)
    explainer.layer = layer

    return explainer


def _save_IntegratedGradients(explainer: 'IntegratedGradients', path: PathLike) -> None:
    model = explainer.model
    layer = explainer.layer
    explainer.model = explainer.layer = None
    with open(Path(path, 'explainer.dill'), 'wb') as f:
        dill.dump(explainer, f, recurse=True)
    explainer.model = model
    explainer.layer = layer


def _load_AnchorImage(path: PathLike, predictor: Callable, meta: dict) -> 'AnchorImage':
    # segmentation function
    with open(Path(path, 'segmentation_fn.dill'), 'rb') as f:
        segmentation_fn = dill.load(f)

    with open(Path(path, 'explainer.dill'), 'rb') as f:
        explainer = dill.load(f)

    explainer.segmentation_fn = segmentation_fn
    explainer.reset_predictor(predictor)

    return explainer


def _save_AnchorImage(explainer: 'AnchorImage', path: PathLike) -> None:
    # save the segmentation function separately (could be user-supplied or built-in), must be picklable
    segmentation_fn = explainer.segmentation_fn
    explainer.segmentation_fn = None
    with open(Path(path, 'segmentation_fn.dill'), 'wb') as f:
        dill.dump(segmentation_fn, f, recurse=True)

    predictor = explainer.predictor
    explainer.predictor = None
    with open(Path(path, 'explainer.dill'), 'wb') as f:
        dill.dump(explainer, f, recurse=True)
    explainer.segmentation_fn = segmentation_fn
    explainer.predictor = predictor


def _load_AnchorText(path: PathLike, predictor: Callable, meta: dict) -> 'AnchorText':
    # load the spacy model
    import spacy
    nlp = spacy.load(Path(path, 'nlp'))

    with open(Path(path, 'explainer.dill'), 'rb') as f:
        explainer = dill.load(f)

    explainer.nlp = nlp

    # explainer._synonyms_generator contains spacy Lexemes which contain unserializable Cython constructs
    # so we re-initialize the object here
    from alibi.explainers.anchor_text import Neighbors
    explainer._synonyms_generator = Neighbors(nlp_obj=nlp)
    explainer.reset_predictor(predictor)

    return explainer


def _save_AnchorText(explainer: 'AnchorText', path: PathLike) -> None:
    # save the spacy model
    nlp = explainer.nlp
    nlp.to_disk(Path(path, 'nlp'))

    _synonyms_generator = explainer._synonyms_generator
    predictor = explainer.predictor

    explainer.nlp = None
    explainer._synonyms_generator = None
    explainer.predictor = None

    with open(Path(path, 'explainer.dill'), 'wb') as f:
        dill.dump(explainer, f, recurse=True)

    explainer.nlp = nlp
    explainer._synonyms_generator = _synonyms_generator
    explainer.predictor = predictor


def _save_KernelShap(explainer: 'KernelShap', path: PathLike) -> None:
    # TODO: save internal shap objects using native pickle?
    _simple_save(explainer, path)


def _save_TreelShap(explainer: 'TreeShap', path: PathLike) -> None:
    # TODO: save internal shap objects using native pickle?
    _simple_save(explainer, path)


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
