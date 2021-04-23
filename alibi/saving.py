import json
from os import PathLike
from pathlib import Path
import sys
from typing import Callable, TYPE_CHECKING, Union

import dill
import numpy as np
import tensorflow as tf

if TYPE_CHECKING:
    from alibi.api.interfaces import Explainer
    from alibi.explainers import (
        ALE,
        AnchorImage,
        AnchorTabular,
        AnchorText,
        IntegratedGradients
    )
    import keras

thismodule = sys.modules[__name__]
# https://www.python.org/dev/peps/pep-0519/#provide-specific-type-hinting-support
# PathLike = Union[str, bytes, os.PathLike]

NOT_SUPPORTED = ["DistributedAnchorTabular",
                 "CEM",
                 "CounterFactual",
                 "CounterFactualProto",
                 "TreeShap"]


# TODO: tricky to use singledispatch and explainer types due to circular imports,
#  manual dispatch on name instead
# @singledispatch
# def save_explainer(explainer, path) -> None:
#    pass
#
# @save_explainer.register
# def _(explainer: explainers.AnchorTabular, path: PathLike) -> None:
#    raise NotImplementedError(f'Saving not implemented for {explainer.name}')

def load_explainer(path: PathLike, predictor) -> 'Explainer':
    # load metadata
    with open(Path(path, 'meta.json'), 'r') as f:
        meta = json.load(f)

    # get the explainer specific load function
    name = meta['name']
    load_fn = getattr(thismodule, '_load_' + name)
    return load_fn(path, predictor, meta)


def save_explainer(explainer: 'Explainer', path: PathLike) -> None:
    name = explainer.meta['name']
    if name in NOT_SUPPORTED:
        raise NotImplementedError(f'Saving for {name} not yet supported')

    path = Path(path)

    # create directory
    path.mkdir(parents=True, exist_ok=True)

    # save metadata
    meta = explainer.meta
    with open(Path(path, 'meta.json'), 'w') as f:
        json.dump(meta, f, cls=NumpyEncoder)

    # get explainer specific save function
    save_fn = getattr(thismodule, '_save_' + name)

    # save
    save_fn(explainer, path)


def _load_ALE(path: PathLike, predictor: Callable, meta: dict) -> 'ALE':
    from alibi.explainers import ALE
    init_kwargs = meta['params']
    init_kwargs.pop('min_bin_points')
    ale = ALE(predictor, **init_kwargs)
    return ale


def _save_ALE(explainer: 'ALE', path: PathLike) -> None:
    # ALE state is contained in metadata which is already saved
    pass


def _load_IntegratedGradients(path: PathLike, predictor: Union[tf.keras.Model, 'keras.Model'],
                              meta: dict) -> 'IntegratedGradients':
    from alibi.explainers import IntegratedGradients
    layer_num = meta['params']['layer']
    if layer_num == 0:
        layer = None
    else:
        layer = predictor.layers[layer_num]

    ig = IntegratedGradients(model=predictor,
                             layer=layer,
                             method=meta['params']['method'],
                             n_steps=meta['params']['n_steps'],
                             internal_batch_size=meta['params']['internal_batch_size'])
    return ig


def _save_IntegratedGradients(explainer: 'IntegratedGradients', path: PathLike) -> None:
    # IG state is contained in the metadata which is already saved
    pass


def _load_AnchorImage(path: PathLike, predictor: Callable, meta: dict) -> 'AnchorImage':
    from alibi.explainers import AnchorImage

    # black-box segmentation function
    if meta['params']['custom_segmentation']:
        with open(Path(path, 'segmentation_fn.dill'), 'rb') as f:
            segmentation_fn = dill.load(f)
    # built-in segmentation function
    else:
        segmentation_fn = meta['params']['segmentation_fn']

    # image-shape should be a tuple
    meta['params']['image_shape'] = tuple(meta['params']['image_shape'])

    ai = AnchorImage(predictor=predictor,
                     image_shape=meta['params']['image_shape'],
                     segmentation_fn=segmentation_fn,
                     segmentation_kwargs=meta['params']['segmentation_kwargs'],
                     images_background=meta['params']['images_background'],
                     seed=meta['params']['seed'])

    return ai


def _save_AnchorImage(explainer: 'AnchorImage', path: PathLike) -> None:
    # if black-box segmentation function used, we save it, it must be picklable
    if explainer.meta['params']['custom_segmentation']:
        with open(Path(path, 'segmentation_fn.dill'), 'wb') as f:
            dill.dump(explainer.segmentation_fn, f, recurse=True)


def _load_AnchorText(path: PathLike, predictor: Callable, meta: dict) -> 'AnchorText':
    from alibi.explainers import AnchorText
    import spacy

    # load the spacy model
    nlp = spacy.load(Path(path, 'nlp'))

    # TODO: re-initialization takes some time due initializing Neighbours, this should be saved as part
    #  of the state in the future, see also https://github.com/SeldonIO/alibi/issues/251#issuecomment-649484225
    atext = AnchorText(nlp=nlp,
                       predictor=predictor,
                       seed=meta['params']['seed'])

    return atext


def _save_AnchorText(explainer: 'AnchorText', path: PathLike) -> None:
    # save the spacy model
    nlp = explainer.nlp
    nlp.to_disk(Path(path, 'nlp'))


def _load_AnchorTabular(path: PathLike, predictor: Callable, meta: dict) -> 'AnchorTabular':
    from alibi.explainers import AnchorTabular

    # TODO: HACK: `categorical_names` should have integer keys, but json saves them as strings
    cmap = meta['params']['categorical_names']
    if cmap is not None:
        cmap = {int(k): v for k, v in cmap.items()}
        meta['params']['categorical_names'] = cmap

    # disc_perc should be a tuple
    meta['params']['disc_perc'] = tuple(meta['params']['disc_perc'])

    # load the training data
    with open(Path(path, 'train_data.npy'), 'rb') as f:
        train_data = np.load(f)

    atab = AnchorTabular(predictor=predictor,
                         feature_names=meta['params']['feature_names'],
                         categorical_names=meta['params']['categorical_names'],
                         seed=meta['params']['seed'])

    # TODO: calling `fit` here is fast and the data is available as it's stored internally by the explainer
    #  though we may still want to improve this and not have to call `fit` in the future
    atab.fit(train_data=train_data, disc_perc=meta['params']['disc_perc'])

    return atab


def _save_AnchorTabular(explainer: 'AnchorTabular', path: PathLike) -> None:
    # AnchorTabular saves a copy of the numpy training data, so we extract it and save it separately
    X_train = explainer.samplers[0].train_data
    with open(Path(path, 'train_data.npy'), 'wb') as f:
        np.save(f, X_train)


# def save_explainer(explainer: 'Explainer', path: PathLike) -> None:
#    # TODO: allow specifying deep or shallow copy instead of try/except?
#    if explainer.name in NOT_SUPPORTED:
#        raise NotImplementedError(f'Saving not yet supported for {explainer.name}')
#    try:
#        # try to recurse and save the white-box explainer referenced by the black-box predictor function
#        with open(path, 'wb') as f:
#            dill.dump(explainer, f, recurse=True)
#    except TypeError:
#        # need to close file before attempting to save again
#        with open(path, 'wb') as f:
#            dill.dump(explainer, f, recurse=False)
#            import warnings
#            warnings.warn('Could not save a deep copy of the explainer. '
#                          'This is most likely due to the type of the model used. '
#                          'You may need to call `explainer.reset_predictor(predictor)`'
#                          'after loading the saved explainer.')


# Older prototype dispatching manually on name
# try:
#    save_fun = getattr(thismodule, '_save_' + explainer.name)
#    save_fun(explainer, path)
# except AttributeError:
#    raise NotImplementedError(f'Saving not implemented for {explainer.name}')


# def _save_AnchorTabular(explainer: 'AnchorTabular', path: PathLike) -> None:
#    with open(path, 'wb') as f:
#        dill.dump(explainer, f, recurse=True)

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
