import json
from os import PathLike
from pathlib import Path
import sys
from typing import Callable, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from alibi.api.interfaces import Explainer
    from alibi.explainers import ALE

thismodule = sys.modules[__name__]
# https://www.python.org/dev/peps/pep-0519/#provide-specific-type-hinting-support
# PathLike = Union[str, bytes, os.PathLike]

NOT_SUPPORTED = ["AnchorTabular",
                 "DistributedAnchorTabular",
                 "AnchorText",
                 "AnchorImage",
                 "CEM",
                 "CounterFactual",
                 "CounterFactualProto",
                 "plot_ale",
                 "IntegratedGradients",
                 "KernelShap",
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
    ale = ALE(predictor, **meta['params']['init_kwargs'])
    return ale


def _save_ALE(explainer: 'Explainer', path: PathLike) -> None:
    # ALE state is contained in metadata which is already saved
    pass


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
