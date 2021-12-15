import copy
import json
import os
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
        TreeShap,
        CounterfactualRL,
        CounterfactualRLTabular
    )

from alibi.version import __version__

# do not extend pickle dispatch table so as not to change pickle behaviour
dill.extend(use_dill=False)

thismodule = sys.modules[__name__]

NOT_SUPPORTED = ["DistributedAnchorTabular",
                 "CEM",
                 "Counterfactual",
                 "CounterfactualProto"]


def load_explainer(path: Union[str, os.PathLike], predictor) -> 'Explainer':
    """
    Load an explainer from disk.

    Parameters
    ----------
    path
        Path to a directory containing the saved explainer.
    predictor
        Model or prediction function used to originally initialize the explainer.

    Returns
    -------
    An explainer instance.
    """
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


def save_explainer(explainer: 'Explainer', path: Union[str, os.PathLike]) -> None:
    """
    Save an explainer to disk. Uses the `dill` module.

    Parameters
    ----------
    explainer
        Explainer instance to save to disk.
    path
        Path to a directory. A new directory will be created if one does not exist.
    """
    name = explainer.meta['name']
    if name in NOT_SUPPORTED:
        raise NotImplementedError(f'Saving for {name} not yet supported')

    path = Path(path)

    # create directory
    path.mkdir(parents=True, exist_ok=True)

    # save metadata
    meta = copy.deepcopy(explainer.meta)
    with open(Path(path, 'meta.dill'), 'wb') as f:
        dill.dump(meta, f)

    try:
        # get explainer specific save function
        save_fn = getattr(thismodule, '_save_' + name)
    except AttributeError:
        # no explainer specific functionality required, just set predictor to `None` and dump
        save_fn = _simple_save
    save_fn(explainer, path)


def _simple_save(explainer: 'Explainer', path: Union[str, os.PathLike]) -> None:
    predictor = explainer.predictor  # type: ignore[attr-defined] # TODO: declare this in the Explainer interface
    explainer.predictor = None  # type: ignore[attr-defined]
    with open(Path(path, 'explainer.dill'), 'wb') as f:
        dill.dump(explainer, f, recurse=True)
    explainer.predictor = predictor  # type: ignore[attr-defined]


def _simple_load(path: Union[str, os.PathLike], predictor, meta) -> 'Explainer':
    with open(Path(path, 'explainer.dill'), 'rb') as f:
        explainer = dill.load(f)
    explainer.reset_predictor(predictor)
    return explainer


def _load_IntegratedGradients(path: Union[str, os.PathLike], predictor: Union[tf.keras.Model],
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


def _save_IntegratedGradients(explainer: 'IntegratedGradients', path: Union[str, os.PathLike]) -> None:
    model = explainer.model
    layer = explainer.layer
    explainer.model = explainer.layer = None
    with open(Path(path, 'explainer.dill'), 'wb') as f:
        dill.dump(explainer, f, recurse=True)
    explainer.model = model
    explainer.layer = layer


def _load_AnchorImage(path: Union[str, os.PathLike], predictor: Callable, meta: dict) -> 'AnchorImage':
    # segmentation function
    with open(Path(path, 'segmentation_fn.dill'), 'rb') as f:
        segmentation_fn = dill.load(f)

    with open(Path(path, 'explainer.dill'), 'rb') as f:
        explainer = dill.load(f)

    explainer.segmentation_fn = segmentation_fn
    explainer.reset_predictor(predictor)

    return explainer


def _save_AnchorImage(explainer: 'AnchorImage', path: Union[str, os.PathLike]) -> None:
    # save the segmentation function separately (could be user-supplied or built-in), must be picklable
    segmentation_fn = explainer.segmentation_fn
    explainer.segmentation_fn = None
    with open(Path(path, 'segmentation_fn.dill'), 'wb') as f:
        dill.dump(segmentation_fn, f, recurse=True)

    predictor = explainer.predictor
    explainer.predictor = None  # type: ignore[assignment]
    with open(Path(path, 'explainer.dill'), 'wb') as f:
        dill.dump(explainer, f, recurse=True)
    explainer.segmentation_fn = segmentation_fn
    explainer.predictor = predictor


def _load_AnchorText(path: Union[str, os.PathLike], predictor: Callable, meta: dict) -> 'AnchorText':
    from alibi.explainers import AnchorText

    # load explainer
    with open(Path(path, 'explainer.dill'), 'rb') as f:
        explainer = dill.load(f)

    perturb_opts = explainer.perturb_opts
    sampling_strategy = explainer.sampling_strategy
    nlp_sampling = [AnchorText.SAMPLING_UNKNOWN, AnchorText.SAMPLING_SIMILARITY]

    if sampling_strategy in nlp_sampling:
        # load the spacy model
        import spacy
        model = spacy.load(Path(path, 'nlp'))
    else:
        # load language model
        import alibi.utils.lang_model as lang_model
        model_class = explainer.model_class
        model = getattr(lang_model, model_class)(preloading=False)
        model.from_disk(Path(path, 'language_model'))

    # construct perturbation
    perturbation = AnchorText.CLASS_SAMPLER[sampling_strategy](model, perturb_opts)

    # set model, predictor, perturbation
    explainer.model = model
    explainer.reset_predictor(predictor)
    explainer.perturbation = perturbation
    return explainer


def _save_AnchorText(explainer: 'AnchorText', path: Union[str, os.PathLike]) -> None:
    from alibi.explainers import AnchorText

    model = explainer.model
    predictor = explainer.predictor
    perturbation = explainer.perturbation
    sampling_strategy = explainer.sampling_strategy

    nlp_sampling = [AnchorText.SAMPLING_UNKNOWN, AnchorText.SAMPLING_SIMILARITY]
    dir_name = 'nlp' if sampling_strategy in nlp_sampling else 'language_model'
    model.to_disk(Path(path, dir_name))

    explainer.model = None  # type: ignore[assignment]
    explainer.predictor = None  # type: ignore[assignment]
    explainer.perturbation = None

    with open(Path(path, 'explainer.dill'), 'wb') as f:
        dill.dump(explainer, f, recurse=True)

    explainer.model = model
    explainer.predictor = predictor
    explainer.perturbation = perturbation


def _save_KernelShap(explainer: 'KernelShap', path: Union[str, os.PathLike]) -> None:
    # TODO: save internal shap objects using native pickle?
    _simple_save(explainer, path)


def _save_TreelShap(explainer: 'TreeShap', path: Union[str, os.PathLike]) -> None:
    # TODO: save internal shap objects using native pickle?
    _simple_save(explainer, path)


def _save_CounterfactualRL(explainer: 'CounterfactualRL', path: Union[str, os.PathLike]) -> None:
    from alibi.utils.frameworks import Framework
    from alibi.explainers import CounterfactualRL
    CounterfactualRL._verify_backend(explainer.params["backend"])

    # get backend module
    backend = explainer.backend

    # define extension
    ext = ".tf" if explainer.params["backend"] == Framework.TENSORFLOW else ".pth"

    # save encoder and decoder (autoencoder components)
    encoder = explainer.params["encoder"]
    decoder = explainer.params["decoder"]
    backend.save_model(path=Path(path, "encoder" + ext), model=explainer.params["encoder"])
    backend.save_model(path=Path(path, "decoder" + ext), model=explainer.params["decoder"])

    # save actor
    actor = explainer.params["actor"]
    optimizer_actor = explainer.params["optimizer_actor"]  # TODO: save the actor optimizer?
    backend.save_model(path=Path(path, "actor" + ext), model=explainer.params["actor"])

    # save critic
    critic = explainer.params["critic"]
    optimizer_critic = explainer.params["optimizer_critic"]  # TODO: save the critic optimizer?
    backend.save_model(path=Path(path, "critic" + ext), model=explainer.params["critic"])

    # save locally prediction function
    predictor = explainer.params["predictor"]

    # save callbacks
    callbacks = explainer.params["callbacks"]  # TODO: what to do with the callbacks?

    # set encoder, decoder, actor, critic, prediction_func, and backend to `None`
    explainer.params["encoder"] = None
    explainer.params["decoder"] = None
    explainer.params["actor"] = None
    explainer.params["critic"] = None
    explainer.params["optimizer_actor"] = None
    explainer.params["optimizer_critic"] = None
    explainer.params["predictor"] = None
    explainer.params["callbacks"] = None
    explainer.backend = None

    # Save explainer. All the pre/post-processing function will be saved in the explainer.
    # TODO: find a better way? (I think this is ok if the functions are not too complex)
    with open(Path(path, "explainer.dill"), 'wb') as f:
        dill.dump(explainer, f)

    # set back encoder, decoder, actor and critic back
    explainer.params["encoder"] = encoder
    explainer.params["decoder"] = decoder
    explainer.params["actor"] = actor
    explainer.params["critic"] = critic
    explainer.params["optimizer_actor"] = optimizer_actor
    explainer.params["optimizer_critic"] = optimizer_critic
    explainer.params["predictor"] = predictor
    explainer.params["callbacks"] = callbacks
    explainer.backend = backend


def _helper_load_CounterfactualRL(path: Union[str, os.PathLike],
                                  predictor: Callable,
                                  explainer):
    # define extension
    from alibi.utils.frameworks import Framework
    ext = ".tf" if explainer.params["backend"] == Framework.TENSORFLOW else ".pth"

    # load the encoder and decoder (autoencoder components)
    explainer.params["encoder"] = explainer.backend.load_model(Path(path, "encoder" + ext))
    explainer.params["decoder"] = explainer.backend.load_model(Path(path, "decoder" + ext))

    # load the actor and critic
    explainer.params["actor"] = explainer.backend.load_model(Path(path, "actor" + ext))
    explainer.params["critic"] = explainer.backend.load_model(Path(path, "critic" + ext))

    # reset predictor
    explainer.reset_predictor(predictor)
    return explainer


def _load_CounterfactualRL(path: Union[str, os.PathLike],
                           predictor: Callable,
                           meta: dict) -> 'CounterfactualRL':
    # load explainer
    with open(Path(path, "explainer.dill"), "rb") as f:
        explainer = dill.load(f)

    # load backend
    from alibi.utils.frameworks import Framework
    from alibi.explainers import CounterfactualRL
    CounterfactualRL._verify_backend(explainer.params["backend"])

    # select backend module
    if explainer.params["backend"] == Framework.TENSORFLOW:
        import alibi.explainers.backends.tensorflow.cfrl_base as backend
    else:
        import alibi.explainers.backends.pytorch.cfrl_base as backend  # type: ignore

    # set explainer backend
    explainer.backend = backend

    # load the rest of the explainer
    return _helper_load_CounterfactualRL(path, predictor, explainer)


def _save_CounterfactualRLTabular(explainer: 'CounterfactualRL', path: Union[str, os.PathLike]) -> None:
    _save_CounterfactualRL(explainer=explainer, path=path)


def _load_CounterfactualRLTabular(path: Union[str, os.PathLike],
                                  predictor: Callable,
                                  meta: dict) -> 'CounterfactualRLTabular':
    # load explainer
    with open(Path(path, "explainer.dill"), "rb") as f:
        explainer = dill.load(f)

    # load backend
    from alibi.utils.frameworks import Framework
    from alibi.explainers import CounterfactualRL
    CounterfactualRL._verify_backend(explainer.params["backend"])

    # select backend module
    if explainer.params["backend"] == Framework.TENSORFLOW:
        import alibi.explainers.backends.tensorflow.cfrl_tabular as backend
    else:
        import alibi.explainers.backends.pytorch.cfrl_tabular as backend  # type: ignore

    # set explainer backend
    explainer.backend = backend

    # load the rest of the explainer
    return _helper_load_CounterfactualRL(path, predictor, explainer)


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
