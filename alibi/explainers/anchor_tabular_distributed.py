import copy
import logging
from collections import OrderedDict, defaultdict
from itertools import accumulate
from typing import (Any, Callable, DefaultDict, Dict, List, Optional, Set,
                    Tuple, Type, Union)

import numpy as np

from alibi.api.defaults import DEFAULT_DATA_ANCHOR, DEFAULT_META_ANCHOR
from alibi.api.interfaces import Explainer, Explanation, FitMixin
from alibi.exceptions import (AlibiPredictorCallException,
                              AlibiPredictorReturnTypeError)
from alibi.utils.discretizer import Discretizer
from alibi.utils.distributed import RAY_INSTALLED
from alibi.utils.mapping import ohe_to_ord, ord_to_ohe
from alibi.utils.wrappers import ArgmaxTransformer

from .anchor_base import AnchorBaseBeam, DistributedAnchorBaseBeam
from .anchor_explanation import AnchorExplanation
from .anchor_tabular import AnchorTabular, TabularSampler


import ray


class RemoteSampler:
    """ A wrapper that facilitates the use of `TabularSampler` for distributed sampling."""
    ray = ray

    def __init__(self, *args):
        self.train_id, self.d_train_id, self.sampler = args
        self.sampler = self.sampler.deferred_init(self.train_id, self.d_train_id)

    def __call__(self, anchors_batch: Union[Tuple[int, tuple], List[Tuple[int, tuple]]], num_samples: int,
                 compute_labels: bool = True) -> List:
        """
        Wrapper around :py:meth:`alibi.explainers.anchor_tabular.TabularSampler.__call__`. It allows sampling a batch
        of anchors in the same process, which can improve performance.

        Parameters
        ----------
        anchors_batch, num_samples, compute_labels
            A list of result tuples. See :py:meth:`alibi.explainers.anchor_tabular.TabularSampler.__call__`
            for details.
        """

        if isinstance(anchors_batch, tuple):  # DistributedAnchorBaseBeam._get_samples_coverage call
            return self.sampler(anchors_batch, num_samples, compute_labels=compute_labels)
        elif len(anchors_batch) == 1:  # batch size = 1
            return [self.sampler(*anchors_batch, num_samples, compute_labels=compute_labels)]
        else:  # batch size > 1
            batch_result = []
            for anchor in anchors_batch:
                batch_result.append(self.sampler(anchor, num_samples, compute_labels=compute_labels))

            return batch_result

    def set_instance_label(self, X: np.ndarray) -> int:
        """
        Sets the remote sampler instance label.

        Parameters
        ----------
        X
            The instance to be explained.

        Returns
        -------
        label
            The label of the instance to be explained.
        """

        self.sampler.set_instance_label(X)
        label = self.sampler.instance_label

        return label

    def set_n_covered(self, n_covered: int) -> None:
        """
        Sets the remote sampler number of examples to save for inspection.

        Parameters
        ----------
        n_covered
            Number of examples where the result (and partial anchors) apply.
        """

        self.sampler.set_n_covered(n_covered)

    def _get_sampler(self) -> TabularSampler:
        """
        A getter that returns the underlying tabular object.

        Returns
        -------
        The tabular sampler object that is used in the process.
        """
        return self.sampler

    def build_lookups(self, X: np.ndarray):
        """
        Wrapper around :py:meth:`alibi.explainers.anchor_tabular.TabularSampler.build_lookups`.

        Parameters
        --------
        X
            See :py:meth:`alibi.explainers.anchor_tabular.TabularSampler.build_lookups`.

        Returns
        -------
        See :py:meth:`alibi.explainers.anchor_tabular.TabularSampler.build_lookups`.
        """

        cat_lookup_id, ord_lookup_id, enc2feat_idx_id = self.sampler.build_lookups(X)

        return [cat_lookup_id, ord_lookup_id, enc2feat_idx_id]


class DistributedAnchorTabular(AnchorTabular):
    ray = ray

    def __init__(self,
                 predictor: Callable,
                 feature_names: List[str],
                 categorical_names: Optional[Dict[int, List[str]]] = None,
                 dtype: Type[np.generic] = np.float32,
                 ohe: bool = False,
                 seed: Optional[int] = None) -> None:

        super().__init__(predictor, feature_names, categorical_names, dtype, ohe, seed)
        if not DistributedAnchorTabular.ray.is_initialized():
            DistributedAnchorTabular.ray.init()

    def fit(self,  # type: ignore[override]
            train_data: np.ndarray,
            disc_perc: tuple = (25, 50, 75),
            **kwargs) -> "AnchorTabular":
        """
        Creates a list of handles to parallel processes handles that are used for submitting sampling
        tasks.

        Parameters
        ----------
        train_data, disc_perc, **kwargs
            See :py:meth:`alibi.explainers.anchor_tabular.AnchorTabular.fit` superclass.
        """

        try:
            ncpu = kwargs['ncpu']
        except KeyError:
            logging.warning('DistributedAnchorTabular object has been initalised but kwargs did not contain '
                            'expected argument, ncpu. Defaulting to ncpu=2!')
            ncpu = 2

        # transform one-hot encodings to labels if ohe == True
        train_data = ohe_to_ord(X_ohe=train_data, cat_vars_ohe=self.cat_vars_ohe)[0] if self.ohe else train_data

        disc = Discretizer(train_data, self.numerical_features, self.feature_names, percentiles=disc_perc)
        d_train_data = disc.discretize(train_data)

        self.feature_values.update(disc.feature_intervals)

        sampler_args = (
            self._predictor,
            disc_perc,
            self.numerical_features,
            self.categorical_features,
            self.feature_names,
            self.feature_values,
        )
        train_data_id = DistributedAnchorTabular.ray.put(train_data)
        d_train_data_id = DistributedAnchorTabular.ray.put(d_train_data)
        samplers = [TabularSampler(*sampler_args, seed=self.seed) for _ in range(ncpu)]  # type: ignore[arg-type]
        d_samplers = []
        for sampler in samplers:
            d_samplers.append(
                DistributedAnchorTabular.ray.remote(RemoteSampler).remote(
                    *(train_data_id, d_train_data_id, sampler)
                )
            )
        self.samplers = d_samplers

        # update metadata
        self.meta['params'].update(disc_perc=disc_perc)

        return self

    def _build_sampling_lookups(self, X: np.ndarray) -> None:
        """
        See :py:meth:`alibi.explainers.anchor_tabular.AnchorTabular._build_sampling_lookups` documentation.

        Parameters
        ----------
        X
            See :py:meth:`alibi.explainers.anchor_tabular.AnchorTabular._build_sampling_lookups` documentation.
        """

        lookups = [sampler.build_lookups.remote(X) for sampler in self.samplers][0]
        self.cat_lookup, self.ord_lookup, self.enc2feat_idx = DistributedAnchorTabular.ray.get(lookups)

    def explain(self,
                X: np.ndarray,
                threshold: float = 0.95,
                delta: float = 0.1,
                tau: float = 0.15,
                batch_size: int = 100,
                coverage_samples: int = 10000,
                beam_size: int = 1,
                stop_on_first: bool = False,
                max_anchor_size: Optional[int] = None,
                min_samples_start: int = 1,
                n_covered_ex: int = 10,
                binary_cache_size: int = 10000,
                cache_margin: int = 1000,
                verbose: bool = False,
                verbose_every: int = 1,
                **kwargs: Any) -> Explanation:
        """
        Explains the prediction made by a classifier on instance `X`. Sampling is done in parallel over a number of
        cores specified in `kwargs['ncpu']`.

        Parameters
        ----------
        X, threshold, delta, tau, batch_size, coverage_samples, beam_size, stop_on_first, max_anchor_size, \
        min_samples_start, n_covered_ex, binary_cache_size, cache_margin, verbose, verbose_every, **kwargs
            See :py:meth:`alibi.explainers.anchor_tabular.AnchorTabular.explain`.

        Returns
        -------
        See :py:meth:`alibi.explainers.anchor_tabular.AnchorTabular.explain` superclass.
        """
        # transform one-hot encodings to labels if ohe == True
        X = ohe_to_ord(X_ohe=X.reshape(1, -1), cat_vars_ohe=self.cat_vars_ohe)[0].reshape(-1) if self.ohe else X

        # get params for storage in meta
        params = locals()
        remove = ['X', 'self']
        for key in remove:
            params.pop(key)

        for sampler in self.samplers:
            label = sampler.set_instance_label.remote(X)
            sampler.set_n_covered.remote(n_covered_ex)

        self.instance_label = DistributedAnchorTabular.ray.get(label)

        # build feature encoding and mappings from the instance values to database rows where similar records are found
        # get anchors and add metadata
        self._build_sampling_lookups(X)
        mab = DistributedAnchorBaseBeam(
            samplers=self.samplers,
            sample_cache_size=binary_cache_size,
            cache_margin=cache_margin,
            **kwargs,
        )
        result = mab.anchor_beam(
            delta=delta, epsilon=tau,
            desired_confidence=threshold,
            beam_size=beam_size,
            min_samples_start=min_samples_start,
            max_anchor_size=max_anchor_size,
            batch_size=batch_size,
            coverage_samples=coverage_samples,
            verbose=verbose,
            verbose_every=verbose_every,
        )  # type: Any
        self.mab = mab

        return self._build_explanation(X, result, self.instance_label, params)

    def reset_predictor(self, predictor: Callable) -> None:
        """
        Resets the predictor function.

        Parameters
        ----------
        predictor
            New model prediction function.
        """
        raise NotImplementedError("Resetting predictor is currently not supported for distributed explainers.")
        # TODO: to support resetting a predictor we would need to re-run most of the code in `fit` instantiating the
        # instances of RemoteSampler anew
