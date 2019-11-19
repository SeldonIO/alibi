import logging
import numpy as np
from collections import OrderedDict, defaultdict
from itertools import accumulate
from typing import Any, Callable, Dict, List, Set, Tuple, Union

import ray

from .anchor_base import AnchorBaseBeam, DistributedAnchorBaseBeam
from .anchor_explanation import AnchorExplanation
from alibi.utils.data import ArgmaxTransformer
from alibi.utils.discretizer import Discretizer

# TODO: Fix typing issues, add all output types


class TabularSampler(object):
    """ A sampler that uses an underlying training set to draw records that have a subset of features with
    values specified in an instance to be expalined, X. """
    def __init__(self, predictor: Callable, disc_perc: Tuple[Union[int, float]], numerical_features: List[int],
                 categorical_features: List[int], feature_names: list, feature_values: dict, n_covered_ex: int = 10) \
            -> None:
        """
        Parameters
        ----------
        predictor
            an object exposing a .predict method, used to predict labels on the samples
        disc_perc
            percentiles used for numerical feat. discretisation
        numerical_features
            numerical features column IDs
        categorical_features
            categorical features column IDs
        feature_names
            feature names
        feature_values
            key: categorical feature column ID, value: values for the feature
        n_covered_ex
            for each anchor, a number of samples where the prediction agrees/disagrees
            with the prediction on instance to be explained are stored
        """

        self.instance_label = None
        self.predictor = predictor
        self.n_covered_ex = n_covered_ex

        self.numerical_features = numerical_features
        self.disc_perc = disc_perc
        self.feature_names = feature_names
        self.categorical_features = categorical_features
        self.feature_values = feature_values

        self.val2idx = {}       # type: Any
        self.cat_lookup = {}    # type: Dict[int, int]
        self.ord_lookup = {}    # type: Dict[int, set]
        self.enc2feat_idx = {}  # type: Dict[int, int]

    def deferred_init(self, train_data: Union[np.ndarray, Any], d_train_data: Union[np.array, Any]) -> Any:
        """
        Initialise the Tabular sampler object with data, discretizer, feature statistics and
        build an index from feature values and bins to database rows for each feature.

        Parameters
        ----------
        train_data:
            data from which samples are drawn. Can be a numpy array or a ray future
        d_train_data:
            discretized version for training data. Can be a numpy array or a ray future

        Returns
        -------
            an initialised sampler

        """
        self._set_data(train_data, d_train_data)
        self._set_discretizer(self.disc_perc)
        self._set_numerical_feats_stats()
        self.val2idx = self._get_data_index()

        return self

    def _set_data(self, train_data: Union[np.ndarray, Any], d_train_data: Union[np.array, Any]) -> None:
        """
        Initialise sampler training set and discretized training set, set number of records.
        """
        self.train_data = train_data
        self.d_train_data = d_train_data
        self.n_records = train_data.shape[0]

    def _set_discretizer(self, disc_perc: Tuple[Union[int, float], ...]) -> None:
        """
        Fit a discretizer to training data. Used to discretize returned samples.
        """
        self.disc = Discretizer(self.train_data,
                                self.numerical_features,
                                self.feature_names,
                                percentiles=disc_perc,
                                )

    def _set_numerical_feats_stats(self) -> None:
        """
        Compute min and max for numerical features so that sampling from this range can be performed if
        a sampling request has bin that is not in the training data.
        """
        self.min, self.max = np.full(self.train_data.shape[1], np.nan), np.full(self.train_data.shape[1], np.nan)
        self.min[self.numerical_features] = np.min(self.train_data[:, self.numerical_features], axis=0)
        self.max[self.numerical_features] = np.max(self.train_data[:, self.numerical_features], axis=0)

    def _get_data_index(self) -> Dict[int, Dict[int, np.ndarray]]:
        """
        Create a mapping where key is feat. col ID. and value is a dict where each int represents a bin value
        or value of categorical variable. Each value in this dict is an array of training data rows where that
        value is found

        Returns
        -------
        val2idx
            mapping as described above
        """
        # key (int): feat. col ID. value is a dict where each int represents a bin value or value of categorical
        # variable. Each value in this dict is a set of training data rows where that value is found
        val2idx = {f_id: defaultdict(None) for f_id in self.numerical_features + self.categorical_features}
        for feat in val2idx:
            for value in range(len(self.feature_values[feat])):
                val2idx[feat][value] = (self.d_train_data[:, feat] == value).nonzero()[0]

        return val2idx

    def __call__(self, anchor: Tuple[int, tuple], num_samples: int,  c_labels=True):# -> List[np.ndarray, np.ndarray, float, int]:
        """
        Draw samples from training data that contain the categorical features and discretized
        numerical features in anchor.

        Parameters
        ----------
        anchor
            the integer represents the order of the anchor in a request array. The tuple contains
            encoded feature indices
        num_samples
            Number of samples used when sampling from training set
        c_labels
            if False, labels are not returned by the sampling function

        Returns
        -------
        raw_data
            Sampled data from training set
        data
            Sampled data where ordinal features are binned (1 if in bin, 0 otherwise)
        labels
            Create labels using model predictions if compute_labels equals True
        anchor
            The index of anchor sampled in request array (used to speed up parallelisation)
        """

        raw_data, d_raw_data, coverage = self._sample_from_train(anchor[1], num_samples)

        # use the sampled, discretized raw data to construct a data matrix with the categorical ...
        # ... and binned ordinal data (1 if in bin, 0 otherwise)
        data = np.zeros((num_samples, len(self.enc2feat_idx)), int)
        for i in self.enc2feat_idx:
            if i in self.cat_lookup:
                data[:, i] = (d_raw_data[:, self.enc2feat_idx[i]] == self.cat_lookup[i])
            else:
                d_records_sampled = d_raw_data[:, self.enc2feat_idx[i]]
                lower_bin, upper_bin = min(list(self.ord_lookup[i])), max(list(self.ord_lookup[i]))
                idxs = np.where((lower_bin <= d_records_sampled) & (d_records_sampled <= upper_bin))
                data[idxs, i] = 1

        if c_labels:
            labels = self.compute_prec(raw_data)
            covered_true = raw_data[labels, :][:self.n_covered_ex]
            covered_false = raw_data[np.logical_not(labels), :][:self.n_covered_ex]
            return [covered_true, covered_false, labels.astype(int), data, coverage, anchor[0]]
        else:
            return [data]   # only binarised data is used for coverage computation

    def compute_prec(self, samples: np.ndarray) -> np.ndarray:
        """
        Compute the agreement between a classifier prediction on an instance to be explained and the
        prediction on a set of samples which have a subset of features fixed to a given value (aka
        compute the precision of anchors)

        Parameters
        ----------
        samples:
            samples whose labels are to be compared with the instance label

        Returns
        -------
            an array of integers indicating whether the prediction was the same as the instance label
        """

        return self.predictor(samples) == self.instance_label

    def _sample_from_train(self, anchor: tuple, num_samples: int) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Implements functionality described in __call__.

        Parameters
        ----------
        anchor:
            Each int is an encoded feature id
        num_samples
            Number of samples

        Returns
        -------

        samples
            Sampled data from training set
        d_samples
            Like samples, but continuous data is converted to oridinal discrete data (binned)
        coverage
            the coverage of the anchor in the training data
        """

        # Initialise samples randomly
        init_sample_idx = np.random.choice(range(self.train_data.shape[0]), num_samples, replace=True)
        samples = self.train_data[init_sample_idx]
        d_samples = self.d_train_data[init_sample_idx]
        if not anchor:
            return samples, d_samples, -1.0

        # bins one can sample from for each numerical feature (key: feat id)
        allowed_bins = {}  # type: Dict[int, Set[int]]
        # index of database rows (values) for each feature in anchor (key: feat id)
        allowed_rows = {}  # type: Dict[int, Any[int]]
        rand_sampled_feats = []  # feats for which there are not training records in the desired bin/with that value
        cat_enc_ids = [enc_id for enc_id in anchor if enc_id in self.cat_lookup.keys()]
        ord_enc_ids = [enc_id for enc_id in anchor if enc_id in self.ord_lookup.keys()]

        if cat_enc_ids:
            cat_feat_vals = [self.cat_lookup[idx] for idx in cat_enc_ids]
            cat_feat_ids = [self.enc2feat_idx[idx] for idx in cat_enc_ids]
            allowed_rows = {f_id: self.val2idx[f_id][f_val] for f_id, f_val in zip(cat_feat_ids, cat_feat_vals)}
            for feat_id, enc_id, val in zip(cat_feat_ids, cat_enc_ids, cat_feat_vals):
                if allowed_rows[feat_id].size == 0:
                    rand_sampled_feats.append((feat_id, 'c', val))
                    cat_feat_ids.remove(feat_id)

        ord_feat_ids = [self.enc2feat_idx[idx] for idx in ord_enc_ids]
        # determine bins from which ordinal data should be drawn
        for feat_id, enc_id in zip(ord_feat_ids, ord_enc_ids):
            # if encoded indices ref to the same feat, intersect the allowed bins to determine which bins to sample from
            if feat_id not in allowed_bins:
                allowed_bins[feat_id] = self.ord_lookup[enc_id]
            else:
                allowed_bins[feat_id] = allowed_bins[feat_id].intersection(self.ord_lookup[enc_id])

        # dict where keys are feature col. ids and values are lists containing row indices in train data which contain
        # data coming from the same bin (or range of bins)
        for feat_id in allowed_bins:  # NB: should scale since we don'features query the whole DB every time!
            allowed_rows[feat_id] = np.concatenate([self.val2idx[feat_id][bin_id] for bin_id in allowed_bins[feat_id]])
            if allowed_rows[feat_id].size == 0:  # no instances in training data are in the specified bins ...
                rand_sampled_feats.append((feat_id, 'o', None))
        uniq_feat_ids = list(OrderedDict.fromkeys([self.enc2feat_idx[enc_idx] for enc_idx in anchor]))
        uniq_feat_ids = [feat for feat in uniq_feat_ids if feat not in [f for f, _, _ in rand_sampled_feats]]

        # for each partial anchor count number of samples available and find their indices
        partial_anchor_rows = list(accumulate([allowed_rows[feat] for feat in uniq_feat_ids],
                                              np.intersect1d))
        n_partial_anchors = np.array([len(n_records) for n_records in reversed(partial_anchor_rows)])
        coverage = n_partial_anchors[-1]/self.n_records
        # search num_samples in the list containing the number of training records containing each sub-anchor
        num_samples_pos = np.searchsorted(n_partial_anchors, num_samples)
        if num_samples_pos == 0:  # training set has more than num_samples records containing the anchor
            samples_idxs = np.random.choice(partial_anchor_rows[-1], num_samples)
            samples[:, uniq_feat_ids] = self.train_data[np.ix_(samples_idxs, uniq_feat_ids)]
            d_samples[:, uniq_feat_ids] = self.d_train_data[np.ix_(samples_idxs, uniq_feat_ids)]
            return samples, d_samples, coverage

        # search partial anchors in the training set and replace the remainder of the features
        start, n_anchor_feats = 0, len(partial_anchor_rows)
        uniq_feat_ids = list(reversed(uniq_feat_ids))
        start_idx = np.nonzero(n_partial_anchors)[0][0]  # skip anchors with no samples in the database
        end_idx = np.searchsorted(np.cumsum(n_partial_anchors), num_samples)
        for idx, n_samp in enumerate(n_partial_anchors[start_idx:end_idx + 1], start=start_idx):
            if num_samples >= n_samp:
                samp_idxs = partial_anchor_rows[n_anchor_feats - idx - 1]
                num_samples -= n_samp
            else:
                if num_samples <= partial_anchor_rows[n_anchor_feats - idx - 1].shape[0]:
                    samp_idxs = np.random.choice(partial_anchor_rows[n_anchor_feats - idx - 1], num_samples)
                else:
                    samp_idxs = np.random.choice(partial_anchor_rows[n_anchor_feats - idx - 1],
                                                 num_samples,
                                                 replace=True,
                                                 )
                n_samp = num_samples
            samples[start:start + n_samp, uniq_feat_ids[idx:]] = self.train_data[np.ix_(samp_idxs, uniq_feat_ids[idx:])]
            if idx > 0:
                feats_to_replace = uniq_feat_ids[:idx]
                to_replace = [np.random.choice(allowed_rows[feat], n_samp, replace=True) for feat in feats_to_replace]
                samples[start: start + n_samp, feats_to_replace] = np.array(to_replace).transpose()
            start += n_samp

        if rand_sampled_feats:
            for feat, var_type, val in rand_sampled_feats:
                if var_type == 'c':
                    fmt = "WARNING: No data records have {} feature with value {}. Setting all samples' values to {}!"
                    print(fmt.format(feat, val, val))
                    samples[:, feat] = val
                else:
                    fmt = "WARNING: For feature {}, no training data record had discretized values in bins {}." \
                          " Sampling uniformly at random from the feature range!"
                    print(fmt.format(feat, allowed_bins[feat]))
                    min_vals, max_vals = self.min[feat], self.max[feat]
                    samples[:, feat] = np.random.uniform(low=min_vals,
                                                         high=max_vals,
                                                         size=(num_samples,)
                                                         )

        return samples, self.disc.discretize(samples), coverage

    def build_lookups(self, X: np.ndarray): # -> List[Dict, Dict, Dict]:
        """
        An encoding of the feature IDs is created by assigning each bin of a discretized numerical variable and each
        categorical variable a unique index. For a dataset containg, e.g., a numerical variable with 5 bins and
        3 categorical variables, indices 0 - 4 represent bins of the numerical variable whereas indices 5, 6, 7
        represent the encoded indices of the categorical variables (but see note for caviats). The encoding is
        necessary so that the different ranges of the numerical variable can be sampled during anchor construction.

        Note: Each continuous variable has n_bins - 1 corresponding entries in ord_lookup.

        Parameters
        ---------
        X
            instance to be explained

        Returns
        -------
            a list containing three dictionaries, whose keys are encoded feature IDs:
             - cat_lookup: maps categorical variables to their value in X
             - ord_lookup: maps discretized numerical variables to the bins they can be sampled from given X
             - enc2feat_idx: maps the encoded IDs to the original (training set) feature column IDs
        """

        X = self.disc.discretize(X.reshape(1, -1))[0]  # map continuous features to ordinal discrete variables

        if not self.numerical_features:  # data contains only categorical variables
            self.cat_lookup = dict(zip(self.categorical_features, X))
            self.enc2feat_idx = dict(zip(*[self.categorical_features] * 2))
            return [self.cat_lookup, self.ord_lookup, self.enc2feat_idx]

        first_numerical_idx = np.searchsorted(self.categorical_features, self.numerical_features[0]).item()
        if first_numerical_idx > 0:  # First column(s) might contain categorical data
            for cat_enc_idx in range(0, first_numerical_idx):
                self.cat_lookup[cat_enc_idx] = X[cat_enc_idx]
                self.enc2feat_idx[cat_enc_idx] = cat_enc_idx

        ord_enc_idx = first_numerical_idx - 1  # -1 as increment comes first
        for i, feature in enumerate(self.numerical_features):
            n_bins = len(self.feature_values[feature])
            for bin_val in range(n_bins):
                ord_enc_idx += 1
                self.enc2feat_idx[ord_enc_idx] = feature
                # if feat. value falls in same or lower bin, sample from same or lower bin only ...
                if X[feature] <= bin_val != n_bins - 1:
                    self.ord_lookup[ord_enc_idx] = set(i for i in range(bin_val + 1))
                # if feat. value falls in a higher bin, sample from higher bins only
                elif X[feature] > bin_val:
                    self.ord_lookup[ord_enc_idx] = set(i for i in range(bin_val + 1, n_bins))
                else:
                    del self.enc2feat_idx[ord_enc_idx]
                    ord_enc_idx -= 1  # when a discretized feat. of the instance to be explained falls in the last bin

            # check if a categorical feature follows the current numerical feature & update mappings
            if i < len(self.numerical_features) - 1:
                n_categoricals = self.numerical_features[i + 1] - self.numerical_features[i] - 1
                if n_categoricals > 0:
                    cat_feat_idx = feature + 1
                    for cat_enc_idx in range(ord_enc_idx + 1, ord_enc_idx + 1 + n_categoricals):
                        self.cat_lookup[cat_enc_idx] = X[cat_feat_idx]
                        self.enc2feat_idx[cat_enc_idx] = cat_feat_idx
                        cat_feat_idx += 1
                    ord_enc_idx += n_categoricals

        # check if the last columns are categorical variables and update mappings
        last_num_idx = np.searchsorted(self.categorical_features, self.numerical_features[-1]).item()
        if last_num_idx != len(self.categorical_features):
            cat_enc_idx = max(self.ord_lookup.keys()) + 1
            for cat_feat_idx in range(self.numerical_features[-1] + 1, self.categorical_features[-1] + 1):
                self.cat_lookup[cat_enc_idx] = X[cat_feat_idx]
                self.enc2feat_idx[cat_enc_idx] = cat_feat_idx
                cat_enc_idx += 1

        return [self.cat_lookup, self.ord_lookup, self.enc2feat_idx]


@ray.remote
class RemoteSampler(object):
    """ A wrapper that facilitates the use of TabularSampler for distributed sampling."""
    def __init__(self, *args):
        self.train_id, self.d_train_id, self.sampler = args
        self.sampler = self.sampler.deferred_init(self.train_id, self.d_train_id)

    def __call__(self, anchors_batch: Union[Tuple[int, tuple], List[Tuple[int, tuple]]], num_samples: int,
                 c_labels: bool = True):  # TODO: output typing
        """
        Wrapper around TabularSampler.__call__. It allows sampling a batch of anchors in the same process,
        which can improve performance.

        Parameters
        ----------
        anchors_batch:
            a list of anchor tuples. see TabularSampler.__call__ for details.
        num_samples:
            see TabularSampler.__call__
        c_labels
            see TabularSampler.__call__
        """

        if isinstance(anchors_batch, tuple):  # DistributedAnchorBaseBeam._get_samples_coverage call
            return self.sampler(anchors_batch, num_samples, c_labels=c_labels)
        elif len(anchors_batch) == 1:     # batch size = 1
            return [self.sampler(*anchors_batch, num_samples, c_labels=c_labels)]
        else:                             # batch size > 1
            batch_result = []
            for anchor in anchors_batch:
                batch_result.append(self.sampler(anchor, num_samples, c_labels=c_labels))

            return batch_result

    def build_lookups(self, X):
        """
        Wrapper around TabularSampler.build_lookups

        Parameters
        --------
        X
            see TabularSampler.build_lookups

        Returns
        -------
            see TabularSampler.build_lookups

        """

        cat_lookup_id, ord_lookup_id, enc2feat_idx_id = self.sampler.build_lookups(X)
        return [cat_lookup_id, ord_lookup_id, enc2feat_idx_id]


class AnchorTabular(object):

    def __init__(self, predictor: Callable, feature_names: list, categorical_names: dict = None,
                 seed: int = None) -> None:
        """
        Parameters
        ----------
        predictor
            Model prediction function
        feature_names
            List with feature names
        categorical_names
            Dictionary where keys are feature columns and values are the categories for the feature
        seed
            Used to set the random number generator for repeatability purposes

        """

        np.random.seed(seed)

        # check if predictor returns predicted class or prediction probabilities for each class
        # if needed adjust predictor so it returns the predicted class
        if np.argmax(predictor(np.zeros([1, len(feature_names)])).shape) == 0:
            self.predictor = predictor
        else:
            transformer = ArgmaxTransformer(predictor)
            self.predictor = transformer

        # define column indices of categorical and numerical (aka continuous) features
        if categorical_names:
            self.categorical_features = sorted(categorical_names.keys())
        else:
            self.categorical_features = []

        self.numerical_features = [x for x in range(len(feature_names)) if x not in self.categorical_features]

        self.feature_names = feature_names
        if categorical_names:
            self.feature_values = categorical_names.copy()  # dict with {col: categorical feature values}
        else:
            self.feature_values = {}

        self.samplers = []

    def fit(self, train_data: np.ndarray, disc_perc: tuple = (25, 50, 75), **kwargs) -> None:
        """
        Fit discretizer to train data to bin numerical features into ordered bins and compute statistics for numerical
        features. Create a mapping between the bin numbers of each discretised numerical feature and the row id in the
        training set where it occurs


        Parameters
        ----------
        train_data
            Representative sample from the training data
        disc_perc
            List with percentiles (int) used for discretization
        """

        # discretization of ordinal features
        disc = Discretizer(train_data, self.numerical_features, self.feature_names, percentiles=disc_perc)
        d_train_data = disc.discretize(train_data)
        self.feature_values.update(disc.feature_intervals)
        sampler = TabularSampler(self.predictor,
                                 disc_perc,
                                 self.numerical_features,
                                 self.categorical_features,
                                 self.feature_names,
                                 self.feature_values,
                                 )
        self.samplers = [sampler.deferred_init(train_data, d_train_data)]

    def _build_sampling_lookups(self, X: np.ndarray) -> None:
        """
        Build a series of lookup tables used to draw samples with feature subsets identical to
        given subsets of X (see TabularSampler.build_sampling_lookups for details)

        Parameters
        ----------
        X:
            instance to be explained
        """

        lookups = [sampler.build_lookups(X) for sampler in self.samplers][0]
        self.cat_lookup, self.ord_lookup, self.enc2feat_idx = lookups

    def explain(self, X: np.ndarray, threshold: float = 0.95, delta: float = 0.1, tau: float = 0.15,
                batch_size: int = 100, beam_size: int = 1, max_anchor_size: int = None, min_samples_start: int = 1,
                desired_label: int = None, **kwargs: Any) -> dict:
        """
        Explain prediction made by classifier on instance X.

        Parameters
        ----------
        X
            Instance to be explained
        threshold
            Minimum precision threshold
        delta
            Used to compute beta
        tau
            Margin between lower confidence bound and minimum precision or upper bound
        batch_size
            Batch size used for sampling
        beam_size
            The number of anchors extended at each step of new anchors construction
        max_anchor_size
            Maximum number of features in anchor
        min_samples_start
            Min number of initial samples
        desired_label
            Label to use as true label for the instance to be explained

        Returns
        -------
        explanation
            Dictionary containing the anchor explaining the instance with additional metadata
        """

        # if no true label available; true label = predicted label
        true_label = desired_label
        if true_label is None:
            self.instance_label = self.predictor(X.reshape(1, -1))[0]

        for sampler in self.samplers:
            sampler.instance_label = self.instance_label

        # build feature encoding and mappings from the instance values to database rows where similar records are found
        # get anchors and add metadata
        self._build_sampling_lookups(X)

        mab = AnchorBaseBeam(samplers=self.samplers,
                             **kwargs,
                             )
        anchor = mab.anchor_beam(delta=delta,
                                 epsilon=tau,
                                 desired_confidence=threshold,
                                 max_anchor_size=max_anchor_size,
                                 min_samples_start=min_samples_start,
                                 beam_size=beam_size,
                                 batch_size=batch_size,
                                 coverage_samples=10000,  # TODO: DO NOT HARDCODE THESE
                                 data_store_size=10000,
                                 )  # type: Any

        return self.return_anchor(X, anchor, true_label)

    def return_anchor(self, X: np.ndarray, anchor: dict, true_label: int) -> dict:
        """
        Preprocess search output and return an explanation object containing metdata

        Parameters
        ----------
        X:
            instance to be explained
        anchor:
            dictionary with explanation search output and metadata
        true_label:
            label of the instance to be explained (inferred if not given)

        Return
        ------
            a dictionary containing human readable explanation, metadata, and precision/coverage info
        """

        self.add_names_to_exp(anchor)
        if true_label is None:
            anchor['prediction'] = self.instance_label
        else:
            anchor['prediction'] = self.predictor(X.reshape(1, -1))[0]
        anchor['instance'] = X
        exp = AnchorExplanation('tabular', anchor)
        return {'names': exp.names(),
                'precision': exp.precision(),
                'coverage': exp.coverage(),
                'raw': exp.exp_map,
                'meta': {'name': self.__class__.__name__}
                }

    def add_names_to_exp(self, explanation: dict) -> None:
        """
        Add feature names to explanation dictionary.

        Parameters
        ----------
        explanation
            Dict with anchors and additional metadata
        """
        anchor_idxs = explanation['feature']
        explanation['names'] = []
        explanation['feature'] = [self.enc2feat_idx[idx] for idx in anchor_idxs]
        ordinal_ranges = {self.enc2feat_idx[idx]: [float('-inf'), float('inf')] for idx in anchor_idxs}
        for idx in set(anchor_idxs) - self.cat_lookup.keys():
            if 0 in self.ord_lookup[idx]:  # tells if the feature in X falls in a higher or lower bin
                ordinal_ranges[self.enc2feat_idx[idx]][1] = min(ordinal_ranges[self.enc2feat_idx[idx]][1],
                                                                max(list(self.ord_lookup[idx])))
            else:
                ordinal_ranges[self.enc2feat_idx[idx]][0] = max(ordinal_ranges[self.enc2feat_idx[idx]][0],
                                                                min(list(self.ord_lookup[idx])) - 1)

        handled = set()  # type: Set[int]
        for idx in anchor_idxs:
            feat_id = self.enc2feat_idx[idx]
            if idx in self.cat_lookup:
                v = int(self.cat_lookup[idx])
                fname = '%s = ' % self.feature_names[feat_id]
                if feat_id in self.feature_values:
                    v = int(v)
                    if ('<' in self.feature_values[feat_id][v]
                            or '>' in self.feature_values[feat_id][v]):
                        fname = ''
                    fname = '%s%s' % (fname, self.feature_values[feat_id][v])
                else:
                    fname = '%s%.2f' % (fname, v)
            else:
                if feat_id in handled:
                    continue
                geq, leq = ordinal_ranges[feat_id]
                fname = ''
                geq_val = ''
                leq_val = ''
                if geq > float('-inf'):
                    if geq == len(self.feature_values[feat_id]) - 1:
                        geq = geq - 1
                    name = self.feature_values[feat_id][geq + 1]
                    if '<' in name:
                        geq_val = name.split()[0]
                    elif '>' in name:
                        geq_val = name.split()[-1]
                if leq < float('inf'):
                    name = self.feature_values[feat_id][leq]
                    if leq == 0:
                        leq_val = name.split()[-1]
                    elif '<' in name:
                        leq_val = name.split()[-1]
                if leq_val and geq_val:
                    fname = '%s < %s <= %s' % (geq_val, self.feature_names[feat_id],
                                               leq_val)
                elif leq_val:
                    fname = '%s <= %s' % (self.feature_names[feat_id], leq_val)
                elif geq_val:
                    fname = '%s > %s' % (self.feature_names[feat_id], geq_val)
                handled.add(feat_id)
            explanation['names'].append(fname)


class DistributedAnchorTabular(AnchorTabular):

    def __init__(self, predictor: Callable, feature_names: list, categorical_names: dict = None,
                 seed: int = None) -> None:

        super(DistributedAnchorTabular, self).__init__(predictor, feature_names, categorical_names, seed)
        import ray
        ray.init()

    def fit(self, train_data: np.ndarray, disc_perc: tuple = (25, 50, 75), **kwargs) -> None:
        """
        Creates a list of handles to parallel processes handles that are used for submitting sampling
        tasks

        Parameters
        ----------
            see superclass implementation
        """

        try:
            ncpu = kwargs['ncpu']
        except KeyError:
            logging.warning('DistributedAnchorTabular object has been initalised but kwargs did not contain '
                            'expected argument, ncpu. Defaulting to ncpu=2!'
                            )
            ncpu = 2

        disc = Discretizer(train_data, self.numerical_features, self.feature_names, percentiles=disc_perc)
        d_train_data = disc.discretize(train_data)
        self.feature_values.update(disc.feature_intervals)
        sampler_args = (self.predictor,
                        disc_perc,
                        self.numerical_features,
                        self.categorical_features,
                        self.feature_names,
                        self.feature_values,
                        )
        train_data_id = ray.put(train_data)
        d_train_data_id = ray.put(d_train_data)
        samplers = [TabularSampler(*sampler_args) for _ in range(ncpu)]
        self.samplers = [RemoteSampler.remote(*(train_data_id, d_train_data_id, sampler)) for sampler in samplers]

    def _build_sampling_lookups(self, X):
        lookups = [sampler.build_lookups.remote(X) for sampler in self.samplers][0]
        self.cat_lookup, self.ord_lookup, self.enc2feat_idx = ray.get(lookups)

    def explain(self, X: np.ndarray, threshold: float = 0.95, delta: float = 0.1, tau: float = 0.15,
                batch_size: int = 100, beam_size: int = 1, max_anchor_size: int = None, min_samples_start: int = 1,
                desired_label: int = None, **kwargs: Any) -> dict:
        """
        Explains the prediction made by a classifier on instance X. Sampling is done in parallel over a number of
        cores specified in kwargs['ncpu'].

        Parameters
        ----------
            see superclass implementation

        Returns
        -------
            see superclass implementation
        """

        # if no true label available; true label = predicted label
        true_label = desired_label
        if true_label is None:
            self.instance_label = self.predictor(X.reshape(1, -1))[0]

        for sampler in self.samplers:
            sampler.instance_label = self.instance_label

        # build feature encoding and mappings from the instance values to database rows where similar records are found
        # get anchors and add metadata
        self._build_sampling_lookups(X)

        mab = DistributedAnchorBaseBeam(samplers=self.samplers,
                                        **kwargs,
                                        )
        anchor = mab.anchor_beam(delta=delta,
                                 epsilon=tau,
                                 desired_confidence=threshold,
                                 min_samples_start=min_samples_start,
                                 max_anchor_size=max_anchor_size,
                                 beam_size=beam_size,
                                 batch_size=batch_size,
                                 coverage_samples=10000,  # TODO: DO NOT HARDCODE THESE
                                 data_store_size=10000,
                                 )  # type: Any
        return self.return_anchor(X, anchor, true_label)
