import copy
import logging
import numpy as np
from collections import OrderedDict, defaultdict
from itertools import accumulate
from typing import Any, Callable, DefaultDict, Dict, List, Set, Tuple, Union

from alibi.api.interfaces import Explainer, Explanation, FitMixin
from alibi.api.defaults import DEFAULT_META_ANCHOR, DEFAULT_DATA_ANCHOR
from .anchor_base import AnchorBaseBeam, DistributedAnchorBaseBeam
from .anchor_explanation import AnchorExplanation
from alibi.utils.wrappers import ArgmaxTransformer
from alibi.utils.discretizer import Discretizer
from alibi.utils.distributed import RAY_INSTALLED


class TabularSampler:
    """ A sampler that uses an underlying training set to draw records that have a subset of features with
    values specified in an instance to be expalined, X."""

    def __init__(self, predictor: Callable, disc_perc: Tuple[Union[int, float], ...], numerical_features: List[int],
                 categorical_features: List[int], feature_names: list, feature_values: dict, n_covered_ex: int = 10,
                 seed: int = None) -> None:
        """
        Parameters
        ----------
        predictor
            A callable that takes a tensor of N data points as inputs and returns N outputs.
        disc_perc
            Percentiles used for numerical feat. discretisation.
        numerical_features
            Numerical features column IDs.
        categorical_features
            Categorical features column IDs.
        feature_names
            Feature names.
        feature_values
            Key: categorical feature column ID, value: values for the feature.
        n_covered_ex
            For each result, a number of samples where the prediction agrees/disagrees
            with the prediction on instance to be explained are stored.
        seed
            If set, fixes the random number sequence.
        """

        np.random.seed(seed)

        self.instance_label = None  # type: int
        self.predictor = predictor
        self.n_covered_ex = n_covered_ex

        self.numerical_features = numerical_features
        self.disc_perc = disc_perc
        self.feature_names = feature_names
        self.categorical_features = categorical_features
        self.feature_values = feature_values

        self.val2idx = {}  # type: Dict[int, DefaultDict[int, Any]]
        self.cat_lookup = {}  # type: Dict[int, int]
        self.ord_lookup = {}  # type: Dict[int, set]
        self.enc2feat_idx = {}  # type: Dict[int, int]

    def deferred_init(self, train_data: Union[np.ndarray, Any], d_train_data: Union[np.array, Any]) -> Any:
        """
        Initialise the Tabular sampler object with data, discretizer, feature statistics and
        build an index from feature values and bins to database rows for each feature.

        Parameters
        ----------
        train_data:
            Data from which samples are drawn. Can be a numpy array or a ray future.
        d_train_data:
            Discretized version for training data. Can be a numpy array or a ray future.

        Returns
        -------
            An initialised sampler.

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

        self.disc = Discretizer(
            self.train_data,
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

    def set_instance_label(self, X: np.ndarray) -> None:
        """
        Sets the sampler label. Necessary for setting the remote sampling process state during explain call.

        Parameters
        ----------
        X
             Instance to be explained.
        """

        label = self.predictor(X.reshape(1, -1))[0]
        self.instance_label = label

    def set_n_covered(self, n_covered: int) -> None:
        """
        Set the number of examples to be saved for each result and partial result during search process.
        The same number of examples is saved in the case where the predictions on perturbed samples and
        original instance agree or disagree.

        Parameters
        ---------
        n_covered
            Number of examples to be saved.
        """

        self.n_covered_ex = n_covered

    def _get_data_index(self) -> Dict[int, DefaultDict[int, np.ndarray]]:
        """
        Create a mapping where key is feat. col ID. and value is a dict where each int represents a bin value
        or value of categorical variable. Each value in this dict is an array of training data rows where that
        value is found.

        Returns
        -------
        val2idx
            Mapping as described above.
        """

        all_features = self.numerical_features + self.categorical_features
        val2idx = {f_id: defaultdict(None) for f_id in all_features}  # type: Dict[int, DefaultDict[int, np.ndarray]]
        for feat in val2idx:
            for value in range(len(self.feature_values[feat])):
                val2idx[feat][value] = (self.d_train_data[:, feat] == value).nonzero()[0]

        return val2idx

    def __call__(self, anchor: Tuple[int, tuple], num_samples: int, compute_labels=True) -> \
            Union[List[Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int]], List[np.ndarray]]:
        """
        Obtain perturbed records by drawing samples from training data that contain the categorical labels and
        discretized numerical features and replacing the remainder of the record with arbitrary values.

        Parameters
        ----------
        anchor
            The integer represents the order of the result in a request array. The tuple contains
            encoded feature indices.
        num_samples
            Number of samples used when sampling from training set.
        compute_labels
            If True, an array of comparisons between predictions on perturbed samples and instance to be
            explained is returned.

        Returns
        -------
            If compute_labels=True, a list containing the following is returned:
             - covered_true: perturbed examples where the anchor applies and the model prediction
                    on perturbation is the same as the instance prediction
             - covered_false: perturbed examples where the anchor applies and the model prediction
                    is NOT the same as the instance prediction
             - labels: num_samples ints indicating whether the prediction on the perturbed sample
                    matches (1) the label of the instance to be explained or not (0)
             - data: Sampled data where ordinal features are binned (1 if in bin, 0 otherwise)
             - coverage: the coverage of the anchor
             - anchor[0]: position of anchor in the batch request
            Otherwise, a list containing the data matrix only is returned.
        """

        raw_data, d_raw_data, coverage = self.perturbation(anchor[1], num_samples)

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

        if compute_labels:
            labels = self.compare_labels(raw_data)
            covered_true = raw_data[labels, :][:self.n_covered_ex]
            covered_false = raw_data[np.logical_not(labels), :][:self.n_covered_ex]
            return [covered_true, covered_false, labels.astype(int), data, coverage, anchor[0]]
        else:
            return [data]  # only binarised data is used for coverage computation

    def compare_labels(self, samples: np.ndarray) -> np.ndarray:
        """
        Compute the agreement between a classifier prediction on an instance to be explained and the
        prediction on a set of samples which have a subset of features fixed to specific values.

        Parameters
        ----------
        samples
            Samples whose labels are to be compared with the instance label.

        Returns
        -------
            An array of integers indicating whether the prediction was the same as the instance label.
        """

        return self.predictor(samples) == self.instance_label

    def perturbation(self, anchor: tuple, num_samples: int) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Implements functionality described in __call__.

        Parameters
        ----------
        anchor:
            Each int is an encoded feature id.
        num_samples
            Number of samples.

        Returns
        -------

        samples
            Sampled data from training set.
        d_samples
            Like samples, but continuous data is converted to oridinal discrete data (binned).
        coverage
            The coverage of the result in the training data.
        """

        # initialise samples randomly
        init_sample_idx = np.random.choice(range(self.train_data.shape[0]), num_samples, replace=True)
        samples = self.train_data[init_sample_idx]
        d_samples = self.d_train_data[init_sample_idx]

        if not anchor:
            return samples, d_samples, -1.0

        # find the training set row indices where each feature in the anchor has same value as instance
        # for discretized continuous features in the anchor, find which bins it should be sampled from
        # find any features what have values/are in bins that don't exist in the training set
        allowed_bins, allowed_rows, unk_feat_vals = self.get_features_index(anchor)

        #  count number of samples available and find the indices for each partial anchor & the full anchor
        uniq_feat_ids = list(OrderedDict.fromkeys([self.enc2feat_idx[enc_idx] for enc_idx in anchor]))
        uniq_feat_ids = [feat for feat in uniq_feat_ids if feat not in [f for f, _, _ in unk_feat_vals]]
        partial_anchor_rows = list(accumulate(
            [allowed_rows[feat] for feat in uniq_feat_ids],
            np.intersect1d),
        )
        nb_partial_anchors = np.array([len(n_records) for n_records in reversed(partial_anchor_rows)])
        coverage = nb_partial_anchors[-1] / self.n_records

        # if there are enough train records containing the anchor, replace the original records and return...
        num_samples_pos = np.searchsorted(nb_partial_anchors, num_samples)
        if num_samples_pos == 0:
            samples_idxs = np.random.choice(partial_anchor_rows[-1], num_samples)
            samples[:, uniq_feat_ids] = self.train_data[np.ix_(samples_idxs, uniq_feat_ids)]
            d_samples[:, uniq_feat_ids] = self.d_train_data[np.ix_(samples_idxs, uniq_feat_ids)]

            return samples, d_samples, coverage

        # ... otherwise, replace the record with partial anchors first and then sample the remainder of the feats
        # from  the same bin or set them to the same value as in the instance to be explained
        # NB: this function modifies samples since it gets a reference to this array so it doesn't return
        self.replace_features(
            samples,
            allowed_rows,
            uniq_feat_ids,
            partial_anchor_rows,
            nb_partial_anchors,
            num_samples,
        )

        if unk_feat_vals:
            self.handle_unk_features(allowed_bins, num_samples, samples, unk_feat_vals)

        return samples, self.disc.discretize(samples), coverage

    def handle_unk_features(self, allowed_bins: Dict[int, Set[int]], num_samples: int, samples: np.ndarray,
                            unk_feature_values: List[Tuple[int, str, Union[Any, int]]]) -> None:
        """
        Replaces unknown feature values with defaults. For categorical variables, the replacement value is
        the same as the value of the unknown feature. For continuous variables, a value is sampled uniformly
        at random from the feature range.

        Parameters
        ----------
        allowed_bins:
            See get_feature_index method.
        num_samples:
            Number of replacement values.
        samples:
            Contains the samples whose values are to be replaced.
        unk_feature_values:
            List of tuples where: [0] is original feature id, [1] feature type, [2] if var is categorical,
            replacement value, otherwise None
        """

        for feat, var_type, val in unk_feature_values:
            if var_type == 'c':
                fmt = "WARNING: No data records have {} feature with value {}. Setting all samples' values to {}!"
                print(fmt.format(feat, val, val))
                samples[:, feat] = val
            else:
                fmt = "WARNING: For feature {}, no training data record had discretized values in bins {}." \
                      " Sampling uniformly at random from the feature range!"
                print(fmt.format(feat, allowed_bins[feat]))
                min_vals, max_vals = self.min[feat], self.max[feat]
                samples[:, feat] = np.random.uniform(low=min_vals, high=max_vals, size=(num_samples,))

    def replace_features(self, samples: np.ndarray, allowed_rows: Dict[int, Any], uniq_feat_ids: List[int],
                         partial_anchor_rows: List[np.ndarray], nb_partial_anchors: np.ndarray,
                         num_samples: int) -> None:
        """
        The method creates perturbed samples by first replacing all partial anchors with partial anchors drawn
        from the training set. Then remainder of the features are then replaced with random values drawn from
        the same bin for discretized continuous features and same value for categorical features.

        Parameters
        ----------
        samples
            Randomly drawn samples, where the anchor does not apply.
        allowed_rows
            Maps feature ids to the rows indices in training set where the feature has same value as instance (cat.)
            or is in the same bin.
        uniq_feat_ids
            Multiple encoded features in the anchor can map to the same original feature id. Unique features in the
            anchor. This is the list of unique original features id in the anchor.
        partial_anchor_rows
            The rows in the training set where each partial anchor applies. Last entry is an array of row indices where
            the entire anchor applies.
        nb_partial_anchors
            The number of training records which contain each partial anchor.
        num_samples:
            Number of perturbed samples to be returned.
        """

        requested_samples = num_samples
        start, n_anchor_feats = 0, len(partial_anchor_rows)
        uniq_feat_ids = list(reversed(uniq_feat_ids))
        start_idx = np.nonzero(nb_partial_anchors)[0][0]  # skip anchors with no samples in the database
        end_idx = np.searchsorted(np.cumsum(nb_partial_anchors), num_samples)

        # replace partial anchors with partial anchors drawn from the training dataset
        # samp_idxs are arrays of training set row indices from where partial anchors are extracted for replacement
        for idx, n_samp in enumerate(nb_partial_anchors[start_idx:end_idx + 1], start=start_idx):
            if num_samples >= n_samp:
                samp_idxs = partial_anchor_rows[n_anchor_feats - idx - 1]
                num_samples -= n_samp
            else:
                if num_samples <= partial_anchor_rows[n_anchor_feats - idx - 1].shape[0]:
                    samp_idxs = np.random.choice(partial_anchor_rows[n_anchor_feats - idx - 1], num_samples)
                else:
                    samp_idxs = np.random.choice(
                        partial_anchor_rows[n_anchor_feats - idx - 1],
                        num_samples,
                        replace=True,
                    )
                n_samp = num_samples
            samples[start:start + n_samp, uniq_feat_ids[idx:]] = self.train_data[np.ix_(samp_idxs, uniq_feat_ids[idx:])]

            # deal with partial anchors; idx = 0 means that we actually sample the entire anchor
            if idx > 0:

                # choose replacement values at random from training set
                feats_to_replace = uniq_feat_ids[:idx]
                samp_idxs = np.zeros((len(feats_to_replace), n_samp)).astype(int)  # =: P x Q
                for i, feat_idx in enumerate(feats_to_replace):
                    samp_idxs[i, :] = np.random.choice(allowed_rows[feat_idx], n_samp, replace=True)

                # N x F ->  P X Q x F -> P X Q
                # First slice takes the data rows indicated in rows of samp_idxs; each row corresponds to a diff. feat
                # Second slice takes the column indicate in feats_to_replace from each of P Q x F sub-tensors
                to_replace_vals = self.train_data[samp_idxs][np.arange(len(feats_to_replace)), :, feats_to_replace]
                samples[start: start + n_samp, feats_to_replace] = to_replace_vals.transpose()
            start += n_samp

        # possible that the dataset doesn't contain enough partial examples. Eg, in anchor is (10,) and have
        # 50 examples in training set but require batch size of 100, all features have to be replaced  one by one in
        # last 50 samples
        max_samples_available = nb_partial_anchors.sum()
        if max_samples_available <= requested_samples:
            n_samp = samples.shape[0] - start

            samp_idxs = np.zeros((len(uniq_feat_ids), n_samp)).astype(int)
            for i, feat_idx in enumerate(uniq_feat_ids):
                samp_idxs[i, :] = np.random.choice(allowed_rows[feat_idx], n_samp, replace=True)

            to_replace_vals = self.train_data[samp_idxs][np.arange(len(uniq_feat_ids)), :, uniq_feat_ids]
            samples[start:, uniq_feat_ids] = to_replace_vals.transpose()

    def get_features_index(self, anchor: tuple) -> \
            Tuple[Dict[int, Set[int]], Dict[int, Any], List[Tuple[int, str, Union[Any, int]]]]:
        """
        Given an anchor, this function finds the row indices in the training set where the feature has
        the same value as the feature in the instance to be explained (for ordinal variables, the row
        indices are those of rows which contain records with feature values in the same bin). The algorithm
        uses both the feature *encoded* ids in anchor and the feature ids in the input data set. The two
        are mapped by self.enc2feat_idx.

        Parameters
        ----------
        anchor
            The anchor for which the training set row indices are to be retrieved. The ints represent
            encoded feature ids.

        Returns
        -------
        allowed_bins
            Maps original feature ids to the bins that the feature should be sampled from given the input anchor.
        allowed_rows
            Maps original feature ids to the training set rows where these features have the same value as the anchor.
        unk_feat_values
            When a categorical variable with the specified value/discretized variable in the specified bin is not found
            in the training set, a tuple is added to unk_feat_values to indicate the original feature id, its type
            ('c'=categorical, o='discretized continuous') and the value/bin it should be sampled from.
        """

        # bins one can sample from for each numerical feature (key: feat id)
        allowed_bins = {}  # type: Dict[int, Set[int]]
        # index of database rows (values) for each feature in result (key: feat id)
        allowed_rows = {}  # type: Dict[int, Any[int]]
        unk_feat_values = []  # feats for which there are not training records in the desired bin/with that value
        cat_enc_ids = [enc_id for enc_id in anchor if enc_id in self.cat_lookup.keys()]
        ord_enc_ids = [enc_id for enc_id in anchor if enc_id in self.ord_lookup.keys()]
        if cat_enc_ids:
            cat_feat_vals = [self.cat_lookup[idx] for idx in cat_enc_ids]
            cat_feat_ids = [self.enc2feat_idx[idx] for idx in cat_enc_ids]
            allowed_rows = {f_id: self.val2idx[f_id][f_val] for f_id, f_val in zip(cat_feat_ids, cat_feat_vals)}
            for feat_id, enc_id, val in zip(cat_feat_ids, cat_enc_ids, cat_feat_vals):
                if allowed_rows[feat_id].size == 0:
                    unk_feat_values.append((feat_id, 'c', val))
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
        for feat_id in allowed_bins:  # NB: should scale since we don't query the whole DB every time!
            allowed_rows[feat_id] = np.concatenate([self.val2idx[feat_id][bin_id] for bin_id in allowed_bins[feat_id]])
            if allowed_rows[feat_id].size == 0:  # no instances in training data are in the specified bins ...
                unk_feat_values.append((feat_id, 'o', None))

        return allowed_bins, allowed_rows, unk_feat_values

    def build_lookups(self, X: np.ndarray) -> List[Dict]:
        """
        An encoding of the feature IDs is created by assigning each bin of a discretized numerical variable and each
        categorical variable a unique index. For a dataset containg, e.g., a numerical variable with 5 bins and
        3 categorical variables, indices 0 - 4 represent bins of the numerical variable whereas indices 5, 6, 7
        represent the encoded indices of the categorical variables (but see note for caviats). The encoding is
        necessary so that the different ranges of the numerical variable can be sampled during result construction.
        Note that the encoded indices represent the predicates used during the anchor construction process (i.e., and
        anchor is a collection of encoded indices.

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
            self.enc2feat_idx = dict(zip(*[self.categorical_features] * 2))  # type: ignore
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


class RemoteSampler:
    """ A wrapper that facilitates the use of TabularSampler for distributed sampling."""
    if RAY_INSTALLED:
        import ray
        ray = ray  # set module as class variable to used only in this context

    def __init__(self, *args):
        self.train_id, self.d_train_id, self.sampler = args
        self.sampler = self.sampler.deferred_init(self.train_id, self.d_train_id)

    def __call__(self, anchors_batch: Union[Tuple[int, tuple], List[Tuple[int, tuple]]], num_samples: int,
                 compute_labels: bool = True) -> List:
        """
        Wrapper around TabularSampler.__call__. It allows sampling a batch of anchors in the same process,
        which can improve performance.

        Parameters
        ----------
        anchors_batch:
            A list of result tuples. see TabularSampler.__call__ for details.
        num_samples:
            See TabularSampler.__call__.
        compute_labels
            See TabularSampler.__call__.
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

    def build_lookups(self, X):
        """
        Wrapper around TabularSampler.build_lookups.

        Parameters
        --------
        X
            See TabularSampler.build_lookups.

        Returns
        -------
            See TabularSampler.build_lookups.
        """

        cat_lookup_id, ord_lookup_id, enc2feat_idx_id = self.sampler.build_lookups(X)

        return [cat_lookup_id, ord_lookup_id, enc2feat_idx_id]


class AnchorTabular(Explainer, FitMixin):

    def __init__(self, predictor: Callable, feature_names: list, categorical_names: dict = None,
                 seed: int = None) -> None:
        """
        Parameters
        ----------
        predictor
            A callable that takes a tensor of N data points as inputs and returns N outputs.
        feature_names
            List with feature names.
        categorical_names
            Dictionary where keys are feature columns and values are the categories for the feature.
        seed
            Used to set the random number generator for repeatability purposes.
        """
        super().__init__(meta=copy.deepcopy(DEFAULT_META_ANCHOR))

        self.feature_names = feature_names
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
            self.feature_values = categorical_names.copy()  # dict with {col: categorical feature values}

        else:
            self.categorical_features = []
            self.feature_values = {}

        self.numerical_features = [x for x in range(len(feature_names)) if x not in self.categorical_features]

        self.samplers = []  # type: list
        self.seed = seed
        self.instance_label = None

        # update metadata
        self.meta['params'].update(seed=seed)

    def fit(self, train_data: np.ndarray, disc_perc: Tuple[Union[int, float], ...] = (25, 50, 75),  # type:ignore
            **kwargs) -> "AnchorTabular":
        """
        Fit discretizer to train data to bin numerical features into ordered bins and compute statistics for
        numerical features. Create a mapping between the bin numbers of each discretised numerical feature and the
        row id in the training set where it occurs.

        Parameters
        ----------
        train_data
            Representative sample from the training data.
        disc_perc
            List with percentiles (int) used for discretization.
        """

        # discretization of continuous features
        disc = Discretizer(train_data, self.numerical_features, self.feature_names, percentiles=disc_perc)
        d_train_data = disc.discretize(train_data)
        self.feature_values.update(disc.feature_intervals)

        sampler = TabularSampler(
            self.predictor,
            disc_perc,
            self.numerical_features,
            self.categorical_features,
            self.feature_names,
            self.feature_values,
            seed=self.seed,
        )
        self.samplers = [sampler.deferred_init(train_data, d_train_data)]

        # update metadata
        self.meta['params'].update(disc_perc=disc_perc)

        return self

    def _build_sampling_lookups(self, X: np.ndarray) -> None:
        """
        Build a series of lookup tables used to draw samples with feature subsets identical to
        given subsets of X (see TabularSampler.build_sampling_lookups for details).

        Parameters
        ----------
        X:
            Instance to be explained.
        """

        lookups = [sampler.build_lookups(X) for sampler in self.samplers][0]
        self.cat_lookup, self.ord_lookup, self.enc2feat_idx = lookups

    def explain(self,
                X: np.ndarray,
                threshold: float = 0.95,
                delta: float = 0.1,
                tau: float = 0.15,
                batch_size: int = 100,
                coverage_samples: int = 10000,
                beam_size: int = 1,
                stop_on_first: bool = False,
                max_anchor_size: int = None,
                min_samples_start: int = 100,
                n_covered_ex: int = 10,
                binary_cache_size: int = 10000,
                cache_margin: int = 1000,
                verbose: bool = False,
                verbose_every: int = 1,
                **kwargs: Any) -> Explanation:
        """
        Explain prediction made by classifier on instance X.

        Parameters
        ----------
        X
            Instance to be explained.
        threshold
            Minimum precision threshold.
        delta
            Used to compute beta.
        tau
            Margin between lower confidence bound and minimum precision or upper bound.
        batch_size
            Batch size used for sampling.
        coverage_samples
            Number of samples used to estimate coverage from during result search.
        beam_size
            The number of anchors extended at each step of new anchors construction.
        stop_on_first
            If True, the beam search algorithm will return the first anchor that has satisfies the
            probability constraint.
        max_anchor_size
            Maximum number of features in result.
        min_samples_start
            Min number of initial samples.
        n_covered_ex
            How many examples where anchors apply to store for each anchor sampled during search
            (both examples where prediction on samples agrees/disagrees with desired_label are stored).
        binary_cache_size
            The result search pre-allocates binary_cache_size batches for storing the binary arrays
            returned during sampling.
        cache_margin
            When only max(cache_margin, batch_size) positions in the binary cache remain empty, a new cache
            of the same size is pre-allocated to continue buffering samples.
        verbose
            Display updates during the anchor search iterations.
        verbose_every
            Frequency of displayed iterations during anchor search process.


        Returns
        -------
        explanation
            Dictionary containing the result explaining the instance with additional metadata.
        """
        # get params for storage in meta
        params = locals()
        remove = ['X', 'self']
        for key in remove:
            params.pop(key)

        for sampler in self.samplers:
            sampler.set_instance_label(X)
            sampler.set_n_covered(n_covered_ex)
        self.instance_label = self.samplers[0].instance_label

        # build feature encoding and mappings from the instance values to database rows where
        # similar records are found get anchors and add metadata
        self._build_sampling_lookups(X)

        # get anchors
        mab = AnchorBaseBeam(
            samplers=self.samplers,
            sample_cache_size=binary_cache_size,
            cache_margin=cache_margin,
            **kwargs)
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

        return self.build_explanation(X, result, self.instance_label, params)

    def build_explanation(self, X: np.ndarray, result: dict, predicted_label: int, params: dict) -> Explanation:
        """
        Preprocess search output and return an explanation object containing metdata

        Parameters
        ----------
        X:
            Instance to be explained.
        result:
            Dictionary with explanation search output and metadata.
        predicted_label:
            Label of the instance to be explained (inferred if not given).
        params
            Parameters passed to `explain`

        Return
        ------
             Dictionary containing human readable explanation, metadata, and precision/coverage info.
        """

        self.add_names_to_exp(result)
        result['prediction'] = predicted_label
        result['instance'] = X
        exp = AnchorExplanation('tabular', result)

        # output explanation dictionary
        data = copy.deepcopy(DEFAULT_DATA_ANCHOR)
        data.update(
            anchor=exp.names(),
            precision=exp.precision(),
            coverage=exp.coverage(),
            raw=exp.exp_map
        )

        # create explanation object
        explanation = Explanation(meta=copy.deepcopy(self.meta), data=data)

        # params passed to explain
        explanation.meta['params'].update(params)
        return explanation

    def add_names_to_exp(self, explanation: dict) -> None:
        """
        Add feature names to explanation dictionary.

        Parameters
        ----------
        explanation
            Dict with anchors and additional metadata.
        """

        anchor_idxs = explanation['feature']
        explanation['names'] = []
        explanation['feature'] = [self.enc2feat_idx[idx] for idx in anchor_idxs]
        ordinal_ranges = {self.enc2feat_idx[idx]: [float('-inf'), float('inf')] for idx in anchor_idxs}
        for idx in set(anchor_idxs) - self.cat_lookup.keys():
            feat_id = self.enc2feat_idx[idx]  # feature col. id
            if 0 in self.ord_lookup[idx]:  # tells if the feature in X falls in a higher or lower bin
                ordinal_ranges[feat_id][1] = min(
                    ordinal_ranges[feat_id][1], max(list(self.ord_lookup[idx]))
                )
            else:
                ordinal_ranges[feat_id][0] = max(
                    ordinal_ranges[feat_id][0], min(list(self.ord_lookup[idx])) - 1
                )

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
    if RAY_INSTALLED:
        import ray
        ray = ray  # set module as class variable to used only in this context

    def __init__(self, predictor: Callable, feature_names: list, categorical_names: dict = None,
                 seed: int = None) -> None:

        super().__init__(predictor, feature_names, categorical_names, seed)
        if not DistributedAnchorTabular.ray.is_initialized():
            DistributedAnchorTabular.ray.init()

    def fit(self, train_data: np.ndarray, disc_perc: tuple = (25, 50, 75), **kwargs) -> "AnchorTabular":  # type: ignore
        """
        Creates a list of handles to parallel processes handles that are used for submitting sampling
        tasks.

        Parameters
        ----------
            See superclass implementation.
        """

        try:
            ncpu = kwargs['ncpu']
        except KeyError:
            logging.warning('DistributedAnchorTabular object has been initalised but kwargs did not contain '
                            'expected argument, ncpu. Defaulting to ncpu=2!')
            ncpu = 2

        disc = Discretizer(train_data, self.numerical_features, self.feature_names, percentiles=disc_perc)
        d_train_data = disc.discretize(train_data)

        self.feature_values.update(disc.feature_intervals)

        sampler_args = (
            self.predictor,
            disc_perc,
            self.numerical_features,
            self.categorical_features,
            self.feature_names,
            self.feature_values,
        )
        train_data_id = DistributedAnchorTabular.ray.put(train_data)
        d_train_data_id = DistributedAnchorTabular.ray.put(d_train_data)
        samplers = [TabularSampler(*sampler_args, seed=self.seed) for _ in range(ncpu)]  # type: ignore
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
        See superclass documentation.

        Parameters
        ----------
        X:
            See superclass documentation.
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
                max_anchor_size: int = None,
                min_samples_start: int = 1,
                n_covered_ex: int = 10,
                binary_cache_size: int = 10000,
                cache_margin: int = 1000,
                verbose: bool = False,
                verbose_every: int = 1,
                **kwargs: Any) -> Explanation:
        """
        Explains the prediction made by a classifier on instance X. Sampling is done in parallel over a number of
        cores specified in kwargs['ncpu'].

        Parameters
        ----------
            See superclass implementation.

        Returns
        -------
            See superclass implementation.
        """
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

        return self.build_explanation(X, result, self.instance_label, params)
