from .anchor_base import AnchorBaseBeam
from .anchor_explanation import AnchorExplanation
from alibi.utils.discretizer import Discretizer
from collections import OrderedDict, defaultdict
import bisect
import itertools
import numpy as np
import random
from typing import Callable, Tuple, Any, Set


class AnchorTabular(object):

    def __init__(self, predict_fn: Callable, feature_names: list, categorical_names: dict = None,
                 seed: int = None) -> None:
        """
        Initialize the anchor tabular explainer.

        Parameters
        ----------
        predict_fn
            Model prediction function
        feature_names
            List with feature names
        categorical_names
            Dictionary where keys are feature columns and values are the categories for the feature
        seed
            Used to set the random number generator for repeatability purposes
        """

        random.seed(seed)
        np.random.seed(seed)

        # check if predict_fn returns predicted class or prediction probabilities for each class
        # if needed adjust predict_fn so it returns the predicted class
        if np.argmax(predict_fn(np.zeros([1, len(feature_names)])).shape) == 0:
            self.predict_fn = predict_fn
        else:
            self.predict_fn = lambda x: np.argmax(predict_fn(x), axis=1)

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
            self.feature_names = {}

        self.val2idx = {}
        self.cat_lookup = {}
        self.ord_lookup = {}
        self.enc2feat_idx = {}

    def fit(self, train_data: np.ndarray, disc_perc: tuple = (25, 50, 75)) -> None:
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
        self.train_data = train_data

        # discretization of ordinal features
        self.disc = Discretizer(self.train_data, self.numerical_features, self.feature_names, percentiles=disc_perc)
        self.d_train_data = self.disc.discretize(self.train_data)
        self.feature_values.update(self.disc.feature_intervals)

        self.min, self.max = np.full(train_data.shape[1], np.nan), np.full(train_data.shape[1], np.nan)
        self.min[self.numerical_features] = np.min(train_data[:, self.numerical_features], axis=0)
        self.max[self.numerical_features] = np.max(train_data[:, self.numerical_features], axis=0)

        # key (int): feat. col ID. value is a dict where each int represents a bin value or value of categorical
        # variable. Each value in this dict is a set of training data rows where that value is found
        val2idx = {col_id: defaultdict(lambda: None) for col_id in self.numerical_features + self.categorical_features}
        for feat in val2idx:
            for value in range(len(self.feature_values[feat])):
                val2idx[feat][value] = set((self.d_train_data[:, feat] == value).nonzero()[0].tolist())
        self.val2idx = val2idx

    def build_sampling_lookups(self, X: np.ndarray) -> None:
        """ An encoding of the feature IDs is created by assigning each bin of a discretized numerical variable and each
         categorical variable a unique index. For a dataset containg, e.g., a numerical variable with 5 bins and
         3 categorical variables, indices 0 - 4 represent bins of the numerical variable whereas indices 5, 6, 7
         represent the encoded indices of the categorical variables (but see note for caviats). The encoding is
         necessary so that the different ranges of the numerical variable can be sampled during anchor construction.
        These encoded feature IDs are keys of:
            - cat_lookup: maps categorical variables to their value in X (instance to be explained)
            - ord_lookup: maps discretized numerical variables to the bins they can be sampled from given X
            - enc2feat_idx: maps the encoded IDs to the original feature IDs

        Note: Each continuous variable has n_bins - 1 corresponding entries in ord_lookup.

        Parameters
        ---------
        X
            instance to be explained

        """

        X = self.disc.discretize(X.reshape(1, -1))[0]  # map continuous features to ordinal discrete variables

        if not self.numerical_features:  # data contains only categorical variables
            self.cat_lookup = dict(zip(self.categorical_features, X))
            self.enc2feat_idx = dict(zip(*[self.categorical_features]*2))
            return

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

    def sample_from_train(self, anchor: list, val2idx: dict, ord_lookup: dict, cat_lookup: dict, enc2feat_idx: dict,
                          num_samples: int) -> (np.ndarray, np.ndarray):
        """
        Sample data from training set but keep features which are anchor in the proposed anchor the same
        as the feature value or bin (for ordinal features) as the instance to be explained.s

        Parameters
        ----------
        anchor:
            Each int is an encoded feature id
        val2idx
            Mapping with keys feature column id (int). The values are dict with key (int) representing a bin number
            (discretized variables) or value (categorical variables) (0-indexed) and values a list of ints representing
            the row numbers in the training set where a record has the value indicated by the key.
        ord_lookup:
            Mapping of feature encoded ids to the bins from which data should be sampled (see build_sampling_lookups
            for details)
        cat_lookup
            Mapping of feature encoded ids to the values the corresponding feature in the instance to be explained takes
            (see build_sampling_lookups for details)
        enc2feat_idx
            Mapping between encoded feature IDs and feature IDs in the dataset
        num_samples
            Number of samples used when sampling from training set

        Returns
        -------

        samples
            Sampled data from training set
        d_samples
            Like samples, but continuous data is converted to oridinal discrete data (binned)
        """

        train = self.train_data
        d_train = self.d_train_data

        # Initialise samples randomly
        init_sample_idx = np.random.choice(range(train.shape[0]), num_samples, replace=True)
        samples = train[init_sample_idx]
        d_samples = d_train[init_sample_idx]
        if not anchor:
            return samples, d_samples

        allowed_bins = {}  # bins one can sample from for each numerical feature (key: feat id)
        allowed_rows = {}  # index of database rows (values) for each feature in anchor (key: feat id)
        rand_sampled_feats = []  # feats for which there are not training records in the desired bin/with that value
        cat_enc_ids = [enc_id for enc_id in anchor if enc_id in cat_lookup.keys()]
        ord_enc_ids = [enc_id for enc_id in anchor if enc_id in ord_lookup.keys()]

        if cat_enc_ids:
            cat_feat_vals = [cat_lookup[idx] for idx in cat_enc_ids]
            cat_feat_ids = [enc2feat_idx[idx] for idx in cat_enc_ids]
            allowed_rows = {f_id: val2idx[f_id][f_val] for f_id, f_val in zip(cat_feat_ids, cat_feat_vals)}
            for feat_id, enc_id, val in zip(cat_feat_ids, cat_enc_ids, cat_feat_vals):
                if not allowed_rows[feat_id]:
                    rand_sampled_feats.append((feat_id, 'c', val))
                    cat_feat_ids.remove(feat_id)

        ord_feat_ids = [enc2feat_idx[idx] for idx in ord_enc_ids]
        # determine bins from which ordinal data should be drawn
        for feat_id, enc_id in zip(ord_feat_ids, ord_enc_ids):
            # if encoded indices ref to the same feat, intersect the allowed bins to determine which bins to sample from
            if feat_id not in allowed_bins:
                allowed_bins[feat_id] = ord_lookup[enc_id]
            else:
                allowed_bins[feat_id] = allowed_bins[feat_id].intersection(ord_lookup[enc_id])

        # dict where keys are feature col. ids and values are lists containing row indices in train data which contain
        # data coming from the same bin (or range of bins)
        for feat_id in allowed_bins:  # NB: should scale since we don't query the whole DB every time!
            allowed_rows[feat_id] = set(itertools.chain(*[val2idx[feat_id][bin_id] for bin_id in allowed_bins[feat_id]]))
            if not allowed_rows[feat_id]:  # no instances in training data are in the specified bins ...
                rand_sampled_feats.append((feat_id, 'o', None))

        uniq_feat_ids = list(OrderedDict.fromkeys([enc2feat_idx[enc_idx] for enc_idx in anchor]))
        uniq_feat_ids = [feat for feat in uniq_feat_ids if feat not in [f for f, _, _ in rand_sampled_feats]]
        # for each partial anchor count number of samples available and find their indices
        partial_anchor_rows = [allowed_rows[uniq_feat_ids[0]]]
        n_partial_anchors = [len(partial_anchor_rows[-1])]

        for feature in uniq_feat_ids[1:]:
            partial_anchor_rows.append(partial_anchor_rows[-1].intersection(allowed_rows[feature]))
            n_partial_anchors.append(len(partial_anchor_rows[-1]))

        n_partial_anchors = list(reversed(n_partial_anchors))

        # search num_samples in the list containing the number of training records containing each sub-anchor
        num_samples_pos = bisect.bisect_left(n_partial_anchors, num_samples)
        if num_samples_pos == 0:  # training set has more than num_samples records containing the anchor
            samples_idxs = random.sample(partial_anchor_rows[-1], num_samples)
            samples[:, uniq_feat_ids] = train[np.ix_(samples_idxs, uniq_feat_ids)]
            d_samples[:, uniq_feat_ids] = d_train[np.ix_(samples_idxs, uniq_feat_ids)]
            return samples, d_samples

        # search partial anchors in the training set and replace the remainder of the features
        start, n_anchor_feats = 0, len(partial_anchor_rows)
        uniq_feat_ids = list(reversed(uniq_feat_ids))
        start_idx = np.nonzero(n_partial_anchors)[0][0]  # skip anchors with no samples in the database
        end_idx = np.searchsorted(np.cumsum(n_partial_anchors), num_samples)
        for idx, n_samp in enumerate(n_partial_anchors[start_idx:end_idx + 1], start=start_idx):
            if num_samples >= n_samp:
                samp_idxs = list(partial_anchor_rows[n_anchor_feats - idx - 1])
                num_samples -= n_samp
            else:
                if num_samples <= len(list(partial_anchor_rows[n_anchor_feats - idx - 1])):
                    samp_idxs = random.sample(partial_anchor_rows[n_anchor_feats - idx - 1], k=num_samples)
                else:
                    samp_idxs = random.choices(list(partial_anchor_rows[n_anchor_feats - idx - 1]), k=num_samples)
                n_samp = num_samples
            samples[start:start + n_samp, uniq_feat_ids[idx:]] = train[np.ix_(samp_idxs, uniq_feat_ids[idx:])]
            feats_to_replace = uniq_feat_ids[:idx]
            to_replace = [random.choices(list(allowed_rows[feat]), k=n_samp) for feat in feats_to_replace]
            samples[start: start + n_samp, feats_to_replace] = np.array(to_replace).transpose()
            start += n_samp

        if rand_sampled_feats:
            for feat, var_type, val in rand_sampled_feats:
                if var_type == 'c':
                    fmt = "WARNING: No data records have {} feature with value {}. Setting all samples' values to {}!"
                    print(fmt.format(feat, val, val))
                    samples[:, feat] = val
                else:
                    fmt = "WARNING: For features {}, no training data record had discretized values in bins {}." \
                          " Sampling uniformly at random from the feature range!"
                    print(fmt.format(rand_sampled_feats, [allowed_bins[f] for f in rand_sampled_feats]))
                    min_vals, max_vals = self.min[feat], self.max[feat]
                    samples[:, feat] = np.random.uniform(low=min_vals,
                                                         high=max_vals,
                                                         size=(num_samples, ))

        return samples, self.disc.discretize(samples)

    def sampler(self, anchor: list, num_samples: int, compute_labels: bool = True) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create sampling function from training data.

        Parameters
        ----------
        anchor
            Ints representing encoded feature ids
        num_samples
            Number of samples used when sampling from training set
        compute_labels
            Boolean whether to use labels coming from model predictions as 'true' labels

        Returns
        -------
        raw_data
            Sampled data from training set
        data
            Sampled data where ordinal features are binned (1 if in bin, 0 otherwise)
        labels
            Create labels using model predictions if compute_labels equals True
        """

        raw_data, d_raw_data = self.sample_from_train(anchor, self.val2idx, self.ord_lookup, self.cat_lookup,
                                                      self.enc2feat_idx, num_samples)

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

        # create labels using model predictions as true labels
        labels = np.array([])
        if compute_labels:
            labels = (self.predict_fn(raw_data) == self.instance_label).astype(int)

        return raw_data, data, labels

    def explain(self, X: np.ndarray, threshold: float = 0.95, delta: float = 0.1,
                tau: float = 0.15, batch_size: int = 100, max_anchor_size: int = None,
                desired_label: int = None, **kwargs: Any) -> dict:
        """
        Explain instance and return anchor with metadata.

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
        max_anchor_size
            Maximum number of features in anchor
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
            self.instance_label = self.predict_fn(X.reshape(1, -1))[0]

        # build feature encoding and mappings from the instance values to database rows where similar records are found
        self.build_sampling_lookups(X)

        # get anchors and add metadata
        anchor = AnchorBaseBeam.anchor_beam(self.sampler, delta=delta, epsilon=tau,
                                            batch_size=batch_size, desired_confidence=threshold,
                                            max_anchor_size=max_anchor_size, **kwargs)  # type: Any
        self.add_names_to_exp(anchor)
        if true_label is None:
            anchor['prediction'] = self.instance_label
        else:
            anchor['prediction'] = self.predict_fn(X.reshape(1, -1))[0]
        anchor['instance'] = X
        exp = AnchorExplanation('tabular', anchor)

        return {'names': exp.names(),
                'precision': exp.precision(),
                'coverage': exp.coverage(),
                'raw': exp.exp_map,
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
