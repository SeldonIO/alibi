from .anchor_base import AnchorBaseBeam
from .anchor_explanation import AnchorExplanation
from alibi.utils.discretizer import Discretizer
from collections import OrderedDict
import bisect
import itertools
import numpy as np
import random
from typing import Callable, Tuple, Dict, Any, Set

random.seed(23)  # TODO: Do this properly
np.random.seed(23)


class AnchorTabular(object):

    def __init__(self, predict_fn: Callable, feature_names: list, categorical_names: dict = None) -> None:
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
        """
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
        self.feature_values = categorical_names.copy()  # dict with {col: categorical feature values}

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

        # calculate min, max and std for numerical features in training data
        self.min = {}  # type: Dict[int, float]
        self.max = {}  # type: Dict[int, float]
        self.std = {}  # type: Dict[int, float]

        min = np.min(train_data[:, self.numerical_features], axis=0)
        max = np.max(train_data[:, self.numerical_features], axis=0)
        std = np.std(train_data[:, self.numerical_features], axis=0)

        for idx in range(len(min)):
            self.min[self.numerical_features[idx]] = min[idx]
            self.max[self.numerical_features[idx]] = max[idx]
            self.std[self.numerical_features[idx]] = std[idx]

        # key (int): feat. col ID for numerical feat., value (dict) with key(int) bin idx , value: list where each elem
        # is a row idx in the training data where a data record with feature in that bin can be found
        self.ord2idx = {feat_col_id: {} for feat_col_id in self.numerical_features}
        ord_feats = self.d_train_data[:, self.numerical_features]  # nb: ordinal features are just discretised cont. feats.
        for i in range(ord_feats.shape[1]):
            for bin_id in range(len(self.disc.feature_intervals[self.numerical_features[i]])):
                self.ord2idx[self.numerical_features[i]][bin_id] = set((ord_feats[:, i] == bin_id).nonzero()[0].tolist())

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

        Note: Each continuous variable has n_bins - 1 corresponding entries in ord_lookup

        Parameters
        ---------
        X
            instance to be explained

            # TODO: Write test to verify cat and ord keys union results in keys of enc2feat_idx
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

    def sample_from_train(self, anchor: list, ord2idx: dict, ord_lookup: dict, cat_lookup: dict, enc2feat_idx: dict,
                          num_samples: int) -> (np.ndarray, np.ndarray):
        """
        Sample data from training set but keep features which are anchor in the proposed anchor the same
        as the feature value or bin (for ordinal features) as the instance to be explained.s

        # TODO: Improve documentation as the current version is misleading (e.g., we don't sample from the 'same' bin)

        Parameters
        ----------
        anchor:
            Each int is an encoded feature id
        ord2idx
            Mapping with keys feature column id (int). The values are dict with key (int) representing a bin number
            (0-indexed) and values a list of ints representing the row numbers in the training set where a record
            has the value indicated by the key
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

        sample
            Sampled data from training set
        """

        train = self.train_data     # TODO: Only works for data that fits in memory.
        d_train = self.d_train_data

        # Initialise samples randomly
        init_sample_idx = np.random.choice(range(train.shape[0]), num_samples, replace=True)
        samples = train[init_sample_idx]
        d_samples = d_train[init_sample_idx]

        # Set categorical variables to the anchor values
        cat_enc_ids = set(anchor).intersection(cat_lookup.keys())
        if cat_enc_ids:
            cat_feat_vals = [cat_lookup[cat_enc_id] for cat_enc_id in cat_enc_ids]
            cat_feat_ids = [enc2feat_idx[idx] for idx in cat_enc_ids]
            samples[:, cat_feat_ids] = np.stack([cat_feat_vals] * num_samples)
            d_samples[:, cat_feat_ids] = samples[:, cat_feat_ids]

        ord_enc_ids = list(set(anchor).intersection(ord_lookup.keys()))
        if not ord_enc_ids:  # anchor only contains categorical data or is empty
            return samples, d_samples
        allowed_bins = {}  # bins one can sample from for each numerical feature (key: feat id)
        allowed_rows = {}  # rows where each numerical feature requested can be found in (key: feat id)
        ord_feat_ids = [enc2feat_idx[idx] for idx in ord_enc_ids]
        rand_sampled_feats = []  # If there are no rows in the database with a specified feat, sample unif at random

        # determine bins from which ordinal data should be drawn
        for i in range(len(ord_feat_ids)):
            # if encoded indices ref to the same feat, intersect the allowed bins to determine which bins to sample from
            if ord_feat_ids[i] not in allowed_bins:
                allowed_bins[ord_feat_ids[i]] = ord_lookup[ord_enc_ids[i]]
            else:
                allowed_bins[ord_feat_ids[i]] = allowed_bins[ord_feat_ids[i]].intersect(ord_lookup[ord_enc_ids[i]])

        # dict where keys are feature col. ids and values are lists containing row indices in train data which contain
        # data coming from the same bin (or range of bins)
        for feature in allowed_bins:  # TODO: This should scale since we don't query the whole DB every time!
            allowed_rows[feature] = set(itertools.chain(*[ord2idx[feature][bin_idx] for bin_idx in allowed_bins[feature]]))
            if not allowed_rows[feature]:  # no instances in training data are in the specified bins ...
                rand_sampled_feats.append(feature)

        if rand_sampled_feats:  # draw U.A.R. from the feature range if no training data falls in those bins
            min_vals = [self.min[feature] for feature in rand_sampled_feats]
            max_vals = [self.max[feature] for feature in rand_sampled_feats]
            samples[:, rand_sampled_feats] = np.random.uniform(low=min_vals,
                                                               high=max_vals,
                                                               size=(num_samples, len(rand_sampled_feats))).squeeze()

        ord_feat_ids = [feat for feat in ord_feat_ids if feat not in rand_sampled_feats]
        if not ord_feat_ids:  # sampled everything U.A.R, because no data in training set fell in specified bins ...
            return samples, self.disc.discretize(samples)

        ord_feat_ids_uniq = list(OrderedDict.fromkeys(ord_feat_ids))
        # for each partial anchor count number of samples available and find their indices
        partial_anchor_rows = [allowed_rows[ord_feat_ids_uniq[0]]]
        n_partial_anchors = [len(partial_anchor_rows[-1])]
        for feature in ord_feat_ids_uniq[1:]:
            partial_anchor_rows.append(partial_anchor_rows[-1].intersection(allowed_rows[feature]))
            n_partial_anchors.append(len(partial_anchor_rows[-1]))

        n_partial_anchors = list(reversed(n_partial_anchors))
        # search num_samples in the list containing the number of training records containing each sub-anchor
        num_samples_pos = bisect.bisect_left(n_partial_anchors, num_samples)
        if num_samples_pos == 0:  # training set has more than num_samples records containing the anchor
            samples_idxs = random.sample(partial_anchor_rows[-1], num_samples)
            samples[:, ord_feat_ids_uniq] = train[samples_idxs, ord_feat_ids_uniq]
            d_samples[:, ord_feat_ids_uniq] = d_train[samples_idxs, ord_feat_ids_uniq]
            return samples, d_samples

        # find maximal length sub-anchor that allows one to draw num_samples
        sub_anchor_max_len_pos = len(n_partial_anchors) - num_samples_pos
        # draw n_samples containing the maximal length sub-anchor
        sample_idxs = random.sample(set.intersection(*partial_anchor_rows[:sub_anchor_max_len_pos]), num_samples)
        anchored_feats = ord_feat_ids_uniq[:sub_anchor_max_len_pos]
        feats_to_replace = ord_feat_ids_uniq[sub_anchor_max_len_pos:]
        samples[:, anchored_feats] = train[sample_idxs, anchored_feats]
        # the remainder variables get replaced with random draws from the feature distributions
        to_replace = [random.choices(allowed_rows[feature], k=num_samples) for feature in feats_to_replace]
        samples[:, feats_to_replace] = np.array(to_replace).transpose()

        # TODO: Review this and ensure it is correct. Ensure discretisation works as intended
        # TODO: Understand what is serialised during parallelisation and see if using self.disc.discretize increases
        #  the overhead
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

        raw_data, d_raw_data = self.sample_from_train(anchor, self.ord2idx, self.ord_lookup, self.cat_lookup,
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
        exp = AnchorBaseBeam.anchor_beam(self.sampler, delta=delta, epsilon=tau,
                                         batch_size=batch_size, desired_confidence=threshold,
                                         max_anchor_size=max_anchor_size, **kwargs)  # type: Any
        self.add_names_to_exp(X, exp)
        if true_label is None:
            exp['prediction'] = self.instance_label
        else:
            exp['prediction'] = self.predict_fn(X.reshape(1, -1))[0]
        exp['instance'] = X
        exp = AnchorExplanation('tabular', exp)

        # output explanation dictionary
        explanation = {}
        explanation['names'] = exp.names()
        explanation['precision'] = exp.precision()
        explanation['coverage'] = exp.coverage()
        explanation['raw'] = exp.exp_map
        return explanation

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
        explanation['feature'] = [self.enc2feat_idx[idx] for idx in anchor_idxs]  # TODO: this contains duplicates, beware

        ordinal_ranges = {self.enc2feat_idx[idx]: [float('-inf'), float('inf')] for idx in anchor_idxs}
        for idx in anchor_idxs:
            if 0 in self.ord_lookup[idx]:  # tells if the feature falls in a higher or lower bin
                ordinal_ranges[self.enc2feat_idx[idx]][1] = min(ordinal_ranges[self.enc2feat_idx[idx]][1],
                                                                max(list(self.ord_lookup[idx])))
            else:
                ordinal_ranges[self.enc2feat_idx[idx]][0] = max(ordinal_ranges[self.enc2feat_idx[idx]][0],
                                                                min(list(self.ord_lookup[idx])) - 1)

        handled = set()  # type: Set[int]
        for idx in anchor_idxs:
            feat_id = self.enc2feat_idx[idx]
            if idx in self.cat_lookup:
                v = int(self.cat_lookup[feat_id])
                fname = '%s = ' % self.feature_names[feat_id]
                if feat_id in self.feature_values:
                    v = int(v)                    # TODO: Discuss, what's the point of this all: f is always in feature values due to update?
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
