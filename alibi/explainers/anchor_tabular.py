from .anchor_base import AnchorBaseBeam
from .anchor_explanation import AnchorExplanation
from alibi.utils.discretizer import Discretizer
from collections import OrderedDict
import bisect
import itertools
import numpy as np
import random
from typing import Callable, Tuple, Dict, Any, Set

# TODO: Fix random seed

class AnchorTabular(object):

    def __init__(self, predict_fn: Callable, feature_names: list, categorical_names: dict = {}) -> None:
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
        self.categorical_features = sorted(categorical_names.keys())
        self.numerical_features = [x for x in range(len(feature_names)) if x not in self.categorical_features]

        self.feature_names = feature_names
        self.categorical_names = categorical_names.copy()  # dict with {col: categorical feature options}

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
        # TODO: This is not scalable? What if the dataset does not fit in memory?
        self.train_data = train_data

        # discretization of ordinal features
        # TODO: Change discretizer so that it takes self.numerical features as input
        self.disc = Discretizer(self.train_data, self.categorical_features, self.feature_names, percentiles=disc_perc)
        self.d_train_data = self.disc.discretize(self.train_data)
        self.categorical_names.update(self.disc.names)

        # calculate min, max and std for numerical features in training data
        self.min = {}  # type: Dict[int, float]
        self.max = {}  # type: Dict[int, float]
        self.std = {}  # type: Dict[int, float]

        min = np.min(train_data[self.numerical_features], axis=0)
        max = np.max(train_data[self.numerical_features], axis=0)
        std = np.std(train_data[self.numerical_features], axis=0)

        for idx in range(len(min)):
            self.min[self.numerical_features[idx]] = min[idx]
            self.max[self.numerical_features[idx]] = max[idx]
            self.std[self.numerical_features[idx]] = std[idx]

        # key (int): feat. col ID for numerical feat., value (dict) with key(int) bin idx , value: list where each elem
        # is a row idx in the training data where a data record with feature in that bin can be found
        self.ord2idx = {feat_col_id: {} for feat_col_id in self.numerical_features}
        ord_feats = self.d_train_data[self.numerical_features]  # nb: ordinal features are just discretised cont. feats.
        for i in range(ord_feats.shape[1]):
            for bin_id in range(len(self.disc.names[self.numerical_features[i]])):
                self.ord2idx[self.numerical_features[i]][bin_id] = set((ord_feats[:, i] == bin_id).nonzero()[0].tolist())

    def build_sampling_lookups(self, X: np.ndarray) -> (dict, dict, dict):
        """ An encoding of the feature IDs is created by assigning each bin of a discretized numerical variable a unique
        index. For a dataset containg, e.g., a numerical variable with 5 bins and 3 categorical variables, indices 0 - 4
        represent bins of the numerical variable whereas indices 5, 6, 7 represent the encoded indices of the categ.
        variables. The encoding is necessary so that the different ranges of the categorical variable can be sampled
        during anchor construction. These encoded feature IDs are keys of:
            - a dictionary mapping categorical variables to their value in X (instance to be explained)
            - a dictionary mapping discretized numerical variables to the bins they can be sampled from given X
            - a dictionary mapping the encoded IDs to the original feature IDs

        NB: TODO: Document the weird handling of X[f] <= bin_val && bin_val == n_bins - 1

        Parameters
        ---------
        X
            instance to be explained

        Returns
        -------
        cat_lookup
            key: encoded feature idx (int) and value (int) of the categorical variable
        ord_lookup
            key: encoded feature idx (int) and value (set) containing the set of bins that samples should be drawn from
            if a sampling request contains an encoded feature value amongst the keys in ord_lookup
        enc2feat_idx
            keys (int) are the union of keys in cat_lookup and ord_lookup, values are the feature column ids in the
            training dataset (e.g., {0: {0}, 1: {0}, 2: {0}} says that encoded features 0, 1 & 2 are indices of the bins
            for a numerical feature in column 0 of the training set)
            # TODO: Write test to verify this union condition
        """

        cat_lookup = {}
        ord_lookup = {}
        enc2feat_idx = {}

        if not self.numerical_features:  # data contains only categorical variables
            cat_lookup = dict(zip(self.categorical_features, X))
            enc2feat_idx = dict(zip(*[self.categorical_features]*2))
            return cat_lookup, ord_lookup, enc2feat_idx

        first_numerical_idx = np.searchsorted(self.categorical_features, self.numerical_features[0]).item()
        if first_numerical_idx > 0:  # First column(s) might contain categorical data
            for cat_enc_idx in range(0, first_numerical_idx):
                cat_lookup[cat_enc_idx] = X[cat_enc_idx]
                enc2feat_idx[cat_enc_idx] = cat_enc_idx

        ord_enc_idx = first_numerical_idx - 1  # -1 as increment comes first
        for i, feature in enumerate(self.numerical_features):
            n_bins = len(self.categorical_names[feature])  # TODO: We should keep the two separate - this is confusing
            for bin_val in range(n_bins):
                ord_enc_idx += 1
                enc2feat_idx[ord_enc_idx] = feature
                # if feat. value falls in same or lower bin, sample from same or lower bin only ...
                if X[feature] <= bin_val != n_bins - 1:
                    ord_lookup[ord_enc_idx] = set(i for i in range(bin_val + 1))
                # if feat. value falls in a higher bin, sample from higher bins only
                elif X[feature] > bin_val:
                    ord_lookup[ord_enc_idx] = set(i for i in range(bin_val + 1, n_bins + 1))   # TODO: Should it be n_bins + 1 or n_bins ?
                else:
                    del enc2feat_idx[ord_enc_idx]
                    ord_enc_idx -= 1  # when a discretized feat. of the instance to be explained falls in the last bin

            # check if a categorical feature follows the current numerical feature & update mappings
            if i < len(self.numerical_features) - 1:
                n_categoricals = self.numerical_features[i + 1] - self.numerical_features[i] - 1
                if n_categoricals > 0:
                    cat_feat_idx = feature + 1
                    for cat_enc_idx in range(ord_enc_idx + 1, ord_enc_idx + 1 + n_categoricals):
                        cat_lookup[cat_enc_idx] = X[cat_feat_idx]
                        enc2feat_idx[cat_enc_idx] = cat_feat_idx
                        cat_feat_idx += 1
                    ord_enc_idx += n_categoricals + 1

        # check if the last columns are categorical variables and update mappings
        last_num_idx = np.searchsorted(self.categorical_features, self.numerical_features[-1]).item()
        if last_num_idx != len(self.categorical_features):
            cat_enc_idx = max(ord_lookup.keys()) + 1
            for cat_feat_idx in range(self.numerical_features[-1] + 1, self.categorical_features[-1] + 1):
                cat_lookup[cat_enc_idx] = X[cat_feat_idx]
                enc2feat_idx[cat_enc_idx] = cat_feat_idx
                cat_enc_idx += 1

        return cat_lookup, ord_lookup, enc2feat_idx

    def sample_from_train(self, anchor: list, ord2idx: dict, ord_lookup: dict, cat_lookup: dict, enc2feat_idx: dict,
                          num_samples: int) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        Sample data from training set but keep features which are present in the proposed anchor the same
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

        # TODO: What happens if the empty anchor is sent in ... Deal with that case here or outside?

        train = self.train_data         # TODO: For parallel algorithm this will need to change - we required data to live in shared memory? otherwise we need to pickle everything which will slow things down?
        d_train = self.d_train_data

        # Set categorical variables to the anchor values
        samples = np.zeros(num_samples, train.shape[1])
        cat_enc_ids = list(set(anchor).intersection(cat_lookup.keys()))
        cat_feat_vals = [cat_lookup[cat_enc_id] for cat_enc_id in cat_enc_ids]
        cat_feat_ids = [enc2feat_idx[idx] for idx in cat_enc_ids]
        samples[:, cat_feat_ids] = np.stack([cat_feat_vals] * num_samples)

        ord_enc_ids = list(set(anchor).intersection(ord_lookup.keys()))
        # TODO: These dicts need to be ordered? Check if so.
        # TODO: Handle the case where we literally need to fill the remainder of the rows with random crap because
        #  the request does not have any numerical variables ...
        allowed_bins = {}  # bins one can sample from for each numerical feature (key: feat id)
        allowed_rows = {}  # rows where each numerical feature requested can be found in (key: feat id)
        ord_feat_ids = [enc2feat_idx[id] for id in ord_enc_ids]
        idxs = set()
        rand_sampled_feats = [] # If there are no rows in the database with a specified feat, sample unif at random


        # determine bins from which ordinal data should be drawn
        for i in range(len(ord_feat_ids)):  # TODO: Ensure that the encoded feature indices coming from AnchorBaseBeam are sorted!
            # if encoded indices ref to the same feat, intersect the allowed bins to determine which bins to sample from
            if ord_feat_ids[i] not in allowed_bins:
                allowed_bins[ord_feat_ids[i]] = ord_lookup[ord_enc_ids[i]]
            else:
                allowed_bins[ord_feat_ids[i]] = allowed_bins[ord_feat_ids[i]].intersect(ord_lookup[ord_enc_ids[i]])

        # dict where keys are feature col. ids and values are lists containing row indices in train data which contain
        # data coming from the same bin (or range of bins)
        # TODO: This is a saving as it allows to immediately check which data records we can draw from to replace
        #  records that do not come from correct bin(s). In calculating options one queries the entire database currently
        for feature in allowed_bins:
            allowed_rows[feature] = set(itertools.chain(*[ord2idx[feature][bin_idx] for bin_idx in allowed_bins[feature]]))
            if not allowed_rows[feature]:  # no instances in training data are in the specified bins ...
                rand_sampled_feats.append(feature)

        if rand_sampled_feats: # draw uniformly at random from the feature range if no training data falls in those bins
            min_vals = [self.min[feature] for feature in rand_sampled_feats]
            max_vals = [self.max[feature] for feature in rand_sampled_feats]
            samples[:, rand_sampled_feats] = np.random.uniform(low=min_vals,
                                                               high=max_vals,
                                                               size=(num_samples, len(rand_sampled_feats))).squeeze()

        ord_feat_ids = [feat for feat in ord_feat_ids if feat not in rand_sampled_feats]

        if not ord_feat_ids:
            return samples

        ord_feat_ids_uniq = list(OrderedDict.fromkeys(ord_feat_ids))
        # NB: 'partial' anchor is at feature level not encoded feature level
        n_partial_anchors, partial_anchor_rows = [len(allowed_rows[ord_feat_ids[0]])], [allowed_rows[ord_feat_ids[0]]]
        # TODO: Make sure this is correct ...
        for feature in ord_feat_ids_uniq[1:]:
            partial_anchor_rows.append(partial_anchor_rows[-1].intersection(allowed_rows[feature]))
            n_partial_anchors.append(len(partial_anchor_rows[-1]))

        n_partial_anchors = list(reversed(n_partial_anchors))

        # search num_samples in the list containing the number of training records containing each sub-anchor
        num_samples_pos = bisect.bisect_left(n_partial_anchors, num_samples)
        if num_samples_pos == 0: # training set has more than num_samples records containing the anchor
            samples_idxs = random.sample(partial_anchor_rows[-1], num_samples)
            samples[:, ord_feat_ids_uniq] = train[samples_idxs, ord_feat_ids_uniq]

            return samples

        # find maximal length sub-anchor that allows one to draw num_samples
        sub_anchor_max_len_pos = len(n_partial_anchors) - num_samples_pos
        # draw n_samples containing the maximal length sub-anchor
        sample_idxs = random.sample(set.intersection(*partial_anchor_rows[:sub_anchor_max_len_pos]))
        # the remainder variables get replaced with random draws from the







        anchor_matches_ids = set.intersection(*allowed_rows.values()) # rows in training set where anchor was found
        if len(anchor_matches_ids) >= num_samples:
            sample_idxs = random.sample(anchor_matches_ids, num_samples)
            samples = train[sample_idxs, :]


            return samples

        remaining_samples = num_samples - len(anchor_matches_ids)





        if not idxs:  # Anchor has zero coverage, no point extending that path
            return np.array([]), np.array([]), np.array([])
        n_sampled = len(idxs)

        if n_sampled >= num_samples:
            sample_idxs = random.sample(idxs, num_samples)
            samples = train[sample_idxs, :]
        else:  # Not enough samples contain the anchor
            samples[:n_sampled, :] = train[idxs, :]
            num_remaining = num_samples - n_sampled
            for index in reversed(range(len(intermediate_idxs) - 1)):
                additional_samples = intermediate_idxs[index] - intermediate_idxs[index+1]
                n_available_samples = len(additional_samples)
                if n_available_samples >= num_remaining:
                    # Sample records which contain a partial anchor
                    sampled_idx = random.sample(intermediate_idxs[index], num_remaining)
                    samples[n_sampled:, :] = train[sampled_idx, :]
                    # Replace the remaining features with values sampled from the appropriate bins
                    to_replace = intermediate_idxs

                    samples[n_sampled:, ord_feat_ids_uniq[index + 1:]] =
                    # TODO: Set feature to the anchor value!!!















        # sample
        # TODO: Deal with numerical variables
        # TODO: Deal with categorical variables
        cat_enc_ids = list(set(anchor).intersection(cat_lookup.keys()))
        cat_feat_ids = [enc2feat_idx[idx] for idx in cat_enc_ids]

        # TODO: Strategy for sampling when there are no records
        # TODO: Ensure equivalence with previous algorithm behaviour

        return sample

    def get_sample_fn(self, X: np.ndarray, desired_label: int = None) -> Tuple[Callable, dict]:
        """
        Create sampling function and mapping dictionary between categorized data and the feature types and values.

        Parameters
        ----------
        X
            Instance to be explained
        desired_label
            Label to use as true label for the instance to be explained

        Returns
        -------
        sample_fn
            Function returning raw and categorized sampled data, and labels
        mapping
            Dict: key = feature column or bin for ordinal features in categorized data; value = tuple containing
                  (feature column, flag for categorical/ordinal feature, feature value or bin value)
        """
        # if no true label available; true label = predicted label
        true_label = desired_label
        if true_label is None:
            true_label = self.predict_fn(X.reshape(1, -1))[0]

        def sample_fn(present: list, num_samples: int, compute_labels: bool = True) \
                -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            """
            Create sampling function from training data.

            Parameters
            ----------
            present
                List with keys from mapping
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
            # initialize dicts for 'eq', 'leq', 'geq' tuple value from previous mapping
            # key = feature column; value = feature or bin (for ordinal features) value
            conditions_eq = {}  # type: Dict[int, float]
            conditions_leq = {}  # type: Dict[int, float]
            conditions_geq = {}  # type: Dict[int, float]
            for x in present:
                f, op, v = mapping[x]  # (feature, 'eq'/'leq'/'geq', feature value)
                if op == 'eq':  # categorical feature
                    conditions_eq[f] = v
                if op == 'leq':  # ordinal feature
                    if f not in conditions_leq:
                        conditions_leq[f] = v
                    conditions_leq[f] = min(conditions_leq[f], v)  # store smallest bin > feature value
                if op == 'geq':  # ordinal feature
                    if f not in conditions_geq:
                        conditions_geq[f] = v
                    conditions_geq[f] = max(conditions_geq[f], v)  # store largest bin < feature value

            # sample data from training set
            # feature values are from same discretized bin or category as the explained instance ...
            # ... if defined in conditions dicts
            raw_data = self.sample_from_train(conditions_eq, {}, conditions_geq, conditions_leq, num_samples)

            # discretize sampled data
            d_raw_data = self.disc.discretize(raw_data)

            # use the sampled, discretized raw data to construct a data matrix with the categorical ...
            # ... and binned ordinal data (1 if in bin, 0 otherwise)
            data = np.zeros((num_samples, len(mapping)), int)
            for i in mapping:
                f, op, v = mapping[i]
                if op == 'eq':
                    data[:, i] = (d_raw_data[:, f] == X[f]).astype(int)
                if op == 'leq':
                    data[:, i] = (d_raw_data[:, f] <= v).astype(int)
                if op == 'geq':
                    data[:, i] = (d_raw_data[:, f] > v).astype(int)

            # create labels using model predictions as true labels
            labels = np.array([])
            if compute_labels:
                labels = (self.predict_fn(raw_data) == true_label).astype(int)
            return raw_data, data, labels

        return sample_fn

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

        X = self.disc.discretize(X.reshape(1, -1))[0]  # map continuous features to ordinal discrete variables
        self.build_sampling_lookups(X)

        # build sampling function and

        sample_fn = self.get_sample_fn(X, desired_label=desired_label)


        # get anchors and add metadata
        exp = AnchorBaseBeam.anchor_beam(sample_fn, delta=delta, epsilon=tau,
                                         batch_size=batch_size, desired_confidence=threshold,
                                         max_anchor_size=max_anchor_size, **kwargs)  # type: Any
        self.add_names_to_exp(X, exp)
        exp['instance'] = X
        exp['prediction'] = self.predict_fn(X.reshape(1, -1))[0]
        exp = AnchorExplanation('tabular', exp)

        # output explanation dictionary
        explanation = {}
        explanation['names'] = exp.names()
        explanation['precision'] = exp.precision()
        explanation['coverage'] = exp.coverage()
        explanation['raw'] = exp.exp_map
        return explanation

    def add_names_to_exp(self, X, explanation: dict) -> None:
        """
        Add feature names to explanation dictionary.

        Parameters
        ----------
        explanation
            Dict with anchors and additional metadata
        """

        mapping = self.create_mapping(X)

        idxs = explanation['feature']
        explanation['names'] = []
        explanation['feature'] = [mapping[idx][0] for idx in idxs]

        ordinal_ranges = {}  # type: Dict[int, list]
        for idx in idxs:
            f, op, v = mapping[idx]
            if op == 'geq' or op == 'leq':
                if f not in ordinal_ranges:
                    ordinal_ranges[f] = [float('-inf'), float('inf')]
            if op == 'geq':
                ordinal_ranges[f][0] = max(ordinal_ranges[f][0], v)
            if op == 'leq':
                ordinal_ranges[f][1] = min(ordinal_ranges[f][1], v)

        handled = set()  # type: Set[int]
        for idx in idxs:
            f, op, v = mapping[idx]
            if op == 'eq':
                fname = '%s = ' % self.feature_names[f]
                if f in self.categorical_names:
                    v = int(v)
                    if ('<' in self.categorical_names[f][v]
                            or '>' in self.categorical_names[f][v]):
                        fname = ''
                    fname = '%s%s' % (fname, self.categorical_names[f][v])
                else:
                    fname = '%s%.2f' % (fname, v)
            else:
                if f in handled:
                    continue
                geq, leq = ordinal_ranges[f]
                fname = ''
                geq_val = ''
                leq_val = ''
                if geq > float('-inf'):
                    if geq == len(self.categorical_names[f]) - 1:
                        geq = geq - 1
                    name = self.categorical_names[f][geq + 1]
                    if '<' in name:
                        geq_val = name.split()[0]
                    elif '>' in name:
                        geq_val = name.split()[-1]
                if leq < float('inf'):
                    name = self.categorical_names[f][leq]
                    if leq == 0:
                        leq_val = name.split()[-1]
                    elif '<' in name:
                        leq_val = name.split()[-1]
                if leq_val and geq_val:
                    fname = '%s < %s <= %s' % (geq_val, self.feature_names[f],
                                               leq_val)
                elif leq_val:
                    fname = '%s <= %s' % (self.feature_names[f], leq_val)
                elif geq_val:
                    fname = '%s > %s' % (self.feature_names[f], geq_val)
                handled.add(f)
            explanation['names'].append(fname)
