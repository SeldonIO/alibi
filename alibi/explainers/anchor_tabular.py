from .anchor_base import AnchorBaseBeam
from .anchor_explanation import AnchorExplanation
from alibi.utils.discretizer import Discretizer
import numpy as np
from typing import Callable, Tuple, Dict, Any, Set


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

    def create_mapping(self, X: np.ndarray):
        """Maps the continuos features to a tuple containing (feat_col_id, 'eq', feat_value) and the discretised feats.
        to a series of tuples containing (feat_col_id, 'leq'/'geq', value) where value is a number in [0, n_bins]. 'leq'
        and 'geq' indicate if the value in the tuple is <= or >, respectively, than the bin value of the instance
        to be explained

        Parameters
        ----------
        X
            instance to be explained

        Returns
        -------
        mapping
            dict: key = feature column or bin for ordinal features in categorized data; value = tuple containing
                  (feature column, flag for categorical/ordinal feature, feature value or bin value)
        """
        # TODO: See if this is necessary for adding names to anchor or if this can be done in a more simple fashion
        # discretize ordinal features of instance to be explained
        # create mapping = (feature column, flag for categorical/ordinal feature, feature value or bin value)
        mapping = {}  # type: Dict[int,Tuple[int, str, float]]
        X = self.disc.discretize(X.reshape(1, -1))[0]
        for f in self.numerical_features:
            for v in range(len(self.categorical_names[f])):  # loop over nb of bins for the ordinal features
                idx = len(mapping)
                if X[f] <= v and v != len(self.categorical_names[f]) - 1:  # feature value <= bin value
                    mapping[idx] = (f, 'leq', v)  # store bin value
                elif X[f] > v:  # feature value > bin value
                    mapping[idx] = (f, 'geq', v)  # store bin value
                else:
                    idx = len(mapping)
                    mapping[idx] = (f, 'eq', X[f])  # store feature value

        return mapping

    def build_sampling_lookups(self, X: np.ndarray) -> (dict, dict, dict):
        """ An encoding of the features is created by assigning each bin of a discretized numerical variable a unique
        index. For a dataset containg a numerical variable with 5 bins and 3 categorical variables, indices 0 - 4
        represent bins of the numerical variable whereas indices 5, 6, 7 represent the encoded indices of the categ.
        variables. The encoding is necessary so that the different ranges of the categorical variable can be sampled
        during anchor construction. These encoded IDs are keys of:
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

        ord_enc_idx = first_numerical_idx[0] - 1  # -1 as increment comes first
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
    
    def sample_from_train(self, conditions_eq: dict, conditions_neq: dict,
                          conditions_geq: dict, conditions_leq: dict, num_samples: int) -> np.ndarray:
        """
        Sample data from training set but keep features which are present in the proposed anchor the same
        as the feature value or bin (for ordinal features) as the instance to be explained.

        Parameters
        ----------
        conditions_eq
            key = feature column; value = categorical feature value
        conditions_neq
            Not used at the moment
        conditions_geq
            key = feature column; value = bin value of ordinal feature where bin value < feature value
        conditions_leq
            key = feature column; value = bin value of ordinal feature where bin value >= feature value
        num_samples
            Number of samples used when sampling from training set

        Returns
        -------
        sample
            Sampled data from training set
        """
        train = self.train_data
        d_train = self.d_train_data

        # sample from train and d_train data sets with replacement
        idx = np.random.choice(range(train.shape[0]), num_samples, replace=True)
        sample = train[idx]
        d_sample = d_train[idx]

        # for each sampled instance, use the categorical feature values specified in conditions_eq ...
        # ... which is equal to the feature value in the instance to be explained
        for f in conditions_eq:
            sample[:, f] = np.repeat(conditions_eq[f], num_samples)

        # for the features in condition_geq: make sure sampled feature comes from correct ordinal bin
        for f in conditions_geq:

            # idx of samples where feature value is in a lower bin than the observation to be explained
            idx = d_sample[:, f] <= conditions_geq[f]

            # add idx where feature value is in a higher bin than the observation
            if f in conditions_leq:
                idx = (idx + (d_sample[:, f] > conditions_leq[f])).astype(bool)

            if idx.sum() == 0:
                continue  # if all values in sampled data have same bin as instance to be explained

            # options: idx in train set where with feature value in same bin than instance to be explained
            options = d_train[:, f] > conditions_geq[f]
            if f in conditions_leq:
                options = options * (d_train[:, f] <= conditions_leq[f])

            # if no options, uniformly sample between min and max of feature ...
            if options.sum() == 0:
                min_ = conditions_geq.get(f, self.min[f])
                max_ = conditions_leq.get(f, self.max[f])
                to_rep = np.random.uniform(min_, max_, idx.sum())
            else:  # ... otherwise draw random samples from training set
                to_rep = np.random.choice(train[options, f], idx.sum(), replace=True)

            # replace sample values for ordinal features where feature values are in a different bin ...
            # ... than the instance to be explained by random values from training set from the correct bin
            sample[idx, f] = to_rep

        # for the features in condition_leq: make sure sampled feature comes from correct ordinal bin
        for f in conditions_leq:

            if f in conditions_geq:
                continue

            idx = d_sample[:, f] > conditions_leq[f]  # idx where feature value is in a higher bin than the observation

            if idx.sum() == 0:
                continue  # if all values in sampled data have same bin as instance to be explained

            # options: idx in train set where with feature value in same bin than instance to be explained
            options = d_train[:, f] <= conditions_leq[f]

            # if no options, uniformly sample between min and max of feature ...
            if options.sum() == 0:
                min_ = conditions_geq.get(f, self.min[f])
                max_ = conditions_leq.get(f, self.max[f])
                to_rep = np.random.uniform(min_, max_, idx.sum())
            else:  # ... otherwise draw random samples from training set
                to_rep = np.random.choice(train[options, f], idx.sum(), replace=True)
            sample[idx, f] = to_rep

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
