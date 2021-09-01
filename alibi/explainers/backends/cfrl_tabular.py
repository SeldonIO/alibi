"""
This module contains utility functions for the Counterfactual with Reinforcement Learning tabular class,
:py:class:`alibi.explainers.cfrl_tabular`, that are common for both Tensorflow and Pytorch backends.
"""

import numpy as np
import pandas as pd  # type: ignore
from typing import List, Dict, Union, Tuple, Callable, Optional, TYPE_CHECKING

from sklearn.preprocessing import StandardScaler, OneHotEncoder  # type: ignore
from sklearn.compose import ColumnTransformer  # type: ignore
from scipy.special import softmax  # type: ignore

if TYPE_CHECKING:
    import torch
    import tensorflow as tf


def get_conditional_dim(feature_names: List[str], category_map: Dict[int, List[str]]) -> int:
    """
    Computes the dimension of the conditional vector.

    Parameters
    ----------
    feature_names
        List of feature names. This should be provided by the dataset.
    category_map
        Dictionary of category mapping. The keys are column indexes and the values are lists containing the
        possible feature values. This should be provided by the dataset.

    Returns
    -------
        Dimension of the conditional vector
    """
    cat_feat = int(np.sum([len(vals) for vals in category_map.values()]))
    num_feat = len(feature_names) - len(category_map)
    return 2 * num_feat + cat_feat


def split_ohe(X_ohe: 'Union[np.ndarray, torch.Tensor, tf.Tensor]',
              category_map: Dict[int, List[str]]) -> Tuple[List, List]:
    """
    Splits a one-hot encoding array in a list of numerical heads and a list of categorical heads. Since by
    convention the numerical heads are merged in a single head, if the function returns a list of numerical heads,
    then the size of the list is 1.

    Parameters
    ----------
    X_ohe
        One-hot encoding representation. This can be any type of tensor: `np.ndarray`, `torch.Tensor`, `tf.Tensor`.
    category_map
        Dictionary of category mapping. The keys are column indexes and the values are lists containing the
        possible values of a feature.

    Returns
    -------
    X_ohe_num_split
        List of numerical heads. If different than `None`, the list's size is 1.
    X_ohe_cat_split
        List of categorical one-hot encoded heads.
    """
    assert hasattr(X_ohe, "shape"), "X_ohe needs to have `shape` attribute."
    X_ohe_num_split, X_ohe_cat_split = [], []
    offset = 0

    # Compute the number of columns spanned by the categorical one-hot encoded heads, and the number of columns
    # spanned by the numerical heads.
    cat_feat = int(np.sum([len(vals) for vals in category_map.values()]))
    num_feat = X_ohe.shape[1] - cat_feat

    # If the number of numerical features is different than 0, then the first `num_feat` columns correspond
    # to the numerical features
    if num_feat > 0:
        X_ohe_num_split.append(X_ohe[:, :num_feat])
        offset = num_feat

    # If there exist categorical features, then extract them one by one
    if cat_feat > 0:
        for id in sorted(category_map.keys()):
            X_ohe_cat_split.append(X_ohe[:, offset:offset + len(category_map[id])])
            offset += len(category_map[id])

    return X_ohe_num_split, X_ohe_cat_split


def generate_numerical_condition(X_ohe: np.ndarray,
                                 feature_names: List[str],
                                 category_map: Dict[int, List[str]],
                                 ranges: Dict[str, List[float]],
                                 immutable_features: List[str],
                                 conditional: bool = True) -> np.ndarray:
    """
    Generates numerical features conditional vector. For numerical features with a minimum value `a_min` and a
    maximum value `a_max`, we include in the conditional vector the values `-p_min`, `p_max`, where `p_min, p_max`
    are in `[0, 1]`. The range `[-p_min, p_max]` encodes a shift and scale-invariant representation of the interval
    `[a - p_min * (a_max - a_min), a + p_max * (a_max - a_min)], where `a` is the original feature value. During
    training, `p_min` and `p_max` are sampled from `Beta(2, 2)` for each unconstrained feature. Immutable features
    can be encoded by `p_min = p_max = 0` or listed in `immutable_features` list. Features allowed to increase or
    decrease only correspond to setting `p_min = 0` or `p_max = 0`, respectively. For example, allowing the `age`
    feature to increase by up to 5 years is encoded by taking `p_min = 0`, `p_max=0.1`, assuming the minimum age of
    `10` and the maximum age of `60` years in the training set: `5 = 0.1 * (60 - 10)`.

    Parameters
    ----------
    X_ohe
        One-hot encoding representation of the element(s) for which the conditional vector will be generated.
        This argument is used to extract the number of conditional vector. The choice of `X_ohe` instead of a
        `size` argument is for consistency purposes with `categorical_cond` function.
    feature_names
        List of feature names. This should be provided by the dataset.
    category_map
        Dictionary of category mapping. The keys are column indexes and the values are lists containing the
        possible feature values.
    ranges:
        Dictionary of ranges for numerical features. Each value is a list containing two elements, first one
        negative and the second one positive.
    immutable_features
        Dictionary of immutable features. The keys are the column indexes and the values are booleans: `True` if
        the feature is immutable, `False` otherwise.
    conditional
        Boolean flag to generate a conditional vector. If `False` the conditional vector does not impose any
        restrictions on the feature value.

    Returns
    -------
        Conditional vector for numerical features.
    """
    num_cond = []
    size = X_ohe.shape[0]

    for feature_id, feature_name in enumerate(feature_names):
        # skip categorical features
        if feature_id in category_map:
            continue

        if feature_name in immutable_features:
            # immutable feature
            range_low, range_high = 0., 0.
        else:
            range_low = ranges[feature_name][0] if feature_name in ranges else -1.
            range_high = ranges[feature_name][1] if feature_name in ranges else 1.

        # Check if the ranges are valid.
        if range_low > 0:
            raise ValueError(f"Lower bound range for {feature_name} should be negative.")
        if range_high < 0:
            raise ValueError(f"Upper bound range for {feature_name} should be positive.")

        # Generate lower and upper bound coefficients.
        coeff_lower = np.random.beta(a=2, b=2, size=size).reshape(-1, 1) if conditional else np.ones((size, 1))
        coeff_upper = np.random.beta(a=2, b=2, size=size).reshape(-1, 1) if conditional else np.ones((size, 1))

        # Generate lower and upper bound conditionals.
        num_cond.append(coeff_lower * range_low)
        num_cond.append(coeff_upper * range_high)

    # Construct numerical conditional vector by concatenating all numerical conditions.
    return np.concatenate(num_cond, axis=1)


def generate_categorical_condition(X_ohe: np.ndarray,
                                   feature_names: List[str],
                                   category_map: Dict[int, List],
                                   immutable_features: List[str],
                                   conditional: bool = True) -> np.ndarray:
    """
    Generates categorical features conditional vector. For a categorical feature of cardinality `K`, we condition the
    subset of allowed feature through a binary mask of dimension `K`. When training the counterfactual generator,
    the mask values are sampled from `Bern(0.5)`. For immutable features, only the original input feature value is
    set to one in the binary mask. For example, the immutability of the `marital_status` having the current
    value `married` is encoded through the binary sequence `[1, 0, 0]`, given an ordering of the possible feature
    values `[married, unmarried, divorced]`.

    Parameters
    ----------
    X_ohe
        One-hot encoding representation of the element(s) for which the conditional vector will be generated.
        The elements are required since some features can be immutable. In that case, the mask vector is the
        one-hot encoding itself for that particular feature.
    feature_names
        List of feature names. This should be provided by the dataset.
    category_map
        Dictionary of category mapping. The keys are column indexes and the values are lists containing the
        possible feature values.
    immutable_features
        List of immutable features.
    conditional
        Boolean flag to generate a conditional vector. If `False` the conditional vector does not impose any
        restrictions on the feature value.

    Returns
    -------
        Conditional vector for categorical feature.
    """

    C_cat = []   # define list of conditional vector for each feature
    cat_idx = 0  # categorical feature index

    # Split the one-hot representation into a list where each element corresponds to an feature.
    _, X_ohe_cat_split = split_ohe(X_ohe, category_map)

    # Create mask for each categorical column.
    for feature_id, feature_name in enumerate(feature_names):
        # Skip numerical features
        if feature_id not in category_map:
            continue

        # Initialize mask with the original value
        mask = X_ohe_cat_split[cat_idx].copy()

        # If the feature is not immutable, add noise to modify the mask
        if feature_name not in immutable_features:
            mask += np.random.rand(*mask.shape) if conditional else np.ones_like(mask)

        # Construct binary mask
        mask = (mask > 0.5).astype(np.float32)
        C_cat.append(mask)

        # Move to the next categorical index
        cat_idx += 1

    return np.concatenate(C_cat, axis=1)


def generate_condition(X_ohe: np.ndarray,
                       feature_names: List[str],
                       category_map: Dict[int, List[str]],
                       ranges: Dict[str, List[float]],
                       immutable_features: List[str],
                       conditional: bool = True) -> np.ndarray:
    """
    Generates conditional vector.

    Parameters
    ----------
    X_ohe
        One-hot encoding representation of the element(s) for which the conditional vector will be generated.
        This method assumes that the input array, `X_ohe`, is has the first columns corresponding to the
        numerical features, and the rest are one-hot encodings of the categorical columns. The numerical and the
        categorical columns are ordered by the original column index( e.g. numerical = (1, 4),
        categorical=(0, 2, 3)).
    feature_names
        List of feature names.
    category_map
        Dictionary of category mapping. The keys are column indexes and the values are lists containing the
        possible feature values.
    ranges
        Dictionary of ranges for numerical features. Each value is a list containing two elements, first one
        negative and the second one positive.
    immutable_features
        List of immutable map features.
    conditional
        Boolean flag to generate a conditional vector. If `False` the conditional vector does not impose any
        restrictions on the feature value.

    Returns
    -------
        Conditional vector.
    """
    # Define conditional vector buffer
    C = []

    # Generate numerical condition vector.
    if len(feature_names) > len(category_map):
        C_num = generate_numerical_condition(X_ohe=X_ohe,
                                             feature_names=feature_names,
                                             category_map=category_map,
                                             ranges=ranges,
                                             immutable_features=immutable_features,
                                             conditional=conditional)
        C.append(C_num)

    # Generate categorical condition vector.
    if len(category_map):
        C_cat = generate_categorical_condition(X_ohe=X_ohe,
                                               feature_names=feature_names,
                                               category_map=category_map,
                                               immutable_features=immutable_features,
                                               conditional=conditional)
        C.append(C_cat)

    # Concatenate numerical and categorical conditional vectors.
    return np.concatenate(C, axis=1)


def sample_numerical(X_hat_num_split: List[np.ndarray],
                     X_ohe_num_split: List[np.ndarray],
                     C_num_split: List[np.ndarray],
                     stats: Dict[int, Dict[str, float]]) -> List[np.ndarray]:
    """
    Samples numerical features according to the conditional vector. This method clips the values between the
    desired ranges specified in the conditional vector, and ensures that the values are between the minimum and
    the maximum values from train training datasets stored in the dictionary of statistics.

    Parameters
    ----------
    X_hat_num_split
        List of reconstructed numerical heads from the auto-encoder. This list should contain a single element
        as all the numerical features are part of a singe linear layer output.
    X_ohe_num_split
        List of original numerical heads. The list should contain a single element as part of the convention
        mentioned in the description of `X_ohe_hat_num`.
    C_num_split
        List of conditional vector for numerical heads. The list should contain a single element as part of the
        convention mentioned in the description of `X_ohe_hat_num`.
    stats
        Dictionary of statistic of the training data. Contains the minimum and maximum value of each numerical
        feature in the training set. Each key is an index of the column and each value is another dictionary
        containing `min` and `max` keys.

    Returns
    -------
    X_ohe_hat_num
        List of clamped input vectors according to the conditional vectors and the dictionary of statistics.
    """
    num_cols = X_hat_num_split[0].shape[1]  # number of numerical columns
    sorted_cols = sorted(stats.keys())  # ensure that the column ids are sorted

    for i, col_id in zip(range(num_cols), sorted_cols):
        # Extract the minimum and the maximum value for the current column from the training set.
        min, max = stats[col_id]["min"], stats[col_id]["max"]

        if C_num_split is not None:
            # Extract the minimum and the maximum value according to the conditional vector.
            lhs = X_ohe_num_split[0][:, i] + C_num_split[0][:, 2 * i] * (max - min)
            rhs = X_ohe_num_split[0][:, i] + C_num_split[0][:, 2 * i + 1] * (max - min)

            # Clamp output according to the conditional vector.
            X_hat_num_split[0][:, i] = np.clip(X_hat_num_split[0][:, i], a_min=lhs, a_max=rhs)

        # Clamp output according to the minimum and maximum value from the training set.
        X_hat_num_split[0][:, i] = np.clip(X_hat_num_split[0][:, i], a_min=min, a_max=max)

    return X_hat_num_split


def sample_categorical(X_hat_cat_split: List[np.ndarray],
                       C_cat_split: List[np.ndarray]) -> List[np.ndarray]:
    """
    Samples categorical features according to the conditional vector. This method sample conditional according to
    the masking vector the most probable outcome.

    Parameters
    ----------
    X_hat_cat_split
        List of reconstructed categorical heads from the auto-encoder. The categorical columns contain logits.
    C_cat_split
        List of conditional vector for categorical heads.

    Returns
    -------
    X_ohe_hat_cat
        List of one-hot encoded vectors sampled according to the conditional vector.
    """
    X_ohe_hat_cat = []  # initialize the returning list
    rows = np.arange(X_hat_cat_split[0].shape[0])  # initialize the returning list

    for i in range(len(X_hat_cat_split)):
        # compute probability distribution
        proba = softmax(X_hat_cat_split[i], axis=1)
        proba = proba * C_cat_split[i] if (C_cat_split is not None) else proba

        # sample the most probable outcome conditioned on the conditional vector
        cols = np.argmax(proba, axis=1)
        samples = np.zeros_like(proba)
        samples[rows, cols] = 1
        X_ohe_hat_cat.append(samples)

    return X_ohe_hat_cat


def sample(X_hat_split: List[np.ndarray],
           X_ohe: np.ndarray,
           C: Optional[np.ndarray],
           category_map: Dict[int, List[str]],
           stats: Dict[int, Dict[str, float]]) -> List[np.ndarray]:
    """
    Samples an instance from the given reconstruction according to the conditional vector and
    the dictionary of statistics.

    Parameters
    ----------
    X_hat_split
        List of reconstructed columns from the auto-encoder. The categorical columns contain logits.
    X_ohe
        One-hot encoded representation of the input.
    C
        Conditional vector.
    category_map
        Dictionary of category mapping. The keys are column indexes and the values are lists containing the possible
        values for a feature.
    stats
        Dictionary of statistic of the training data. Contains the minimum and maximum value of each numerical
        feature in the training set. Each key is an index of the column and each value is another dictionary
        containing `min` and `max` keys.

    Returns
    -------
    X_ohe_hat_split
        Most probable reconstruction sample according to the autoencoder, sampled according to the conditional vector
        and the dictionary of statistics. This method assumes that the input array, `X_ohe` , has the first columns
        corresponding to the numerical features, and the rest are one-hot encodings of the categorical columns.
    """
    X_ohe_num_split, X_ohe_cat_split = split_ohe(X_ohe, category_map)
    C_num_split, C_cat_split = split_ohe(C, category_map) if (C is not None) else (None, None)

    X_ohe_hat_split = []  # list of sampled numerical columns and sampled categorical columns
    num_feat, cat_feat = len(X_ohe_num_split), len(X_ohe_cat_split)

    if num_feat > 0:
        # Sample numerical columns
        X_ohe_hat_split += sample_numerical(X_hat_num_split=X_hat_split[:num_feat],
                                            X_ohe_num_split=X_ohe_num_split,
                                            C_num_split=C_num_split,
                                            stats=stats)

    if cat_feat > 0:
        # Sample categorical columns
        X_ohe_hat_split += sample_categorical(X_hat_cat_split=X_hat_split[-cat_feat:],
                                              C_cat_split=C_cat_split)

    return X_ohe_hat_split


def get_he_preprocessor(X: np.ndarray,
                        feature_names: List[str],
                        category_map: Dict[int, List[str]],
                        feature_types: Dict[str, type] = None
                        ) -> Tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    """
    Heterogeneous dataset preprocessor. The numerical features are standardized and the categorical features
    are one-hot encoded.

    Parameters
    ----------
    X
        Data to fit.
    feature_names
        List of feature names. This should be provided by the dataset.
    category_map
        Dictionary of category mapping. The keys are column indexes and the values are lists containing the
        possible feature values. This should be provided by the dataset.
    feature_types
        Dictionary of type for the numerical features.

    Returns
    -------
    preprocessor
        Data preprocessor.
    inv_preprocessor
        Inverse data preprocessor (e.g., `inv_preprocessor(preprocessor(x)) = x` )
    """
    if feature_types is None:
        feature_types = dict()

    # Separate columns in numerical and categorical
    categorical_ids = list(category_map.keys())
    numerical_ids = [i for i in range(len(feature_names)) if i not in category_map.keys()]

    # Define standard scaler and one-hot encoding transformations
    num_transf = StandardScaler()
    cat_transf = OneHotEncoder(
        categories=[range(len(x)) for x in category_map.values()],
        handle_unknown="ignore"
    )

    # Define preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transf, numerical_ids),
            ("cat", cat_transf, categorical_ids)
        ],
        sparse_threshold=0
    )
    preprocessor.fit(X)

    num_feat_ohe = len(numerical_ids)  # number of numerical columns
    cat_feat_ohe = sum([len(v) for v in category_map.values()])  # number of categorical columns

    # Define inverse preprocessor
    def get_inv_preprocessor(X_ohe: np.ndarray):
        X_inv = []

        if "num" in preprocessor.named_transformers_ and len(numerical_ids):
            num_transf = preprocessor.named_transformers_["num"]
            X_ohe_num = X_ohe[:, :num_feat_ohe] if preprocessor.transformers[0][0] == "num" else \
                X_ohe[:, -num_feat_ohe:]
            X_inv.append(num_transf.inverse_transform(X_ohe_num))

        if "cat" in preprocessor.named_transformers_ and len(categorical_ids):
            cat_transf = preprocessor.named_transformers_["cat"]
            X_ohe_cat = X_ohe[:, :cat_feat_ohe] if preprocessor.transformers[0][0] == "cat" else \
                X_ohe[:, -cat_feat_ohe:]
            X_inv.append(cat_transf.inverse_transform(X_ohe_cat))

        # Concatenate all columns. at this point the columns are not ordered correctly
        np_X_inv = np.concatenate(X_inv, axis=1)

        # Construct permutation to order the columns correctly
        perm = [i for i in range(len(feature_names)) if i not in category_map.keys()]
        perm += [i for i in range(len(feature_names)) if i in category_map.keys()]

        inv_perm = [0] * len(perm)
        for i in range(len(perm)):
            inv_perm[perm[i]] = i

        np_X_inv = np_X_inv[:, inv_perm].astype(object)
        for i, fn in enumerate(feature_names):
            type = feature_types[fn] if fn in feature_types else float
            np_X_inv[:, i] = np_X_inv[:, i].astype(type)

        return np_X_inv

    return preprocessor.transform, get_inv_preprocessor


def get_statistics(X: np.ndarray,
                   preprocessor: Callable[[np.ndarray], np.ndarray],
                   category_map: Dict[int, List[str]]) -> Dict[int, Dict[str, float]]:
    """
    Computes statistics.

    Parameters
    ----------
    X
        Instances for which to compute statistic.
    preprocessor
        Data preprocessor. The preprocessor should standardize the numerical values and convert categorical ones
        into one-hot encoding representation. By convention, numerical features should be first, followed by the
        rest of categorical ones.
    category_map
        Dictionary of category mapping. The keys are column indexes and the values are lists containing the
        possible feature values. This should be provided by the dataset.

    Returns
    -------
        Dictionary of statistics. For each numerical column, the minimum and maximum value is returned.
    """
    stats = dict()

    # Extract numerical features
    num_features_ids = [id for id in range(X.shape[1]) if id not in category_map]

    # Preprocess data (standardize + one-hot encoding)
    X_ohe = preprocessor(X)

    for i, feature_id in enumerate(num_features_ids):
        min, max = np.min(X_ohe[:, i]), np.max(X_ohe[:, i])
        stats[feature_id] = {"min": min, "max": max}

    return stats


def get_numerical_conditional_vector(X: np.ndarray,
                                     condition: Dict[str, List[Union[float, str]]],
                                     preprocessor: Callable[[np.ndarray], np.ndarray],
                                     feature_names: List[str],
                                     category_map: Dict[int, List[str]],
                                     stats: Dict[int, Dict[str, float]],
                                     ranges: Dict[str, List[float]] = None,
                                     immutable_features: List[str] = None,
                                     diverse=False) -> List[np.ndarray]:
    """
    Generates a conditional vector. The condition is expressed a a delta change of the feature.
    For numerical features, if the `Age` feature is allowed to increase up to 10 more years, the delta change is
    `[0, 10]`.  If the `Hours per week` is allowed to decrease down to `-5` and increases up to `+10`, then the
    delta change is `[-5, +10]`. Note that the interval must go include `0`.

    Parameters
    ----------
    X
        Instances for which to generate the conditional vector in the original input format.
    condition
        Dictionary of conditions per feature. For numerical features it expects a range that contains the original
        value. For categorical features it expects a list of feature values per features that includes the original
        value.
    preprocessor
        Data preprocessor. The preprocessor should standardize the numerical values and convert categorical ones
        into one-hot encoding representation. By convention, numerical features should be first, followed by the
        rest of categorical ones.
    feature_names
        List of feature names. This should be provided by the dataset.
    category_map
        Dictionary of category mapping. The keys are column indexes and the values are lists containing the
        possible feature values.  This should be provided by the dataset.
    stats
        Dictionary of statistic of the training data. Contains the minimum and maximum value of each numerical
        feature in the training set. Each key is an index of the column and each value is another dictionary
        containing `min` and `max` keys.
    ranges
        Dictionary of ranges for numerical feature. Each value is a list containing two elements, first one
        negative and the second one positive.
    immutable_features
        List of immutable features.
    diverse
        Whether to generate a diverse set of conditional vectors. A diverse set of conditional vector can generate
        a diverse set of counterfactuals for a given input instance.

    Returns
    -------
        List of conditional vectors for each numerical feature.
    """
    if ranges is None:
        ranges = dict()

    if immutable_features is None:
        immutable_features = list()

    # Extract numerical features
    num_features_ids = [id for id in range(X.shape[1]) if id not in category_map]
    num_features_names = [feature_names[id] for id in num_features_ids]

    # Need to standardize numerical features. Thus, we use the preprocessor
    X_low, X_high = X.copy(), X.copy()

    for feature_id, feature_name in enumerate(feature_names):
        if feature_id in category_map:
            continue

        if feature_name in condition:
            if int(condition[feature_name][0]) > 0:  # int conversion because of mypy error (the value can be str too)
                raise ValueError(f"Lower bound on the conditional vector for {feature_name} should be negative.")

            if int(condition[feature_name][1]) < 0:  # int conversion because of mypy error (the value can be str too)
                raise ValueError(f"Upper bound on the conditional vector for {feature_name} should be positive.")

            X_low[:, feature_id] += condition[feature_name][0]
            X_high[:, feature_id] += condition[feature_name][1]

    # Preprocess the vectors (standardize + one-hot encoding)
    X_low_ohe = preprocessor(X_low)
    X_high_ohe = preprocessor(X_high)
    X_ohe = preprocessor(X)

    # Initialize conditional vector buffer.
    C = []

    # Scale the numerical features in [0, 1] and add them to the conditional vector
    for i, (feature_id, feature_name) in enumerate(zip(num_features_ids, num_features_names)):
        if feature_name in immutable_features:
            range_low, range_high = 0., 0.
        elif feature_name in ranges:
            range_low, range_high = ranges[feature_name][0], ranges[feature_name][1]
        else:
            range_low, range_high = -1., 1.

        if (feature_name in condition) and (feature_name not in immutable_features):
            # Mutable feature with conditioning
            min, max = stats[feature_id]["min"], stats[feature_id]["max"]
            X_low_ohe[:, i] = (X_low_ohe[:, i] - X_ohe[:, i]) / (max - min)
            X_high_ohe[:, i] = (X_high_ohe[:, i] - X_ohe[:, i]) / (max - min)

            # Clip in [0, 1]
            X_low_ohe[:, i] = np.clip(X_low_ohe[:, i], a_min=range_low, a_max=0)
            X_high_ohe[:, i] = np.clip(X_high_ohe[:, i], a_min=0, a_max=range_high)
        else:
            # This means no conditioning
            X_low_ohe[:, i] = range_low
            X_high_ohe[:, i] = range_high

        if diverse:
            # Note that this is still a feasible counterfactual
            X_low_ohe[:, i] *= np.random.rand(*X_low_ohe[:, i].shape)
            X_high_ohe[:, i] *= np.random.rand(*X_high_ohe[:, i].shape)

        # Append feature conditioning
        C += [X_low_ohe[:, i].reshape(-1, 1), X_high_ohe[:, i].reshape(-1, 1)]

    return C


def get_categorical_conditional_vector(X: np.ndarray,
                                       condition: Dict[str, List[Union[float, str]]],
                                       preprocessor: Callable[[np.ndarray], np.ndarray],
                                       feature_names: List[str],
                                       category_map: Dict[int, List[str]],
                                       immutable_features: List[str] = None,
                                       diverse=False) -> List[np.ndarray]:
    """
    Generates a conditional vector. The condition is expressed a a delta change of the feature.
    For categorical feature, if the `Occupation` can change to `Blue-Collar` or `White-Collar` the delta change
    is `['Blue-Collar', 'White-Collar']`. Note that the original value is optional as it is included by default.

    Parameters
    ----------
    X
        Instances for which to generate the conditional vector in the original input format.
    condition
        Dictionary of conditions per feature. For numerical features it expects a range that contains the original
        value. For categorical features it expects a list of feature values per features that includes the original
        value.
    preprocessor
        Data preprocessor. The preprocessor should standardize the numerical values and convert categorical ones
        into one-hot encoding representation. By convention, numerical features should be first, followed by the
        rest of categorical ones.
    feature_names
        List of feature names. This should be provided by the dataset.
    category_map
        Dictionary of category mapping. The keys are column indexes and the values are lists containing the
        possible feature values.  This should be provided by the dataset.
    immutable_features
        List of immutable features.
    diverse
        Whether to generate a diverse set of conditional vectors. A diverse set of conditional vector can generate
        a diverse set of counterfactuals for a given input instance.

    Returns
    -------
        List of conditional vectors for each categorical feature.
    """
    if immutable_features is None:
        immutable_features = list()

    # Define conditional vector buffer
    C = []

    # extract categorical features
    cat_features_ids = [id for id in range(X.shape[1]) if id in category_map]
    cat_feature_names = [feature_names[id] for id in cat_features_ids]

    # Extract list of categorical one-hot encoded columns
    X_ohe = preprocessor(X)
    _, X_ohe_cat_split = split_ohe(X_ohe, category_map)

    # For each categorical feature add the masking vector
    for i, (feature_id, feature_name) in enumerate(zip(cat_features_ids, cat_feature_names)):
        mask = np.zeros_like(X_ohe_cat_split[i])

        if feature_name not in immutable_features:
            if feature_name in condition:
                indexes = [category_map[feature_id].index(str(feature_value)) for feature_value in
                           condition[feature_name]]  # conversion to str because of mypy (can be also float)
                mask[:, indexes] = 1
            else:
                # Allow any value
                mask[:] = 1

        if diverse:
            # Note that by masking random entries we still have a feasible counterfactual
            mask *= np.random.randint(low=0, high=2, size=mask.shape)

        # Ensure that the original value is a possibility
        mask = ((mask + X_ohe_cat_split[i]) > 0).astype(int)

        # Append feature conditioning
        C.append(mask)
    return C


def get_conditional_vector(X: np.ndarray,
                           condition: Dict[str, List[Union[float, str]]],
                           preprocessor: Callable[[np.ndarray], np.ndarray],
                           feature_names: List[str],
                           category_map: Dict[int, List[str]],
                           stats: Dict[int, Dict[str, float]],
                           ranges: Dict[str, List[float]] = None,
                           immutable_features: List[str] = None,
                           diverse=False) -> np.ndarray:
    """
    Generates a conditional vector. The condition is expressed a a delta change of the feature.

    For numerical features, if the `Age` feature is allowed to increase up to 10 more years, the delta change is
    `[0, 10]`.  If the `Hours per week` is allowed to decrease down to `-5` and increases up to `+10`, then the
    delta change is `[-5, +10]`. Note that the interval must go include `0`.

    For categorical feature, if the `Occupation` can change to `Blue-Collar` or `White-Collar` the delta change
    is `['Blue-Collar', 'White-Collar']`. Note that the original value is optional as it is included by default.

    Parameters
    ----------
    X
        Instances for which to generate the conditional vector in the original input format.
    condition
        Dictionary of conditions per feature. For numerical features it expects a range that contains the original
        value. For categorical features it expects a list of feature values per features that includes the original
        value.
    preprocessor
        Data preprocessor. The preprocessor should standardize the numerical values and convert categorical ones
        into one-hot encoding representation. By convention, numerical features should be first, followed by the
        rest of categorical ones.
    feature_names
        List of feature names. This should be provided by the dataset.
    category_map
        Dictionary of category mapping. The keys are column indexes and the values are lists containing the
        possible feature values.  This should be provided by the dataset.
    stats
        Dictionary of statistic of the training data. Contains the minimum and maximum value of each numerical
        feature in the training set. Each key is an index of the column and each value is another dictionary
        containing `min` and `max` keys.
    ranges
        Dictionary of ranges for numerical feature. Each value is a list containing two elements, first one
        negative and the second one positive.
    immutable_features
        List of immutable features.
    diverse
        Whether to generate a diverse set of conditional vectors. A diverse set of conditional vector can generate
        a diverse set of counterfactuals for a given input instance.

    Returns
    -------
        Conditional vector.
    """
    if ranges is None:
        ranges = dict()

    if immutable_features is None:
        immutable_features = list()

    # Reshape the vector.
    X = X.reshape(1, -1) if len(X.shape) == 1 else X

    # Check that the second dimension matches the number of features.
    if X.shape[1] != len(feature_names):
        raise ValueError(f"Unexpected number of features. The expected number "
                         f"is {len(feature_names)}, but the input has {X.shape[1]} features.")

    # Get list of numerical conditional vectors.
    C_num = get_numerical_conditional_vector(X=X,
                                             condition=condition,
                                             preprocessor=preprocessor,
                                             feature_names=feature_names,
                                             category_map=category_map,
                                             stats=stats,
                                             ranges=ranges,
                                             immutable_features=immutable_features,
                                             diverse=diverse)

    # Get list of categorical conditional vectors.
    C_cat = get_categorical_conditional_vector(X=X,
                                               condition=condition,
                                               preprocessor=preprocessor,
                                               feature_names=feature_names,
                                               category_map=category_map,
                                               immutable_features=immutable_features,
                                               diverse=diverse)

    # concat all conditioning
    return np.concatenate(C_num + C_cat, axis=1)


def apply_category_mapping(X: np.ndarray, category_map: Dict[int, List[str]]) -> np.ndarray:
    """
    Applies a category mapping for the categorical feature in the array. It transforms ints back to strings
    to be readable.

    Parameters
    -----------
    X
        Array containing the columns to be mapped.
    category_map
        Dictionary of category mapping. Keys are columns index, and values are list of feature values.

    Returns
    -------
        Transformed array.
    """
    pd_X = pd.DataFrame(X)

    for key in category_map:
        pd_X[key].replace(range(len(category_map[key])), category_map[key], inplace=True)

    return pd_X.to_numpy()
