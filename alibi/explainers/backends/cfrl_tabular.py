import numpy as np
import pandas as pd
from typing import List, Dict, Union, Tuple, Callable

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from scipy.special import softmax


class CounterfactualRLTabularBackend:
    @staticmethod
    def conditional_dim(feature_names: List[str], category_map: Dict[int, List[str]]) -> int:
        """
        Computes the dimension of the conditional vector

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

    @staticmethod
    def split_ohe(x_ohe: np.ndarray,
                  category_map: Dict[int, List[str]]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Splits a one-hot encoding array in a list of numerical heads and a list of categorical heads. Since by
        convention the numerical heads are merged in a single head, if the function returns a list of numerical heads,
        then the size of the list is 1.

        Parameters
        ----------
        x_ohe
            One-hot encoding representation.
        category_map
            Dictionary of category mapping. The keys are column indexes and the values are lists containing the
            possible values of a feature.

        Returns
        -------
        x_ohe_num_split
            List of numerical heads. If different than `None`, the list's size is 1.
        x_ohe_cat_split
            List of categorical one-hot encoded heads.
        """
        x_ohe_num_split, x_ohe_cat_split = [], []
        offset = 0

        # Compute the number of columns spanned by the categorical one-hot encoded heads, and the number of columns
        # spanned by the numerical heads.
        cat_feat = int(np.sum([len(vals) for vals in category_map.values()]))
        num_feat = x_ohe.shape[1] - cat_feat

        # If the number of numerical features is different than 0, then the first `num_feat` columns correspond
        # to the numerical features
        if num_feat > 0:
            x_ohe_num_split.append(x_ohe[:, :num_feat])
            offset = num_feat

        # If there exist categorical features, then extract them one by one
        if cat_feat > 0:
            for id in category_map:
                x_ohe_cat_split.append(x_ohe[:, offset:offset + len(category_map[id])])
                offset += len(category_map[id])

        return x_ohe_num_split, x_ohe_cat_split

    @staticmethod
    def numerical_condition(x_ohe: np.ndarray,
                            feature_names: List[str],
                            category_map: Dict[int, List[str]],
                            ranges: Dict[str, List[float]],
                            immutable_features: List[str],
                            conditional: bool = True) -> np.ndarray:
        """
        Generates numerical features conditional vector.

        Parameters
        ----------
        x_ohe
            One-hot encoding representation of the element(s) for which the conditional vector will be generated.
            This argument is used to extract the number of conditional vector. The choice of `x_ohe` instead of a
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
        num_cond
            Conditional vector for numerical features.
        """
        num_cond = []
        size = x_ohe.shape[0]

        for feature_id, feature_name in enumerate(feature_names):
            # skip categorical features
            if feature_id in category_map:
                continue

            if feature_name in immutable_features:
                # immutable feature
                range_low, range_high = 0, 0
            else:
                range_low = ranges[feature_name][0] if feature_name in ranges else -1
                range_high = ranges[feature_name][1] if feature_name in ranges else 1

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
        num_cond = np.concatenate(num_cond, axis=1)
        return num_cond

    @staticmethod
    def categorical_condition(x_ohe: np.ndarray,
                              feature_names: List[str],
                              category_map: Dict[int, List],
                              immutable_features: List[str],
                              conditional: bool = True) -> np.ndarray:
        """
        Generates categorical features conditional vector.

        Parameters
        ----------
        x_ohe
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
        cat_cond
            Conditional vector for categorical feature.
        """
        cat_cond = []
        cat_idx = 0

        # Split the one-hot representation into a list where each element corresponds to an feature.
        _, x_ohe_cat_split = CounterfactualRLTabularBackend.split_ohe(x_ohe, category_map)

        # Create mask for each categorical column.
        for feature_id, feature_name in enumerate(feature_names):
            # skip numerical features
            if feature_id not in category_map:
                continue

            # initialize mask with the original value
            mask = x_ohe_cat_split[cat_idx].copy()

            # if the feature is not immutable, add noise to modify the mask
            if feature_name not in immutable_features:
                mask += np.random.rand(*mask.shape) if conditional else np.ones_like(mask)

            # construct binary mask
            mask = (mask > 0.5).astype(np.float32)
            cat_cond.append(mask)

            # move to the next categorical index
            cat_idx += 1

        cat_cond = np.concatenate(cat_cond, axis=1)
        return cat_cond

    @staticmethod
    def generate_condition(x_ohe: np.ndarray,
                           feature_names: List[str],
                           category_map: Dict[int, List[str]],
                           ranges: Dict[str, List[float]],
                           immutable_features: List[str],
                           conditional: bool = True) -> np.ndarray:
        """
        Generates conditional vector.

        Parameters
        ----------
        x_ohe
            One-hot encoding representation of the element(s) for which the conditional vector will be generated.
            This method assumes that the input array, `x_ohe`, is encoded as follows: first columns correspond to the
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
        cond
            Conditional vector.
        """
        # Generate numerical condition vector.
        num_cond = CounterfactualRLTabularBackend.numerical_condition(x_ohe=x_ohe,
                                                                      feature_names=feature_names,
                                                                      category_map=category_map,
                                                                      ranges=ranges,
                                                                      immutable_features=immutable_features,
                                                                      conditional=conditional)

        # Generate categorical condition vector.
        cat_cond = CounterfactualRLTabularBackend.categorical_condition(x_ohe=x_ohe,
                                                                        feature_names=feature_names,
                                                                        category_map=category_map,
                                                                        immutable_features=immutable_features,
                                                                        conditional=conditional)

        # Concatenate numerical and categorical conditional vectors.
        cond = np.concatenate([num_cond, cat_cond], axis=1)
        return cond

    @staticmethod
    def sample_numerical(x_hat_num_split: List[np.ndarray],
                         x_ohe_num_split: List[np.ndarray],
                         cond_num_split: List[np.ndarray],
                         stats: Dict[int, Dict[str, float]]) -> List[np.ndarray]:
        """
        Samples numerical features according to the conditional vector. This method clips the values between the
        desired ranges specified in the conditional vector, and ensures that the values are between the minimum and
        the maximum values from train training datasets stored in the dictionary of statistics.

        Parameters
        ----------
        x_hat_num_split
            List of reconstructed numerical heads from the auto-encoder. This list should contain a single element
            as all the numerical features are part of a singe linear layer output.
        x_ohe_num_split
            List of original numerical heads. The list should contain a single element as part of the convention
            mentioned in the description of `x_ohe_hat_num`.
        cond_num_split
            List of conditional vector for numerical heads. The list should contain a single element as part of the
            convention mentioned in the description of `x_ohe_hat_num`.
        stats
            Dictionary of statistic of the training data. Contains the minimum and maximum value of each numerical
            feature in the training set. Each key is an index of the column and each value is another dictionary
            containing `min` and `max` keys.

        Returns
        -------
        x_ohe_hat_num
            List of clamped input vectors according to the conditional vectors and the dictionary of statistics .
        """
        num_cols = x_hat_num_split[0].shape[1]  # number of numerical columns
        sorted_cols = sorted(stats.keys())  # ensure that the column ids are sorted

        for i, col_id in zip(range(num_cols), sorted_cols):
            # Extract the minimum and the maximum value for the current column from the training set.
            min, max = stats[col_id]["min"], stats[col_id]["max"]

            # Extract the minimum and the maximum value according to the conditional vector.
            lhs = x_ohe_num_split[0][:, i] + cond_num_split[0][:, 2 * i] * (max - min)
            rhs = x_ohe_num_split[0][:, i] + cond_num_split[0][:, 2 * i + 1] * (max - min)

            # Clamp output according to the conditional vector.
            x_hat_num_split[0][:, i] = np.clip(x_hat_num_split[0][:, i], a_min=lhs, a_max=rhs)

            # Clamp output according to the minimum and maximum value from the training set.
            x_hat_num_split[0][:, i] = np.clip(x_hat_num_split[0][:, i], a_min=min, a_max=max)

        return x_hat_num_split

    @staticmethod
    def sample_categorical(x_hat_cat_split: List[np.ndarray],
                           cond_cat_split: List[np.ndarray]) -> List[np.ndarray]:
        """
        Samples categorical features according to the conditional vector. This method sample conditional according to
        the masking vector the most probable outcome.

        Parameters
        ----------
        x_hat_cat_split
            List of reconstructed categorical heads from the auto-encoder. The categorical columns contain logits.
        cond_cat_split
            List of conditional vector for categorical heads.

        Returns
        -------
        x_ohe_hat_cat
            List of one-hot encoded vectors sampled according to the conditional vector.
        """
        x_out = []  # initialize the returning list
        rows = np.arange(x_hat_cat_split[0].shape[0])  # initialize the returning list

        for i in range(len(x_hat_cat_split)):
            # compute probability distribution
            proba = softmax(x_hat_cat_split[i], axis=1)

            # sample the most probable outcome conditioned on the conditional vector
            cols = np.argmax(cond_cat_split[i] * proba, axis=1)
            samples = np.zeros_like(proba)
            samples[rows, cols] = 1
            x_out.append(samples)

        return x_out

    @staticmethod
    def sample(x_hat_split: List[np.ndarray],
               x_ohe: np.ndarray,
               cond: np.ndarray,
               stats: Dict[int, Dict[str, float]],
               category_map: Dict[int, List[str]]) -> List[np.ndarray]:
        """
        Samples an instance from the given reconstruction according to the conditional vector and
        the dictionary of statistics.

        Parameters
        ----------
        x_hat_split
            List of one-hot encoded reconstructed columns form the auto-encoder. The categorical columns contain logits.
        x_ohe
            One-hot encoded representation of the input.
        cond
            Conditional vector.
        stats
            Dictionary of statistic of the training data. Contains the minimum and maximum value of each numerical
            feature in the training set. Each key is an index of the column and each value is another dictionary
            containing `min` and `max` keys.
        category_map
            Dictionary of category mapping. The keys are column indexes and the values are lists containing the possible
            values for a feature.

        Returns
        -------
        Most probable sample according to the auto-encoder reconstruction, sampled according to the conditional vector
        and the dictionary of statistics. This method assumes that the input array, `x_ohe`, is encoded as follows:
        first columns correspond to the numerical features, and the rest are one-hot encodings of the categorical
        columns.
        """
        x_ohe_num_split, x_ohe_cat_split = CounterfactualRLTabularBackend.split_ohe(x_ohe, category_map)
        cond_num_split, cond_cat_split = CounterfactualRLTabularBackend.split_ohe(cond, category_map)

        sampled_num, sampled_cat = [], []  # list of sampled numerical columns and sampled categorical columns
        num_feat, cat_feat = len(x_ohe_num_split), len(x_ohe_cat_split)

        if num_feat > 0:
            # sample numerical columns
            sampled_num = CounterfactualRLTabularBackend.sample_numerical(x_hat_num_split=x_hat_split[:num_feat],
                                                                          x_ohe_num_split=x_ohe_num_split,
                                                                          cond_num_split=cond_num_split,
                                                                          stats=stats)

        if cat_feat > 0:
            # sample categorical columns
            sampled_cat = CounterfactualRLTabularBackend.sample_categorical(x_hat_cat_split=x_hat_split[-cat_feat:],
                                                                            cond_cat_split=cond_cat_split)

        return sampled_num + sampled_cat

    @staticmethod
    def he_preprocessor(x: np.ndarray,
                        feature_names: List[str],
                        category_map: Dict[int, List[str]],
                        feature_types: Dict[int, type] = dict()
                        ) -> Tuple[Callable[[np.ndarray], np.ndarray],
                                   Callable[[np.ndarray], np.ndarray]]:
        """
        Heterogeneous dataset preprocessor. The numerical features are standardized and the categorical features
        are one-hot encoded.

        Parameters
        ----------
        x
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
            Inverse data preprocessor (e.g., inv_preprocessor(preprocssor(x)) = x)
        """
        # separate columns in numerical and categorical
        categorical_ids = list(category_map.keys())
        numerical_ids = [i for i in range(len(feature_names)) if i not in category_map.keys()]

        # define standard scaler and one-hot encoding transformations
        num_transf = StandardScaler()
        cat_transf = OneHotEncoder(
            categories=[range(len(x)) for x in category_map.values()],
            handle_unknown="ignore"
        )

        # define preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_transf, numerical_ids),
                ("cat", cat_transf, categorical_ids)
            ],
            sparse_threshold=0
        )
        preprocessor.fit(x)

        num_feat_ohe = len(numerical_ids)  # number of numerical columns
        cat_feat_ohe = sum([len(v) for v in category_map.values()])  # number of categorical columns

        # define inverse preprocessor
        def inv_preprocessor(x_ohe: np.ndarray):
            x_inv = []

            if "num" in preprocessor.named_transformers_:
                num_transf = preprocessor.named_transformers_["num"]
                x_inv.append(num_transf.inverse_transform(x_ohe[:, :num_feat_ohe]))

            if "cat" in preprocessor.named_transformers_:
                cat_transf = preprocessor.named_transformers_["cat"]
                x_inv.append(cat_transf.inverse_transform(x_ohe[:, -cat_feat_ohe:]))

            # concatenate all columns. at this point the columns are not ordered correctly
            x_inv = np.concatenate(x_inv, axis=1)

            # construct permutation to order the columns correctly
            perm = [i for i in range(len(feature_names)) if i not in category_map.keys()]
            perm += [i for i in range(len(feature_names)) if i in category_map.keys()]

            inv_perm = [0] * len(perm)
            for i in range(len(perm)):
                inv_perm[perm[i]] = i

            x_inv = x_inv[:, inv_perm].astype(object)
            for i in range(len(feature_names)):
                type = feature_types[i] if i in feature_types else int
                x_inv[:, i] = x_inv[:, i].astype(type)

            return x_inv

        return preprocessor.transform, inv_preprocessor

    @staticmethod
    def statistics(x: np.ndarray,
                   preprocessor: Callable[[np.ndarray], np.ndarray],
                   category_map: Dict[int, List[str]]) -> Dict[int, Dict[str, float]]:
        """
        Computes statistics.

        Parameters
        ----------
        x
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

        # extract numerical features
        num_features_ids = [id for id in range(x.shape[1]) if id not in category_map]

        # preprocess data (standardize + one-hot encoding)
        x_ohe = preprocessor(x)

        for i, feature_id in enumerate(num_features_ids):
            min, max = np.min(x_ohe[:, i]), np.max(x_ohe[:, i])
            stats[feature_id] = {"min": min, "max": max}

        return stats

    @staticmethod
    def conditional_vector(x: np.ndarray,
                           condition: Dict[str, List[Union[int, str]]],
                           preprocessor: Callable[[np.ndarray], np.ndarray],
                           feature_names: List[str],
                           category_map: Dict[int, List[str]],
                           stats: Dict[int, Dict[str, float]],
                           ranges: Dict[str, List[float]] = dict(),
                           immutable_features: List[str] = list(),
                           diverse=False) -> np.ndarray:
        """
        Generates a conditional vector. The condition is expressed a a delta change of the feature. For example, if
        `Age`=26 and the feature is allowed to increase up to 10 more years. Similar for categorical features,
        the current value can be omitted.

        Parameters
        ----------
        x
            Instances for which to generate the conditional vector.
        condition
            Dictionary of conditions per feature. For numerical features it expects a rang that contains the original
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
        # reshape the vector
        x = x.reshape(1, -1) if len(x.shape) == 1 else x

        # extract numerical features
        num_features_ids = [id for id in range(x.shape[1]) if id not in category_map]
        num_features_names = [feature_names[id] for id in num_features_ids]

        # extract categorical features
        cat_features_ids = [id for id in range(x.shape[1]) if id in category_map]
        cat_feature_names = [feature_names[id] for id in cat_features_ids]

        # need to standardize numerical features. Thus, we use the preprocessor
        x_low, x_high = x.copy(), x.copy()

        for feature_id, feature_name in enumerate(feature_names):
            if feature_id in category_map:
                continue

            if feature_name in condition:
                if condition[feature_name][0] > 0:
                    raise ValueError(f"Lower bound on the conditional vector for {feature_name} should be negative.")

                if condition[feature_name][1] < 0:
                    raise ValueError(f"Upper bound on the conditional vector for {feature_name} should be positive.")

                x_low[:, feature_id] += condition[feature_name][0]
                x_high[:, feature_id] += condition[feature_name][1]

        # preprocess the vectors (standardize + one-hot encoding)
        x_low_ohe = preprocessor(x_low)
        x_high_ohe = preprocessor(x_high)
        x_ohe = preprocessor(x)

        # initialize conditional vector
        cond = []

        # scale the numerical features in [0, 1] and add them to the conditional vector
        for i, (feature_id, feature_name) in enumerate(zip(num_features_ids, num_features_names)):
            if feature_name in immutable_features:
                range_low, range_high = 0, 0
            elif feature_name in ranges:
                range_low, range_high = ranges[feature_name][0], ranges[feature_name][1]
            else:
                range_low, range_high = -1, 1

            if (feature_name in condition) and (feature_name not in immutable_features):
                # mutable feature with conditioning
                min, max = stats[feature_id]["min"], stats[feature_id]["max"]
                x_low_ohe[:, i] = (x_low_ohe[:, i] - x_ohe[:, i]) / (max - min)
                x_high_ohe[:, i] = (x_high_ohe[:, i] - x_ohe[:, i]) / (max - min)

                # clip in [0, 1]
                x_low_ohe[:, i] = np.clip(x_low_ohe[:, i], a_min=range_low, a_max=0)
                x_high_ohe[:, i] = np.clip(x_high_ohe[:, i], a_min=0, a_max=range_high)
            else:
                # this means no conditioning
                x_low_ohe[:, i] = range_low
                x_high_ohe[:, i] = range_high

            if diverse:
                # not that this is still a feasible counterfactual
                x_low_ohe[:, i] *= np.random.rand(*x_low_ohe[:, i].shape)
                x_high_ohe[:, i] *= np.random.rand(*x_high_ohe[:, i].shape)

            # append feature conditioning
            cond += [x_low_ohe[:, i].reshape(-1, 1), x_high_ohe[:, i].reshape(-1, 1)]

        # extract list of categorical one-hot encoded columns
        _, x_ohe_cat_split = CounterfactualRLTabularBackend.split_ohe(x_ohe, category_map)

        # for each categorical feature add the masking vector
        for i, (feature_id, feature_name) in enumerate(zip(cat_features_ids, cat_feature_names)):
            mask = np.zeros_like(x_ohe_cat_split[i])

            if feature_name not in immutable_features:
                if feature_name in condition:
                    indexes = [category_map[feature_id].index(feature_value) for feature_value in
                               condition[feature_name]]
                    mask[:, indexes] = 1
                else:
                    # allow any value
                    mask[:] = 1

            if diverse:
                # note that by masking random entries we still have a feasible counterfactual
                mask *= np.random.randint(low=0, high=2, size=mask.shape)

            # ensure that the original value is a possibility
            mask = ((mask + x_ohe_cat_split[i]) > 0).astype(int)

            # append feature conditioning
            cond.append(mask)

        # concat all conditioning
        cond = np.concatenate(cond, axis=1)
        return cond

    @staticmethod
    def category_mapping(x: np.ndarray, category_map: Dict[int, List[str]]):
        """
        Applies a category mapping for the categorical feature in the array. It transforms ints back to strings
        to be readable

        Parameters
        -----------
        x
            Array containing the columns to be mapped.
        category_map
            Dictionary of category mapping. Keys are columns index, and values are list of feature values.
        Returns
        -------
        Transformed array.
        """
        x = pd.DataFrame(x)

        for key in category_map:
            x[key].replace(range(len(category_map[key])), category_map[key], inplace=True)

        return x.to_numpy()
