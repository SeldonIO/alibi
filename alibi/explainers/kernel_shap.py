import copy
import logging
import shap

import numpy as np
import pandas as pd

from alibi.api.defaults import DEFAULT_META_KERNEL_SHAP, DEFAULT_DATA_KERNEL_SHAP
from alibi.api.interfaces import Explanation, Explainer, FitMixin
from alibi.utils.wrappers import methdispatch
from functools import partial
from scipy import sparse
from shap.common import DenseData, DenseDataWithIndex
from typing import Callable, Dict, List, Optional, Sequence, Union, Tuple

logger = logging.getLogger(__name__)

KERNEL_SHAP_PARAMS = [
    'link',
    'group_names',
    'groups',
    'weights',
    'summarise_background',
    'summarise_result',
    'kwargs',
]

KERNEL_SHAP_BACKGROUND_THRESHOLD = 300


def rank_by_importance(shap_values: List[np.ndarray],
                       feature_names: Union[List[str], Tuple[str], None] = None) -> Dict:
    """
    Given the shap values estimated for a multi-output model, this feature ranks
    features according to their importance. The feature importance is the average
    absolute value for a given feature.

    Parameters
    ----------
    shap_values
        Each element corresponds to a samples x features array of shap values corresponding
        to each model output.
    feature_names
        Each element is the name of the column with the corresponding index in each of the
        arrays in the shap_values list.

    Returns
    -------
    importances
        A dictionary containing a key for each model output ('0', '1', ...) and a key for
        aggregated model output ('aggregated'). Each value is a dictionary contains a 'ranked_effect' field,
        populated with an array of values representing the average magnitude for each shap value,
        ordered from highest (most important) to the lowest (least important) feature. The 'names'
        field contains the corresponding feature names.
    """

    if len(shap_values[0].shape) == 1:
        shap_values = [np.atleast_2d(arr) for arr in shap_values]

    if not feature_names:
        feature_names = ['feature_{}'.format(i) for i in range(shap_values[0].shape[1])]
    else:
        if len(feature_names) != shap_values[0].shape[1]:
            msg = "The feature names provided do not match the number of shap values estimated. " \
                  "Received {} feature names but estimated {} shap values!"
            logger.warning(msg.format(len(feature_names), shap_values[0].shape[1]))
            feature_names = ['feature_{}'.format(i) for i in range(shap_values[0].shape[1])]

    importances = {}  # type: Dict[str, Dict[str, np.ndarray]]
    avg_mag = []  # type: List

    # rank the features by average shap value for each class in turn
    for class_idx in range(len(shap_values)):
        avg_mag_shap = np.abs(shap_values[class_idx]).mean(axis=0)
        avg_mag.append(avg_mag_shap)
        feature_order = np.argsort(avg_mag_shap)[::-1]
        most_important = avg_mag_shap[feature_order]
        most_important_names = [feature_names[i] for i in feature_order]
        importances[str(class_idx)] = {
            'ranked_effect': most_important,
            'names': most_important_names,
        }

    # rank feature by average shap value for aggregated classes
    combined_shap = np.sum(avg_mag, axis=0)
    feature_order = np.argsort(combined_shap)[::-1]
    most_important_c = combined_shap[feature_order]
    most_important_c_names = [feature_names[i] for i in feature_order]
    importances['aggregated'] = {
        'ranked_effect': most_important_c,
        'names': most_important_c_names
    }

    return importances


def sum_categories(values: np.ndarray, start_idx: Tuple[int], enc_feat_dim: Tuple[int]):
    """
    For each entry in start_idx, the function sums the following k columns where k is the
    corresponding entry in the enc_feat_dim sequence. The columns whose indices are not in
    start_idx are left unchanged.

    Parameters
    ----------
    values
        The array whose columns will be summed.
    start_idx
        The start indices of the columns to be summed.
    enc_feat_dim
        The number of columns to be summed, one for each start index.

    Returns
    -------
    new_values
        An array whose columns have been summed according to the entries in start_idx and enc_feat_dim.
    """

    if start_idx is None or enc_feat_dim is None:
        raise ValueError("Both the start indices or the encoding dimension need to be specified!")

    if not len(enc_feat_dim) == len(start_idx):
        raise ValueError("The lengths of the sequences of start indices and encodings must be equal!")

    n_encoded_levels = sum(enc_feat_dim)
    if n_encoded_levels > values.shape[-1]:
        raise ValueError("The sum of the encoded features dimensions exceeds data dimension!")

    def _get_slices(start: Tuple, dim: Tuple, arr_trailing_dim: int) -> List[int]:
        """
        Given start indices, encoding dimensions and the array trailing shape, this function returns
        an array where contiguous numbers are slices. This array is used to reduce along an axis
        only the slices `slice(start[i], start[i]+dim[i], 1)` from a tensor and leave all other slices
        unchanged.
        """

        slices = []  # type: List[int]
        # first columns may not be reduced
        if start[0] > 0:
            slices.extend(tuple(range(start[0])))

        # add all slices to reduce
        slices.extend([start[0], start[0] + dim[0]])
        for s_idx, d in zip(start[1:], dim[1:]):
            last_idx = slices[-1]
            # some columns might not be reduced
            if last_idx < s_idx - 1:
                slices.extend(tuple(range(last_idx + 1, s_idx)))
                last_idx += (s_idx - last_idx - 2)
            # handle contiguous slices 
            if s_idx == last_idx:
                slices.append(s_idx + d)
            else:
                slices.extend((s_idx, s_idx + d))

        # avoid index error
        if start[-1] + dim[-1] == arr_trailing_dim:
            slices.pop()
            return slices

        # last few columns may not be reduced
        last_idx = slices[-1]
        if last_idx < arr_trailing_dim:
            slices.extend(tuple(range(last_idx + 1, arr_trailing_dim)))

        return slices

    def _reduction(arr, axis, indices=None):
        return np.add.reduceat(arr, indices, axis)

    # create array of slices to be reduced
    slices = _get_slices(start_idx, enc_feat_dim, values.shape[-1])
    if len(values.shape) == 3:
        reduction = partial(_reduction, indices=slices)
        return np.apply_over_axes(reduction, values, axes=(2, 1))
    return np.add.reduceat(values, slices, axis=1)


class KernelShap(Explainer, FitMixin):

    def __init__(self,
                 predictor: Callable,
                 link: str = 'identity',
                 feature_names: Union[List[str], Tuple[str], None] = None,
                 categorical_names: Optional[Dict[int, List[str]]] = None,
                 task: str = 'classification',
                 seed: int = None):
        """
        A wrapper around the shap.KernelExplainer class. This extends the current shap library functionality
        by allowing the user to specify variable groups in order to deal with one-hot encoded categorical
        variables. The user can also specify whether to aggregate the shap values estimate for the encoded levels
        of categorical variables during the explain call.

        Parameters
        ----------
        predictor
            A callable that takes as an input a samples x features array and outputs a samples x n_outputs
            outputs. The n_outputs should represent model output in margin space. If the model outputs
            probabilities, then the link should be set to 'logit' to ensure correct force plots.
        link
            Valid values are 'identity' or 'logit'. A generalized linear model link to connect the feature
            importance values to the model output. Since the feature importance values, phi, sum up to the
            model output, it often makes sense to connect them to the ouput with a link function where
            link(output) = sum(phi). If the model output is a probability then the LogitLink link function
            makes the feature importance values have log-odds units. Therefore, for a model which outputs
            probabilities, link='logit' makes the feature effects have log-odds (evidence) units where
            link='identity' means that the feature effects have probability units. Please see
            https://github.com/slundberg/shap/blob/master/notebooks/kernel_explainer/Squashing%20Effect.ipynb
            for an in-depth discussion about the semantics of explaining the model in the probability vs the
            margin space.
        feature_names
            Used to infer group names when categorical data is treated by grouping and `group_names` input to `fit`
            is not specified, assuming it has the same len as `groups` argument of `fit` method. It is also used to
            compute the `names` field, which appears as a key in each of the values of the `importances` sub-field of
            the response `raw` field.
        categorical_names
            Keys are feature column indices in the `background_data` matrix (see `fit`). Each value contains strings
            with the names of the categories for the feature. Used to select the method for background data
            summarisation (if specified, subsampling is performed as opposed to kmeans clustering). In the future it
            may be used for visualisation.
        task
            Can have values 'classification' and 'regression'. It is only used to set the contents of the `prediction`
            field in the `data['raw']` response field.
        seed
            Fixes the random number stream, which influences which subsets are sampled during shap value estimation.
        """

        super().__init__(meta=copy.deepcopy(DEFAULT_META_KERNEL_SHAP))

        self.link = link
        self.predictor = predictor
        self.feature_names = feature_names if feature_names else []
        self.categorical_names = categorical_names if categorical_names else {}
        self.task = task
        self.seed = seed
        self._update_metadata({"task": self.task})

        # if the user specifies groups but no names, the groups are automatically named
        self.use_groups = False
        # changes if feature groups indices are passed but not names
        self.create_group_names = False
        # if sum of groups entries matches first dimension as opposed to second, warn user
        self.transposed = False
        # if weights are not correctly specified, they are ignored
        self.ignore_weights = False
        # sums up shap values for each level of categorical var
        self.summarise_result = False
        # selects a subset of the background data to avoid excessively slow runtimes
        self.summarise_background = False
        # checks if it has been fitted:
        self._fitted = False

    def _check_inputs(self,
                      background_data: Union[shap.common.Data, pd.DataFrame, np.ndarray, sparse.spmatrix],
                      group_names: Union[Tuple, List, None],
                      groups: Optional[List[Union[Tuple[int], List[int]]]],
                      weights: Union[Union[List[float], Tuple[float]], np.ndarray, None]) -> None:
        """
        If user specifies parameter grouping, then we check input is correct or inform
        them if the settings they put might not behave as expected.
        """

        if isinstance(background_data, shap.common.Data):
            # don't provide checks for situations where the user passes
            # the data object directly
            if not self.summarise_background:
                self.use_groups = False
                return
            # if summarisation took place, we do the checks to ensure everything is correct
            else:
                background_data = background_data.data

        if isinstance(background_data, np.ndarray) and background_data.ndim == 1:
            background_data = np.atleast_2d(background_data)

        if background_data.shape[0] > KERNEL_SHAP_BACKGROUND_THRESHOLD:
            msg = "Large datasets can cause slow runtimes for shap. The background dataset " \
                  "provided has {} records. Consider passing a subset or allowing the algorithm " \
                  "to automatically summarize the data by setting the summarise_background=True or" \
                  "setting summarise_background to 'auto' which will default to {} samples!"
            logger.warning(msg.format(background_data.shape[0], KERNEL_SHAP_BACKGROUND_THRESHOLD))

        if group_names and not groups:
            logger.info(
                "Specified group_names but no corresponding sequence 'groups' with indices "
                "for each group was specified. All groups will have len=1."
            )
            if not len(group_names) in background_data.shape:
                msg = "Specified {} group names but data dimension is {}. When grouping " \
                      "indices are not specifies the number of group names should equal " \
                      "one of the data dimensions! Igoring grouping inputs!"
                logger.warning(msg.format(len(group_names), background_data.shape))
                self.use_groups = False

        if groups and not group_names:
            logger.warning(
                "No group names specified but groups specified! Automatically "
                "assigning 'group_' name for every index group specified!")
            if self.feature_names:
                n_groups = len(groups)
                n_features = len(self.feature_names)
                if n_features != n_groups:
                    msg = "Number of feature names specified did not match the number of groups." \
                          "Specified {} groups and {} features names. Creating default names for " \
                          "specified groups"
                    logger.warning(msg.format(n_groups, n_features))
                    self.create_group_names = True
                else:
                    group_names = self.feature_names
            else:
                self.create_group_names = True

        if groups:
            if not (isinstance(groups[0], tuple) or isinstance(groups[0], list)):
                msg = "groups should be specified as List[Union[Tuple[int], List[int]]] where each " \
                      "sublist represents a group and int represent group instance. Specified group " \
                      "elements have type {}. Ignoring grouping inputs!"
                logger.warning(msg.format(type(groups[0])))
                self.use_groups = False

            expected_dim = sum(len(g) for g in groups)
            if background_data.ndim == 1:
                actual_dim = background_data.shape[0]
            else:
                actual_dim = background_data.shape[1]
            if expected_dim != actual_dim:
                if background_data.shape[0] == expected_dim:
                    logger.warning(
                        "The sum of the group indices list did not match the "
                        "data dimension along axis=1 but matched dimension "
                        "along axis=0. Consider transposing the data!"
                    )
                    self.transposed = True
                else:
                    msg = "The sum of the group sizes specified did not match the number of features. " \
                          "Sum of group sizes: {}. Number of features: {}. Ignoring grouping inputs!"
                    logger.warning(msg.format(expected_dim, actual_dim))
                    self.use_groups = False

            if group_names:
                n_groups = len(groups)
                n_group_names = len(group_names)
                if n_group_names != n_groups:
                    msg = "The number of group names specified does not match the number of groups. " \
                          "Received {} groups and {} names! Ignoring grouping inputs!"
                    logger.warning(msg.format(n_groups, n_group_names))
                    self.use_groups = False

        if weights is not None:
            if background_data.ndim == 1 or background_data.shape[0] == 1:
                logger.warning(
                    "Specified weights but the background data has only one record. "
                    "Weights will be ignored!"
                )
                self.ignore_weights = True
            else:
                data_dim = background_data.shape[0]
                feat_dim = background_data.shape[1]
                weights_dim = len(weights)
                if data_dim != weights_dim:
                    if not (feat_dim == weights_dim and self.transposed):
                        msg = "The number of weights specified did not match data dimension. " \
                              "Number of weights: {}. Number of datapoints: {}. Weights will " \
                              "be ignored!"
                        logger.warning(msg.format(weights_dim, data_dim))
                        self.ignore_weights = True

            # NB: we have already summarised the data at this point
            if self.summarise_background:

                weights_dim = len(weights)
                if background_data.ndim == 1:
                    n_background_samples = 1
                else:
                    if not self.transposed:
                        n_background_samples = background_data.shape[0]
                    else:
                        n_background_samples = background_data.shape[1]

                if weights_dim != n_background_samples:
                    msg = "The number of weights vector provided ({}) did not match the number of " \
                          "summary data points ({}). The weights provided will be ignored!"
                    logger.warning(msg.format(weights_dim, n_background_samples))

                    self.ignore_weights = True

    def _summarise_background(self,
                              background_data: Union[shap.common.Data, pd.DataFrame, np.ndarray, sparse.spmatrix],
                              n_background_samples: int) -> \
            Union[shap.common.Data, pd.DataFrame, np.ndarray, sparse.spmatrix]:
        """
        Summarises the background data to n_background_samples in order to reduce the computational cost. If the
        background data is a shap.common.Data object, no summarisation is performed.

        Returns
        -------
            If the user has specified grouping, then the input object is subsampled and an object of the same
            type is returned. Otherwise, a shap.common.Data object containing the result of a kmeans algorithm
            is wrapped in a shap.common.DenseData object and returned. The samples are weighted according to the
            frequency of the occurence of the clusters in the original data.
        """

        if isinstance(background_data, shap.common.Data):
            msg = "Received option to summarise the data but the background_data object " \
                  "was an instance of shap.common.Data. No summarisation will take place!"
            logger.warning(msg)
            return background_data

        if background_data.ndim == 1:
            msg = "Received option to summarise the data but the background_data object only had " \
                  "one record with {} features. No summarisation will take place!"
            logger.warning(msg.format(len(background_data)))
            return background_data

        self.summarise_background = True

        # if the input is sparse, we assume there are categorical variables and use random sampling, not kmeans
        if self.use_groups or self.categorical_names or isinstance(background_data, sparse.spmatrix):
            return shap.sample(background_data, nsamples=n_background_samples)
        else:
            logger.info(
                "When summarising with kmeans, the samples are weighted in proportion to their "
                "cluster occurrence frequency. Please specify a different weighting of the samples "
                "through the by passing a weights of len=n_background_samples to the constructor!"
            )
            return shap.kmeans(background_data, n_background_samples)

    @methdispatch
    def _get_data(self,
                  background_data: Union[shap.common.Data, pd.DataFrame, np.ndarray, sparse.spmatrix],
                  group_names: Sequence,
                  groups: List[Sequence[int]],
                  weights: Sequence[Union[float, int]],
                  **kwargs):
        """
        Groups the data if grouping options are specified, returning a shap.common.Data object in this
        case. Otherwise, the original data is returned and handled internally by the shap library.
        """

        raise TypeError("Type {} is not supported for background data!".format(type(background_data)))

    @_get_data.register(shap.common.Data)
    def _(self, background_data, *args, **kwargs) -> shap.common.Data:
        """
        Initialises background data if user passes a shap.common.Data object as input.
        Input  is returned as this is a native object to shap.

        Note: If self.summarise_background = True, then a shap.common.Data object is
        returned if the user passed a shap.common.data object to fit or didn't specify groups.
        """

        group_names, groups, weights = args
        if weights is not None and self.summarise_background:
            if not self.ignore_weights:
                background_data.weights = weights
            if self.use_groups:
                background_data.groups = groups
                background_data.group_names = group_names
                background_data.group_size = len(groups)

        return background_data

    @_get_data.register(np.ndarray)  # type: ignore
    def _(self, background_data, *args, **kwargs) -> Union[np.ndarray, shap.common.Data]:
        """
        Initialises background data if the user passes a numpy array as input.
        If the user specifies feature grouping then a shap.common.DenseData object
        is returned. Weights are handled separately to avoid triggering assertion
        correct inside shap library. Otherwise, the original data is returned and
        is handled by the shap library internally.
        """

        group_names, groups, weights = args
        new_args = (group_names, groups, weights) if weights is not None else (group_names, groups)
        if self.use_groups:
            return DenseData(background_data, *new_args)
        else:
            return background_data

    @_get_data.register(sparse.spmatrix)  # type: ignore
    def _(self, background_data, *args, **kwargs) -> Union[shap.common.Data, sparse.spmatrix]:
        """
        Initialises background data if user passes a sparse matrix as input. If the
        user specifies feature grouping, then the sparse array is converted to a dense
        array. Otherwise, the original array is returned and handled internally by shap
        library.
        """

        group_names, groups, weights = args
        new_args = (group_names, groups, weights) if weights is not None else (group_names, groups)

        if self.use_groups:
            logger.warning(
                "Grouping is not currently compatible with sparse matrix inputs. "
                "Converting background data sparse array to dense matrix."
            )
            background_data = background_data.toarray()
            return DenseData(
                background_data,
                *new_args,
            )

        return background_data

    @_get_data.register(pd.core.frame.DataFrame)  # type: ignore
    def _(self, background_data, *args, **kwargs) -> Union[shap.common.Data, pd.core.frame.DataFrame]:
        """
        Initialises background data if user passes a pandas.core.frame.DataFrames as input.
        If the user has specified groups and given a data frame, initialise shap.common.DenseData
        explicitly as this is not handled by shap library internally. Otherwise, data initialisation,
        is left to the shap library.
        """

        _, groups, weights = args
        new_args = (groups, weights) if weights is not None else (groups,)
        if self.use_groups:
            logger.info("Group names are specified by column headers, group_names will be ignored!")
            keep_index = kwargs.get("keep_index", False)
            if keep_index:
                return DenseDataWithIndex(
                    background_data.values,
                    list(background_data.columns),
                    background_data.index.values,
                    background_data.index.name,
                    *new_args,
                )
            else:
                return DenseData(
                    background_data.values,
                    list(background_data.columns),
                    *new_args,
                )
        else:
            return background_data

    @_get_data.register(pd.core.frame.Series)  # type: ignore
    def _(self, background_data, *args, **kwargs) -> Union[shap.common.Data, pd.core.frame.Series]:
        """
        Initialises background data if user passes a pandas Series object as input.
        Original object is returned as this is initialised internally by shap is there
        is no group structure specified. Otherwise the a shap.common.DenseData object
        is initialised.
        """

        _, groups, _ = args
        if self.use_groups:
            return DenseData(
                background_data.values.reshape(1, len(background_data)),
                list(background_data.index),
                groups,
            )

        return background_data

    def _update_metadata(self, data_dict: dict, params: bool = False) -> None:
        """
        This function updates the metadata of the explainer using the data from
        the data_dict. If the params option is specified, then each key-value
        pair is added to the metadata 'params' dictionary only if the key is
        included in KERNEL_SHAP_PARAMS.

        Parameters
        ----------
        data_dict
            Dictionary containing the data to be stored in the metadata.
        params
            If True, the method updates the 'params' attribute of the metatadata.
        """

        if params:
            for key in data_dict.keys():
                if key not in KERNEL_SHAP_PARAMS:
                    continue
                else:
                    self.meta['params'].update([(key, data_dict[key])])
        else:
            self.meta.update(data_dict)

    def fit(self,  # type: ignore
            background_data: Union[np.ndarray, sparse.spmatrix, pd.DataFrame, shap.common.Data],
            summarise_background: Union[bool, str] = False,
            n_background_samples: int = KERNEL_SHAP_BACKGROUND_THRESHOLD,
            group_names: Union[Tuple[str], List[str], None] = None,
            groups: Optional[List[Union[Tuple[int], List[int]]]] = None,
            weights: Union[Union[List[float], Tuple[float]], np.ndarray, None] = None,
            **kwargs) -> "KernelShap":
        """
        This takes a background dataset (usually a subsample of the training set) as an input along with several
        user specified options and initialises a KernelShap explainer. The runtime of the algorithm depends on the
        number of samples in this dataset and on the number of features in the dataset. To reduce the size of the
        dataset, the summarise_background option and n_background_samples should be used. To reduce the feature
        dimensionality, encoded categorical variables can be treated as one during the feature perturbation process;
        this decreases the effective feature dimensionality, can reduce the variance of the shap values estimation and
        reduces slightly the number of calls to the predictor. Further runtime savings can be achieved by changing the
        nsamples parameter in the call to explain. Runtime reduction comes with an accuracy tradeoff, so it is better
        to experiment with a runtime reduction method and understand results stability before using the system.

        Parameters
        -----------
        background_data
            Data used to estimate feature contributions and baseline values for force plots. The rows of the
            background data should represent samples and the columns features.
        summarise_background
            A large background dataset impacts the runtime and memory footprint of the algorithm. By setting
            this argument to True, only n_background_samples from the provided data are selected. If group_names or
            groups arguments are specified, the algorithm assumes that the data contains categorical variables so
            the records are selected uniformly at random. Otherwise, shap.kmeans (a wrapper around sklearn kmeans
            implementation) is used for selection. If set to 'auto', a default of BACKGROUND_WARNING_THRESHOLD
            samples is selected.
        n_background_samples
            The number of samples to keep in the background dataset if summarise_background=True.
        groups:
            A list containing sublists specifying the indices of features belonging to the same group.
        group_names:
            If specified, this array is used to treat groups of features as one during feature perturbation.
            This feature can be useful, for example, to treat encoded categorical variables as one and can
            result in computational savings (this may require adjusting the nsamples parameter).
        weights:
            A sequence or array of weights. This is used only if grouping is specified and assigns a weight
            to each point in the dataset.
        kwargs:
            Expected keyword arguments include "keep_index" and should be used if a data frame containing an
            index column is passed to the algorithm.
        """

        np.random.seed(self.seed)

        self._fitted = True
        # user has specified variable groups
        use_groups = groups is not None or group_names is not None
        self.use_groups = use_groups

        if summarise_background:
            if isinstance(summarise_background, str):
                if not isinstance(background_data, shap.common.Data):
                    n_samples = background_data.shape[0]
                else:
                    n_samples = background_data.data.shape[0]
                n_background_samples = min(n_samples, KERNEL_SHAP_BACKGROUND_THRESHOLD)
            background_data = self._summarise_background(background_data, n_background_samples)

        # check user inputs to provide warnings if input is incorrect
        self._check_inputs(background_data, group_names, groups, weights)
        if self.create_group_names:
            group_names = ['group_{}'.format(i) for i in range(len(groups))]
        # disable grouping or data weights if inputs are not correct
        if self.ignore_weights:
            weights = None
        if not self.use_groups:
            group_names, groups = None, None
        else:
            self.feature_names = group_names

        # perform grouping if requested by the user
        self.background_data = self._get_data(background_data, group_names, groups, weights, **kwargs)
        self._explainer = shap.KernelExplainer(
            self.predictor,
            self.background_data,
            link=self.link,
        )  # type: shap.KernelExplainer
        self.expected_value = self._explainer.expected_value
        if not self._explainer.vector_out:
            logger.warning(
                "Predictor returned a scalar value. Ensure the output represents a probability or decision score "
                "as opposed to a classification label!"
            )

        # update metadata
        params = {
            'groups': groups,
            'group_names': group_names,
            'weights': weights,
            'kwargs': kwargs,
            'summarise_background': self.summarise_background,
            'grouped': self.use_groups,
            'transpose': self.transposed,
        }
        self._update_metadata(params, params=True)

        return self

    def explain(self,
                X: Union[np.ndarray, pd.DataFrame, sparse.spmatrix],
                summarise_result: bool = False,
                cat_vars_start_idx: Tuple[int] = None,
                cat_vars_enc_dim: Tuple[int] = None,
                **kwargs) -> Explanation:
        """
        Explains the instances in the array X.

        Parameters
        ----------
        X
            Instances to be explained.
        summarise_result
            Specifies whether the shap values corresponding to dimensions of encoded categorical variables should be
            summed so that a single shap value is returned for each categorical variable. Both the start indices of
            the categorical variables (`cat_vars_start_idx`) and the encoding dimensions (`cat_vars_enc_dim`)
            have to be specified
        cat_vars_start_idx
            The start indices of the categorical variables. If specified, cat_vars_enc_dim should also be specified.
        cat_vars_enc_dim
            The length of the encoding dimension for each categorical variable.
        kwargs
            Keyword arguments specifying explain behaviour. Valid arguments are:

                -`nsamples`: controls the number of predictor calls and therefore runtime.
                -`l1_reg`: controls the explanation sparsity.

            For more details, please see the shap library documentation_ .

            .. _documentation https://shap.readthedocs.io/en/latest/.

        Returns
        -------
        explanation
            An explanation object containing the algorithm results.
        """

        if not self._fitted:
            raise TypeError(
                "Called explain on an unfitted object! Please fit the "
                "explainer using the .fit method first!"
            )

        # convert data to dense format if sparse
        if self.use_groups and isinstance(X, sparse.spmatrix):
            X = X.toarray()

        shap_values = self._explainer.shap_values(X, **kwargs)
        expected_value = self.expected_value
        # for scalar model outputs a single numpy array is returned
        if isinstance(shap_values, np.ndarray):
            shap_values = [shap_values]
        if isinstance(expected_value, float):
            expected_value = [self.expected_value]

        explanation = self.build_explanation(
            X,
            shap_values,
            expected_value,
            summarise_result=summarise_result,
            cat_vars_start_idx=cat_vars_start_idx,
            cat_vars_enc_dim=cat_vars_enc_dim,
        )

        return explanation

    def build_explanation(self,
                          X: Union[np.ndarray, pd.DataFrame, sparse.spmatrix],
                          shap_values: List[np.ndarray],
                          expected_value: List[float],
                          **kwargs) -> Explanation:
        """
        Create an explanation object.

        Parameters
        ----------
        X
            Array of instances to be explained.
        shap_values
            Each entry is a n_instances x n_features array, and the length of the list equals the dimensionality
            of the predictor output. The rows of each array correspond to the shap values for the instances with
            the corresponding row index in X
        expected_value
            A list containing the expected value of the prediction for each class.

        Returns
        -------
            An explanation containing a meta field with basic classifier metadata.

        """

        cat_vars_start_idx = kwargs.get('cat_vars_start_idx', ())  # type: Tuple[int]
        cat_vars_enc_dim = kwargs.get('cat_vars_enc_dim', ())  # type: Tuple[int]
        summarise_result = kwargs.get('summarise_result', False)  # type: bool

        self._check_result_summarisation(summarise_result, cat_vars_start_idx, cat_vars_enc_dim)
        if self.summarise_result:
            summarised_shap = []
            for shap_array in shap_values:
                summarised_shap.append(sum_categories(shap_array, cat_vars_start_idx, cat_vars_enc_dim))
            shap_values = summarised_shap

        # TODO: DEFINE COMPLETE SCHEMA FOR THE METADATA (ONGOING)
        # TODO: Plotting default should be same space as the explanation? How do we figure out what space they
        #  explain in?

        raw_predictions = self._explainer.linkfv(self.predictor(X))

        if self.task != 'regression':
            argmax_pred = np.argmax(np.atleast_2d(raw_predictions), axis=1)
        else:
            argmax_pred = []
        importances = rank_by_importance(shap_values, feature_names=self.feature_names)

        if isinstance(X, sparse.spmatrix):
            X = X.toarray()
        else:
            X = np.array(X)

        # output explanation dictionary
        data = copy.deepcopy(DEFAULT_DATA_KERNEL_SHAP)
        data.update(
            shap_values=shap_values,
            expected_value=expected_value,
            link=self.link,
            categorical_names=self.categorical_names,
            feature_names=self.feature_names
        )
        data['raw'].update(
            raw_prediction=raw_predictions,
            prediction=argmax_pred,
            instances=X,
            importances=importances
        )

        return Explanation(meta=copy.deepcopy(self.meta), data=data)

    def _check_result_summarisation(self,
                                    summarise_result: bool,
                                    cat_vars_start_idx: Tuple[int],
                                    cat_vars_enc_dim: Tuple[int]) -> None:
        """
        This function checks whether the result summarisation option is correct given the inputs and explainer setup.

        Parameters
        ----------
        summarise_result:
            See `explain` documentation.
        cat_vars_start_idx:
            See `explain` documentation.
        cat_vars_enc_dim:
            See `explain` documentation.
        """

        self.summarise_result = summarise_result
        if summarise_result:
            if not cat_vars_start_idx or not cat_vars_enc_dim:
                logger.warning(
                    "Results cannot be summarised as either the"
                    "start indices for categorical variables or"
                    "the encoding dimensions were not passed!"
                )
                self.summarise_result = False
            elif self.use_groups:
                logger.warning(
                    "Specified both groups as well as summarisation for categorical variables. "
                    "By grouping, only one shap value is estimated for each categorical variable. "
                    "Summarisation is thus not necessary!"
                )
                self.summarise_result = False

        self._update_metadata({"summarise_result": self.summarise_result}, params=True)
