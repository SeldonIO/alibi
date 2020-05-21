import copy
import logging
import shap

import numpy as np
import pandas as pd

from alibi.api.defaults import DEFAULT_META_KERNEL_SHAP, DEFAULT_DATA_KERNEL_SHAP, DEFAULT_META_TREE_SHAP, \
    DEFAULT_DATA_TREE_SHAP
from alibi.api.interfaces import Explanation, Explainer, FitMixin
from alibi.utils.wrappers import methdispatch
from functools import partial
from scipy import sparse
from scipy.special import expit
from shap.common import DenseData, DenseDataWithIndex
from typing import Any, Callable, Dict, List, Optional, Sequence, Union, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import catboost  # noqa F401

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
    Given the shap values estimated for a multi-output model, this function ranks
    features according to their importance. The feature importance is the average
    absolute value for a given feature.

    Parameters
    ----------
    shap_values
        Each element corresponds to a samples x features array of shap values corresponding
        to each model output.
    feature_names
        Each element is the name of the column with the corresponding index in each of the
        arrays in the `shap_values` list.

    Returns
    -------
    importances
        A dictionary of the form::

            {
                '0': {'ranked_effect': array([0.2, 0.5, ...]), 'names': ['feat_3', 'feat_5', ...]},
                '1': {'ranked_effect': array([0.3, 0.2, ...]), 'names': ['feat_6', 'feat_1', ...]},
                ...
                'aggregated': {'ranked_effect': array([0.9, 0.7, ...]), 'names': ['feat_3', 'feat_6', ...]}
            }

        The keys of the first level represent the index of the model output. The feature effects in
        `ranked_effect` and the corresponding feature names in `names` are sorted from highest (most
        important) to lowest (least important). The values in the `aggregated` field are obtained by
        summing the shap values for all the model outputs and then computing the effects. Given an
        output, the effects are defined as the average magnitude of the shap values across the instances
        to be explained.
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


def sum_categories(values: np.ndarray, start_idx: Sequence[int], enc_feat_dim: Sequence[int]):
    """
    This function is used to reduce specified slices in a two- or three- dimensional tensor.

    For two-dimensional `values` arrays, for each entry in start_idx, the function sums the
    following k columns where k is the corresponding entry in the enc_feat_dim sequence.
    The columns whose indices are not in start_idx are left unchanged. This arises when the slices
    contain the shap values for each dimension of an encoded categorical variable and a single shap
    value for each variable is desired.

    For three-dimensional `values` arrays, the reduction is applied for each rank 2 subtensor, first along
    the column dimension and then across the row dimension. This arises when summarising shap interaction values.
    Each rank 2 tensor is a E x E matrix of shap interaction values, where E is the dimension of the data after
    one-hot encoding. The result of applying the reduction yields a rank 2 tensor of dimension F x F, where F is the
    number of features (ie, the feature dimension of the data matrix before encoding). By applying this transformation,
    a single value describing the interaction of categorical features i and j and a single value describing the
    intearction of j and i is returned.

    Parameters
    ----------
    values
        A two or three dimensional array to be reduced, as described above.
    start_idx
        The start indices of the columns to be summed.
    enc_feat_dim
        The number of columns to be summed, one for each start index.
    Returns
    -------
    new_values
        An array whose columns have been summed according to the entries in `start_idx` and `enc_feat_dim`.
    """

    if start_idx is None or enc_feat_dim is None:
        raise ValueError("Both the start indices or the encoding dimension need to be specified!")

    if not len(enc_feat_dim) == len(start_idx):
        raise ValueError("The lengths of the sequences of start indices and encodings must be equal!")

    n_encoded_levels = sum(enc_feat_dim)
    if n_encoded_levels > values.shape[-1]:
        raise ValueError("The sum of the encoded features dimensions exceeds data dimension!")

    if len(values.shape) not in (2, 3):
        raise ValueError(
            f"Shap value summarisation can only be applied to tensors of shap values (dim=2) or shap "
            f"interaction values (dim=3). The tensor to be summarised had dimension {values.shape}!"
        )

    def _get_slices(start: Sequence[int], dim: Sequence[int], arr_trailing_dim: int) -> List[int]:
        """
        Given start indices, encoding dimensions and the array trailing shape, this function returns
        an array where contiguous numbers are slices. This array is used to reduce along an axis
        only the slices `slice(start[i], start[i] + dim[i], 1)` from a tensor and leave all other slices
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
        A wrapper around the `shap.KernelExplainer` class. It extends the current `shap` library functionality
        by allowing the user to specify variable groups in order to treat one-hot encoded categorical as one during
        sampling. The user can also specify whether to aggregate the `shap` values estimate for the encoded levels
        of categorical variables as an optional argument to `explain`, if grouping arguments are not passed to `fit`.

        Parameters
        ----------
        predictor
            A callable that takes as an input a samples x features array and outputs a samples x n_outputs
            model outputs. The n_outputs should represent model output in margin space. If the model outputs
            probabilities, then the link should be set to 'logit' to ensure correct force plots.
        link
            Valid values are `'identity'` or `'logit'`. A generalized linear model link to connect the feature
            importance values to the model output. Since the feature importance values, :math:`\phi`, sum up to the
            model output, it often makes sense to connect them to the ouput with a link function where
            :math:`link(output - expected\_value) = sum(\phi)`. Therefore, for a model which outputs probabilities,
            `link='logit'` makes the feature effects have log-odds (evidence) units and `link='identity'` means that the
            feature effects have probability units. Please see this `example`_ for an in-depth discussion about the
            semantics of explaining the model in the probability or margin space.

            .. _example:
               https://github.com/slundberg/shap/blob/master/notebooks/kernel_explainer/Squashing%20Effect.ipynb

        feature_names
            Used to infer group names when categorical data is treated by grouping and `group_names` input to `fit`
            is not specified, assuming it has the same length as the `groups` argument of `fit` method. It is also used
            to compute the `names` field, which appears as a key in each of the values of
            `explanation.data['raw']['importances']`.
        categorical_names
            Keys are feature column indices in the `background_data` matrix (see `fit`). Each value contains strings
            with the names of the categories for the feature. Used to select the method for background data
            summarisation (if specified, subsampling is performed as opposed to k-means clustering). In the future it
            may be used for visualisation.
        task
            Can have values `'classification'` and `'regression'`. It is only used to set the contents of
            `explanation.data['raw']['prediction']`
        seed
            Fixes the random number stream, which influences which subsets are sampled during shap value estimation.
        """  # noqa W605

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
        background data is a `shap.common.Data object`, no summarisation is performed.

        Returns
        -------
            If the user has specified grouping, then the input object is subsampled and an object of the same
            type is returned. Otherwise, a `shap.common.Data` object containing the result of a k-means algorithm
            is wrapped in a `shap.common.DenseData` object and returned. The samples are weighted according to the
            frequency of the occurrence of the clusters in the original data.
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
        Initialises background data if the user passes a `shap.common.Data` object as input.

        Notes
        _____

        If `self.summarise_background = True`, then a `shap.common.Data` object is
        returned if the user passed a `shap.common.Data` object to `fit` or didn't specify groups.
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
        Initialises background data if the user passes an `np.ndarray` object as input.
        If the user specifies feature grouping then a `shap.common.DenseData` object
        is returned. Weights are handled separately to avoid triggering assertion
        correct inside `shap` library. Otherwise, the original data is returned and
        is handled by the `shap` library internally.
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
        Initialises background data if the user passes a sparse matrix as input. If the
        user specifies feature grouping, then the sparse array is converted to a dense
        array. Otherwise, the original array is returned and handled internally by `shap`
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
        Initialises background data if the user passes a `pandas.core.frame.DataFrame` as input.
        If the user has specified groups and given a data frame, it initialises a `shap.common.DenseData`
        object explicitly as this is not handled by `shap` library internally. Otherwise, data initialisation,
        is left to the `shap` library.
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
        Initialises background data if the user passes a `pandas.Series` object as input.
        Original object is returned as this is initialised internally by `shap` is there
        is no group structure specified. Otherwise, a `shap.common.DenseData` object
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
        the `data_dict`. If the params option is specified, then each key-value
        pair is added to the metadata `'params'` dictionary only if the key is
        included in `KERNEL_SHAP_PARAMS`.

        Parameters
        ----------
        data_dict
            Dictionary containing the data to be stored in the metadata.
        params
            If True, the method updates the `'params'` attribute of the metatadata.
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
        user specified options and initialises a `KernelShap` explainer. The runtime of the algorithm depends on the
        number of samples in this dataset and on the number of features in the dataset. To reduce the size of the
        dataset, the `summarise_background` option and `n_background_samples` should be used. To reduce the feature
        dimensionality, encoded categorical variables can be treated as one during the feature perturbation process;
        this decreases the effective feature dimensionality, can reduce the variance of the shap values estimation and
        reduces slightly the number of calls to the predictor. Further runtime savings can be achieved by changing the
        `nsamples` parameter in the call to explain. Runtime reduction comes with an accuracy trade-off, so it is better
        to experiment with a runtime reduction method and understand results stability before using the system.

        Parameters
        -----------
        background_data
            Data used to estimate feature contributions and baseline values for force plots. The rows of the
            background data should represent samples and the columns features.
        summarise_background
            A large background dataset impacts the runtime and memory footprint of the algorithm. By setting
            this argument to `True`, only `n_background_samples` from the provided data are selected. If group_names or
            groups arguments are specified, the algorithm assumes that the data contains categorical variables so
            the records are selected uniformly at random. Otherwise, `shap.kmeans` (a wrapper around `sklearn` k-means
            implementation) is used for selection. If set to `'auto'`, a default of
            `KERNEL_SHAP_BACKGROUND_THRESHOLD` samples is selected.
        n_background_samples
            The number of samples to keep in the background dataset if `summarise_background=True`.
        groups:
            A list containing sub-lists specifying the indices of features belonging to the same group.
        group_names:
            If specified, this array is used to treat groups of features as one during feature perturbation.
            This feature can be useful, for example, to treat encoded categorical variables as one and can
            result in computational savings (this may require adjusting the `nsamples` parameter).
        weights:
            A sequence or array of weights. This is used only if grouping is specified and assigns a weight
            to each point in the dataset.
        kwargs:
            Expected keyword arguments include `keep_index` (bool) and should be used if a data frame containing an
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
                cat_vars_start_idx: Sequence[int] = None,
                cat_vars_enc_dim: Sequence[int] = None,
                **kwargs) -> Explanation:
        """
        Explains the instances in the array `X`.

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
            The start indices of the categorical variables. If specified, `cat_vars_enc_dim` should also be specified.
        cat_vars_enc_dim
            The length of the encoding dimension for each categorical variable. If specified `cat_vars_start_idx` should
            also be specified.
        kwargs
            Keyword arguments specifying explain behaviour. Valid arguments are:

                - `nsamples`: controls the number of predictor calls and therefore runtime.

                - `l1_reg`: the algorithm is exponential in the feature dimension. If set to `auto` the algorithm will \
                first run a feature selection algorithm to select the top features, provided the fraction of sampled \
                sets of missing features is less than 0.2 from the number of total subsets. The Akaike Information \
                Criterion is used in this case. See our examples for more details about available settings for this \
                parameter. Note that by first running a feature selection step, the shapley values of the remainder of \
                the features will be different to those estimated from the entire set.

                For more details, please see the shap library `documentation`_ .

                .. _documentation:
                   https://shap.readthedocs.io/en/latest/.

        Returns
        -------
        explanation
            An explanation object containing the algorithm results.
        """  # noqa W605

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
        Create an explanation object.  If output summarisation is required and all inputs necessary for this operation
        are passed, the raw shap values are summed first so that a single shap value is returned for each categorical
        variable, as opposed to a shap value per dimension of categorical variable encoding.

        Parameters
        ----------
        X
            Instances to be explained.
        shap_values
            Each entry is a n_instances x n_features array, and the length of the list equals the dimensionality
            of the predictor output. The rows of each array correspond to the shap values for the instances with
            the corresponding row index in `X`. The length of the list equals the number of model outputs.
        expected_value
            A list containing the expected value of the prediction for each class. Its length should be equal to that of
            `shap_values`.

        Returns
        -------
        explanation
            An explanation object containing the shap values and prediction in the `data` field, along with a `meta`
            field containing additional data. See usage `examples`_ for details.

            .. _examples:
               https://docs.seldon.io/projects/alibi/en/latest/methods/KernelSHAP.html

        """

        # TODO: DEFINE COMPLETE SCHEMA FOR THE METADATA (ONGOING)
        # TODO: Plotting default should be same space as the explanation? How do we figure out what space they
        #  explain in?

        cat_vars_start_idx = kwargs.get('cat_vars_start_idx', ())  # type: Sequence[int]
        cat_vars_enc_dim = kwargs.get('cat_vars_enc_dim', ())  # type: Sequence[int]
        summarise_result = kwargs.get('summarise_result', False)  # type: bool
        if summarise_result:
            self._check_result_summarisation(summarise_result, cat_vars_start_idx, cat_vars_enc_dim)
        if self.summarise_result:
            summarised_shap = []
            for shap_array in shap_values:
                summarised_shap.append(sum_categories(shap_array, cat_vars_start_idx, cat_vars_enc_dim))
            shap_values = summarised_shap

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
            expected_value=np.array(expected_value),
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
        self._update_metadata({"summarise_result": self.summarise_result}, params=True)

        return Explanation(meta=copy.deepcopy(self.meta), data=data)

    def _check_result_summarisation(self,
                                    summarise_result: bool,
                                    cat_vars_start_idx: Sequence[int],
                                    cat_vars_enc_dim: Sequence[int]) -> None:
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
                    "Summarisation is not necessary!"
                )
                self.summarise_result = False


# TODO: Look into pyspark support requirements if requested
# TODO: catboost.Pool not supported for fit stage (due to summarisation) but can do if there is a user need

TREE_SHAP_PARAMS = [
    'model_output',
    'summarise_background',
    'summarise_result',
    'approximate',
    'interactions',
    'explain_loss',
    'algorithm',
    'kwargs'
]
TREE_SHAP_BACKGROUND_WARNING_THRESHOLD = 1000
TREE_SHAP_MODEL_OUTPUT = ['raw', 'probability', 'probability_doubled', 'log_loss']


class TreeShap(Explainer, FitMixin):

    def __init__(self,
                 predictor: Any,
                 model_output: str = 'raw',
                 feature_names: Union[List[str], Tuple[str], None] = None,
                 categorical_names: Optional[Dict[int, List[str]]] = None,
                 task: str = 'classification',
                 seed: int = None):
        """
        A wrapper around the `shap.TreeExplainer` class. It adds the following functionality:

            1. Input summarisation options to allow control over background dataset size and hence runtime
            2. Output summarisation for sklearn models with one-hot encoded categorical variables.

        Users are strongly encouraged to familiarise themselves with the algorithm by reading the method
        overview in the documentation.

        Parameters
        ----------
        predictor
            A fitted model to be explained. XGBoost, LightGBM, CatBoost and most tree-based
            scikit-learn models are supported. In the future, Pyspark could also be supported.
            Please open an issue if this is a use case for you.
        model_output
            Supported values are: `'raw'`, `'probability'`, `'probability_doubled'`, `'log_loss'`: 

                - `'raw'`: the raw model of the output, which varies by task, is explained. This option \
                should always be used if the `fit` is called without arguments. It should also be set to compute \
                shap interaction values. For regression models it is the standard output, for binary classification \
                in XGBoost it is the log odds ratio. \

                - `'probability'`: the probability output is explained. This option should only be used if `fit` was \
                was called with the `background_data` argument set. The effect of specifying this parameter is that \
                the `shap` library will use this information to transform the shap values computed in margin space (aka \
                using the raw output) to shap values that sum to the probability output by the model plus the model expected \
                output probability. This requires knowledge of the type of output for `predictor` which is inferred by the \
                `shap` library from the model type (e.g., most sklearn models with exception of \
                `sklearn.tree.DecisionTreeClassifier`, `sklearn.ensemble.RandomForestClassifier`, \
                `sklearn.ensemble.ExtraTreesClassifier` output logits) or on the basis of the mapping implemented in \
                the `shap.TreeEnsemble` constructor. Only trees that output log odds and probabilities are supported \
                currently.  

                - `'probability_doubled'`: used for binary classification problem in situations where the model outputs \
                the logits/probabilities for the positive class but shap values for both outcomes are desired. This \
                option should be used only if `fit` was called with the `background_data` argument set. In \
                this case the expected value for the negative class is 1 - expected_value for positive class and \
                the shap values for the negative class are the negative values of the positive class shap values. \
                As before, the explanation happens in the margin space, and the shap values are subsequently adjusted. \
                convert the model output to probabilities. The same considerations as for `probability` apply for this \
                output type too. 

                - `'log_loss'`: logarithmic loss is explained. This option shoud be used only if `fit` was called with the \
                `background_data` argument set and requires specifying labels, `y`, when calling `explain`.  If the \
                objective is squared error, then the transformation :math:`(output - y)^2` is applied. For binary \
                cross-entropy objective, the transformation :math:`log(1 + exp(output)) - y * output` with  \
                :math:`y \in \{0, 1\}`. Currently only binary cross-entropy and squared error losses can be explained. \

        feature_names
            Used to compute the `names` field, which appears as a key in each of the values of the `importances`
            sub-field of the response `raw` field.
        categorical_names
            Keys are feature column indices. Each value contains strings with the names of the categories
            for the feature. Used to select the method for background data summarisation (if specified,
            subsampling is performed as opposed to kmeans clustering). In the future it may be used for visualisation.
        task
            Can have values `'classification'` and `'regression'`. It is only used to set the contents of the 
            `prediction` field in the `data['raw']` response field.

        Notes
        -----
        Tree SHAP is an additive attribution method so it is best suited to explaining output in margin space
        (the entire real line). For discussion related to explaining models in output vs probability space, please
        consult this  resource_.

        .. _resource: https://github.com/slundberg/shap/blob/master/notebooks/kernel_explainer/Squashing%20Effect.ipynb
    """  # noqa W605

        super().__init__(meta=copy.deepcopy(DEFAULT_META_TREE_SHAP))
        if model_output in TREE_SHAP_MODEL_OUTPUT:
            self.model_output = model_output
        else:
            logger.warning(f"Unrecognised model output {model_output}. Defaulting to model_output='raw'")
            self.model_output = 'raw'
        self.predictor = predictor
        self.feature_names = feature_names if feature_names else []
        self.categorical_names = categorical_names if categorical_names else {}
        self.task = task
        self.seed = seed

        # sums up shap values for each level of categorical var
        self.summarise_result = False
        # selects a subset of the background data to avoid excessively slow runtimes
        self.summarise_background = False
        # checks if it has been fitted:
        self._fitted = False

        self._update_metadata({"task": self.task})

    def _update_metadata(self, data_dict: dict, params: bool = False) -> None:
        """
        This function updates the metadata of the explainer using the data from
        the `data_dict`. If `params=True`, then each key-value pair is added
        to the metadata `params` dictionary only if the key is included in
        `TREE_SHAP_PARAMS`.

        Parameters
        ----------
        data_dict
            Dictionary containing the data to be stored in the metadata.
        params
            If `True`, the method updates the `['params']` attribute of the metadata.
        """

        if params:
            for key in data_dict.keys():
                if key not in TREE_SHAP_PARAMS:
                    continue
                else:
                    self.meta['params'].update([(key, data_dict[key])])
        else:
            self.meta.update(data_dict)

    def fit(self,  # type: ignore
            background_data: Union[np.ndarray, pd.DataFrame, None] = None,
            summarise_background: Union[bool, str] = False,
            n_background_samples: int = TREE_SHAP_BACKGROUND_WARNING_THRESHOLD,
            **kwargs) -> "TreeShap":
        """
        This function instantiates an explainer which can then be use to explain instances using the `explain` method.
        If no background dataset is passed, the explainer uses the path-dependent feature perturbation algorithm
        to explain the values. As such, only the model raw output can be explained and this should be reflected by
        passing `model_output='raw'` when instantiating the explainer. If a background dataset is passed, the
        interventional feature perturbation algorithm is used. Using this algorithm, probability outputs can also be
        explained. Additionally, if the `model_output='log_loss'` option is passed to the explainer constructor, then
        the model loss function can be explained by passing the labels as the `y` argument to the explain method.
        A limited number of loss functions are supported, as detailed in the constructor documentation.

        Parameters
        -----------
        background_data
            Data used to estimate feature contributions and baseline values for force plots. The rows of the
            background data should represent samples and the columns features.
        summarise_background
            A large background dataset may impact the runtime and memory footprint of the algorithm. By setting
            this argument to `True`, only `n_background_samples` from the provided data are selected. If the
            `categorical_names` argument has been passed to the constructor, subsampling of the data is used.
            Otherwise, `shap.kmeans` (a wrapper around `sklearn.kmeans` implementation) is used for selection.
            If set to `'auto'`, a default of `TREE_SHAP_BACKGROUND_WARNING_THRESHOLD` samples is selected.
        n_background_samples
            The number of samples to keep in the background dataset if `summarise_background=True`.
        """

        np.random.seed(self.seed)

        self._fitted = True
        if isinstance(background_data, pd.DataFrame):
            self.feature_names = list(background_data.columns)
        if background_data is not None:
            if isinstance(background_data, np.ndarray) and background_data.ndim == 1:
                background_data = np.atleast_2d(background_data)
            if summarise_background:
                n_samples = background_data.shape[0]
                if isinstance(summarise_background, str):
                    if n_samples > TREE_SHAP_BACKGROUND_WARNING_THRESHOLD:
                        background_data = self._summarise_background(
                            background_data,
                            TREE_SHAP_BACKGROUND_WARNING_THRESHOLD,
                        )
                else:
                    if n_samples > n_background_samples:
                        background_data = self._summarise_background(
                            background_data,
                            n_background_samples
                        )
            else:
                self._check_inputs(background_data)

        perturbation = 'interventional' if background_data is not None else 'tree_path_dependent'
        self.background_data = background_data
        self._explainer = shap.TreeExplainer(
            self.predictor,
            data=self.background_data,
            model_output=self.model_output,
            feature_perturbation=perturbation,
        )  # type: shap.TreeExplainer
        self.expected_value = self._explainer.expected_value

        self.scalar_output = False
        if self._explainer.model.num_outputs == 1:
            if self.task == 'classification':
                logger.warning(
                    "Predictor returned a scalar value. Ensure the output represents a probability or decision score "
                    "as opposed to a classification label!"
                )
            self.scalar_output = True

        # update metadata
        params = {
            'summarise_background': self.summarise_background,
            'algorithm': perturbation,
            'kwargs': kwargs,
        }
        self._update_metadata(params, params=True)

        return self

    @staticmethod
    def _check_inputs(background_data: Union[pd.DataFrame, np.ndarray]) -> None:
        """
        This function warns the user if slow runtime can occur due to the background dataset.

        Parameters
        ----------
        background_data
            Background dataset.
        """

        if background_data.shape[0] > TREE_SHAP_BACKGROUND_WARNING_THRESHOLD:
            msg = "Large datasets may cause slow runtimes for shap. The background dataset " \
                  "provided has {} records. If the runtime is too slow, consider passing a " \
                  "subset or allowing the algorithm to automatically summarize the data by " \
                  "setting the summarise_background=True or setting summarise_background to " \
                  "'auto' which will default to {} samples!"
            logger.warning(msg.format(background_data.shape[0], TREE_SHAP_BACKGROUND_WARNING_THRESHOLD))

    def _summarise_background(self,
                              background_data: Union[pd.DataFrame, np.ndarray],
                              n_background_samples: int) -> Union[np.ndarray, pd.DataFrame, shap.common.DenseData]:
        """
        Summarises the background data to n_background_samples in order to reduce the computational cost.

        Returns
        -------
            If the `categorical_names` argument to the constructor is specified, then an object of the same type as
            input containing only `n_background_samples` is returned. Otherwise, a `shap.common.Data` containing an
            `np.ndarray` object of `n_background_samples` in the `data` field is returned.

        """

        self.summarise_background = True

        if self.categorical_names:
            return shap.sample(background_data, nsamples=n_background_samples)
        else:
            return shap.kmeans(background_data, n_background_samples)

    def explain(self,
                X: Union[np.ndarray, pd.DataFrame, 'catboost.Pool'],
                y: Optional[np.ndarray] = None,
                interactions: bool = False,
                approximate: bool = False,
                check_additivity: bool = True,
                tree_limit: Optional[int] = None,
                summarise_result: bool = False,
                cat_vars_start_idx: Optional[Sequence[int]] = None,
                cat_vars_enc_dim: Optional[Sequence[int]] = None,
                **kwargs) -> "Explanation":
        """
        Explains the instances in `X`. `y` should be passed if the model loss function is to be explained,
        which can be useful in order to understand how various features affect model performance over
        time. This is only possible if the explainer has been fitted with a background dataset and
        requires setting `model_output='log_loss'`.

        Parameters
        ----------
        X
            Instances to be explained.
        y
            Labels corresponding to rows of `X`. Should be passed only if a background dataset was passed to the
            `fit` method.
        interactions
            If `True`, the shap value for every feature of every instance in `X` is decomposed into
            `X.shape[1] - 1` shap value interactions and one main effect. This is only supported if `fit` is called
            with `background_dataset=None`.
        approximate
            If `True`, an approximation to the shap values that does not account for feature order is computed. This
            was proposed by `Ando Sabaas`_ here . Check `this`_ resource for more details. This option is currently
            only supported for `xgboost` and `sklearn` models.

            .. _Ando Sabaas:
               https://github.com/andosa/treeinterpreter

            .. _this:
               https://static-content.springer.com/esm/art%3A10.1038%2Fs42256-019-0138-9/MediaObjects/42256_2019_138_MOESM1_ESM.pdf

        check_additivity
            If `True`, output correctness is ensured if `model_output='raw'` has been passed to the constructor.
        tree_limit
            Explain the output of a subset of the first `tree_limit` trees in an ensemble model.
        summarise_result
            This should be set to True only when some of the columns in `X` represent encoded dimensions of a
            categorical variable and one single shap value per categorical variable is desired. Both
            `cat_vars_start_idx` and `cat_vars_enc_dim` should be specified as detailed below to allow this.
        cat_vars_start_idx
            The start indices of the categorical variables.
        cat_vars_enc_dim
            The length of the encoding dimension for each categorical variable.
        """  # noqa: E501

        if not self._fitted:
            raise TypeError(
                "Called explain on an unfitted object! Please fit the "
                "explainer using the .fit method first!"
            )

        self.tree_limit = tree_limit
        self.approximate = approximate
        self.interactions = interactions
        self.summarise_result = summarise_result

        if interactions:
            self._check_interactions(approximate, self.background_data, y)
            if self._explainer.model.model_type == 'xgboost':
                shap_output = self._xgboost_interactions(X)
            else:
                shap_output = self._explainer.shap_interaction_values(X, tree_limit=tree_limit)
        else:
            self._check_explainer_setup(self.background_data, self.model_output, y)
            shap_output = self._explainer.shap_values(
                X,
                y=y,
                tree_limit=tree_limit,
                approximate=self.approximate,
                check_additivity=check_additivity,
            )
        expected_value = self.expected_value
        if isinstance(shap_output, np.ndarray):
            shap_output = [shap_output]
        if isinstance(expected_value, float):
            expected_value = [self.expected_value]

        explanation = self.build_explanation(
            X,
            shap_output,
            expected_value,
            summarise_result=summarise_result,
            cat_vars_start_idx=cat_vars_start_idx,
            cat_vars_enc_dim=cat_vars_enc_dim,
        )

        self._update_metadata(
            {'interactions': self.interactions,
             'explain_loss': True if y is not None else False,
             'approximate': self.approximate,
             },
            params=True,
        )

        return explanation

    def _xgboost_interactions(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, List[np.ndarray]]:
        """
        `shap` library handling of `xgboost` causes a `ValueError` due to `xgboost` (features name mismatch)
        if you call `shap_interaction_values` with a numpy array (likely only if the user declares their
        `xgboost.DMatrix` object with the feature_names keyword argument). This method converts the
        incoming numpy array to an `xgboost.DMatrix` object with feature names that match the predictor.
        """

        import xgboost

        dexplain = xgboost.DMatrix(X, feature_names=self.predictor.feature_names)
        shap_output = self._explainer.shap_interaction_values(dexplain, tree_limit=self.tree_limit)

        return shap_output

    def _check_interactions(self, approximate: bool,
                            background_data: Union[np.ndarray, pd.DataFrame, None],
                            y: Optional[np.ndarray]) -> None:
        """
        Checks if the inputs to the explain method match the explainer setup if shap interaction values
        are to be explained.

        Parameters
        ----------
        approximate
            See `explain` documentation.
        background_data
            See `fit` documentation.
        y
            See `explain` documentation.

        Raises
        ------
        NotImplementedError
            If a background dataset is passed to the `fit` method or argument `y` is specified to the `explain`
            method. These algorithms are not yet supported in the `shap` library.

        Warns
        -----
            If approximate values are requested. These values are not defined for interactions.
        """

        self.approximate = approximate
        if approximate:
            logger.warning("Approximate shap values are not defined for shap interaction values, "
                           "ignoring argument!")
            self.approximate = False
        if background_data is not None:
            raise NotImplementedError(
                "Interactions can currently only be computed if no background dataset is specified. "
                "Re-instantiate the explainer and run fit without any arguments to compute shap "
                "interaction values!")
        if y is not None:
            raise NotImplementedError(
                "Interactions can currently only be computed if no background dataset is specified "
                "but explaining loss functions requires a background dataset. Re-instantiate the "
                "explainer with model_output='log_loss' and run fit(background_data=my_data) to "
                "explain loss values!!"
            )

    def _check_explainer_setup(self,
                               background_data: Union[np.ndarray, pd.DataFrame, None],
                               model_output: str,
                               y: Optional[np.ndarray]) -> None:
        """
        Checks whether the inputs to the `explain` method match the explainer setup if shap interaction values are
        not required

        Parameters
        ----------
        background_data
            See `fit` method documentation.
        model_output
            See `TreeShap` constructor documentation.
        y
            See `explain` method documentation.


        Raises
        ------
        NotImplementedError
            If the users passes labels to the `explain` method but does not specify a background dataset or
            if the user does not pass a background dataset to `fit` and specifies a `model_output` other than
            `'raw'` when initialising the explainer.
        ValueError
            If the user passes labels to the `explain` method but has not set `model_output='log_loss'` when
            initialising the explainer.
        """

        # check settings are correct for loss value explanations
        if y is not None:
            if background_data is None:
                raise NotImplementedError(
                    "Loss values can only currently be explained with respect to the value over a background dataset. "
                    "Re-instantiate the explainer and run fit(background_data=my_data)!")
            if model_output != 'log_loss':
                raise ValueError(
                    "Model output should be set to 'log_loss' in order to explain loss values. Re-instantiate the model"
                    "with the option `model_output='log_loss' passed to the constructor, call  "
                    "fit(background_data=my_data) and then explain with the default arguments."
                )
        # check model output data is compatible with background data setting
        else:
            if background_data is None:
                if model_output != 'raw':
                    raise NotImplementedError(
                        f"Without a background dataset, only raw output can be explained currently. "
                        f"To explain output {model_output}, select a background dataset, re-instanstiate the "
                        f"explainer with the desired model output option and then call fit(background_data=my_data)!"
                    )

    def build_explanation(self,
                          X: Union[np.ndarray, pd.DataFrame, 'catboost.Pool'],
                          shap_output: List[np.ndarray],
                          expected_value: List[float],
                          **kwargs) -> Explanation:

        """
        Create an explanation object. If output summarisation is required and all inputs necessary for this operation
        are passed, the raw shap values are summed first so that a single shap value is returned for each categorical
        variable, as opposed to a shap value per dimension of categorical variable encoding. Similarly, the
        shap interaction values are summed such that they represent the interaction between categorical variables as
        opposed to levels of categorical variables. If the interaction option has been specified during `explain`,
        this method computes the shap values given the interactions prior to creating the response.

        Parameters
        ----------
        X
            Instances to be explained.
        shap_output
            If `explain` is callled with `interactions=True` then the list contains tensors of dimensionality
            `n_instances x n_features x n_features` of shap interaction values. Otherwise, it contains tensors of
            dimension `n_instances x n_features` representing shap values. The length of the list equals the number of
            model outputs.
        expected_value
            A list containing the expected value of the prediction for each class. Its length is equal to that of
            `shap_output`.

        Returns
        -------
        explanation
            An `Explanation` object containing the shap values and prediction in the `data` field, along with a
            `meta` field containing additional data. See usage examples `here`_ for details.

            .. _here:
               https://docs.seldon.io/projects/alibi/en/latest/methods/TreeSHAP.html

        """

        y = kwargs.get('y')
        if y is None:
            y = np.array([])
        cat_vars_start_idx = kwargs.get('cat_vars_start_idx', ())  # type: Sequence[int]
        cat_vars_enc_dim = kwargs.get('cat_vars_enc_dim', ())  # type: Sequence[int]
        summarise_result = kwargs.get('summarise_result', False)  # type: bool

        # check if interactions were computed
        if len(shap_output[0].shape) == 3:
            shap_interaction_values = shap_output
            # shap values are the sum over all shap interaction values for each instance
            shap_values = [interactions.sum(axis=2) for interactions in shap_output]
        else:
            shap_interaction_values = [np.array([])]
            shap_values = shap_output
        if summarise_result:
            self._check_result_summarisation(summarise_result, cat_vars_start_idx, cat_vars_enc_dim)
        if self.summarise_result:
            summarised_shap = []
            for shap_array in shap_values:
                summarised_shap.append(sum_categories(shap_array, cat_vars_start_idx, cat_vars_enc_dim))
            shap_values = summarised_shap
            if shap_interaction_values[0].size != 0:
                summarised_shap_interactions = []
                for shap_array in shap_interaction_values:
                    summarised_shap_interactions.append(
                        sum_categories(shap_array, cat_vars_start_idx, cat_vars_enc_dim)
                    )
                shap_interaction_values = summarised_shap_interactions

        # NB: Can't get the raw prediction from model when model_output = 'log_loss` as shap library does
        # not support this (issue raised). We may be able to support this if there's a compelling need.
        # NB: raw output of a regression or classification task will not work for pyspark (predict not implemented)
        if self.model_output == 'log_loss':
            loss = self._explainer.model.predict(X, y, tree_limit=self.tree_limit)
            raw_predictions = []  # type: Union[List, np.ndarray]
        else:
            loss = []
            raw_predictions = self._explainer.model.predict(X, tree_limit=self.tree_limit)
            # flatten array of predictions if the trailing dimension is 1
            if raw_predictions.shape[-1] == 1:
                raw_predictions = raw_predictions.squeeze(-1)

        # predicted class
        argmax_pred = []  # type: Union[List, np.ndarray]
        if self.task != 'regression':
            if not isinstance(raw_predictions, list):
                if self.scalar_output:
                    if self.model_output == 'raw':
                        probas = expit(raw_predictions)
                    else:
                        probas = raw_predictions
                    argmax_pred = (probas > 0.5).astype(int)
                else:
                    argmax_pred = np.argmax(np.atleast_2d(raw_predictions), axis=1)

        importances = rank_by_importance(shap_values, feature_names=self.feature_names)

        if self._explainer.model.model_type == 'catboost':
            import catboost  # noqa: F811
            if isinstance(X, catboost.Pool):
                X = X.get_features()
        # output explanation dictionary
        data = copy.deepcopy(DEFAULT_DATA_TREE_SHAP)
        data.update(
            shap_values=shap_values,
            shap_interaction_values=shap_interaction_values,
            expected_value=expected_value,
            model_output=self.model_output,
            categorical_names=self.categorical_names,
            feature_names=self.feature_names,
        )
        data['raw'].update(
            raw_prediction=raw_predictions,
            loss=loss,
            prediction=argmax_pred,
            instances=np.array(X),
            labels=y,
            importances=importances,
        )

        self._update_metadata({"summarise_result": self.summarise_result}, params=True)

        return Explanation(meta=copy.deepcopy(self.meta), data=data)

    def _check_result_summarisation(self,
                                    summarise_result: bool,
                                    cat_vars_start_idx: Sequence[int],
                                    cat_vars_enc_dim: Sequence[int]):
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
        if not cat_vars_start_idx or not cat_vars_enc_dim:
            logger.warning(
                "Results cannot be summarised as either the"
                "start indices for categorical variables or"
                "the encoding dimensions were not passed!"
            )
            self.summarise_result = False
