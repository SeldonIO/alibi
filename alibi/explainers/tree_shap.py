import shap
import logging
import copy

import numpy as np
import pandas as pd

from alibi.api.defaults import DEFAULT_META_TREE_SHAP, DEFAULT_DATA_TREE_SHAP
from alibi.api.interfaces import Explanation, Explainer, FitMixin
from typing import Any, Dict, List, Optional, Sequence, Union, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import catboost


logger = logging.getLogger(__name__)


TREE_SHAP_PARAMS = [
    'model_output',
    'summarise_background',
    'summarise_result',
    'kwargs'
]


TREE_SHAP_BACKGROUND_WARNING_THRESHOLD = 300

# TODO: Add support for pyspark if necessary.


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
            logging.warning(msg.format(len(feature_names), shap_values[0].shape[1]))
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
    if n_encoded_levels > values.shape[1]:
        raise ValueError("The sum of the encoded features dimensions exceeds data dimension!")

    new_values = np.zeros((values.shape[0], values.shape[1] - n_encoded_levels + len(enc_feat_dim)))

    # find all the other indices of categorical columns other than those specified
    cat_cols = []
    for start, feat_dim in zip(start_idx, enc_feat_dim):
        for i in range(1, feat_dim):
            cat_cols.append(start + i)

    # sum the columns corresponding to a categorical variable
    enc_idx, new_vals_idx = 0, 0
    for idx in range(values.shape[1]):
        if idx in start_idx:
            feat_dim = enc_feat_dim[enc_idx]
            enc_idx += 1
            stop_idx = idx + feat_dim
            new_values[:, new_vals_idx] = np.sum(values[:, idx:stop_idx], axis=1)
        elif idx in cat_cols:
            continue
        else:
            new_values[:, new_vals_idx] = values[:, idx]
        new_vals_idx += 1

    return new_values


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
            2. Output summarisation for sklearn models with one-hot encoded categorical variables


        # TODO: ADD LINK TO DOCS HERE ...

        Parameters
        ----------
        predictor
            A fitted model to be explained. XGBoost, LightGBM, CatBoost and most tree-based
            scikit-learn models are supported. In the future, Pyspark could also be supported.
            Please open an issue if this is a use case for you.
        model_output
            Supported values are: 'raw', 'probability', 'probability_doubled', 'log_loss':

            - 'raw': the raw model of the output, which varies by method, is explained. This option
            should always be used if the `fit` is called without arguments. It should also be set to compute
            shap interaction values. For regression models it is the standard output, for binary classification
            in XGBoost it is the log odds ratio.
            - 'probability': the probability output is explained. This option should only be used if `fit`
            was called. If the tree outputs log-odds, then an inverse logit transformation is applied to
            - 'probability_doubled': used for binary classification problem in situations where the model outputs
            the logits/probabilities for the positive class but shap values for both outcomes are desired. This
            option should be used only if `fit` was called. In
            this case the expected value for the negative class is 1 - expected_value for positive class and
            the shap values for the negative class are the negative values of the positive class shap values.
            convert the model output to probabilities. The tree output is inferred either from the model type
            (e.g., most sklearn models with exception of `sklearn.tree.DecisionTreeClassifier`,
            `"sklearn.ensemble.RandomForestClassifier`, `sklearn.ensemble.ExtraTreesClassifier` output
            logits) or on the basis of the mapping implemented in the `shap.TreeEnsemble` constructor.
            - 'log_loss': logarithmic loss is explained. This option shoud be used only if `fit` was called and
            labels, y, are provided to the fit method. If the objective is squared error, then the transformation
            (output - y)*(output -y) is applied. For binary cross-entropy objective, the transformation
            :math:`log(1 + exp(output)) - y * output` with  :math:`y \in \{0, 1\}`. Currently only binary cross-entropy
            and squared error losses can be explained.

        feature_names
            Used to compute the `names` field, which appears as a key in each of the values of the `importances`
            sub-field of the response `raw` field.
        categorical_names
            Keys are feature column indices. Each value contains strings with the names of the categories
            for the feature. Used to select the method for background data summarisation (if specified,
            subsampling is performed as opposed to kmeans clustering). In the future it may be used for visualisation.
        task
            Can have values 'classification' and 'regression'. It is only used to set the contents of the `prediction`
            field in the `data['raw']` response field.

        Notes
        -----
        Tree SHAP is an additive attribution method so it is best suited to explaining output in margin space
        (the entire real line). For discussion related to explaining models in output vs probability space, please
        consult this  resource_.

        .. _resource: https://github.com/slundberg/shap/blob/master/notebooks/kernel_explainer/Squashing%20Effect.ipynb
    """

    # TODO: Default to something sensible and warn user if they don't call fit and pass something crazy in
    #    model_output
    # TODO: Default to model output raw if the user provides loss but no labels
    # TODO: Define meta, data and params
    # TODO: Implement summarisation
    # TODO: Default to `raw` if model_output not as expected
    # TODO: deal with interactions (should be kwarg to explain)

        super().__init__(meta=copy.deepcopy(DEFAULT_META_TREE_SHAP))

        self.model_output = model_output
        self.predictor = predictor
        self.feature_names = feature_names if feature_names else []
        self.categorical_names = categorical_names if categorical_names else {}
        self.task = task
        self.seed = seed
        self._update_metadata({"task": self.task})

        # sums up shap values for each level of categorical var
        self.summarise_result = False
        # selects a subset of the background data to avoid excessively slow runtimes
        self.summarise_background = False
        # checks if it has been fitted:
        self._fitted = False

    def _summarise_background(self) -> np.ndarray: pass

    def _update_metadata(self, data_dict: dict, params: bool = False) -> None:
        """
        This function updates the metadata of the explainer using the data from
        the data_dict. If the params option is specified, then each key-value
        pair is added to the metadata 'params' dictionary only if the key is
        included in TREE_SHAP_PARAMS.

        Parameters
        ----------
        data_dict
            Dictionary containing the data to be stored in the metadata.
        params
            If True, the method updates the 'params' attribute of the metatadata.
        """

        if params:
            for key in data_dict.keys():
                if key not in TREE_SHAP_PARAMS:
                    continue
                else:
                    self.meta['params'].update([(key, data_dict[key])])
        else:
            self.meta.update(data_dict)

    def fit(self,
            background_data: Union[np.ndarray, pd.DataFrame, None] = None,
            summarise_background: Union[bool, str] = False,
            n_background_samples: int = TREE_SHAP_BACKGROUND_WARNING_THRESHOLD,
            **kwargs) -> "TreeShap":

        # TODO: we could support fitting if catboost.Pool was passed but that goes in
        # docs for ppl to raise issues if they need it
        np.random.seed(self.seed)

        self._fitted = True
        if isinstance(background_data, pd.DataFrame):
            self.feature_names = list(background_data.columns)
        # TODO: summarisation here
        # TODO: check inputs wrt to model output type, and whether data is passed or not.
        #  This check should override model output type
        self.background_data = background_data
        self._explainer = shap.TreeExplainer(
            self.predictor,
            data=self.background_data,
            model_output=self.model_output,
        )  # type: shap.TreeExplainer
        self.expected_value = self._explainer.expected_value

        # TODO: Update metadata here
        # update metadata
        params = {
            'kwargs': kwargs,
            'summarise_background': self.summarise_background,
        }
        self._update_metadata(params, params=True)

        return self

    def explain(self,
                X: Union[np.ndarray, pd.DataFrame, catboost.Pool],
                y: Optional[np.ndarray] = None,
                interactions: bool = False,
                approximate: bool = False,
                check_additivity: bool = True,
                tree_limit: Optional[int] = None,
                summarise_result: bool = False,
                cat_vars_start_idx: Optional[List[int]] = None,
                cat_vars_enc_dim: Optional[List[int]] = None,
                **kwargs) -> "Explanation":
        """
        Explains the instances in X. `y` should be passed if the model loss function is to be explained,
        which can be useful in order to understand how various features affect model performance over
        time. This is only possible if the explainer has been fitted with a background dataset and
        requires setting `model_output='log_loss'`.

        Parameters
        ----------
        X
            Instances to be explained.
        y
            Labels corresponding to rows of X. Should be passed only if a background dataset was passed to the
            fit method.
        interactions
            If True, the shap value for every feature of every instance in X is decomposed into `X.shape[1] - 1`
            shap value interactions and one shap main effects. This is only supported if no background dataset is
            passed to the algorithm.
        approximate
            If True, an approximation to the shap values that does not account for feature order is computed. This
            was proposed by Sabaas here_. Check this_ resource for more details. This option is currently only supported
            for `xgboost` and `sklearn` models.
        check_additivity
            If True, output correctness is ensured if `model_output=raw` has been passed to the constructor.
        tree_limit
            Explain the output of a subset of the first `tree_limit` trees in an ensamble model.
        summarise_result
            This should be set to True only when some of the columns in `X` represent encoded dimensions of a
            categorical variable and one single shap value per categorical variable is desired. Both `cat_vars_start_idx`
            and `cat_vars_enc_dim` should be specified as detailed below to allow this.
        cat_vars_start_idx
            The start indices of the categorical variables
        cat_vars_enc_dim
            The length of the encoding dimension for each
            categorical variable.

        .. _this: https://static-content.springer.com/esm/art%3A10.1038%2Fs42256-019-0138-9/MediaObjects/42256_2019_138_
        MOESM1_ESM.pdf)

        .. _here: https://github.com/andosa/treeinterpreter
        """

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
        expected_value  = self.expected_value
        if isinstance(shap_output, np.ndarray):
            shap_output = [shap_output]
        if isinstance(expected_value, float):
            expected_value = [self.expected_value]
        self._check_output_summarisation(cat_vars_start_idx, cat_vars_enc_dim)
        if self.summarise_result:
            pass
        # TODO: Continue here


    def _check_interactions(self, approximate: bool, background_data, y: Optional[np.ndarray]) -> None:

        if approximate:
            logger.warning("Approximate shap values are not defined for shap interaction values, "
                           "ignoring argument!")
            self.approximate = False
        if background_data is not None:
            # TODO: @Janis: Here we can just disable the interactions, give a warning and just
            #  return the shap values?
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

        # check settings are correct for loss value explanations
        if y is not None:
            if background_data is None:
                raise NotImplementedError(
                    "Loss values can only currently be explained with respect to the value over a background dataset. "
                    "Re-instantiate the explainer and run fit(background_data=my_data)!")
            if model_output != 'log_loss':
                raise ValueError(
                    "Model output should be set to 'log_loss' in order to explain loss values. Re-instantiate the model"
                    "with the option `model_output='log_loss' passed to the constructor.`"
                )
        # check model output data is compatible with background data setting
        else:
            if background_data is None:
                if model_output != 'raw':
                    raise NotImplementedError(
                        f"Without a background dataset, only raw output can be explained currently. "
                        f"To explain output {model_output}, select an background dataset, re-instanstiate the "
                        f"explainer with the desired model output option and then call fit(background_data=my_data)!"
                    )

    def build_explanation(self,
                          X: Union[np.ndarray, pd.DataFrame, catboost.Pool],
                          shap_output: List[np.ndarray],
                          expected_value: List[float],
                          kwargs) -> Explanation:

        # TODO: In docstring, explain what shap_output can be

        y = kwargs.get('y')
        if not y:
            y = np.array([])

        if isinstance(X, catboost.Pool):
            X = X.get_features()

        # check if interactions were computed
        if len(shap_output[0].shape) == 3:
            shap_interaction_values = shap_output
            # shap values are the sum over all shap interaction values for each instance
            shap_values = [interactions.sum(axis=2) for interactions in shap_output]
        else:
            shap_interaction_values = np.array([])
            shap_values = shap_output

        raw_predictions, loss = [], []
        # raw output of a regression or classification task. Will not work for pyspark.
        if self.model_output != 'log_loss':
            raw_predictions = self._explainer.model.predict(X, tree_limit=self.tree_limit)
        # loss function values explained
        if y is not None or self.model_output == 'log_loss':
            loss = self._explainer.model.predict(X, y, tree_limit=self.tree_limit)
        # predicted class
        if self.task != 'regression' and raw_predictions:
            argmax_pred = np.argmax(np.atleast_2d(raw_predictions), axis=1)
        else:
            argmax_pred = []
        importances = rank_by_importance(shap_values, feature_names=self.feature_names)

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
            instances=X,
            labels=y,
            importances=importances,
        )

        return Explanation(meta=copy.deepcopy(self.meta), data=data)

    def _check_output_summarisation(self, cat_var_start_idx, cat_var_enc_dim):
        if not cat_var_start_idx or cat_var_enc_dim:
            logger.warning(
                "Results cannot be summarised as either the"
                "start indices for categorical variables or"
                "the encoding dimensions were not passed!"
            )
            self.summarise_result = False

# TODO: Test:
#  Probability doubled ?
#  That the output contains the data you expect: all fields in data['raw']
#   are as expected and contain the correct data depending on other settings
#   such as e.g., task (should be test_build_explanation)

