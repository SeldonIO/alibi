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

        Parameters
        ----------
        predictor
            A fitted model to be explained. XGBoost, LightGBM, CatBoost and most tree-based
            scikit-learn models are supported. In the future, Pyspark could also be supported.
            Please open an issue if this is a use case for you.
        model_output
            Supported values are: 'raw', 'probability', 'probability_doubled', 'log_loss':

            - 'raw': the raw model of the output, which varies by method, is explained. This option
            should always be used if the `fit` method is not called. For regression models it is the
            standard output, for binary classification in XGBoost it is the log odds ratio.
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
            :math:`log(1 + exp(output)) - y * output` with  :math:`y \in \{0, 1\}`

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

    def _check_inputs(self) -> None: pass

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
                interactions=False,
                approximate=False,
                check_additivity=True,
                tree_limit=None,
                summarise_result: bool = False,
                cat_vars_start_idx: List[int] = None,
                cat_vars_enc_dim: List[int] = None,
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



        """

        if not self._fitted:
            raise TypeError(
                "Called explain on an unfitted object! Please fit the "
                "explainer using the .fit method first!"
            )

        self.tree_limit = tree_limit

        # TODO: Deal with the case where they e.g., want to explain loss or want
        #  interactions but pass a background dataset
        # TODO: For which algorithm does approximate work?
        # TODO: Interactions and approximate don't go well together - warn about this and default
        # TODO: Interactions require specific settings in terms of model outputs - set those and warn
        #  users if they are different to the settings they want to explain
        # TODO: Interactions are not supported for loss functions, disable and return shap values

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

# TODO: Test:
#  Probability doubled ?
#  That the output contains the data you expect: all fields in data['raw']
#   are as expected and contain the correct data depending on other settings
#   such as e.g., task (should be test_build_explanation)

