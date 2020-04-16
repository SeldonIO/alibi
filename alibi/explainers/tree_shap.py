import shap
import logging
import copy

import numpy as np
import pandas as pd

from alibi.api.defaults import DEFAULT_META_TREE_SHAP, DEFAULT_DATA_TREE_SHAP
from alibi.api.interfaces import Explanation, Explainer, FitMixin
from shap import TreeExplainer
from typing import Any, Dict, List, Optional, Sequence, Union, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import catboost

TREE_SHAP_PARAMS = [
    'link',
    'summarise_background',
    'summarise_result',
    'kwargs'
]

BACKGROUND_WARNING_THRESHOLD = 300


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
                 seed: int = None):
        """
        A wrapper around the shap.TreeExplainer class. It adds the following functionality:
            1. Input summarisation options to allow control over background dataset size and hence runtime
            2. Output summarisation for sklearn models with one-hot encoded categorical variables

        Parameters
        ----------
        predictor
            A fitted model to be explained. XGBoost, LightGBM, CatBoost, Pyspark and most tree-based
            scikit-learn models are supported.
        model_output
            Supported values are: 'raw', 'probability', 'probability_doubled', 'log_loss':

            - raw: the raw model of the output, which varies by method, is explained. This option
            should always be used if the `fit` method is not called. For regression models it is the
            standard output, for binary classification in XGBoost it is the log odds ratio.
            - probability: the probability output is explained. This option should only be used if `fit`
            was called. If the tree outputs log-odds, then an inverse logit transformation is applied to
            - probability_doubled: used for binary classification problem in situations where the model outputs
            the logits/probabilities for the positive class but shap values for both outcomes are desired. This
            option should be used only if `fit` was called. In
            this case the expected value for the negative class is 1 - expected_value for positive class and
            the shap values for the negative class are the negative values of the positive class shap values.
            convert the model output to probabilities. The tree output is inferred either from the model type
            (e.g., most sklearn models with exception of `sklearn.tree.DecisionTreeClassifier`,
            `"sklearn.ensemble.RandomForestClassifier`, `sklearn.ensemble.ExtraTreesClassifier` output
            logits) or on the basis of the mapping implemented in the `shap.TreeEnsemble` constructor.
            - log_loss: logarithmic loss is explained. This option shoud be used only if `fit` was called and
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
    # TODO: Default to a `raw` if model_output not as expected
        self.model_output = model_output
        self._explainer = TreeExplainer(predictor, model_output=model_output)
        self.feature_names = feature_names if feature_names else []
        self.categorical_names = categorical_names if categorical_names else {}
        self.seed = seed

        # checks if it has been fitted:
        self._fitted = False

    def _check_inputs(self) -> None: pass

    def _summarise_background(self) -> np.ndarray: pass

    def _update_metadata(self) -> None: pass

    def fit(self,
            background_data: Union[np.ndarray, catboost, None] = None,
            **kwargs) -> "TreeShap":


    def explain(self,
                X: Union[np.ndarray, pd.DataFrame, catboost.Pool],
                y: Optional[np.ndarray] = None,
                tree_limit=None,
                ) -> "Explanation":

        self.tree_limit = tree_limit

    def build_explanation(self,
                          X: Union[np.ndarray, pd.DataFrame, catboost.Pool],
                          shap_values: List[np.ndarray],
                          expected_value: List[float],
                          kwargs) -> Explanation:

        y = kwargs.get('y', None)

        if isinstance(X, catboost.Pool):
            X = X.get_features()

        raw_predictions, loss_predictions = None, None
        if self.model_output != 'log_loss':
            raw_predictions = self._explainer.model.predict(X, tree_limit=self.tree_limit)
        if y is not None or self.model_output == 'log_loss':
            loss_predictions = self._explainer.model.predict(X, y, tree_limit=self.tree_limit)


        data = copy.deepcopy(DEFAULT_DATA_TREE_SHAP)
        data['raw'].update(
            raw_prediction=raw_predictions,
            loss_predictions=loss_predictions,

        )

        # TODO: Check what flatoutput does in self._explainer.predict and see if you need to
        #   adjust before putting in the response


# TODO: how to access raw predictions - are they available in TreeExplainer?
# TODO: can't call predict is arrays which shape (n,)
# TODO: add both loss and raw output to the schema
# TODO: Test:
#  Probability doubled ?
#  That the output contains the data you expect

