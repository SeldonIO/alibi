import shap
import logging

import numpy as np

from alibi.api.interfaces import Explanation, Explainer, FitMixin
from typing import Callable, Dict, List, Optional, Sequence, Union, Tuple

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
                 predictor: Callable,
                 model_output: str = 'raw',
                 feature_names: Union[List[str], Tuple[str], None] = None,
                 categorical_names: Optional[Dict[int, List[str]]] = None,
                 seed: int = None):
        """
        A wrapper around the shap.TreeExplainer class. It adds the following functionality:
            * Input summarisation options to allow control over background dataset size and hence runtime
            * Output summarisation for sklearn models with one-hot encoded categorical variables


        :param predictor:
        :param model_output:
        :param feature_names:
        :param categorical_names:
        :param seed:
        """
