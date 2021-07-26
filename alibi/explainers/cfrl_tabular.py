import torch
import torch.nn as nn
import torch.nn.functional as F

import tensorflow as tf
import tensorflow.keras as keras

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from alibi.explainers.cfrl_base import CounterfactualRLBase, Postprocessing
from alibi.models.pytorch.autoencoder import AE as PytorchAE
from alibi.models.tensorflow.autoencoder import AE as TensorflowAE

import numpy as np
import pandas as pd
from itertools import count
from scipy.special import softmax
from functools import partial
from typing import Tuple, List, Dict, Callable, Union, Any


def conditional_dim(feature_names: List[str],
                    category_map: Dict[int, List[str]]) -> int:
    """
    Computes the dimension of the conditional vector

    Parameters
    ----------
    feature_names
        List of feature names. This should be provided by the dataset.
    category_map
        Dictionary of category mapping. The keys are column indexes and the values are lists containing the possible
        values for an attribute. This should be provided by the dataset.

    Returns
    -------
    Dimension of the conditional vector
    """
    cat_attr = int(np.sum([len(vals) for vals in category_map.values()]))
    num_attr = len(feature_names) - len(category_map)
    return 2 * num_attr + cat_attr


def split_ohe(x_ohe: Union[np.ndarray, torch.Tensor, tf.Tensor],
              category_map: Dict[int, List[str]]
              ) -> Tuple[List[Union[np.ndarray, torch.Tensor, tf.Tensor]],
                         List[Union[np.ndarray, torch.Tensor, tf.Tensor]]]:
    """
    Splits a one-hot encoding array in a list of numerical heads and a list of categorical heads. Since by convention
    the numerical heads are merged in a single head, if the function returns a list of numerical heads, then the size
    of the list is 1.

    Parameters
    ----------
    x_ohe
        One-hot encoding representation.
    category_map
        Dictionary of category mapping. The keys are column indexes and the values are lists containing the possible
        values for an attribute.

    Returns
    -------
    x_ohe_num_split
        List of numerical heads. If different than `None`, the list's size is 1.
    x_ohe_cat_split
        List of categorical one-hot encoded heads.
    """
    x_ohe_num_split, x_ohe_cat_split = [], []
    offset = 0

    # Compute the number of columns spanned by the categorical one-hot encoded heads, and the number of columns spanned
    # by the numerical heads.
    cat_attr = int(np.sum([len(vals) for vals in category_map.values()]))
    num_attr = x_ohe.shape[1] - cat_attr

    # If the number of numerical attributes is different than 0, then the first `num_attr` columns correspond to the
    # numerical attributes
    if num_attr > 0:
        x_ohe_num_split.append(x_ohe[:, :num_attr])
        offset = num_attr

    # If there exist categorical attributes, then extract them one by one
    if cat_attr > 0:
        for id in category_map:
            x_ohe_cat_split.append(x_ohe[:, offset:offset + len(category_map[id])])
            offset += len(category_map[id])

    return x_ohe_num_split, x_ohe_cat_split


def numerical_condition(x_ohe: np.ndarray,
                        feature_names: List[str],
                        category_map: Dict[int, List[str]],
                        ranges: Dict[str, List[float]],
                        immutable_attr: List[str],
                        conditional: bool = True) -> np.ndarray:
    """
    Generates numerical attributes conditional vector.

    Parameters
    ----------
    x_ohe
        One-hot encoding representation of the element(s) for which the conditional vector will be generated.
        This argument is used to extract the number of conditional vector. The choice of `x_ohe` instead of a
        `size` argument is for consistency purposes with `categorical_cond` function.
    feature_names
        List of feature names. This should be provided by the dataset.
    category_map
        Dictionary of category mapping. The keys are column indexes and the values are lists containing the possible
        values for an attribute.
    ranges:
        Dictionary of ranges for numerical attributes. Each value is a list containing two elements, first one negative
        and the second one positive.
    immutable_attr
        Dictionary of immutable attributes. The keys are the column indexes and the values are booleans: `True` if
        the attribute is immutable, `False` otherwise.
    conditional
        Boolean flag to generate a conditional vector. If `False` the conditional vector does not impose any
        restrictions on the attribute value.

    Returns
    -------
    num_cond
        Conditional vector for numerical attributes.
    """
    num_cond = []
    size = x_ohe.shape[0]

    for feature_id, feature_name in enumerate(feature_names):
        # skip categorical features
        if feature_id in category_map:
            continue

        if feature_name in immutable_attr:
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


def categorical_condition(x_ohe: np.ndarray,
                          feature_names: List[str],
                          category_map: Dict[int, List],
                          immutable_attr: List[str],
                          conditional: bool = True) -> np.ndarray:
    """
    Generates categorical attributes conditional vector.

    Parameters
    ----------
    x_ohe
        One-hot encoding representation of the element(s) for which the conditional vector will be generated.
        The elements are required since some attributes can be immutable. In that case, the mask vector is the
        one-hot encoding itself for that particular attribute.
    feature_names
        List of feature names. This should be provided by the dataset.
    category_map
        Dictionary of category mapping. The keys are column indexes and the values are lists containing the possible
        values for an attribute.
    immutable_attr
        List of immutable attributes.
    conditional
        Boolean flag to generate a conditional vector. If `False` the conditional vector does not impose any
        restrictions on the attribute value.

    Returns
    -------
    cat_cond
        Conditional vector for categorical attributes.
    """
    cat_cond = []
    cat_idx = 0

    # Split the one-hot representation into a list where each element corresponds to an attribute.
    _, x_ohe_cat_split = split_ohe(x_ohe, category_map)

    # Create mask for each categorical column.
    for feature_id, feature_name in enumerate(feature_names):
        # skip numerical features
        if feature_id not in category_map:
            continue

        # initialize mask with the original value
        mask = x_ohe_cat_split[cat_idx].copy()

        # if the feature is not immutable, add noise to modify the mask
        if feature_name not in immutable_attr:
            mask += np.random.rand(*mask.shape) if conditional else np.ones_like(mask)

        # construct binary mask
        mask = (mask > 0.5).astype(np.float32)
        cat_cond.append(mask)

        # move to the next categorical index
        cat_idx += 1

    cat_cond = np.concatenate(cat_cond, axis=1)
    return cat_cond


def generate_condition(x_ohe: np.ndarray,
                       feature_names: List[str],
                       category_map: Dict[int, List[str]],
                       ranges: Dict[str, List[float]],
                       immutable_attr: List[str],
                       conditional: bool = True) -> np.ndarray:
    """
    Generates conditional vector.

    Parameters
    ----------
    x_ohe
        One-hot encoding representation of the element(s) for which the conditional vector will be generated.
        This method assumes that the input array, `x_ohe`, is encoded as follows: first columns correspond to the
        numerical attributes, and the rest are one-hot encodings of the categorical columns. The numerical and the
        categorical columns are ordered by the original column index( e.g. numerical = (1, 4), categorical=(0, 2, 3)).
    feature_names
        List of feature names.
    category_map
        Dictionary of category mapping. The keys are column indexes and the values are lists containing the possible
        values for an attribute.
    ranges
        Dictionary of ranges for numerical attributes. Each value is a list containing two elements, first one negative
        and the second one positive.
    immutable_attr
        List of immutable map attributes.
    conditional
        Boolean flag to generate a conditional vector. If `False` the conditional vector does not impose any
        restrictions on the attribute value.

    Returns
    -------
    cond
        Conditional vector.
    """
    # Generate numerical condition vector.
    num_cond = numerical_condition(x_ohe=x_ohe,
                                   feature_names=feature_names,
                                   category_map=category_map,
                                   ranges=ranges,
                                   immutable_attr=immutable_attr,
                                   conditional=conditional)

    # Generate categorical condition vector.
    cat_cond = categorical_condition(x_ohe=x_ohe,
                                     feature_names=feature_names,
                                     category_map=category_map,
                                     immutable_attr=immutable_attr,
                                     conditional=conditional)

    # Concatenate numerical and categorical conditional vectors.
    cond = np.concatenate([num_cond, cat_cond], axis=1)
    return cond


def sample_numerical(x_hat_num_split: List[np.ndarray],
                     x_ohe_num_split: List[np.ndarray],
                     cond_num_split: List[np.ndarray],
                     stats: Dict[int, Dict[str, float]]) -> List[np.ndarray]:
    """
    Samples numerical attributes according to the conditional vector. This method clips the values between the
    desired ranges specified in the conditional vector, and ensures that the values are between the minimum and
    the maximum values from train training datasets stored in the dictionary of statistics.

    Parameters
    ----------
    x_hat_num_split
        List of reconstructed numerical heads from the auto-encoder. This list should contain a single element
        as all the numerical attributes are part of a singe linear layer output.
    x_ohe_num_split
        List of original numerical heads. The list should contain a single element as part of the convention
        mentioned in the description of `x_ohe_hat_num`.
    cond_num_split
        List of conditional vector for numerical heads. The list should contain a single element as part of the
        convention mentioned in the description of `x_ohe_hat_num`.
    stats
        Dictionary of statistic of the training data. Contains the minimum and maximum value of each numerical attribute
        in the training set. Each key is an index of the column and each value is another dictionary containing `min`
        and `max` keys.

    Returns
    -------
    x_ohe_hat_num
        List of clamped input vectors according to the conditional vectors and the dictionary of statistics .
    """
    num_cols = x_hat_num_split[0].shape[1]    # number of numerical columns
    sorted_cols = sorted(stats.keys())            # ensure that the column ids are sorted

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


def sample_categorical(x_hat_cat_split: List[np.ndarray],
                       cond_cat_split: List[np.ndarray]) -> List[np.ndarray]:
    """
    Samples categorical attributes according to the conditional vector. This method sample conditional according to
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
    x_out = []                                             # initialize the returning list
    rows = np.arange(x_hat_cat_split[0].shape[0])      # initialize the returning list

    for i in range(len(x_hat_cat_split)):
        # compute probability distribution
        proba = softmax(x_hat_cat_split[i], axis=1)

        # sample the most probable outcome conditioned on the conditional vector
        cols = np.argmax(cond_cat_split[i] * proba, axis=1)
        samples = np.zeros_like(proba)
        samples[rows, cols] = 1
        x_out.append(samples)

    return x_out


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
        Dictionary of statistic of the training data. Contains the minimum and maximum value of each numerical attribute
        in the training set. Each key is an index of the column and each value is another dictionary containing `min`
        and `max` keys.
    category_map
        Dictionary of category mapping. The keys are column indexes and the values are lists containing the possible
        values for an attribute.

    Returns
    -------
    Most probable sample according to the auto-encoder reconstruction, sampled according to the conditional vector and
    the dictionary of statistics. This method assumes that the input array, `x_ohe`, is encoded as follows: first
    columns correspond to the numerical attributes, and the rest are one-hot encodings of the categorical columns.
    """
    x_ohe_num_split, x_ohe_cat_split = split_ohe(x_ohe, category_map)
    cond_num_split, cond_cat_split = split_ohe(cond, category_map)

    sampled_num, sampled_cat = [], []  # list of sampled numerical columns and sampled categorical columns
    num_attr, cat_attr = len(x_ohe_num_split), len(x_ohe_cat_split)

    if num_attr > 0:
        # sample numerical columns
        sampled_num = sample_numerical(x_hat_num_split=x_hat_split[:num_attr],
                                       x_ohe_num_split=x_ohe_num_split,
                                       cond_num_split=cond_num_split,
                                       stats=stats)

    if cat_attr > 0:
        # sample categorical columns
        sampled_cat = sample_categorical(x_hat_cat_split=x_hat_split[-cat_attr:],
                                         cond_cat_split=cond_cat_split)

    return sampled_num + sampled_cat


def pytorch_sample_differentiable(x_ohe_hat_split: List[torch.Tensor],
                                  category_map: Dict[int, List[str]]) -> List[tf.Tensor]:
    """
    Samples differentiable reconstruction.

    Parameters
    ----------
    x_ohe_hat_split
        List of one-hot encoded reconstructed columns form the auto-encoder.
    category_map
        Dictionary of category mapping. The keys are column indexes and the values are lists containing the possible
        values for an attribute.

    Returns
    -------
    Differentiable reconstruction.
    """
    num_attr = len(x_ohe_hat_split) - len(category_map)
    cat_attr = len(category_map)
    x_out = []

    # pass numerical attributes as they are
    if num_attr > 0:
        x_out.append(x_ohe_hat_split[0])

    # sample categorical attributes
    if cat_attr > 0:
        for head in x_ohe_hat_split[-cat_attr:]:
            out = torch.argmax(head, dim=1)

            # transform to one-hot encoding
            out = F.one_hot(out, num_classes=head.shape[1])
            proba = F.softmax(head, dim=1)
            out = out - proba.detach() + proba
            x_out.append(out)

    return x_out


def tensorflow_sample_differentiable(x_ohe_hat_split: List[tf.Tensor],
                                     category_map: Dict[int, List[str]]) -> List[tf.Tensor]:
    """
    Samples differentiable reconstruction.

    Parameters
    ----------
    x_ohe_hat_split
        List of one-hot encoded reconstructed columns form the auto-encoder.
    category_map
        Dictionary of category mapping. The keys are column indexes and the values are lists containing the possible
        values for an attribute.

    Returns
    -------
    Differentiable reconstruction.
    """
    num_attr = len(x_ohe_hat_split) - len(category_map)
    cat_attr = len(category_map)
    x_out = []

    # pass numerical attributes as they are
    if num_attr > 0:
        x_out.append(x_ohe_hat_split[0])

    # sample categorical attributes
    if cat_attr > 0:
        for head in x_ohe_hat_split[-cat_attr:]:
            out = tf.argmax(head, axis=1)

            # transform to one-hot encoding
            out = tf.one_hot(out, depth=head.shape[1])
            proba = tf.nn.softmax(head, axis=1)
            out = out - tf.stop_gradient(proba) + proba
            x_out.append(out)

    return x_out


def pytorch_l0_ohe(input: torch.Tensor,
                   target: torch.Tensor,
                   reduction: str = 'none') -> torch.Tensor:
    """
    Computes the L0 loss for a one-hot encoding representation.

    Parameters
    ----------
    input
        Input tensor.
    target
        Target tensor
    reduction
        Specifies the reduction to apply to the output: `none` | `mean` | `sum`.

    Returns
    -------
    L0 loss.
    """
    loss = torch.maximum(target - input, torch.zeros_like(input))

    if reduction == 'none':
        return loss

    if reduction == 'mean':
        return torch.mean(loss)

    if reduction == 'sum':
        return torch.sum(loss)

    raise ValueError(f"Reduction {reduction} not implemented.")


def tensorflow_l0_ohe(input: tf.Tensor,
                      target: tf.Tensor,
                      reduction: str = 'none') -> tf.Tensor:
    """
    Computes the L0 loss for a one-hot encoding representation.

    Parameters
    ----------
    input
        Input tensor.
    target
        Target tensor
    reduction
        Specifies the reduction to apply to the output: `none` | `mean` | `sum`.

    Returns
    -------
    L0 loss.
    """
    loss = tf.maximum(target - input, tf.zeros_like(input))

    if reduction == 'none':
        return loss

    if reduction == 'mean':
        tf.reduce_mean(loss)

    if reduction == 'sum':
        return tf.reduce_sum(loss)

    raise ValueError(f"Reduction {reduction} not implemented.")


def pytorch_l1_loss(input: torch.Tensor, target: torch.Tensor, reduction: str = 'none') -> torch.Tensor:
    """
    Computes L1 loss.

    Parameters
    ----------
    input
        Input tensor.
    target
        Target tensor.
    reduction
        Specifies the reduction to apply to the output: `none` | `mean` | `sum`.

    Returns
    -------
    L1 loss.
    """
    return F.l1_loss(input=input, target=target, reduction=reduction)


def tensorflow_l1_loss(input: tf.Tensor, target=tf.Tensor, reduction: str = 'none') -> tf.Tensor:
    """
    Computes the L1 loss.

    Parameters
    ----------
    input
       Input tensor.
    target
       Target tensor
    reduction
       Specifies the reduction to apply to the output: `none` | `mean` | `sum`.

    Returns
    -------
    L1 loss.
    """
    loss = tf.abs(input - target)

    if reduction == 'none':
        return loss

    if reduction == 'mean':
        return tf.reduce_mean(loss)

    if reduction == 'sum':
        return tf.reduce_sum(loss)

    raise ValueError(f"Reduction {reduction} not implemented.")


def pytorch_he_sparsity_loss(x_ohe_hat_split: List[torch.Tensor],
                             x_ohe: torch.Tensor,
                             category_map: Dict[int, List[str]],
                             weight_num: float = 1.0,
                             weight_cat: float = 1.0):
    """
    Computes heterogeneous sparsity loss.

    Parameters
    ----------
    x_ohe_hat_split
        List of one-hot encoded reconstructed columns form the auto-encoder.
    x_ohe
        One-hot encoded representation of the input.
    category_map
        Dictionary of category mapping. The keys are column indexes and the values are lists containing the possible
        values for an attribute.
    weight_num
        Numerical loss weight.
    weight_cat
        Categorical loss weight.

    Returns
    -------
    Heterogeneous sparsity loss.
    """
    # split the input into a list of tensor, where each element corresponds to a network head
    x_ohe_num_split, x_ohe_cat_split = split_ohe(x_ohe=x_ohe, category_map=category_map)

    # sample differentiable output
    x_ohe_hat_split = pytorch_sample_differentiable(x_ohe_hat_split=x_ohe_hat_split,
                                                    category_map=category_map)

    # define numerical and categorical loss
    num_loss, cat_loss = 0, 0
    offset = 0

    # compute numerical loss
    if len(x_ohe_num_split) > 0:
        offset = 1
        num_loss = torch.mean(pytorch_l1_loss(input=x_ohe_hat_split[0],
                                              target=x_ohe_num_split[0],
                                              reduction='none'))

    # compute categorical loss
    if len(x_ohe_cat_split) > 0:
        for i in range(len(x_ohe_cat_split)):
            batch_size = x_ohe_hat_split[i].shape[0]
            cat_loss += torch.sum(pytorch_l0_ohe(input=x_ohe_hat_split[i + offset],
                                                 target=x_ohe_cat_split[i],
                                                 reduction='none')) / batch_size

        cat_loss /= len(x_ohe_cat_split)

    return {"num_loss": weight_num * num_loss, "cat_loss": weight_cat * cat_loss}


def tensorflow_he_sparsity_loss(x_ohe_hat_split: List[tf.Tensor],
                                x_ohe: tf.Tensor,
                                category_map: Dict[int, List[str]],
                                weight_num: float = 1.0,
                                weight_cat: float = 1.0):
    """
    Computes heterogeneous sparsity loss.

    Parameters
    ----------
    x_ohe_hat_split
        List of one-hot encoded reconstructed columns form the auto-encoder.
    x_ohe
        One-hot encoded representation of the input.
    category_map
        Dictionary of category mapping. The keys are column indexes and the values are lists containing the possible
        values for an attribute.
    weight_num
        Numerical loss weight.
    weight_cat
        Categorical loss weight.

    Returns
    -------
    Heterogeneous sparsity loss.
    """
    # split the input into a list of tensor, where each element corresponds to a network head
    x_ohe_num_split, x_ohe_cat_split = split_ohe(x_ohe=x_ohe, category_map=category_map)

    # sample differentiable output
    x_ohe_hat_split = tensorflow_sample_differentiable(x_ohe_hat_split=x_ohe_hat_split,
                                                       category_map=category_map)

    # define numerical and categorical loss
    num_loss, cat_loss = 0, 0
    offset = 0

    # compute numerical loss
    if len(x_ohe_num_split) > 0:
        offset = 1
        num_loss = tf.reduce_mean(tensorflow_l1_loss(input=x_ohe_hat_split[0],
                                                     target=x_ohe_num_split[0],
                                                     reduction='none'))

    # compute categorical loss
    if len(x_ohe_cat_split) > 0:
        for i in range(len(x_ohe_cat_split)):
            batch_size = x_ohe_hat_split[i].shape[0]
            cat_loss += tf.reduce_sum(tensorflow_l0_ohe(input=x_ohe_hat_split[i + offset],
                                                        target=x_ohe_cat_split[i],
                                                        reduction='none')) / batch_size

        cat_loss /= len(x_ohe_cat_split)

    return {"sparsity_num_loss": weight_num * num_loss, "sparsity_cat_loss": weight_cat * cat_loss}


def pytorch_he_consistency_loss(z_cf_pred: torch.Tensor, z_cf_tgt: torch.Tensor, **kwargs):
    """
    Computes heterogeneous consistency loss.

    Parameters
    ----------
    z_cf_pred
        Predicted counterfactual embedding.
    x_cf
        Counterfactual reconstruction. This should be already post-processed.
    ae
        Pre-trained autoencoder.

    Returns
    -------
    Heterogeneous consistency loss.
    """
    # compute consistency loss
    loss = F.mse_loss(z_cf_pred, z_cf_tgt)
    return {"consistency_loss": loss}


def tensorflow_he_consistency_loss(z_cf_pred: tf.Tensor, z_cf_tgt: Union[np.ndarray, tf.Tensor], **kwargs):
    """
    Computes heterogeneous consistency loss.

    Parameters
    ----------
    z_cf_pred
            Counterfactual embedding prediction.
    z_cf_tgt
        Counterfactual embedding target.


    Returns
    -------
    Heterogeneous consistency loss.
    """
    # compute consistency loss
    loss = tf.reduce_mean(tf.square(z_cf_pred - z_cf_tgt))
    return {"consistency_loss": loss}


def he_preprocessor(x: np.ndarray,
                    feature_names: List[str],
                    category_map: Dict[int, List[str]],
                    attr_types: Dict[int, type] = dict()) -> Tuple[Callable[[int, float], float]]: # TODO: check this out
    """
    Heterogeneous dataset preprocessor. The numerical attributes are standardized and the categorical attributes
    are one-hot encoded.

    Parameters
    ----------
    x
        Data to fit.
    feature_names
        List of feature names. This should be provided by the dataset.
    category_map
        Dictionary of category mapping. The keys are column indexes and the values are lists containing the possible
        values for an attribute.  This should be provided by the dataset.
    attr_types
        Dictionary of type for numerical attributes.

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

    num_attr_ohe = len(numerical_ids)                            # number of numerical columns
    cat_attr_ohe = sum([len(v) for v in category_map.values()])  # number of categorical columns

    # define inverse preprocessor
    def inv_preprocessor(x_ohe: np.ndarray):
        x_inv = []

        if "num" in preprocessor.named_transformers_:
            num_transf = preprocessor.named_transformers_["num"]
            x_inv.append(num_transf.inverse_transform(x_ohe[:, :num_attr_ohe]))

        if "cat" in preprocessor.named_transformers_:
            cat_transf = preprocessor.named_transformers_["cat"]
            x_inv.append(cat_transf.inverse_transform(x_ohe[:, -cat_attr_ohe:]))

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
            type = attr_types[i] if i in attr_types else int
            x_inv[:, i] = x_inv[:, i].astype(type)

        return x_inv

    return preprocessor.transform, inv_preprocessor


def statistics(x: np.ndarray,
               preprocessor: Callable,
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
        Dictionary of category mapping. The keys are column indexes and the values are lists containing the possible
        values for an attribute.  This should be provided by the dataset.

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


def conditional_vector(x: np.ndarray,
                       condition: Dict[str, List[Union[int, str]]],
                       preprocessor: Callable,
                       feature_names: List[str],
                       category_map: Dict[int, List[str]],
                       stats: Dict[int, Dict[str, float]],
                       ranges: Dict[str, List[float]] = dict(),
                       immutable_attr: List[str] = list(),
                       diverse=False) -> np.ndarray:
    """
    Generates a conditional vector. The condition is expressed a a delta change of the feature. For example, if
    `Age`=26 and the attribute is allowed to increase up to 10 more years. Similar for categorical attributes,
    the current value can be omitted.

    Parameters
    ----------
    x
        Instances for which to generate the conditional vector.
    condition
        Dictionary of conditions per feature. For numerical features it expects a rang that contains the original value.
        For categorical features it expects a list of feature values per attribute that includes the original value.
    preprocessor
        Data preprocessor. The preprocessor should standardize the numerical values and convert categorical ones
        into one-hot encoding representation. By convention, numerical features should be first, followed by the
        rest of categorical ones.
    feature_names
        List of feature names. This should be provided by the dataset.
    category_map
        Dictionary of category mapping. The keys are column indexes and the values are lists containing the possible
        values for an attribute.  This should be provided by the dataset.
    stats
        Dictionary of statistic of the training data. Contains the minimum and maximum value of each numerical attribute
        in the training set. Each key is an index of the column and each value is another dictionary containing `min`
        and `max` keys.
    ranges
        Dictionary of ranges for numerical attributes. Each value is a list containing two elements, first one negative
        and the second one positive.
    immutable_attr
        List of immutable attributes.
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
        if feature_name in immutable_attr:
            range_low, range_high = 0, 0
        elif feature_name in ranges:
            range_low, range_high = ranges[feature_name][0], ranges[feature_name][1]
        else:
            range_low, range_high = -1, 1

        if (feature_name in condition) and (feature_name not in immutable_attr):
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
    _, x_ohe_cat_split = split_ohe(x_ohe, category_map)

    # for each categorical feature add the masking vector
    for i, (feature_id, feature_name) in enumerate(zip(cat_features_ids, cat_feature_names)):
        mask = np.zeros_like(x_ohe_cat_split[i])

        if feature_name not in immutable_attr:
            if feature_name in condition:
                indexes = [category_map[feature_id].index(feature_value) for feature_value in condition[feature_name]]
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


def category_mapping(x: np.ndarray, category_map: Dict[int, List[str]]):
    """
    Applies a category mapping for the categorical attributes in the array.
    Basically, transforms ints back to strings to be readable
    Parameters
    -----------
    x
        Array containing the columns to be mapped.
    category_map
        Dictionary of category mapping. Keys are columns
        index, and values are list of attribute values.
    Returns
    -------
    Transformed array.
    """
    x = pd.DataFrame(x)

    for key in category_map:
        x[key].replace(range(len(category_map[key])), category_map[key], inplace=True)

    return x.to_numpy()


class SamplePostprocessing(Postprocessing):
    def __init__(self,  category_map: Dict[int, List[str]], stats: Dict[int, Dict[str, float]]):
        super().__init__()
        self.category_map = category_map
        self.stats = stats

    def __call__(self, x_cf: List[np.ndarray], x: np.ndarray, c: np.ndarray):
        return sample(x_hat_split=x_cf,
                      x_ohe=x,
                      cond=c,
                      stats=self.stats,
                      category_map=self.category_map)


class ConcatPostprocessing(Postprocessing):
    def __call__(self, x_cf: List[np.ndarray], x: np.ndarray, c: np.ndarray):
        return np.concatenate(x_cf, axis=1)


class CounterfactualRLTabular(CounterfactualRLBase):
    """ Counterfactual Reinforcement Learning Tabular. """

    def __init__(self,
                 ae: Union[TensorflowAE, PytorchAE],
                 ae_preprocessor: Callable,
                 ae_inv_preprocessor: Callable,
                 actor: Union[keras.Sequential, nn.Sequential],
                 critic: Union[keras.Sequential, nn.Sequential],
                 predict_func: Callable,
                 coeff_sparsity: float,
                 coeff_consistency: float,
                 num_classes: int,
                 category_map: Dict[int, List[str]],
                 feature_names: List[str],
                 stats: Dict[int, Dict[str, float]],  # TODO: infer it in the constructor
                 ranges: Dict[str, Tuple[int, int]],  # TODO: infer it (make it optional)
                 immutable_attr: List[str],           # TODO: ren
                 attr_types: Dict[int, type],         # TODO: delete this
                 backend: str = "tensorflow",
                 weight_num: float = 1.0,
                 weight_cat: float = 1.0,
                 **kwargs):
        """
        Constructor.

        Parameters
        ----------
        ae
           Pre-trained autoencoder.
        actor
           Actor network.
        critic
           Critic network.
        predict_func.
           Prediction function. This corresponds to the classifier.
        coeff_sparsity
           Sparsity loss coefficient.
        coeff_consistency
           Consistency loss coefficient.
        backend
           Deep learning backend: `tensorflow`|`pytorch`. Default `tensorflow`.
        weight_num
            Numerical loss weight.
        weight_cat
            Categorical loss weight.
        """
        super().__init__(ae=ae, actor=actor, critic=critic, predict_func=predict_func, coeff_sparsity=coeff_sparsity,
                         coeff_consistency=coeff_consistency, num_classes=num_classes, backend=backend, **kwargs)

        # Set ae preprocessor and inverse preprocessor.
        self.params["ae_preprocessor"] = ae_preprocessor
        self.params["ae_inv_preprocessor"] = ae_inv_preprocessor

        # Set dataset specific arguments.
        self.params["category_map"] = category_map
        self.params["feature_names"] = feature_names
        self.params["ranges"] = ranges
        self.params["immutable_attr"] = immutable_attr
        self.params["stats"] = stats
        self.params["attr_types"] = attr_types
        self.params["weight_num"] = weight_num
        self.params["weight_cat"] = weight_cat

        # Set postprocessing functions.
        self.params["postprocessing_funcs"] = [
            SamplePostprocessing(stats=self.params["stats"], category_map=self.params["category_map"]),
            ConcatPostprocessing(),
        ]

        if "sparsity_loss" not in kwargs:
            # select backend specific function
            he_sparsity_loss = tensorflow_he_sparsity_loss \
                if self.params["backend"] == CounterfactualRLBase.TENSORFLOW \
                else pytorch_he_sparsity_loss

            # set sparsity loss
            self.params["sparsity_loss"] = partial(he_sparsity_loss,
                                                   category_map=self.params["category_map"],
                                                   weight_num=weight_num,
                                                   weight_cat=weight_cat)

        if "consistency_loss" not in kwargs:
            # select backend specific function
            self.params["consistency_loss"] = tensorflow_he_consistency_loss \
                if self.params["backend"] == CounterfactualRLBase.TENSORFLOW \
                else pytorch_he_consistency_loss

        if "conditional_func" not in kwargs:
            # set conditional function generator
            self.params["conditional_func"] = partial(generate_condition,
                                                      feature_names=self.params["feature_names"],
                                                      category_map=self.params["category_map"],
                                                      ranges=self.params["ranges"],
                                                      immutable_attr=self.params["immutable_attr"])

    def explain(self, x: np.ndarray, y_t: np.ndarray, c: Dict[str, List[Union[str, int]]]) -> "Explanation":
        # TODO: check if `c` can be optional
        # TODO: extend --- a condition for each instance


        # convert dictionary conditioning to array
        c = conditional_vector(x=x,
                               condition=c,
                               preprocessor=self.params["ae_preprocessor"],
                               feature_names=self.params["feature_names"],
                               category_map=self.params["category_map"],
                               stats=self.params["stats"],
                               ranges=self.params["ranges"],
                               immutable_attr=self.params["immutable_attr"])
        return super().explain(x, y_t, c)


    # TODO: try to include this in the explain method, with a flag
    def diversity(self,
                  x: np.ndarray,
                  y_t: np.ndarray,
                  c: Dict[str, List[Union[str, int]]],
                  num_samples: int = 1,
                  batch_size: int = 100,
                  patience: int = 1000,
                  tolerance: float = 1e-3) -> np.ndarray:

        # reshape input array
        x = x.reshape(1, -1)

        if x.shape[1] != len(self.params["feature_names"]):
            raise ValueError("Only a single input can be passed.")

        if len(y_t) != 1:
            raise ValueError("Only a single label can be passed.")

        # generate a batch of data
        x = np.tile(x, (batch_size, 1))
        y_t = np.tile(y_t, batch_size)

        # define counterfactual buffer
        cf_buff = None

        from tqdm import tqdm
        for i in tqdm(count()):
            if i == patience:
                break

            if (cf_buff is not None) and (cf_buff.shape[0] >= num_samples):
                break

            # generate conditional vector
            c_vec = conditional_vector(x=x,
                                       condition=c,
                                       preprocessor=self.params["ae_preprocessor"],
                                       feature_names=self.params["feature_names"],
                                       category_map=self.params["category_map"],
                                       stats=self.params["stats"],
                                       immutable_attr=self.params["immutable_attr"],
                                       diverse=True)

            # generate counterfactuals
            x_cf = super().explain(x, y_t, c_vec)

            # get prediction
            y_cf_m = self.params["predict_func"](x_cf)

            # select only counterfactuals where prediction matches the target
            x_cf = x_cf[y_t == y_cf_m]
            if x_cf.shape[0] == 0:
                continue

            # find unique counterfactuals
            x_cf = np.unique(np.floor(x_cf / tolerance).astype(int), axis=0) * tolerance

            # add them to the unique buffer but make sure not to add duplicates
            if cf_buff is None:
                cf_buff = x_cf
            else:
                cf_buff = np.concatenate([cf_buff, x_cf], axis=0)
                cf_buff = np.unique(np.floor(cf_buff / tolerance).astype(int), axis=0) * tolerance

        return cf_buff[:num_samples] if (cf_buff is not None) else np.array([])


# if __name__ == "__main__":
#     from alibi.datasets import fetch_adult
#
#     adult = fetch_adult()
#     x = adult.data
#     category_map = adult.category_map
#     feature_names = adult.feature_names
#     preprocessor, inv_preprocessor = he_preprocessor(x, feature_names, category_map)
#     stats = statistics(x, preprocessor, category_map)
#
#
#     print(feature_names)
#     print(x[0])
#     print(category_map[feature_names.index('Workclass')])
#
#     condition = {"Age": [-5, 20], "Workclass": ["State-gov", "?", "Local-gov"]}
#     immutable_attr = ['Education', 'Marital Status', 'Occupation', 'Relationship', 'Race', 'Sex', 'Capital Gain',
#                       'Capital Loss', 'Hours per week', 'Country']
#
#     c = conditional_vector(x=x[:10],
#                            condition=condition,
#                            preprocessor=preprocessor,
#                            feature_names=feature_names,
#                            category_map=category_map,
#                            stats=stats,
#                            immutable_attr=immutable_attr)
#
#     print(c[:2])
