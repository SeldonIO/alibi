import torch
import torch.nn as nn
import torch.nn.functional as F

import tensorflow as tf
import tensorflow.keras as keras

import numpy as np
from typing import Dict, List, Union, Tuple, Callable

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from alibi.explainers.cfrl import PTCounterfactualRLBackend, TFCounterfactualRLBackend

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

    # Compute the number of columns spanned by the categorical one-hot encoded heads,
    # and the number of columns spanned by the numerical heads.
    cat_attr = int(np.sum([len(vals) for vals in category_map.values()]))
    num_attr = x_ohe.shape[1] - cat_attr

    # If the number of numerical attributes is different than 0,
    # then the first `num_attr` columns correspond to the numerical attributes
    offset = 0

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
                        ranges: Dict[int, List[float]],
                        conditional: bool = True) -> np.ndarray:
    """
    Generates numerical attributes conditional vector.

    Parameters
    ----------
    x_ohe
        One-hot encoding representation of the element(s) for which the conditional vector will be generated.
        This argument is used to extract the number of conditional vector. The choice of x_ohe instead of a
        `size` argument is for consistency purposes with `categorical_cond` function.
    ranges:
        Dictionary of ranges for numerical attributes. Each value is a list containing two elements, first one negative
        and the second one positive.
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

    for i in ranges:
        # Check if the ranges are valid.
        if ranges[i][0] > 0:
            raise ValueError(f"Lower bound range for {i} should be negative.")
        if ranges[i][1] < 0:
            raise ValueError(f"Upper bound range for {i} should be positive.")

        # Generate lower and upper bound coefficients.
        coeff_lower = np.random.beta(a=2, b=2, size=size).reshape(-1, 1) if conditional else np.ones((size, 1))
        coeff_upper = np.random.beta(a=2, b=2, size=size).reshape(-1, 1) if conditional else np.ones((size, 1))

        # Generate lower and upper bound conditionals.
        num_cond.append(coeff_lower * ranges[i][0])
        num_cond.append(coeff_upper * ranges[i][1])

    # Construct numerical conditional vector by concatenating all numerical conditions.
    num_cond = np.concatenate(num_cond, axis=1)
    return num_cond


def categorical_condition(x_ohe: np.ndarray,
                          category_map: Dict[int, List],
                          immutable_map: Dict[int, bool],
                          conditional: bool = True) -> np.ndarray:
    """
    Generates categorical attributes conditional vector.

    Parameters
    ----------
    x_ohe
        One-hot encoding representation of the element(s) for which the conditional vector will be generated.
        The elements are required since some attributes can be immutable. In that case, the mask vector is the
        one-hot encoding itself for that particular attribute.
    category_map
        Dictionary of category mapping. The keys are column indexes and the values are lists containing the possible
        values for an attribute.
    immutable_map
        Dictionary of immutable map attributes. The keys are the column indexes and the values are booleans: `True` if
        the attribute is immutable, `False` otherwise.
    conditional
        Boolean flag to generate a conditional vector. If `False` the conditional vector does not impose any
        restrictions on the attribute value.

    Returns
    -------
    cat_cond
        Conditional vector for categorical attributes.
    """
    cat_cond = []

    # Split the one-hot representation into a list where each element corresponds to an attribute.
    _, x_ohe_cat_split = split_ohe(x_ohe, category_map)

    # Create mask for each categorical column.
    for i, key in enumerate(immutable_map):
        mask = x_ohe_cat_split[i].copy()

        if not immutable_map[key]:
            mask += np.random.rand(*mask.shape) if conditional else np.ones_like(mask)

        mask = (mask > 0.5).astype(np.float32)
        cat_cond.append(mask)

    cat_cond = np.concatenate(cat_cond, axis=1)
    return cat_cond


def generate_condition(x_ohe: np.ndarray,
                       ranges: Dict[int, List[float]],
                       category_map: Dict[int, List[str]],
                       immutable_map: Dict[int, bool],
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
    ranges
        Dictionary of ranges for numerical attributes. Each value is a list containing two elements, first one negative
        and the second one positive.
    category_map
        Dictionary of category mapping. The keys are column indexes and the values are lists containing the possible
        values for an attribute.
    immutable_map
        Dictionary of immutable map attributes. The keys are the column indexes and the values are booleans: `True` if
        the attribute is immutable, `False` otherwise.
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
                                   ranges=ranges,
                                   conditional=conditional)

    # Generate categorical condition vector.
    cat_cond = categorical_condition(x_ohe=x_ohe,
                                     category_map=category_map,
                                     immutable_map=immutable_map,
                                     conditional=conditional)

    # Concatenate numerical and categorical conditional vectors.
    cond = np.concatenate([num_cond, cat_cond], axis=1)
    return cond


def sample_numerical(x_ohe_hat_num_split: List[np.ndarray],
                     x_ohe_num_split: List[np.ndarray],
                     cond_num_split: List[np.ndarray],
                     stats: Dict[int, Dict[str, float]]) -> List[np.ndarray]:
    """
    Samples numerical attributes according to the conditional vector. This method clips the values between the
    desired ranges specified in the conditional vector, and ensures that the values are between the minimum and
    the maximum values from train training datasets stored in the dictionary of statistics.

    Parameters
    ----------
    x_ohe_hat_num_split
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
    num_cols = x_ohe_hat_num_split[0].shape[1]    # number of numerical columns
    sorted_cols = sorted(stats.keys())            # ensure that the column ids are sorted

    for i, col_id in zip(range(num_cols), sorted_cols):
        # Extract the minimum and the maximum value for the current column from the training set.
        min, max = stats[col_id]["min"], stats[col_id]["max"]

        # Extract the minimum and the maximum value according to the conditional vector.
        lhs = x_ohe_num_split[0][:, i] + cond_num_split[0][:, 2 * i] * (max - min)
        rhs = x_ohe_num_split[0][:, i] + cond_num_split[0][:, 2 * i + 1] * (max - min)

        # Clamp output according to the conditional vector.
        x_ohe_hat_num_split[0][:, i] = np.clip(x_ohe_hat_num_split[0][:, i], a_min=lhs, a_max=rhs)

        # Clamp output according to the minimum and maximum value from the training set.
        x_ohe_hat_num_split[0][:, i] = np.clip(x_ohe_hat_num_split[0][:, i], a_min=min, a_max=max)

    return x_ohe_hat_num_split


def sample_categorical(x_ohe_hat_cat_split: List[np.ndarray],
                       cond_cat_split: List[np.ndarray]) -> List[np.ndarray]:
    """
    Samples categorical attributes according to the conditional vector. This method sample conditional according to
    the masking vector the most probable outcome.

    Parameters
    ----------
    x_ohe_hat_cat_split
        List of reconstructed categorical heads from the auto-encoder.
    cond_cat_split
        List of conditional vector for categorical heads.

    Returns
    -------
    x_ohe_hat_cat
        List of one-hot encoded vectors sampled according to the conditional vector.
    """
    x_out = []                                             # initialize the returning list
    rows = np.arange(x_ohe_hat_cat_split[0].shape[0])      # initialize the returning list

    for i in range(len(x_ohe_hat_cat_split)):
        # sample the most probable outcome conditioned on the conditional vector
        cols = np.argmax(cond_cat_split[i] * x_ohe_hat_cat_split[i], axis=1)
        samples = np.zeros_like(x_ohe_hat_cat_split[i])
        samples[rows, cols] = 1
        x_out.append(samples)

    return x_out


def sample(x_ohe_hat_split: List[np.ndarray],
           x_ohe: np.ndarray,
           cond: np.ndarray,
           stats: Dict[int, Dict[str, float]],
           category_map: Dict[int, List[str]]) -> List[np.ndarray]:
    """
    Samples an instance from the given reconstruction according to the conditional vector and
    the dictionary of statistics.

    Parameters
    ----------
    x_ohe_hat_split
        List of one-hot encoded reconstructed columns form the auto-encoder.
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
        sampled_num = sample_numerical(x_ohe_hat_num_split=x_ohe_hat_split[:num_attr],
                                       x_ohe_num_split=x_ohe_num_split,
                                       cond_num_split=cond_num_split,
                                       stats=stats)

    if cat_attr > 0:
        # sample categorical columns
        sampled_cat = sample_categorical(x_ohe_hat_cat_split=x_ohe_hat_split[-cat_attr:],
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
    return F.l1_loss(input=input, target=target,reduction=reduction)


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
                             wgt_cat: float = 1.0):
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
    wgt_cat
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

    return {"num_loss": num_loss, "cat_loss": wgt_cat * cat_loss}


def tensorflow_he_sparsity_loss(x_ohe_hat_split: List[tf.Tensor],
                                x_ohe: tf.Tensor,
                                category_map: Dict[int, List[str]],
                                wgt_cat: float = 1.0):
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
    wgt_cat
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

    return {"sparsity_num_loss": num_loss, "sparsity_cat_loss": wgt_cat * cat_loss}


def pytorch_he_consistency_loss(z_cf_pred: torch.Tensor,
                                x_cf_split: List[torch.Tensor],
                                x_ohe: torch.Tensor,
                                cond: torch.Tensor,
                                ae: nn.Module,
                                postprocessing_funcs: List[Callable]):
    """
    Computes heterogeneous consistency loss.

    Parameters
    ----------
    x_cf_split
        List of one-hot encoded reconstructed columns form the auto-encoder.
    cond
        Conditional tensor.
    z
        Embedding tensor.
    postprocessing_funcs
        List of postprocessing functions

    Returns
    -------
    Heterogeneous consistency loss.
    """
    x_cf = PTCounterfactualRLBackend.to_numpy(x_cf_split)
    x = PTCounterfactualRLBackend.to_numpy(x_ohe)
    cond = PTCounterfactualRLBackend.to_numpy(cond)

    # post-process the counterfactual
    for pp_func in postprocessing_funcs:
        x_cf = pp_func(x_cf, x, cond)

    # compute counterfactual embedding
    x_cf = torch.Tensor(x_cf)

    with torch.no_grad():
        z_cf_true = ae.encode(x_cf)

    # compute consistency loss
    loss = F.mse_loss(z_cf_pred, z_cf_true)
    return {"consistency_loss": loss}


def tensorflow_he_consistency_loss(z_cf_pred: tf.Tensor,
                                   x_cf_split: List[tf.Tensor],
                                   x_ohe: tf.Tensor,
                                   cond: tf.Tensor,
                                   ae: keras.Model,
                                   postprocessing_funcs: List[Callable]):
    """
    Computes heterogeneous consistency loss.

    Parameters
    ----------
    x_cf_split
        List of one-hot encoded reconstructed columns form the auto-encoder.
    cond
        Conditional tensor.
    z_cf
        Embedding tensor.
    postprocessing_funcs
        List of postprocessing functions

    Returns
    -------
    Heterogeneous consistency loss.
    """
    x_cf = TFCounterfactualRLBackend.to_numpy(x_cf_split)
    x = TFCounterfactualRLBackend.to_numpy(x_ohe)
    cond = TFCounterfactualRLBackend.to_numpy(cond)

    # post-process the counterfactual
    for pp_func in postprocessing_funcs:
        x_cf = pp_func(x_cf, x, cond)

    # compute counterfactual embedding
    z_cf_true = ae.encoder(x_cf)

    # compute consistency loss
    loss = tf.reduce_mean(tf.square(z_cf_pred - z_cf_true))
    return {"consistency_loss": loss}


def he_preprocessor(x: np.ndarray, feature_names: List[str], category_map: Dict[int, List[str]]):
    """
    Heterogeneous dataset preprocessor. The numerical attributes are standardized and the categorical attirbutes
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

        return x_inv[:, inv_perm]

    return preprocessor.transform, inv_preprocessor


if __name__ == "__main__":
    x = np.random.randint(0, 3, (10, 5))
    category_map = {
        2: list(np.unique(x[:, 2])),
        3: list(np.unique(x[:, 3])),
        4: list(np.unique(x[:, 4]))
    }
    feature_names = [0, 1, 2, 3, 4]

    preproc, inv_preproc = he_preprocessor(x, feature_names, category_map)


    print(x)
    print("=============")
    print(inv_preproc(preproc(x)))

    assert np.all(x == inv_preproc(preproc(x)))


    # ranges = {
    #     0: [-1.0, 1.0],
    #     1: [-0.0, 1.0],
    # }
    #
    # category_map = {2: ['a', 'b', 'c'], 3: ['a', 'b']}
    # immutable_map = {2: False, 3: True}
    # stats = {
    #     0: {"min": 0, "max": 1},
    #     1: {"min": 0, "max": 1}
    # }
    #
    # x_ohe = np.array([
    #     [0.2, 0.3, 0, 0, 1, 0, 1],
    #     [0.3, 0.7, 0, 1, 0, 1, 0],
    # ], dtype=np.float32)
    #
    # cond = generate_condition(x_ohe=x_ohe,
    #                           ranges=ranges,
    #                           category_map=category_map,
    #                           immutable_map=immutable_map,
    #                           conditional=True)
    #
    # size = x_ohe.shape[0]
    # x_ohe_hat_split = [
    #     np.random.rand(size, 2).astype(np.float32),
    #     np.random.rand(size, 3).astype(np.float32),
    #     np.random.rand(size, 2).astype(np.float32)
    # ]
    #
    # # x_ohe_hat_split = [tf.constant(x) for x in x_ohe_hat_split]
    # # x_ohe = tf.constant(x_ohe)
    # #
    # # x_ohe_hat_split = tensorflow_sample_differentiable(x_ohe_hat_split=x_ohe_hat_split,
    # #                                          category_map=category_map)
    # # print(tensorflow_he_sparsity_loss(x_ohe_hat_split=x_ohe_hat_split, x_ohe=x_ohe, category_map=category_map))
    #
    # x_ohe_hat_split = [torch.tensor(x) for x in x_ohe_hat_split]
    # x_ohe = torch.tensor(x_ohe)
    # x_ohe_hat_split = pytorch_sample_differentiable(x_ohe_hat_split=x_ohe_hat_split,
    #                                                 category_map=category_map)
    # print(pytorch_he_sparsity_loss(x_ohe_hat_split, x_ohe, category_map=category_map))