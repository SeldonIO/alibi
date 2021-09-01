"""
This module contains utility functions for the Counterfactual with Reinforcement Learning tabular class (`cfrl_tabular`)
for the Tensorflow backend.
"""

from alibi.explainers.backends.cfrl_tabular import split_ohe, generate_condition  # noqa: F401

# The following methods are included since `alibi.explainers.backends.pytorch.cfrl_tabular` is an extension to the
# `alibi.explainers.backends.pytorch.cfrl_base.py`. In the explainer class `alibi.explainers.cfrl_tabular` the
# access to the backend specific methods is performed through `self.backend` which is of `types.ModuleType`. Since
# some of the methods imported below are common for both data modalities and are access through `self.backend`
# we import them here, without being used explicitly in this module.

from alibi.explainers.backends.tensorflow.cfrl_base import get_actor, get_critic, get_optimizer, data_generator, \
    encode, decode, generate_cf, update_actor_critic, add_noise, to_numpy, to_tensor, set_seed, \
    save_model, load_model, initialize_optimizers, initialize_actor_critic  # noqa: F403, F401

import numpy as np
import tensorflow as tf
from typing import List, Dict, Union


def sample_differentiable(X_hat_split: List[tf.Tensor],
                          category_map: Dict[int, List[str]]) -> List[tf.Tensor]:
    """
    Samples differentiable reconstruction.

    Parameters
    ----------
    X_hat_split
        List of reconstructed columns form the auto-encoder.
    category_map
        Dictionary of category mapping. The keys are column indexes and the values are lists containing the possible
        values for an attribute.

    Returns
    -------
        Differentiable reconstruction.
    """
    num_attr = len(X_hat_split) - len(category_map)
    cat_attr = len(category_map)
    X_out = []

    # Pass numerical attributes as they are
    if num_attr > 0:
        X_out.append(X_hat_split[0])

    # Sample categorical attributes
    if cat_attr > 0:
        for head in X_hat_split[-cat_attr:]:
            out = tf.argmax(head, axis=1)

            # Transform to one-hot encoding
            out = tf.one_hot(out, depth=head.shape[1])
            proba = tf.nn.softmax(head, axis=1)
            out = out - tf.stop_gradient(proba) + proba
            X_out.append(out)

    return X_out


def l0_ohe(input: tf.Tensor,
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
    # Order matters as the gradient of zeros will still flow if reversed order. Maybe consider clipping a bit higher?
    eps = 1e-7 / input.shape[1]
    loss = tf.reduce_sum(tf.maximum(eps + tf.zeros_like(input), target - input), axis=1)

    if reduction == 'none':
        return loss

    if reduction == 'mean':
        return tf.reduce_mean(loss)

    if reduction == 'sum':
        return tf.reduce_sum(loss)

    raise ValueError(f"Reduction {reduction} not implemented.")


def l1_loss(input: tf.Tensor, target=tf.Tensor, reduction: str = 'none') -> tf.Tensor:
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


def sparsity_loss(X_hat_split: List[tf.Tensor],
                  X_ohe: tf.Tensor,
                  category_map: Dict[int, List[str]],
                  weight_num: float = 1.0,
                  weight_cat: float = 1.0):
    """
    Computes heterogeneous sparsity loss.

    Parameters
    ----------
    X_hat_split
        List of reconstructed columns form the auto-encoder.
    X_ohe
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
    # Split the input into a list of tensor, where each element corresponds to a network head
    X_ohe_num_split, X_ohe_cat_split = split_ohe(X_ohe=X_ohe,
                                                 category_map=category_map)

    # Sample differentiable output
    X_ohe_hat_split = sample_differentiable(X_hat_split=X_hat_split,
                                            category_map=category_map)

    # Define numerical and categorical loss
    num_loss, cat_loss = 0., 0.
    offset = 0

    # Compute numerical loss
    if len(X_ohe_num_split) > 0:
        offset = 1
        num_loss = tf.reduce_mean(l1_loss(input=X_ohe_hat_split[0],
                                          target=X_ohe_num_split[0],
                                          reduction='none'))

    # Compute categorical loss
    if len(X_ohe_cat_split) > 0:
        for i in range(len(X_ohe_cat_split)):
            cat_loss += tf.reduce_mean(l0_ohe(input=X_ohe_hat_split[i + offset],
                                              target=X_ohe_cat_split[i],
                                              reduction='none'))

        cat_loss /= len(X_ohe_cat_split)

    return {"sparsity_num_loss": weight_num * num_loss, "sparsity_cat_loss": weight_cat * cat_loss}


def consistency_loss(Z_cf_pred: tf.Tensor, Z_cf_tgt: Union[np.ndarray, tf.Tensor], **kwargs):
    """
    Computes heterogeneous consistency loss.

    Parameters
    ----------
    Z_cf_pred
            Counterfactual embedding prediction.
    Z_cf_tgt
        Counterfactual embedding target.


    Returns
    -------
        Heterogeneous consistency loss.
    """
    # Compute consistency loss
    loss = tf.reduce_mean(tf.square(Z_cf_pred - Z_cf_tgt))
    return {"consistency_loss": loss}
