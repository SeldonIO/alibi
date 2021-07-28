from alibi.explainers.backends.cfrl_tabular import split_ohe, generate_condition  # noqa: F401
from alibi.explainers.backends.tflow.cfrl_base import get_actor, get_critic, get_optimizer, data_generator, \
    encode, decode, generate_cf, update_actor_critic, add_noise, to_numpy, to_tensor  # noqa: F403, F401

import torch
import torch.nn.functional as F
from typing import List, Dict


def sample_differentiable(x_ohe_hat_split: List[torch.Tensor],
                          category_map: Dict[int, List[str]]) -> List[torch.Tensor]:
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


def l0_ohe(input: torch.Tensor,
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


def l1_loss(input: torch.Tensor, target: torch.Tensor, reduction: str = 'none') -> torch.Tensor:
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


def sparsity_loss(x_ohe_hat_split: List[torch.Tensor],
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
    x_ohe_num_split, x_ohe_cat_split = split_ohe(x_ohe=x_ohe,
                                                 category_map=category_map)

    # sample differentiable output
    x_ohe_hat_split = sample_differentiable(x_ohe_hat_split=x_ohe_hat_split,
                                            category_map=category_map)

    # define numerical and categorical loss
    num_loss, cat_loss = torch.tensor(0.), torch.tensor(0.)
    offset = 0

    # compute numerical loss
    if len(x_ohe_num_split) > 0:
        offset = 1
        num_loss = torch.mean(l1_loss(input=x_ohe_hat_split[0],
                                      target=x_ohe_num_split[0],
                                      reduction='none'))

    # compute categorical loss
    if len(x_ohe_cat_split) > 0:
        for i in range(len(x_ohe_cat_split)):
            batch_size = x_ohe_hat_split[i].shape[0]
            cat_loss += torch.sum(l0_ohe(input=x_ohe_hat_split[i + offset],
                                         target=x_ohe_cat_split[i],
                                         reduction='none')) / batch_size

        cat_loss /= len(x_ohe_cat_split)

    return {"num_loss": weight_num * num_loss, "cat_loss": weight_cat * cat_loss}


def consistency_loss(z_cf_pred: torch.Tensor, z_cf_tgt: torch.Tensor, **kwargs):
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
