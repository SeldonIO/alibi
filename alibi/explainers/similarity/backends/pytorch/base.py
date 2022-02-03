"""Tensor flow backend for similarity explainers.

Methods unique to the Tensorflow backend are defined here.
"""

import numpy as np
import torch.nn as nn
import torch
from typing import Dict, Any, Callable, Optional, Union
import random


def get_grads(
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> np.ndarray:
    """
    Computes the gradients of the loss function with respect to the model's parameters for a single training and target
    pair.

    Parameters:
    -----------
    model: keras.Model
        The model to compute gradients for.
    x: torch.Tensor
        The input data.
    y: torch.Tensor
        The target data.
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        The loss function to use.
    """
    for param in model.parameters():
        param.grad.data.zero_()

    output = model(x)
    loss = loss_fn(output, y)
    loss.backward()
    return np.concatenate([to_numpy(param.grad).reshape(-1)
                           for param in model.parameters()])


def to_tensor(x: np.ndarray) -> torch.Tensor:
    # TODO: align with CFRL backend
    return torch.tensor(x)


def to_numpy(x: torch.Tensor) -> np.ndarray:
    # TODO: align with CFRL backend
    return x.detach().numpy()


def argmax(x: torch.Tensor) -> torch.Tensor:
    return torch.argmax(x, dim=1)


def set_seed(seed: int = 13):
    # TODO: align with CFRL backend
    """
    Sets a seed to ensure reproducibility

    Parameters
    ----------
    seed
        seed to be set
    """
    # Others
    np.random.seed(seed)
    random.seed(seed)

    # Torch related
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
