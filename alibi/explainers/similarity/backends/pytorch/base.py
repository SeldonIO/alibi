"""Tensor flow backend for similarity explainers.

Methods unique to the Tensorflow backend are defined here.
"""

import numpy as np
import torch.nn as nn
import torch
from typing import Dict, Any, Callable, Optional, Union


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
    output = model(x)
    loss = loss_fn(output, y)
    loss.backward()
    return np.concatenate([param.grad.detach().numpy().reshape(-1)
                           for param in model.parameters()])


def to_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.tensor(x)
