"""Torch backend for similarity explainers.

Methods unique to the Torch backend are defined here. The interface this class defines syncs with the torch backend in
order to ensure that the similarity methods only require to match this interface.
"""

import random
from typing import Callable

import numpy as np
import torch.nn as nn
import torch


class _TorchBackend(object):
    device = None

    @staticmethod
    def get_grads(
            model: nn.Module,
            X: torch.Tensor,
            Y: torch.Tensor,
            loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> np.ndarray:
        """
        Computes the gradients of the loss function with respect to the model's parameters for a single training and
        target pair.

        Parameters:
        -----------
        model:
            The model to compute gradients for.
        X:
            The input data.
        Y:
            The target data.
        loss_fn:
            The loss function to use.

        Returns:
        --------
        grads:
            The gradients of the loss function with respect to the model's parameters. This is returned as a flattened \
            array.
        """

        model.zero_grad()
        initial_model_state = model.training
        model.train(False)
        output = model(X)
        loss = loss_fn(output, Y)
        loss.backward()
        model.train(initial_model_state)
        return np.concatenate([_TorchBackend.to_numpy(param.grad).reshape(-1)
                               for param in model.parameters()])

    @staticmethod
    def to_tensor(X: np.ndarray) -> torch.Tensor:
        """Converts a `numpy` array to a `torch` tensor and assigns to the backend device."""
        return torch.tensor(X).to(_TorchBackend.device)

    @staticmethod
    def set_device(device: str = 'cpu') -> None:
        """Sets the device to use for the backend.

        Sets te device value on the class. Any subsequent calls to the backend will use this device.
        """
        _TorchBackend.device = torch.device(device)

    @staticmethod
    def to_numpy(X: torch.Tensor) -> np.ndarray:
        """Maps a `torch` tensor to a `numpy` array."""
        return X.detach().numpy()

    @staticmethod
    def argmax(X: torch.Tensor) -> torch.Tensor:
        """Returns the index of the maximum value in a tensor."""
        return torch.argmax(X, dim=1)

    @staticmethod
    def set_seed(seed: int = 13):
        """Sets a seed to ensure reproducibility

        Parameters
        ----------
        seed:
            `seed` to be set
        """
        # Others
        np.random.seed(seed)
        random.seed(seed)

        # Torch related
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
