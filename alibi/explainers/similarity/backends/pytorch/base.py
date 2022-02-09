"""Torch backend for similarity explainers.

Methods unique to the Torch backend are defined here. The interface this class defines syncs with the torch backend in
order to ensure that the similarity methods only require to match this interface.
"""

import random
from typing import Callable

import numpy as np
import torch.nn as nn
import torch


class TorchBackend(object):
    device = None

    @staticmethod
    def _get_grads(
            model: nn.Module,
            x: torch.Tensor,
            y: torch.Tensor,
            loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> np.ndarray:
        """
        Computes the gradients of the loss function with respect to the model's parameters for a single training and
        target pair.

        Parameters:
        -----------
        model: torch.nn.Module
            The model to compute gradients for.
        x: torch.Tensor
            The input data.
        y: torch.Tensor
            The target data.
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
            The loss function to use.

        Returns:
        --------
        grads: np.ndarray
            The gradients of the loss function with respect to the model's parameters. This is returned as a flattened
            array.
        """

        for param in model.parameters():
            if isinstance(param.grad, torch.Tensor):
                param.grad.data.zero_()

        output = model(x)
        loss = loss_fn(output, y)
        loss.backward()
        return np.concatenate([TorchBackend._to_numpy(param.grad).reshape(-1)
                               for param in model.parameters()])

    @staticmethod
    def _to_tensor(x: np.ndarray) -> torch.Tensor:
        """
        Converts a numpy array to a torch tensor and assigns to the backend device.
        """
        return torch.tensor(x).to(TorchBackend.device)

    @staticmethod
    def _set_device(device: str = 'cpu') -> None:
        """Sets the device to use for the backend.

        Sets te device value on the class. Any subsequent calls to the backend will use this device.
        """
        TorchBackend.device = torch.device(device)

    @staticmethod
    def _to_numpy(x: torch.Tensor) -> np.ndarray:
        """Maps a torch tensor to a numpy array."""
        return x.detach().numpy()

    @staticmethod
    def _argmax(x: torch.Tensor) -> torch.Tensor:
        """Returns the index of the maximum value in a tensor."""
        return torch.argmax(x, dim=1)

    @staticmethod
    def _set_seed(seed: int = 13):
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
