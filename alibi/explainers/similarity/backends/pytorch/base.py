"""`pytorch` backend for similarity explainers.

Methods unique to the `pytorch` backend are defined here. The interface this class defines syncs with the `tensorflow`
backend in order to ensure that the similarity methods only require to match this interface.
"""

from typing import Callable, Union, Optional

import numpy as np
import torch.nn as nn
import torch


class _PytorchBackend:
    device: Optional[torch.device] = None  # device used by `pytorch` backend

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

        Parameters
        ----------
        model
            The model to compute gradients for.
        X
            The input data.
        Y
            The target data.
        loss_fn
            The loss function to use.

        Returns
        -------
        grads
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
        return np.concatenate([_PytorchBackend.to_numpy(param.grad).reshape(-1)
                               for param in model.parameters()])

    @staticmethod
    def to_tensor(X: np.ndarray) -> torch.Tensor:
        """Converts a `numpy` array to a `pytorch` tensor and assigns to the backend device."""
        return torch.tensor(X).to(_PytorchBackend.device)

    @staticmethod
    def set_device(device: Union[str, int, torch.device, None] = None) -> None:
        """Sets the device to use for the backend.

        Allows the device used by the framework to be set using string, integer or device object directly. This is so
        users can follow the pattern recommended in
        https://pytorch.org/blog/pytorch-0_4_0-migration-guide/#writing-device-agnostic-code for writing
        device-agnostic code.
        """
        if isinstance(device, (int, str)):
            _PytorchBackend.device = torch.device(device)
        elif isinstance(device, torch.device):
            _PytorchBackend.device = device
        elif device is not None:
            raise TypeError(("`device` must be a None, string, integer or "
                            f"torch.device object. Got {type(device)} instead."))

    @staticmethod
    def to_numpy(X: torch.Tensor) -> np.ndarray:
        """Maps a `pytorch` tensor to a `numpy` array."""
        return X.detach().cpu().numpy()

    @staticmethod
    def argmax(X: torch.Tensor, dim=-1) -> torch.Tensor:
        """Returns the index of the maximum value in a tensor."""
        return torch.argmax(X, dim=dim)
