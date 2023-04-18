"""`pytorch` backend for similarity explainers.

Methods unique to the `pytorch` backend are defined here. The interface this class defines syncs with the `tensorflow`
backend in order to ensure that the similarity methods only require to match this interface.
"""

from typing import Any, Callable, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn


class _PytorchBackend:
    device: Optional[torch.device] = None  # device used by `pytorch` backend

    @staticmethod
    def get_grads(
            model: nn.Module,
            X: Union[torch.Tensor, List[Any]],
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

        return np.concatenate([_PytorchBackend._grad_to_numpy(grad=param.grad, name=name)
                               for name, param in model.named_parameters()
                               if param.grad is not None])

    @staticmethod
    def _grad_to_numpy(grad: torch.Tensor, name: Optional[str] = None) -> np.ndarray:
        """Convert gradient to `np.ndarray`.

        Converts gradient tensor to flat `numpy` array. If the gradient is a sparse tensor, it is converted to a dense
        tensor first.
        """
        if grad.is_sparse:
            grad = grad.to_dense()

        if not hasattr(grad, 'numpy'):
            name = f' for the named tensor: {name}' if name else ''
            raise TypeError((f'Could not convert gradient to `numpy` array{name}. To ignore these '
                             'gradients in the similarity computation set ``requires_grad=False`` on the '
                             'corresponding parameter.'))
        return grad.reshape(-1).cpu().numpy()

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
            raise TypeError(("`device` must be a ``None``, `string`, `integer` or "
                            f"`torch.device` object. Got {type(device)} instead."))

    @staticmethod
    def to_numpy(X: torch.Tensor) -> np.ndarray:
        """Maps a `pytorch` tensor to `np.ndarray`."""
        return X.detach().cpu().numpy()

    @staticmethod
    def argmax(X: torch.Tensor, dim=-1) -> torch.Tensor:
        """Returns the index of the maximum value in a tensor."""
        return torch.argmax(X, dim=dim)

    @staticmethod
    def _count_non_trainable(model: nn.Module) -> int:
        """Returns number of non trainable parameters.

        Returns the number of parameters that are non trainable. If no trainable parameter exists we raise
        a `ValueError`.
        """

        num_non_trainable_params = len([param for param in model.parameters() if not param.requires_grad])

        if num_non_trainable_params == len(list(model.parameters())):
            raise ValueError("The model has no trainable parameters. This method requires at least "
                             "one trainable parameter to compute the gradients for. "
                             "Try setting ``.requires_grad_(True)`` on the model or one of its parameters.")

        return num_non_trainable_params
