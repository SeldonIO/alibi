"""
This module contains a Pytorch general implementation of an autoencoder, by combining the encoder and the decoder
module. In addition it provides an implementation of a heterogeneous autoencoder which includes a type checking of the
output.
"""

import torch
import torch.nn as nn

from alibi.models.pytorch.model import Model
from typing import List, Union


class AE(Model):
    """
    Autoencoder. Standard autoencoder architecture. The model is composed from two submodules, the encoder and
    the decoder. The forward pass consist of passing the input to the encoder, obtain the input embedding and
    pass the embedding through the decoder. The abstraction can be used for multiple data modalities.
    """

    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 **kwargs):
        """
        Constructor. Combine encoder and decoder in AE.

        Parameters
        ----------
        encoder
            Encoder network.
        decoder
            Decoder network.
        """
        super().__init__(**kwargs)

        # set encoder and decoder
        self.encoder = encoder
        self.decoder = decoder

        # send to device
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        x_hat
            Reconstruction of the input tensor.
        """
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


class HeAE(AE):
    """
    Heterogeneous autoencoder. The model follows the standard autoencoder architecture and includes and additional
    type check to ensure that the output of the model is a list of tensors. For more details, see
    :py:class:`alibi.models.pytorch.autoencoder.AE`.
    """

    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 **kwargs):
        """
        Constructor. Combine encoder and decoder in HeAE.

        Parameters
        ----------
        encoder
            Encoder network.
        decoder
            Decoder network.
        """
        super().__init__(encoder=encoder, decoder=decoder, **kwargs)

        # send to device
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        List of reconstruction of the input tensor. First element corresponds to the reconstruction of all the \
        numerical features if they exist, and the rest of the elements correspond to each categorical feature.

        """
        x_hat = super().forward(x)

        # TODO: think of a better way to do the check, or maybe just remove it since return type hints
        if not isinstance(x_hat, list):
            raise ValueError("The output of HeAE should be list.")

        return x_hat
