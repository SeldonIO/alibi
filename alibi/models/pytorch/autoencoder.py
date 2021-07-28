import torch
import torch.nn as nn

from alibi.models.pytorch.model import Model
from typing import List


class AE(Model):
    """ Autoencoder. """

    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 **kwargs):
        """
        Constructor. Combine encoder and decoder in AE.

        Parameters
        ----------
        encoder
            Encoder network
        decoder
            Decoder network
        """
        super().__init__(**kwargs)

        # set encoder and decoder
        self.encoder = encoder
        self.decoder = decoder

        # send to device
        self.to(self.device)

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


class HeAE(AE):
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
        x_hat = super().forward(x)

        # TODO: think of a better way to do the check, or maybe just remove it since return type hints
        if not isinstance(x_hat, list):
            raise ValueError("The output of HeAE should be list.")

        return x_hat
