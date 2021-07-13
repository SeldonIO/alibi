from alibi.models.pytorch.model import *
from alibi.models.pytorch.metrics import *


class EncoderAE(nn.Module):
    def __init__(self,
                 encoder_net: nn.Module,
                 name: str = 'encoder_ae') -> None:
        """
        Encoder AE

        Parameters
        ----------
        encoder_net
            Layers for encoder wrapped in a nn.Module class.
        name
            Name of encoder.
        """
        super().__init__()
        self.encoder_net = encoder_net
        self.name = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder_net(x)


class DecoderAE(nn.Module):
    def __init__(self,
                 decoder_net: nn.Module,
                 name: str = 'decoder_ae') -> None:
        """
        Decoder AE

        Parameters
        ----------
        decoder_net
            Layers for decoder wrapped in a nn.Module class.
        name
            Name of decoder.
        """
        super().__init__()
        self.decoder_net = decoder_net
        self.name = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder_net(x)


class AE(Model):
    def __init__(self,
                 encoder_net: nn.Module,
                 decoder_net: nn.Module,
                 name: str = 'ae'):
        """
        Combine encoder and decoder in AE

        Parameters
        ----------
        encoder_net
            Layers for encoder wrapped in a nn.Module class.
        decoder_net
            Layers for decoder wrapped in a nn.Module class.
        name
            Name of auto-encoder model.
        """
        super().__init__()
        self.encoder = EncoderAE(encoder_net)
        self.decoder = DecoderAE(decoder_net)
        self.name = name

        # send to device
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

