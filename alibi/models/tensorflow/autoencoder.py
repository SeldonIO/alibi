"""
This module contains a Tensorflow general implementation of an autoencoder, by combining the encoder and the decoder
module. In addition it provides an implementation of a heterogeneous autoencoder which includes a type checking of the
output.
"""

import tensorflow as tf
import tensorflow.keras as keras
from typing import List, Tuple, Union


class AE(keras.Model):
    """
    Autoencoder. Standard autoencoder architecture. The model is composed from two submodules, the encoder and
    the decoder. The forward pass consists of passing the input to the encoder, obtain the input embedding and
    pass the embedding through the decoder. The abstraction can be used for multiple data modalities.
    """

    def __init__(self,
                 encoder: keras.Model,
                 decoder: keras.Model,
                 **kwargs) -> None:
        """
        Constructor. Combine encoder and decoder in AE

        Parameters
        ----------
        encoder
            Encoder network.
        decoder
            Decoder network.
        """
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, x: tf.Tensor, **kwargs) -> Union[tf.Tensor, List[tf.Tensor]]:
        """
        Forward pass.

        Parameters
        ----------
        x
            Input tensor.
        **kwargs
            Other arguments passed to encoder/decoder `call` method.

        Returns
        -------
        x_hat
            Reconstruction of the input tensor.
        """

        z = self.encoder(x, **kwargs)
        x_hat = self.decoder(z, **kwargs)
        return x_hat


class HeAE(AE):
    """
    Heterogeneous autoencoder. The model follows the standard autoencoder architecture and includes and additional
    type check to ensure that the output of the model is a list of tensors. For more details, see
    :py:class:`alibi.models.pytorch.autoencoder.AE`.
    """

    def __init__(self,
                 encoder: keras.Model,
                 decoder: keras.Model,
                 **kwargs) -> None:
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

    def build(self, input_shape: Tuple[int, ...]) -> None:
        """
        Build method.

        Parameters
        ----------
        input_shape
            Tensor's input shape.
        """
        super().build(input_shape)

        # Check if the output is a list
        input = tf.zeros(input_shape)
        output = self.call(input)

        if not isinstance(output, list):
            raise ValueError("The output of HeAE should be a list.")

    def call(self, x: tf.Tensor, **kwargs) -> List[tf.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        x
            Input tensor.
        **kwargs
            Other arguments passed to the encoder/decoder.

        Returns
        --------
        List of reconstruction of the input tensor. First element corresponds to the reconstruction of all the \
        numerical features if they exist, and the rest of the elements correspond to each categorical feature.
        """
        return super().call(x, **kwargs)
