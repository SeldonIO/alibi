import tensorflow as tf
import tensorflow.keras as keras

from typing import List, Tuple


class AE(keras.Model):
    """ Autoencoder. """

    def __init__(self,
                 encoder: keras.layers.Layer,
                 decoder: keras.layers.Layer,
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

    def call(self, x: tf.Tensor, **kwargs):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


class HeAE(AE):
    """ Heterogeneous autoencoder. """

    def __init__(self,
                 encoder: tf.keras.layers.Layer,
                 decoder: tf.keras.layers.Layer,
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

    def build(self, input_shape: Tuple[int, ...]):
        input = tf.zeros(input_shape)
        output = self.call(input)

        if not isinstance(output, list):
            raise ValueError("The output of HeAE should be a list.")

    def call(self, x: tf.Tensor, **kwargs) -> List[tf.Tensor]:
        return super().call(x, **kwargs)
