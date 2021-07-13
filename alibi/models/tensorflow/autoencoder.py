import tensorflow as tf
import tensorflow.keras as keras


class EncoderAE(keras.layers.Layer):
    def __init__(self,
                 encoder_net: tf.keras.Model,
                 name: str = 'encoder_ae') -> None:
        """
        Encoder AE

        Parameters
        ----------
        encoder_net
            Layers for the encoder wrapped in a tf.keras.Model class.
        name
            Name of encoder.
        """
        super().__init__(name=name)
        self.encoder_net = encoder_net

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return self.encoder_net(x)


class DecoderAE(keras.layers.Layer):
    def __init__(self,
                 decoder_net: tf.keras.Model,
                 name: str = 'decoder_ae') -> None:
        """
        Decoder AE

        Parameters
        ----------
        decoder_net
            Layers for the decoder wrapped in a tf.keras.Model class.
        name
            Name of the decoder.
        """
        super().__init__(name=name)
        self.decoder_net = decoder_net

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return self.decoder_net(x)


class AE(tf.keras.Model):
    def __init__(self,
                 encoder_net: tf.keras.Model,
                 decoder_net: tf.keras.Model,
                 name: str = 'ae') -> None:
        """
        Combine  encoder and decoder in AE

        Parameters
        ----------
        encoder_net
            Layers for the encoder wrapped in a tf.keras.Model class.
        decoder_net
            Layers for the decoder wrapped in a tf.keras.Model class.
        name
            Name of auto-encoder model.
        """
        super().__init__(name=name)
        self.encoder = EncoderAE(encoder_net)
        self.decoder = DecoderAE(decoder_net)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
