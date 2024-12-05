import numpy as np
import pytest
import tensorflow.keras as keras
from alibi.models.tensorflow.autoencoder import HeAE


@pytest.mark.parametrize('input_dim, latent_dim', [[32, 4]])
def test_heae_output(input_dim, latent_dim):
    """ Tests whether an error is raised if the output of the `HeAE` is not a list."""
    x = np.random.randn(1, input_dim)

    encoder = keras.layers.Dense(latent_dim, activation='tanh')
    decoder = keras.layers.Dense(input_dim)
    heae = HeAE(encoder=encoder, decoder=decoder)

    with pytest.raises(ValueError) as err:
        heae(x)

    assert "The output of HeAE should be a list." == err.value.args[0]
