import numpy as np
import pytest
from alibi.models.tensorflow.autoencoder import AE, HeAE
from alibi.models.tensorflow.cfrl_models import (ADULTDecoder, ADULTEncoder,
                                                 MNISTClassifier, MNISTDecoder,
                                                 MNISTEncoder)


@pytest.mark.parametrize('output_dim', [10, 20])
def test_mnist_classifier(output_dim):
    """ Tests whether the `MNISTClassifier` has a deterministic output when
    ``training=False`` and if the output shape matches the expectation. """
    x = np.random.randn(1, 28, 28, 1)
    model = MNISTClassifier(output_dim=output_dim)

    output1 = model(x, training=False)
    output2 = model(x, training=False)
    np.testing.assert_allclose(output1, output2)
    assert output1.shape[-1] == output_dim


@pytest.mark.parametrize('latent_dim', [32, 64])
def test_mnist_autoencoder(latent_dim):
    """ Tests whether the latent representation is in [-1, 1] and has the
    expected shape, and the composition of the encoder and decoder modules. """
    x = np.random.randn(1, 28, 28, 1)
    enc = MNISTEncoder(latent_dim=latent_dim)
    dec = MNISTDecoder()
    ae = AE(encoder=enc, decoder=dec)

    z = enc(x)
    assert z.shape[-1] == latent_dim
    assert np.all((z >= -1) & (z <= 1))

    x_hat1 = dec(z)
    assert x.shape == x_hat1.shape

    x_hat2 = ae(x)
    assert x.shape == x_hat2.shape
    np.testing.assert_allclose(x_hat1, x_hat2)


@pytest.mark.parametrize('input_dim, hidden_dim, latent_dim', [(32, 16, 4)])
def test_adult_encoder(input_dim, hidden_dim, latent_dim):
    """ Tests whether the latent representation is [-1, 1] and has the expected shape,
    and if the fully connected hidden layers have the expected shapes."""
    x = np.random.randn(1, input_dim)
    model = ADULTEncoder(hidden_dim=hidden_dim, latent_dim=latent_dim)

    z = model(x)
    assert z.shape[-1] == latent_dim
    assert np.all((z >= -1) & (z <= 1))

    w1, b1 = model.layers[0].weights
    assert w1.shape == (input_dim, hidden_dim)
    assert b1.shape == (hidden_dim, )

    w2, b2 = model.layers[1].weights
    assert w2.shape == (hidden_dim, latent_dim)
    assert b2.shape == (latent_dim, )


@pytest.mark.parametrize('input_dim, hidden_dim, output_dims', [
    [4, 16, [4, 4, 8, 16]]
])
def test_adult_decoder(input_dim, hidden_dim, output_dims):
    """ Tests whether the reconstruction has the expected length and if each output
    head has the expected shape, and if the fully connected hidden layer have the
    expected shapes. """
    z = np.random.randn(1, 4)
    model = ADULTDecoder(hidden_dim=hidden_dim, output_dims=output_dims)

    x_hat = model(z)
    assert len(x_hat) == len(output_dims)
    assert np.all([x_hat[i].shape[-1] == output_dims[i] for i in range(len(output_dims))])

    w1, b1 = model.layers[0].weights
    assert w1.shape == (input_dim, hidden_dim)
    assert b1.shape == (hidden_dim, )

    for i in range(len(output_dims)):
        w, b = model.layers[i + 1].weights
        assert w.shape == (hidden_dim, output_dims[i])
        assert b.shape == (output_dims[i], )


@pytest.mark.parametrize('input_dim, hidden_dim, latent_dim, output_dims', [
    [32, 16, 4, [4, 4, 8, 16]]
])
def test_adult_autoencoder(input_dim, hidden_dim, latent_dim, output_dims):
    """ Tests heterogeneous autoencoder composition. """
    x = np.random.randn(1, input_dim)

    enc = ADULTEncoder(hidden_dim=hidden_dim, latent_dim=latent_dim)
    dec = ADULTDecoder(hidden_dim=hidden_dim, output_dims=output_dims)
    ae = HeAE(encoder=enc, decoder=dec)

    x_hat1 = ae(x)
    x_hat2 = dec(enc(x))
    for x1, x2 in zip(x_hat1, x_hat2):
        np.testing.assert_allclose(x1, x2)
