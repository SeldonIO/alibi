import pytest
import torch
from alibi.models.pytorch import AE, HeAE
from alibi.models.pytorch.cfrl_models import (ADULTDecoder, ADULTEncoder,
                                              MNISTClassifier, MNISTDecoder,
                                              MNISTEncoder)


@pytest.mark.parametrize('output_dim', [10, 20])
def test_mnist_classifier(output_dim):
    """ Tests whether the `MNISTClassifier` has a deterministic output when is in the
     evaluation mode and if the output shape matches the expectation. """
    x = torch.randn(1, 1, 28, 28)
    model = MNISTClassifier(output_dim=output_dim)

    model.eval()
    output1 = model(x)
    output2 = model(x)
    assert torch.allclose(output1, output2)
    assert output1.shape[-1] == output_dim


@pytest.mark.parametrize('latent_dim', [32, 64])
def test_mnist_autoencoder(latent_dim):
    """ Tests whether the latent representation is in [-1, 1] and has the
    expected shape, and the composition of the encoder and decoder modules. """
    x = torch.randn(1, 1, 28, 28)
    enc = MNISTEncoder(latent_dim=latent_dim)
    dec = MNISTDecoder(latent_dim=latent_dim)
    ae = AE(encoder=enc, decoder=dec)

    z = enc(x)
    assert z.shape[-1] == latent_dim
    assert torch.all((z >= -1) & (z <= 1))

    x_hat1 = dec(z)
    assert x.shape == x_hat1.shape

    x_hat2 = ae(x)
    assert x.shape == x_hat2.shape
    assert torch.allclose(x_hat1, x_hat2)


@pytest.mark.parametrize('input_dim, hidden_dim, latent_dim', [(32, 16, 4)])
def test_adult_encoder(input_dim, hidden_dim, latent_dim):
    """ Tests whether the latent representation is [-1, 1] and has the expected shape,
    and if the fully connected hidden layers have the expected shapes."""
    x = torch.randn(1, input_dim)
    model = ADULTEncoder(hidden_dim=hidden_dim, latent_dim=latent_dim)

    z = model(x)
    assert z.shape[-1] == latent_dim
    assert torch.all((z >= -1) & (z <= 1))

    layers = list(model.children())
    w1, b1 = layers[0].weight, layers[0].bias
    assert w1.shape == (hidden_dim, input_dim)
    assert b1.shape == (hidden_dim, )

    w2, b2 = layers[1].weight, layers[1].bias
    assert w2.shape == (latent_dim, hidden_dim)
    assert b2.shape == (latent_dim, )


@pytest.mark.parametrize('input_dim, hidden_dim, output_dims', [
    [4, 16, [4, 4, 8, 16]]
])
def test_adult_decoder(input_dim, hidden_dim, output_dims):
    """ Tests whether the reconstruction has the expected length and if each output
    head has the expected shape, and if the fully connected hidden layer have the
    expected shapes. """
    z = torch.randn(1, input_dim)
    model = ADULTDecoder(hidden_dim=hidden_dim, output_dims=output_dims)

    x_hat = model(z)
    assert len(x_hat) == len(output_dims)
    assert all([x_hat[i].shape[-1] == output_dims[i] for i in range(len(output_dims))])

    layers = list(model.children())
    w1, b1 = layers[0].weight, layers[0].bias
    assert w1.shape == (hidden_dim, input_dim)
    assert b1.shape == (hidden_dim, )

    for i in range(len(output_dims)):
        w, b = layers[1][i].weight, layers[1][i].bias
        assert w.shape == (output_dims[i], hidden_dim)
        assert b.shape == (output_dims[i], )


@pytest.mark.parametrize('input_dim, hidden_dim, latent_dim, output_dims', [
    [32, 16, 4, [4, 4, 8, 16]]
])
def test_adult_autoencoder(input_dim, hidden_dim, latent_dim, output_dims):
    """ Tests heterogeneous autoencoder composition. """
    x = torch.randn(1, input_dim)

    enc = ADULTEncoder(hidden_dim=hidden_dim, latent_dim=latent_dim)
    dec = ADULTDecoder(hidden_dim=hidden_dim, output_dims=output_dims)
    ae = HeAE(encoder=enc, decoder=dec)

    x_hat1 = ae(x)
    x_hat2 = dec(enc(x))
    for x1, x2 in zip(x_hat1, x_hat2):
        assert torch.allclose(x1, x2)
