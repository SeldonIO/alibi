import pytest
import torch
import torch.nn as nn
from alibi.models.pytorch.autoencoder import HeAE


@pytest.mark.parametrize('input_dim, latent_dim', [[32, 4]])
def test_heae_output(input_dim, latent_dim):
    """ Tests whether an error is raised if the output of the `HeAE` is not a list."""
    x = torch.randn(1, input_dim)
    encoder = nn.Sequential(
        nn.Linear(input_dim, latent_dim),
        nn.ReLU()
    )
    decoder = nn.Linear(latent_dim, input_dim)
    heae = HeAE(encoder=encoder, decoder=decoder)

    with pytest.raises(ValueError) as err:
        heae(x)

    assert 'The output of HeAE should be list.' in str(err.value)
