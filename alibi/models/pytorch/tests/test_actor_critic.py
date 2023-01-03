import pytest
import torch
from alibi.models.pytorch.actor_critic import Actor, Critic


@pytest.mark.parametrize('input_dim, hidden_dim, output_dim', [
    (8, 32, 16),
    (16, 32, 16)
])
def test_actor(input_dim, hidden_dim, output_dim):
    """ Tests whether the actor network output is in [-1, 1] and if the fully
    connected hidden layers have the expected shapes. """
    x = torch.randn(1, input_dim)
    model = Actor(hidden_dim=hidden_dim, output_dim=output_dim)

    output = model(x)
    assert torch.all((-1 <= output) & (output <= 1))

    layers = list(model.children())
    w1, b1 = layers[0].weight, layers[0].bias
    assert w1.shape == (hidden_dim, input_dim)
    assert b1.shape == (hidden_dim, )

    w2, b2 = layers[2].weight, layers[2].bias
    assert w2.shape == (hidden_dim, hidden_dim)
    assert b2.shape == (hidden_dim, )

    w3, b3 = layers[4].weight, layers[4].bias
    assert w3.shape == (output_dim, hidden_dim)
    assert b3.shape == (output_dim, )


@pytest.mark.parametrize('input_dim, hidden_dim', [(8, 16), (16, 32)])
def test_critic(input_dim, hidden_dim):
    """ Tests if the critic fully connected hidden layers have the expected shapes
    and if the output shape is 1. """
    x = torch.randn(1, input_dim)
    model = Critic(hidden_dim=hidden_dim)

    output = model(x)
    assert output.shape[-1] == 1

    layers = list(model.children())
    w1, b1 = layers[0].weight, layers[0].bias
    assert w1.shape == (hidden_dim, input_dim)
    assert b1.shape == (hidden_dim, )

    w2, b2 = layers[2].weight, layers[2].bias
    assert w2.shape == (hidden_dim, hidden_dim)
    assert b2.shape == (hidden_dim, )

    w3, b3 = layers[4].weight, layers[4].bias
    assert w3.shape == (1, hidden_dim)
    assert b3.shape == (1, )
