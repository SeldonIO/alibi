import numpy as np
import pytest
from alibi.models.tensorflow.actor_critic import Actor, Critic


@pytest.mark.parametrize('input_dim, hidden_dim, output_dim', [
    (8, 32, 16),
    (16, 32, 16)
])
def test_actor(input_dim, hidden_dim, output_dim):
    """ Tests whether the actor network output is in [-1, 1] and if the fully
    connected hidden layers have the expected shapes. """
    x = np.random.randn(1, input_dim)
    model = Actor(hidden_dim=hidden_dim, output_dim=output_dim)

    output = model(x)
    assert np.all((-1 <= output) & (output <= 1))

    w1, b1 = model.layers[0].weights
    assert w1.shape == (input_dim, hidden_dim)
    assert b1.shape == (hidden_dim, )

    w2, b2 = model.layers[2].weights
    assert w2.shape == (hidden_dim, hidden_dim)
    assert b2.shape == (hidden_dim, )

    w3, b3 = model.layers[4].weights
    assert w3.shape == (hidden_dim, output_dim)
    assert b3.shape == (output_dim, )


@pytest.mark.parametrize('input_dim, hidden_dim', [(8, 16), (16, 32)])
def test_critic(input_dim, hidden_dim):
    """ Tests if the critic fully connected hidden layers have the expected shapes
    and if the output shape is 1. """
    x = np.random.randn(1, input_dim)
    model = Critic(hidden_dim=hidden_dim)

    output = model(x)
    assert output.shape[-1] == 1

    w1, b1 = model.layers[0].weights
    assert w1.shape == (input_dim, hidden_dim)
    assert b1.shape == (hidden_dim, )

    w2, b2 = model.layers[2].weights
    assert w2.shape == (hidden_dim, hidden_dim)
    assert b2.shape == (hidden_dim, )

    w3, b3 = model.layers[4].weights
    assert w3.shape == (hidden_dim, 1)
    assert b3.shape == (1, )
