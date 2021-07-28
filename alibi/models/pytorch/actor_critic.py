import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """ Actor network """

    def __init__(self, hidden_dim: int, output_dim: int) -> None:
        """
        Constructor.

        Parameters
        ----------
        hidden_dim
            Hidden dimension.
        output_dim
            Output dimension
        """
        super().__init__()
        self.fc1 = nn.LazyLinear(hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.LazyLinear(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.LazyLinear(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = torch.tanh(self.fc3(x))
        return x


class Critic(nn.Module):
    """ Critic network """

    def __init__(self, hidden_dim: int):
        """
        Constructor.

        Parameters
        ----------
        hidden_dim
            Hidden dimension.
        """
        super().__init__()
        self.fc1 = nn.LazyLinear(hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.LazyLinear(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.LazyLinear(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.fc3(x)
        return x
